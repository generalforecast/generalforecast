// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Models the worker loads beside its steps. An `AddModel` is answered with
//! `Loading` at once, timed like a step until then; the worker then loads
//! the model on a thread of its own and reports it `Loaded` by name, while
//! the shard goes on serving writes, reads and the steps of the models it
//! serves already, however long the weights take.

use super::*;

/// A model the worker was told to load, whose `Loaded` report is still due,
/// and the callers waiting for it: a second call for the same name waits on
/// the first rather than loading it again.
pub(super) struct Adding {
    name: String,
    spec: String,
    sent: Instant,
    /// The worker's answer to the `AddModel`, until it acknowledged the load.
    pub(super) reply: Option<oneshot::Receiver<worker::Reply>>,
    waiters: Vec<Reply<()>>,
}

impl Shard {
    /// Has the worker load a model it does not serve yet without waiting
    /// for it; `loaded` answers `reply` once the worker did.
    pub(super) async fn add_model(&mut self, name: String, spec: String, reply: Reply<()>) {
        if self.worker.models.contains(&name) {
            let _ = reply.send(Ok(()));
            return;
        }
        if let Some(adding) = self.adding.iter_mut().find(|adding| adding.name == name) {
            adding.waiters.push(reply);
            return;
        }
        if let Err(error) = self.worker_up() {
            let _ = reply.send(Err(error));
            return;
        }
        let request = WorkerRequest::AddModel {
            name: name.clone(),
            spec: spec.clone(),
        };
        match self.worker.pipeline.send(&request.frame()).await {
            Ok(answer) => self.adding.push_back(Adding {
                name,
                spec,
                sent: Instant::now(),
                reply: Some(answer),
                waiters: vec![reply],
            }),
            Err(error) => {
                let _ = reply.send(Err(ServerError::Internal(error)));
            }
        }
    }

    /// Takes the worker's answer to the oldest load not acknowledged: on
    /// `Loading` the load goes on beside the steps, untimed, until its
    /// `Loaded` report; any other answer fails it.
    pub(super) fn acknowledged(&mut self, answer: worker::Reply) {
        let Some(at) = self.adding.iter().position(|adding| adding.reply.is_some()) else {
            return;
        };
        let name = self.adding[at].name.clone();
        let refused = answer
            .map_err(ServerError::Internal)
            .and_then(|frame| {
                WorkerReply::of(&frame).map_err(|error| ServerError::Internal(error.0))
            })
            .and_then(|reply| match reply {
                WorkerReply::Loading => Ok(()),
                reply => Err(ServerError::Internal(format!(
                    "the worker answered adding model {name:?} with {reply:?}"
                ))),
            })
            .err();
        match refused {
            None => self.adding[at].reply = None,
            Some(error) => {
                let adding = self.adding.remove(at).expect("found above");
                for waiter in adding.waiters {
                    let _ = waiter.send(Err(error.clone()));
                }
            }
        }
    }

    /// Takes the worker's reports of models it loaded beside its steps:
    /// loaded, a model is served from now on and a restarted worker loads it
    /// too; failed, its callers hear why.
    pub(super) fn loaded(&mut self) {
        for report in self.worker.pipeline.take_loaded() {
            let Some(at) = self
                .adding
                .iter()
                .position(|adding| adding.name == report.name)
            else {
                tracing::warn!(
                    model = report.name,
                    "the worker loaded a model nothing asked for"
                );
                continue;
            };
            let adding = self.adding.remove(at).expect("found above");
            let outcome = match report.failure {
                None => {
                    self.worker.models.push(adding.name.clone());
                    self.worker
                        .launch
                        .config
                        .models
                        .insert(adding.name.clone(), adding.spec);
                    Ok(())
                }
                Some(failure) => Err(ServerError::Invalid(format!(
                    "model {:?} could not load: {failure}",
                    adding.name
                ))),
            };
            for waiter in adding.waiters {
                let _ = waiter.send(outcome.clone());
            }
        }
    }

    /// When the oldest load the worker has not acknowledged must have been:
    /// as for a step, the worker's progress reports move it later. A load
    /// the worker acknowledged runs beside its steps and is not timed.
    pub(super) fn adding_deadline(&self) -> Option<Instant> {
        let progressed = self.worker.pipeline.progressed();
        self.adding
            .iter()
            .find(|adding| adding.reply.is_some())
            .map(|adding| {
                adding
                    .sent
                    .max(self.finished)
                    .max(progressed.unwrap_or(adding.sent))
                    + self.step_timeout
            })
    }

    /// The model whose load the worker has not acknowledged, when its answer
    /// is due before the oldest step's, to say what it hung on.
    pub(super) fn loading(&self) -> Option<&str> {
        let adding = self.adding.iter().find(|adding| adding.reply.is_some())?;
        let first = self
            .in_flight
            .front()
            .is_none_or(|step| adding.sent <= step.sent);
        first.then_some(adding.name.as_str())
    }

    /// Fails the callers of every model still loading on a worker that
    /// stopped, acknowledged or not; a later call loads it on the worker
    /// started again.
    pub(super) fn fail_adding(&mut self, reason: &str) {
        for adding in std::mem::take(&mut self.adding) {
            let error =
                ServerError::Internal(format!("model {:?} could not load: {reason}", adding.name));
            for waiter in adding.waiters {
                let _ = waiter.send(Err(error.clone()));
            }
        }
    }
}
