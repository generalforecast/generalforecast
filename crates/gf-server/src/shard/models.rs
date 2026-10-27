// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Models the worker loads while the shard goes on: an `AddModel` is sent
//! and its answer read when it comes, like a step's, so writes, reads of
//! logs and every other command the worker is not needed for are served
//! while a model loads, however long its weights take.

use super::*;

/// A model the worker was told to load, whose answer is still due, and the
/// callers waiting for it: a second call for the same name waits on the
/// first rather than loading it again.
pub(super) struct Adding {
    name: String,
    spec: String,
    sent: Instant,
    pub(super) reply: oneshot::Receiver<worker::Reply>,
    waiters: Vec<Reply<()>>,
}

impl Shard {
    /// Has the worker load a model it does not serve yet without waiting
    /// for it; `added` answers `reply` once the worker did.
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
        let request = WorkerRequest::Controls {
            requests: vec![WorkerRequest::AddModel {
                name: name.clone(),
                spec: spec.clone(),
            }],
        };
        match self.worker.pipeline.send(&request.frame()).await {
            Ok(answer) => self.adding.push_back(Adding {
                name,
                spec,
                sent: Instant::now(),
                reply: answer,
                waiters: vec![reply],
            }),
            Err(error) => {
                let _ = reply.send(Err(ServerError::Internal(error)));
            }
        }
    }

    /// Takes the worker's answer to the oldest model being loaded: loaded,
    /// it is served from now on and a restarted worker loads it too.
    pub(super) fn added(&mut self, answer: worker::Reply) {
        let Some(adding) = self.adding.pop_front() else {
            return;
        };
        // The worker loads in turn, so a step sent after the model starts now.
        self.finished = Instant::now();
        let name = adding.name;
        let outcome = answer
            .map_err(ServerError::Internal)
            .and_then(|frame| {
                WorkerReply::of(&frame).map_err(|error| ServerError::Internal(error.0))
            })
            .and_then(|reply| match reply {
                WorkerReply::Ok => Ok(()),
                WorkerReply::Controlled { failures } => Err(ServerError::Invalid(format!(
                    "model {name:?} could not load: {}",
                    failures
                        .iter()
                        .map(|failure| failure.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                ))),
                reply => Err(ServerError::Internal(format!(
                    "the worker answered adding a model with {reply:?}"
                ))),
            });
        if outcome.is_ok() {
            self.worker.models.push(name.clone());
            self.worker.launch.config.models.insert(name, adding.spec);
        }
        for waiter in adding.waiters {
            let _ = waiter.send(outcome.clone());
        }
    }

    /// When the oldest model being loaded must have answered: as for a step,
    /// the worker's progress reports, compiles while it loads among them,
    /// move it later.
    pub(super) fn adding_deadline(&self) -> Option<Instant> {
        let progressed = self.worker.pipeline.progressed();
        self.adding.front().map(|adding| {
            adding
                .sent
                .max(self.finished)
                .max(progressed.unwrap_or(adding.sent))
                + self.step_timeout
        })
    }

    /// The model the worker is loading, when its answer is due before the
    /// oldest step's, to say what it hung on.
    pub(super) fn loading(&self) -> Option<&str> {
        let adding = self.adding.front()?;
        let first = self
            .in_flight
            .front()
            .is_none_or(|step| adding.sent <= step.sent);
        first.then_some(adding.name.as_str())
    }

    /// Fails the callers of every model still loading on a worker that
    /// stopped; a later call loads it on the worker started again.
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
