// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Starting a worker again once it stopped or hung, and taking it up once it
//! is back.

use super::*;

impl Shard {
    /// Kills the worker and starts it again in the background: requests in
    /// flight fail, writes go on, and predictions wait until it is back.
    pub(super) async fn restart(&mut self, reason: &str) {
        let index = self.worker.launch.index;
        tracing::error!(worker = index, "{reason}; starting the worker again");
        self.metrics.workers_down.add(1);
        // A worker that stopped answering may still hold its GPU and disk file;
        // killing it also fails the replies the steps in flight wait for.
        self.worker.kill().await;
        self.down = Some(Down {
            retry: Instant::now(),
            delay: Duration::from_secs(1),
            starting: None,
        });
        while let Some(step) = self.in_flight.front_mut() {
            let reply = (&mut step.reply)
                .await
                .unwrap_or_else(|_| Err("the worker stopped".to_string()));
            self.finish(reply).await;
        }
        // The worker forgets every table, so answers still due mean nothing.
        self.controls.clear();
        // The worker starts again with only the shard's own tables, so the
        // scratch and ephemeral tables go, and requests waiting on them fail.
        let ephemeral = self.ephemeral.drain();
        self.metrics.ephemeral_tables.sub(ephemeral.len() as u64);
        for table in &ephemeral {
            self.dropped(table);
        }
        let ephemeral = ephemeral.into_iter().map(|table| table.id);
        let dropped: Vec<TableId> = std::mem::take(&mut self.scratch)
            .into_keys()
            .chain(ephemeral)
            .collect();
        for id in dropped {
            let waiting: Vec<RequestId> = self
                .pending
                .iter()
                .filter(|(_, pending)| pending.table == id)
                .map(|(&request, _)| request)
                .collect();
            for request in waiting {
                self.node.cancel(request);
                if let Some(pending) = self.take(request) {
                    let _ = pending.reply.send(Err(ServerError::Unavailable(
                        "the worker stopped; retry shortly".to_string(),
                    )));
                }
            }
            let _ = self.node.remove_table(id);
        }
        self.node.forget_cache();
    }

    /// Takes a started worker, sizing its pools and adding back every table,
    /// or schedules another start, waiting twice as long, up to a minute.
    pub(super) async fn started(&mut self, started: Result<WorkerConnection, String>) {
        let index = self.worker.launch.index;
        let adopted = match started {
            Ok(worker) => self.adopt(worker).await,
            Err(error) => Err(error),
        };
        let down = self.down.as_mut().expect("starts come while down");
        down.starting = None;
        match adopted {
            Ok(()) => {
                self.down = None;
                self.metrics.workers_down.sub(1);
                tracing::info!(worker = index, "the worker is back");
            }
            Err(error) => {
                tracing::error!(worker = index, "starting the worker again: {error}");
                down.retry = Instant::now() + down.delay;
                down.delay = (down.delay * 2).min(Duration::from_secs(60));
            }
        }
    }

    /// Takes a started worker in place of the one that stopped: sizes its
    /// pools and serves it every table. A worker that cannot take them is
    /// killed before the next start, so it holds neither its device memory
    /// nor its disk file against the worker started after it.
    pub(super) async fn adopt(&mut self, worker: WorkerConnection) -> Result<(), String> {
        if worker.models != self.worker.models {
            return Err(format!(
                "the worker now serves {:?} instead of {:?}",
                worker.models, self.worker.models
            ));
        }
        self.worker = worker;
        let configured = self.configure().await;
        if configured.is_err() {
            self.worker.kill().await;
        }
        configured
    }

    /// Sizes the worker's pools and serves it every table.
    pub(super) async fn configure(&mut self) -> Result<(), String> {
        self.allocate().await?;
        for entry in self.tables.values() {
            self.worker
                .request_within(
                    &add_table(entry.id, &entry.definition).frame(),
                    self.step_timeout,
                )
                .await?;
        }
        Ok(())
    }

    /// Refuses work that needs the worker while it is starting again.
    pub(super) fn worker_up(&self) -> Result<(), ServerError> {
        match self.down {
            Some(_) => Err(ServerError::Unavailable(
                "the worker is starting again; retry shortly".to_string(),
            )),
            None => Ok(()),
        }
    }
}
