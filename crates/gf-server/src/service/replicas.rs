// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Copies of a table on other nodes: applying the log entries a router ships,
//! and letting go of a copy whose table lives elsewhere.

use super::frames::invalid;
use super::*;

impl Service {
    /// Applies a stream of log entries from another node; the first message names the table.
    pub(super) async fn replicate(
        &self,
        table: TableName,
        mut stream: Peekable<Streaming<FlightData>>,
        lineage: Lineage,
    ) -> Result<u64, Status> {
        #[derive(Deserialize)]
        struct Sequenced {
            sequence: u64,
            #[serde(default = "last")]
            last: bool,
        }
        fn last() -> bool {
            true
        }
        let mut length = None;
        let mut batch: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut batch_bytes = 0;
        // The parts of an entry read so far, until its last arrives.
        let mut partial: Option<(u64, Vec<u8>)> = None;
        // What the batch and the partial entry hold of the server's budget.
        let mut charge = self.budget.charge();
        loop {
            let next = stream.next().await.transpose()?;
            let mut entry = None;
            if let Some(message) = &next {
                let Sequenced { sequence, last } = serde_json::from_slice(&message.app_metadata)
                    .map_err(|error| invalid(format!("bad replication metadata: {error}")))?;
                let held = partial.as_ref().map_or(0, |(_, bytes)| bytes.len());
                if (held + message.data_body.len()) as u64 > self.budget.limit() {
                    return Err(Status::resource_exhausted(format!(
                        "a log entry is larger than the {} MiB of requests this node holds \
                         at once; raise its request_bytes_in_flight",
                        self.budget.limit() >> 20
                    )));
                }
                charge.grow(message.data_body.len() as u64)?;
                let bytes = match partial.take() {
                    Some((earlier, mut bytes)) if earlier == sequence => {
                        bytes.extend_from_slice(&message.data_body);
                        bytes
                    }
                    Some(_) => return Err(invalid("a log entry ended before its last part")),
                    None => message.data_body.to_vec(),
                };
                if bytes.len() > ENTRY_BYTES {
                    return Err(invalid("a log entry is larger than a log can hold"));
                }
                if last {
                    entry = Some((sequence, bytes));
                } else {
                    partial = Some((sequence, bytes));
                }
            } else if partial.is_some() {
                return Err(invalid("the stream ended inside a log entry"));
            }
            if let Some((sequence, bytes)) = entry {
                if sequence == 0 {
                    let LogEntry::Definition(definition, terms) =
                        decode(&bytes).map_err(|error| invalid(error.to_string()))?
                    else {
                        return Err(invalid("entry 0 of a log is its definition"));
                    };
                    if definition.name != table {
                        return Err(invalid("the definition names another table"));
                    }
                    if lineage
                        .created
                        .as_ref()
                        .is_some_and(|created| definition.created.as_ref() != Some(created))
                    {
                        return Err(invalid("the definition is not the shipped copy's"));
                    }
                    match self.create(definition, terms).await {
                        Ok(()) => {}
                        Err(error) if error.code() == tonic::Code::AlreadyExists => {}
                        Err(error) => return Err(error),
                    }
                }
                // The shard takes a later term from an existing copy's definition.
                batch_bytes += bytes.len();
                batch.push((sequence, bytes));
            }
            let full = batch.len() >= REPLICATE_BATCH || batch_bytes >= REPLICATE_BATCH_BYTES;
            if !batch.is_empty() && (next.is_none() || full) {
                let (placed, _routed) = self.routed(&table).await?;
                let shard = placed.shard;
                let entries = std::mem::take(&mut batch);
                batch_bytes = 0;
                let name = table.clone();
                let lineage = Box::new(lineage.clone());
                length = Some(
                    self.ask(shard, |reply| Command::Replicate {
                        table: name,
                        entries,
                        lineage,
                        reply,
                    })
                    .await?,
                );
                // The shard has the batch now; the partial entry stays charged.
                charge = self.budget.charge();
                if let Some((_, bytes)) = &partial {
                    charge.grow(bytes.len() as u64)?;
                }
            }
            if next.is_none() {
                break;
            }
        }
        match length {
            Some(length) => Ok(length),
            None => {
                let (placed, _routed) = self.routed(&table).await?;
                let shard = placed.shard;
                let described = self
                    .ask(shard, |reply| Command::Describe { table, reply })
                    .await?;
                Ok(described.log_length)
            }
        }
    }

    /// Stops serving a table here and deletes its log.
    ///
    /// The table leaves the placement first, so calls for it find nothing
    /// while the shard drains it, and comes back if the shard cannot. Its
    /// name stays taken meanwhile, so a creation of it waits until the old
    /// table's log is gone rather than racing it on another shard, and its
    /// route is held, so a move of it finishes first or waits.
    pub(super) async fn release(&self, table: TableName, keep: bool) -> Result<(), Status> {
        let routed = self.route(&table).write_owned().await;
        let (placed, reservation) = {
            let mut placement = self.placement.write().await;
            let Some(placed) = placement.remove(&table) else {
                self.routes
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(table.as_str());
                return Err(Status::not_found(format!("table {table:?} does not exist")));
            };
            let (signal, waiting) = watch::channel(());
            self.creating
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(table.clone(), (placed.shard, waiting));
            let reservation = Reservation {
                creating: self.creating.clone(),
                name: table.clone(),
                _placed: signal,
            };
            (placed, reservation)
        };
        // Apart from this call, so a client that goes away cannot leave a
        // table its shard serves but the placement no longer names.
        let service = self.clone();
        tokio::spawn(async move {
            let name = table.clone();
            let released = service
                .ask(placed.shard, |reply| Command::Release {
                    table: name,
                    keep,
                    reply,
                })
                .await;
            let mut placement = service.placement.write().await;
            if let Err(error) = released {
                placement.insert(table, placed);
                return Err(error);
            }
            service.metrics.tables.set(placement.len() as u64);
            service.metrics.forget(&table);
            if let Some(mover) = &service.mover {
                mover.forget(&table);
            }
            service
                .routes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(table.as_str());
            drop(reservation);
            drop(routed);
            Ok(())
        })
        .await
        .map_err(|error| Status::internal(format!("dropping the table: {error}")))?
    }
}
