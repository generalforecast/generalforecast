// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Predictions queued on the shard: queries submitted against a table,
//! their strings resolved to codes, admitted under the queue's budget,
//! and taken off the queue once answered or abandoned.

use super::*;

impl Shard {
    /// Takes queries whose callers have gone, such as ones that timed out,
    /// out of the queue, so they are neither planned nor counted against it.
    pub(super) fn cancel_abandoned(&mut self) {
        let abandoned: Vec<RequestId> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.reply.is_closed())
            .map(|(&request, _)| request)
            .collect();
        for request in abandoned {
            self.node.cancel(request);
            self.take(request);
        }
    }

    /// Writes the codes of queries' strings into their features: the codes
    /// the table's vocabularies hold, and NaN for words they do not.
    pub(super) fn resolve(
        &self,
        table: &str,
        features: &mut [f32],
        strings: &Strings,
    ) -> Result<(), ServerError> {
        self.entry(table)?;
        let entry = &self.tables[table];
        for (column, values) in strings {
            let vocabulary = entry
                .vocabularies
                .get(*column)
                .ok_or_else(|| ServerError::Internal(format!("no column {column} to code")))?;
            fill(
                features,
                &entry.definition,
                *column,
                &vocabulary.codes(values),
            )?;
        }
        Ok(())
    }

    /// Queues a request's queries with the node, unless its table is
    /// quarantined or the queue is full.
    pub(super) fn submit(
        &mut self,
        table: &str,
        features: Vec<f32>,
        before: Option<Vec<i64>>,
        given: Option<Vec<Vec<u128>>>,
    ) -> Result<Submitted, ServerError> {
        let id = self.usable(table)?;
        self.enqueue(id, table, features, before, given)
    }

    /// Queues queries of a table the node holds, within the queue's budget.
    pub(super) fn enqueue(
        &mut self,
        id: TableId,
        table: &str,
        features: Vec<f32>,
        before: Option<Vec<i64>>,
        given: Option<Vec<Vec<u128>>>,
    ) -> Result<Submitted, ServerError> {
        if let Some(reason) = self.quarantined.get(&id) {
            return Err(ServerError::Invalid(format!(
                "table {table:?} takes no queries: it failed {FAILURES} steps running \
                 ({reason}); a write or refit lets it try again"
            )));
        }
        // Queries and the 16-byte keys of any given contexts.
        let keys = given
            .as_ref()
            .map_or(0, |contexts| contexts.iter().map(Vec::len).sum());
        let bytes = features.len() as u64 * 4 + keys as u64 * 16;
        if self.queued_bytes + bytes > self.queue_bytes {
            self.cancel_abandoned();
        }
        if self.queued_bytes + bytes > self.queue_bytes {
            return Err(ServerError::Busy(format!(
                "{} MiB of queries are waiting already; retry shortly",
                self.queued_bytes >> 20
            )));
        }
        let request = match given {
            Some(contexts) => self.node.submit_keys(id, &features, &contexts)?,
            None => self.node.submit_before(id, &features, before.as_deref())?,
        };
        Ok(Submitted {
            request,
            table: id,
            bytes,
        })
    }

    /// Queues a forecast of series, as `submit` queues queries.
    /// The definition of a table the shard serves, durable or ephemeral.
    pub(super) fn definition(&self, id: TableId, table: &str) -> Option<&TableDefinition> {
        self.ephemeral
            .get(id)
            .map(|ephemeral| &ephemeral.definition)
            .or_else(|| self.tables.get(table).map(|entry| &entry.definition))
    }

    pub(super) fn submit_forecast(
        &mut self,
        id: TableId,
        table: &str,
        series: Vec<u64>,
        before: Option<Vec<i64>>,
        horizon: NonZeroUsize,
        groups: Option<Vec<Vec<u64>>>,
    ) -> Result<Submitted, ServerError> {
        if let Some(reason) = self.quarantined.get(&id) {
            return Err(ServerError::Invalid(format!(
                "table {table:?} takes no forecasts: it failed {FAILURES} steps running \
                 ({reason}); a write lets it try again"
            )));
        }
        let definition = self
            .definition(id, table)
            .ok_or_else(|| ServerError::Internal(format!("table {table:?} has no definition")))?;
        let key = (definition.model.clone(), definition.model_features());
        if let Some(profile) = self.profiles.get(&key) {
            let limit = horizon_limit(
                profile_steps(profile.output_patch, "output patch")?,
                profile
                    .pass_limit
                    .map(|steps| profile_steps(steps, "pass limit"))
                    .transpose()?,
            )
            .get();
            if horizon.get() > limit {
                return Err(ServerError::Invalid(format!(
                    "a horizon of {horizon} steps is past the {limit} the model {:?} forecasts \
                     in {HORIZON_PASSES} passes",
                    definition.model
                )));
            }
        }
        // Each forecast's answer: a value per step, while it waits.
        let bytes = (series.len() as u64).saturating_mul(horizon.get() as u64) * 8;
        if self.queued_bytes + bytes > self.queue_bytes {
            self.cancel_abandoned();
        }
        if self.queued_bytes + bytes > self.queue_bytes {
            return Err(ServerError::Busy(format!(
                "{} MiB of queries are waiting already; retry shortly",
                self.queued_bytes >> 20
            )));
        }
        let request = self.node.submit_forecast(
            id,
            &series,
            before.as_deref(),
            horizon,
            groups.as_deref(),
        )?;
        Ok(Submitted {
            request,
            table: id,
            bytes,
        })
    }

    /// Takes a request out of the queue, for its answer.
    pub(super) fn take(&mut self, request: RequestId) -> Option<Pending> {
        let pending = self.pending.remove(&request)?;
        self.queued_bytes -= pending.bytes;
        self.ephemeral.unpin(pending.table, Instant::now());
        Some(pending)
    }

    /// Queues a request for its answer, pinning the ephemeral table it reads.
    pub(super) fn queue_pending(
        &mut self,
        submitted: Submitted,
        reply: Reply<Predictions>,
        trace: String,
    ) {
        let Submitted {
            request,
            table,
            bytes,
        } = submitted;
        self.queued_bytes += bytes;
        self.ephemeral.pin(table);
        self.pending.insert(
            request,
            Pending {
                reply,
                trace,
                table,
                bytes,
                arrived: Instant::now(),
            },
        );
    }

    /// Queues a query a table took, or answers why it did not take it.
    pub(super) fn admit(
        &mut self,
        submitted: Result<Submitted, ServerError>,
        reply: Reply<Predictions>,
        trace: String,
    ) {
        match submitted {
            Ok(submitted) => self.queue_pending(submitted, reply, trace),
            Err(error) => {
                let _ = reply.send(Err(error));
            }
        }
    }
}
