// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Queueing requests, and cancelling them before they are planned.

use super::*;

impl Node {
    /// Queues query rows for a table's next step.
    ///
    /// # Errors
    ///
    /// If the table does not exist or has no rows, or the queries have the wrong width.
    pub fn submit(&mut self, table: TableId, queries: &[f32]) -> Result<RequestId, NodeError> {
        self.submit_before(table, queries, None)
    }

    /// Queues query rows that each read only rows from before their cutoff.
    ///
    /// # Errors
    ///
    /// As `submit`, or if there is not one cutoff per query.
    pub fn submit_before(
        &mut self,
        table: TableId,
        queries: &[f32],
        cutoffs: Option<&[i64]>,
    ) -> Result<RequestId, NodeError> {
        let request = RequestId(self.next_request);
        let state = self.state(table)?;
        if state.spec.series.is_some() {
            return Err(ForecastError::OnlyForecasts.into());
        }
        state.check_queries(queries, cutoffs)?;
        if state.table.is_empty() {
            return Err(NodeError::EmptyTable(table));
        }
        let queued = (request, queries.to_vec(), cutoffs.map(<[i64]>::to_vec));
        // A given request is planned alone, so others wait until it is.
        if state.given.is_some() || !state.givens.is_empty() {
            state.later.push(queued);
        } else {
            state.pending.push(queued);
        }
        self.next_request += 1;
        Ok(request)
    }

    /// Queues queries whose contexts were chosen elsewhere: each query reads
    /// exactly its given rows, encoded together, such as rows gathered across
    /// a table's partitions. The request is planned alone: behind the table's
    /// pending requests, if any, and before requests that come after it.
    ///
    /// # Errors
    ///
    /// As `submit`, `Count` if there is not one context per query, or `Given`
    /// if a context names a row the table does not hold, or a query is given
    /// no rows.
    pub fn submit_given(
        &mut self,
        table: TableId,
        queries: &[f32],
        given: Vec<Context>,
    ) -> Result<RequestId, NodeError> {
        let state = self.state(table)?;
        let width = state.spec.features;
        if queries.len() != given.len() * width {
            return Err(NodeError::Count {
                what: Counted::Contexts,
                expected: queries.len() / width,
                got: given.len(),
            });
        }
        // Rows the table holds; plans leave out those they may not read.
        let live = state.table.snapshot().rows();
        let held = given
            .iter()
            .all(|rows| !rows.is_empty() && (rows - &live).is_empty());
        if !held {
            return Err(NodeError::Given);
        }
        state.check_queries(queries, None)?;
        if state.table.is_empty() {
            return Err(NodeError::EmptyTable(table));
        }
        let request = RequestId(self.next_request);
        self.next_request += 1;
        let state = self.state(table)?;
        let pending = (request, queries.to_vec(), None);
        match &mut state.given {
            // Given requests not yet planned join one batch.
            Some(batch) => {
                state.pending.push(pending);
                batch.extend(given);
            }
            None if state.pending.is_empty() => {
                state.pending.push(pending);
                state.given = Some(given);
            }
            // Behind requests of other contexts, until their turn.
            None => state.givens.push_back((request, queries.to_vec(), given)),
        }
        Ok(request)
    }

    /// Queues forecasts of a series table's series, each from its cutoff, or
    /// from the table's latest time plus a step without one, over `horizon`
    /// steps.
    ///
    /// # Errors
    ///
    /// If the table does not exist or holds no series, no series is named, or
    /// there is not one cutoff per series.
    pub fn submit_forecast(
        &mut self,
        table: TableId,
        series: &[u64],
        cutoffs: Option<&[i64]>,
        horizon: NonZeroUsize,
        groups: Option<&[Vec<u64>]>,
    ) -> Result<RequestId, NodeError> {
        let request = RequestId(self.next_request);
        let state = self.state(table)?;
        let max_horizon = state
            .spec
            .series
            .as_ref()
            .ok_or(ForecastError::NotSeries)?
            .max_horizon;
        if horizon > max_horizon {
            return Err(ForecastError::HorizonTooLong {
                horizon: horizon.get(),
                limit: max_horizon.get(),
            }
            .into());
        }
        if series.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if let Some(cutoffs) = cutoffs
            && cutoffs.len() != series.len()
        {
            return Err(NodeError::Count {
                what: Counted::Cutoffs,
                expected: series.len(),
                got: cutoffs.len(),
            });
        }
        if let Some(groups) = groups
            && groups.len() != series.len()
        {
            return Err(NodeError::Count {
                what: Counted::Groups,
                expected: series.len(),
                got: groups.len(),
            });
        }
        let cutoffs = match cutoffs {
            Some(cutoffs) => cutoffs.iter().map(|&cutoff| Some(cutoff)).collect(),
            None => vec![None; series.len()],
        };
        let groups = groups.map(|groups| {
            groups
                .iter()
                .map(|group| group.iter().map(|&member| SeriesKey(member)).collect())
                .collect()
        });
        state.forecasts.push((
            request,
            SeriesRequest {
                series: series.iter().map(|&series| SeriesKey(series)).collect(),
                cutoffs,
                horizon,
                groups,
            },
        ));
        self.next_request += 1;
        Ok(request)
    }

    /// Queues queries that each read exactly the rows of the keys given
    /// them, encoded together, such as rows an external index chose. A
    /// context past the model's trained length warns, or under `Clamp`, keeps
    /// its first keys up to that length, in the order given: an index's
    /// ranking, best first.
    ///
    /// # Errors
    ///
    /// As `submit_given`, or `UnknownKey` if a key names no row.
    pub fn submit_keys(
        &mut self,
        table: TableId,
        queries: &[f32],
        contexts: &[Vec<u128>],
    ) -> Result<RequestId, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let cap = state.spec.clamp().map_or(usize::MAX, |rows| {
            usize::try_from(rows).unwrap_or(usize::MAX)
        });
        let given = contexts
            .iter()
            .map(|keys| {
                let mut rows = BTreeSet::new();
                for &key in keys {
                    let row = state
                        .table
                        .row(RowKey(key))
                        .ok_or(NodeError::UnknownKey(RowKey(key)))?;
                    if rows.len() < cap {
                        rows.insert(row);
                    }
                }
                Ok(rows.into_iter().collect::<Context>())
            })
            .collect::<Result<Vec<Context>, NodeError>>()?;
        self.submit_given(table, queries, given)
    }

    /// Drops a queued request before it is planned; returns whether it was queued.
    pub fn cancel(&mut self, request: RequestId) -> bool {
        self.tables.values_mut().any(|state| {
            let queued = state.pending.len()
                + state.later.len()
                + state.givens.len()
                + state.forecasts.len();
            state.drop_pending(&[request]);
            state.later.retain(|(id, _, _)| *id != request);
            state.givens.retain(|(id, _, _)| *id != request);
            state.forecasts.retain(|(id, _)| *id != request);
            let left = state.pending.len()
                + state.later.len()
                + state.givens.len()
                + state.forecasts.len();
            let cancelled = left < queued;
            state.promote();
            state.generation += u64::from(cancelled);
            cancelled
        })
    }
}
