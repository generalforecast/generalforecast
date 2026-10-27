// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Request contexts and the tables made for them: ephemeral tables a
//! context's rows are served from, kept in memory by fingerprint and
//! swept once idle, and scratch tables made for one request's given rows.

use super::*;

impl Shard {
    /// The rows a table offers queries, then the queries themselves, as rows
    /// travel between partitions: categorical features as words.
    pub(super) fn offer(
        &mut self,
        table: &str,
        queries: Vec<f32>,
        strings: &Strings,
        share: Option<u64>,
        before: Option<i64>,
        strategy: Option<Strategy>,
    ) -> Result<(ModelRows, Vec<Offer>), ServerError> {
        let id = self.usable(table)?;
        let mut coded = queries.clone();
        self.resolve(table, &mut coded, strings)?;
        let offered = self.node.candidates(id, &coded, share, before, strategy)?;
        let entry = &self.tables[table];
        let width = entry.definition.features();
        let ids: Vec<RowId> = offered.iter().map(|candidate| candidate.row).collect();
        let held = self.node.rows_of(id, &ids)?;
        let count = queries.len() / width.max(1);
        let mut targets: Vec<Option<f32>> = match &held.targets {
            Targets::Classes(classes) => classes.iter().map(|&class| Some(class as f32)).collect(),
            Targets::Values(values) => values.iter().copied().map(Some).collect(),
        };
        targets.resize(ids.len() + count, None);
        let mut rows = ModelRows {
            keys: held.keys.iter().copied().map(Some).collect(),
            features: held.features,
            strings: Vec::new(),
            targets,
        };
        rows.keys.resize(ids.len() + count, None);
        rows.features.extend(&queries);
        for (column, offset) in categorical_offsets(&entry.definition) {
            let words = entry.vocabularies[column].words();
            let asked = strings.iter().find(|(held, _)| *held == column);
            let mut values: Vec<Option<String>> = (0..ids.len())
                .map(|row| {
                    let code = rows.features[row * width + offset];
                    code.is_finite()
                        .then(|| words.get(code as usize).cloned())
                        .flatten()
                })
                .collect();
            values
                .extend((0..count).map(|query| asked.and_then(|(_, words)| words[query].clone())));
            for row in 0..ids.len() + count {
                rows.features[row * width + offset] = f32::NAN;
            }
            rows.strings.push((column, values));
        }
        let offers = offered
            .iter()
            .map(|candidate| Offer {
                query: candidate.query,
                score: candidate.score,
            })
            .chain((0..count).map(|query| Offer {
                query: Some(query as u32),
                score: 0.0,
            }))
            .collect();
        Ok((rows, offers))
    }

    /// Queues queries against the contexts given them, on a table made from
    /// the given rows alone, with `table`'s definition; the table is dropped
    /// once the request leaves.
    pub(super) async fn submit_given(
        &mut self,
        table: &str,
        rows: ModelRows,
        given: Given,
    ) -> Result<Submitted, ServerError> {
        self.usable(table)?;
        let definition = self.tables[table].definition.clone();
        let width = definition.features();
        let mut keys = Vec::new();
        let mut context = Vec::new();
        let mut queries = Vec::new();
        for (row, key) in rows.keys.iter().enumerate() {
            let values = &rows.features[row * width..(row + 1) * width];
            match key {
                Some(key) => {
                    keys.push(*key);
                    context.extend_from_slice(values);
                }
                None => queries.extend_from_slice(values),
            }
        }
        if queries.is_empty() {
            return Err(ServerError::Invalid("no query rows".to_string()));
        }
        let mut held_words = Vec::with_capacity(rows.strings.len());
        let mut asked_words = Vec::with_capacity(rows.strings.len());
        for (column, words) in &rows.strings {
            let (held, asked): (Vec<_>, Vec<_>) = rows
                .keys
                .iter()
                .zip(words)
                .partition(|(key, _)| key.is_some());
            held_words.push((
                *column,
                held.into_iter()
                    .map(|(_, word)| word.clone())
                    .collect::<Vec<_>>(),
            ));
            asked_words.push((
                *column,
                asked
                    .into_iter()
                    .map(|(_, word)| word.clone())
                    .collect::<Vec<_>>(),
            ));
        }
        let targets =
            partition::targets(&rows, definition.classes == 0).map_err(ServerError::Invalid)?;
        // A partition dropped and made again is another creation, which
        // may differ in its model, task or columns.
        let content = (
            definition.name.clone(),
            definition.created.clone(),
            content_hash(&keys, &context, &held_words, &targets),
        );
        // An idle table made from these same rows serves again, caches and all.
        let reused = self
            .scratch
            .iter()
            .find(|(id, scratch)| {
                scratch.content == content
                    && !self.pending.values().any(|pending| pending.table == **id)
            })
            .map(|(&id, _)| id);
        let id = match reused {
            Some(id) => id,
            None => {
                let mut vocabularies = Vec::with_capacity(held_words.len());
                for (column, held) in &held_words {
                    let mut vocabulary = Vocabulary::default();
                    let mut new = Vec::new();
                    let codes = vocabulary
                        .assign(held, &mut new)
                        .map_err(ServerError::Invalid)?;
                    vocabulary.extend(new);
                    fill(&mut context, &definition, *column, &codes)?;
                    vocabularies.push((*column, vocabulary));
                }
                let id = self.serve_scratch(&definition).await?;
                match self.node.insert(id, &keys, &context, targets) {
                    Ok(ids) => {
                        self.scratch.insert(
                            id,
                            Scratch {
                                content,
                                ids,
                                vocabularies,
                                used: 0,
                                metrics: self.metrics.table(definition.name.as_str()),
                            },
                        );
                    }
                    Err(error) => {
                        self.unserve(id).await;
                        return Err(error.into());
                    }
                }
                id
            }
        };
        self.scratch_uses += 1;
        let scratch = self.scratch.get_mut(&id).expect("made or found above");
        scratch.used = self.scratch_uses;
        for ((column, asked), (_, vocabulary)) in asked_words.iter().zip(&scratch.vocabularies) {
            fill(&mut queries, &definition, *column, &vocabulary.codes(asked))?;
        }
        let ids = &scratch.ids;
        let contexts = given
            .contexts
            .iter()
            .map(|positions| {
                positions
                    .iter()
                    .map(|&position| {
                        ids.get(position as usize).copied().ok_or_else(|| {
                            ServerError::Invalid(format!(
                                "a context names row {position} of {}",
                                ids.len()
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let request = self.node.submit_given(id, &queries, contexts)?;
        Ok(Submitted {
            request,
            table: id,
            bytes: queries.len() as u64 * 4,
        })
    }

    /// Queues queries against a request context's ephemeral table, making
    /// the table from the context's rows unless the shard holds it already.
    pub(super) async fn infer(
        &mut self,
        context: RequestContext,
        asked: Asked,
    ) -> Result<(Submitted, Option<Vec<f64>>), ServerError> {
        let RequestContext {
            fingerprint,
            definition,
            keys,
            features,
            targets,
            times,
            shared,
        } = context;
        if definition.durability != Durability::Ephemeral {
            return Err(ServerError::Invalid(
                "a request's context takes an ephemeral table".to_string(),
            ));
        }
        let name = definition.name.clone();
        let id = match self.ephemeral.named(&name) {
            Some(table) if table.definition == definition => table.id,
            Some(_) => {
                return Err(ServerError::Internal(format!(
                    "table {name:?} holds another context"
                )));
            }
            None => {
                self.make_ephemeral(
                    fingerprint,
                    definition,
                    &keys,
                    &features,
                    targets,
                    times.as_deref(),
                    shared,
                )
                .await?
            }
        };
        if let Some((routes, shard)) = &self.routes {
            routes.holds(&fingerprint, *shard);
        }
        let levels = self
            .ephemeral
            .get(id)
            .and_then(|table| table.levels.clone());
        let submitted = match asked {
            Asked::Rows(queries) => self.enqueue(id, name.as_str(), queries, None, None)?,
            Asked::Forecast {
                series,
                before,
                horizon,
                groups,
            } => self.submit_forecast(id, name.as_str(), series, before, horizon, groups)?,
        };
        Ok((submitted, levels))
    }

    /// Serves an ephemeral table in memory, without a log, and adds its rows;
    /// see `RequestContext::shared`.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn make_ephemeral(
        &mut self,
        fingerprint: [u8; 32],
        definition: TableDefinition,
        keys: &[u128],
        features: &[f32],
        targets: Targets,
        times: Option<&[Option<i64>]>,
        shared: bool,
    ) -> Result<TableId, ServerError> {
        let (spec, _, levels) = self.spec(&definition, false).await?;
        let id = self.node.create_table(spec)?;
        if !shared {
            self.node.set_single_use(id)?;
        }
        self.send_to_worker(id, &definition);
        let rows = targets.len() as u64;
        if let Err(error) = self.node.insert_at(id, keys, features, targets, times) {
            self.unserve(id).await;
            return Err(error.into());
        }
        let bytes = keys.len() as u64 * 16 + features.len() as u64 * 4 + rows * 4;
        self.ephemeral.insert(Ephemeral::new(
            id,
            fingerprint,
            definition,
            levels,
            bytes,
            self.metrics.table(EPHEMERAL_METRICS),
        ));
        self.metrics.ephemeral_tables.add(1);
        Ok(id)
    }

    /// Drops ephemeral tables, least recently used first, while more are
    /// kept than the limits allow or they went unused past the idle limit;
    /// tables a request reads are pinned.
    pub(super) async fn sweep_ephemeral(&mut self) {
        for table in self.ephemeral.evict(Instant::now()) {
            self.metrics.ephemeral_tables.sub(1);
            self.dropped(&table);
            self.unserve(table.id).await;
        }
    }

    /// Drops the scratch tables no request needs, past the `SCRATCH_KEPT`
    /// most recently used.
    pub(super) async fn sweep_scratch(&mut self) {
        let mut idle: Vec<(u64, TableId)> = self
            .scratch
            .iter()
            .filter(|(id, _)| !self.pending.values().any(|pending| pending.table == **id))
            .map(|(&id, scratch)| (scratch.used, id))
            .collect();
        idle.sort_unstable_by(|left, right| right.cmp(left));
        for (_, id) in idle.into_iter().skip(SCRATCH_KEPT) {
            self.scratch.remove(&id);
            self.unserve(id).await;
        }
    }

    /// Tells the routes a context's table left this shard.
    pub(super) fn dropped(&self, table: &Ephemeral) {
        if let Some((routes, shard)) = &self.routes {
            routes.dropped(&table.fingerprint, *shard);
        }
    }

    /// Takes an ephemeral table the worker could not serve off the shard,
    /// failing the requests that wait on it.
    pub(super) async fn drop_unserved(&mut self, id: TableId, error: &ServerError) {
        if let Some(table) = self.ephemeral.remove(id) {
            self.metrics.ephemeral_tables.sub(1);
            self.dropped(&table);
        }
        let waiting: Vec<RequestId> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.table == id)
            .map(|(&request, _)| request)
            .collect();
        for request in waiting {
            self.node.cancel(request);
            if let Some(pending) = self.take(request) {
                let _ = pending.reply.send(Err(error.clone()));
            }
        }
        if let Ok(dropped) = self.node.remove_table(id) {
            self.send_drops(&dropped).await;
        }
    }

    /// Whether the next step waits for the one in flight: new contexts
    /// that arrive while it runs go together in the step after it, side by
    /// side, rather than each in a step of its own behind it.
    pub(super) fn holds_new_contexts(&self) -> bool {
        !self.in_flight.is_empty() && self.node.only_new_contexts_wait()
    }
}
