// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A table's life on the shard: checked against its model, served on the
//! node and the worker, created with a log or restored from one, and
//! released or handed off to another shard.

use super::*;

impl Shard {
    /// What the worker measures about a model for a feature count, asked once.
    pub(super) async fn profile(
        &mut self,
        model: &str,
        features: usize,
    ) -> Result<ModelProfile, ServerError> {
        let key = (model.to_string(), features);
        if let Some(profile) = self.profiles.get(&key) {
            return Ok(profile.clone());
        }
        let request = WorkerRequest::Profile {
            model: model.to_string(),
            features,
        };
        let reply = self.ask_worker(&request).await?;
        let WorkerReply::Profile(profile) = reply else {
            return Err(ServerError::Internal(format!(
                "the worker answered a profile with {reply:?}"
            )));
        };
        self.profiles.insert(key, profile.clone());
        Ok(profile)
    }

    /// Asks the worker how many rows the fits of tables waiting for them take,
    /// once per model and feature count, and sets them.
    ///
    /// Returns whether any table waited. A worker that cannot say leaves the
    /// table fitting on no more rows than it could before.
    pub(super) async fn measure_fits(&mut self) -> bool {
        let tables = self.node.take_unmeasured();
        for &table in &tables {
            // A scratch table fits as its partition's table does; a request
            // context is an ephemeral table, which `tables` does not hold.
            let Some(name) = self.name_of(table) else {
                continue;
            };
            let Some(definition) = self.definition(table, name) else {
                continue;
            };
            let key = (definition.model.clone(), definition.model_features());
            let limits = if let Some(&limits) = self.fit_limits.get(&key) {
                limits
            } else {
                let request = WorkerRequest::FitLimits {
                    model: key.0.clone(),
                    features: key.1,
                };
                let rows = |rows: Option<u64>| rows.map_or(usize::MAX, |rows| rows as usize);
                match self.ask_worker(&request).await {
                    Ok(WorkerReply::FitLimits {
                        state_rows,
                        fused_rows,
                    }) => {
                        let limits = (rows(state_rows), rows(fused_rows));
                        self.fit_limits.insert(key, limits);
                        limits
                    }
                    reply => {
                        tracing::warn!(
                            model = key.0.as_str(),
                            "the worker could not measure the model's fits ({reply:?}); \
                             they take no more rows than before"
                        );
                        let bound = self
                            .node
                            .spec(table)
                            .map_or(usize::MAX, |spec| spec.sizes.fit_bound);
                        (bound, bound)
                    }
                }
            };
            let _ = self.node.set_fit_limits(table, limits.0, limits.1);
        }
        !tables.is_empty()
    }

    /// Asks the worker whether its model can fit a table of this task and
    /// width, so a table it would refuse is refused before it exists.
    pub(super) async fn check_model(
        &mut self,
        definition: &TableDefinition,
    ) -> Result<(), ServerError> {
        let request = WorkerRequest::Check {
            model: definition.model.clone(),
            classes: definition.classes,
            features: definition.model_features(),
            series: definition
                .series
                .as_ref()
                .map_or(1, |series| match series.group {
                    crate::definition::GroupSpec::Alone => 1,
                    crate::definition::GroupSpec::Random { size, .. }
                    | crate::definition::GroupSpec::Given { size } => {
                        usize::try_from(size).unwrap_or(usize::MAX)
                    }
                }),
        };
        match self.ask_worker(&request).await? {
            WorkerReply::Ok => Ok(()),
            WorkerReply::Invalid { message } => Err(ServerError::Invalid(format!(
                "model {:?} cannot serve this table: {message}",
                definition.model
            ))),
            reply => Err(ServerError::Internal(format!(
                "the worker answered a check with {reply:?}"
            ))),
        }
    }

    /// Checks a table against its model and serves it on the node and the
    /// worker, without a log yet; returns its id and prediction width.
    pub(super) async fn serve(
        &mut self,
        definition: &TableDefinition,
        check: bool,
    ) -> Result<(TableId, u64, Option<Vec<f64>>), ServerError> {
        let name = &definition.name;
        if self.tables.contains_key(name) {
            return Err(ServerError::Exists(format!("table {name:?} exists")));
        }
        let (spec, outputs, levels) = self.spec(definition, check).await?;
        let rows = self.directory.join(format!("{}.rows", definition.name));
        let creation = definition
            .created
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let sealer = Arc::new(RowSealer::new(&self.key, &definition.name, &creation));
        let row_ids = definition.config().map_err(ServerError::Invalid)?.row_ids;
        let id = self.node.create_table_on_disk(spec, &rows, sealer)?;
        if let Some(row_ids) = row_ids {
            self.node.set_row_ids(id, row_ids)?;
        }
        self.add_to_worker(id, definition).await?;
        Ok((id, outputs, levels))
    }

    /// Serves a table made for one request alone, in memory and without a log.
    pub(super) async fn serve_scratch(
        &mut self,
        definition: &TableDefinition,
    ) -> Result<TableId, ServerError> {
        let (spec, _, _) = self.spec(definition, false).await?;
        let id = self.node.create_table(spec)?;
        self.add_to_worker(id, definition).await?;
        Ok(id)
    }

    /// Has the worker serve a table the node serves, or takes it off the node.
    pub(super) async fn add_to_worker(
        &mut self,
        id: TableId,
        definition: &TableDefinition,
    ) -> Result<(), ServerError> {
        if let Err(error) = self
            .worker
            .request_within(&add_table(id, definition).frame(), self.step_timeout)
            .await
        {
            let _ = self.node.remove_table(id);
            return Err(ServerError::Internal(error));
        }
        Ok(())
    }

    /// Has the worker serve a table the node serves without waiting for its
    /// answer, so tables made while steps are in flight do not queue behind
    /// them, and tables made in one tick go to the worker in one frame;
    /// `settle_controls` reads the answer.
    pub(super) fn send_to_worker(&mut self, id: TableId, definition: &TableDefinition) {
        let reply = self.worker.pipeline.control(add_table(id, definition));
        self.controls.push_back((Control::Add(id), reply));
    }

    /// Sends the worker the controls this tick queued, in one frame; if the
    /// worker stopped, their replies fail and the restart serves the tables
    /// again.
    pub(super) fn flush_controls(&mut self) {
        if let Err(error) = self.worker.pipeline.flush() {
            tracing::error!("sending tables and tiers' changes to the worker: {error}");
        }
    }

    /// Reads the worker's answers to tables added and removed, in the order
    /// sent, as far as they came. A table the worker refused to add leaves
    /// the shard, failing the requests that wait on it.
    pub(super) async fn settle_controls(&mut self) {
        while let Some((_, reply)) = self.controls.front_mut() {
            let answer = match reply.try_recv() {
                Ok(answer) => answer,
                Err(oneshot::error::TryRecvError::Empty) => break,
                Err(oneshot::error::TryRecvError::Closed) => Err("the worker stopped".to_string()),
            };
            let (control, _) = self.controls.pop_front().expect("found above");
            let Err(error) = answer else {
                continue;
            };
            match control {
                Control::Add(id) => {
                    tracing::error!(table = id.0, "adding a table to the worker: {error}");
                    self.drop_unserved(id, &ServerError::Internal(error)).await;
                }
                Control::Remove(id) => {
                    tracing::error!(table = id.0, "removing a table from the worker: {error}");
                }
            }
        }
    }

    /// A table's spec on the node, checked against its model, with its
    /// prediction width.
    pub(super) async fn spec(
        &mut self,
        definition: &TableDefinition,
        check: bool,
    ) -> Result<(TableSpec, u64, Option<Vec<f64>>), ServerError> {
        self.worker_up()?;
        let model = self
            .worker
            .models
            .iter()
            .position(|candidate| *candidate == definition.model)
            .ok_or_else(|| {
                ServerError::Invalid(format!("model {:?} is not served", definition.model))
            })?;
        let config = definition.config().map_err(ServerError::Invalid)?;
        let settings = definition.series_settings().map_err(ServerError::Invalid)?;
        if let Some(roles) = &definition.roles {
            roles
                .check(definition.classes)
                .map_err(ServerError::Invalid)?;
        }
        if let Some(route) = &definition.route_by {
            route
                .check(&definition.columns)
                .map_err(ServerError::Invalid)?;
        }
        // A table of series may hold targets alone, as a local one may.
        if definition.series.is_none() || !definition.columns.is_empty() {
            columns::check(&definition.columns, MAX_FEATURES).map_err(ServerError::Invalid)?;
        }
        if definition.features() > MAX_FEATURES {
            return Err(ServerError::Invalid(format!(
                "a table's columns become at most {MAX_FEATURES} model features, \
                 each datetime column {}",
                columns::DATETIME_PARTS
            )));
        }
        let profile = self
            .profile(&definition.model, definition.model_features())
            .await?;
        let ModelProfile {
            bytes_per_row,
            representation_bytes: bytes_per_representation,
            regression,
            class_slots,
            outputs,
            context_rows,
            state_bytes,
            categorical_state_bytes,
            column_summary,
            unlabeled_context,
            fit_bound,
            ..
        } = profile;
        let served = &definition.model;
        let own = config.summary == SummaryScope::Context;
        if own && !column_summary {
            return Err(ServerError::Invalid(format!(
                "model {served:?} has no column summary to fit per context; \
                 create the table with the table summary"
            )));
        }
        match (settings.is_some(), profile.series) {
            (true, false) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} is not a forecaster; create the table without series"
                )));
            }
            (false, true) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} is a forecaster; create a table of series with it"
                )));
            }
            _ => {}
        }
        if settings.is_some() && config.task != Task::Values {
            return Err(ServerError::Invalid(
                "a table of series predicts values; create it without classes".to_string(),
            ));
        }
        match (config.task, regression) {
            (Task::Classes(_), true) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} predicts values; create the table without classes"
                )));
            }
            (Task::Values, false) => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} predicts classes; give the table its number of classes"
                )));
            }
            (Task::Classes(1), false) => {
                return Err(ServerError::Invalid(
                    "a classifier tells at least 2 classes apart, not 1 class; a regressor's \
                     table takes none"
                        .to_string(),
                ));
            }
            (Task::Classes(classes), false) if u64::from(classes) > class_slots => {
                return Err(ServerError::Invalid(format!(
                    "model {served:?} serves at most {class_slots} classes, not {classes}"
                )));
            }
            _ => {}
        }
        if check {
            self.check_model(definition).await?;
        }
        if profile.series {
            check_quantiles(profile.quantiles.as_deref(), outputs)?;
        }
        let series = settings
            .map(|settings| -> Result<SeriesSpec, ServerError> {
                let max_rows = profile
                    .max_rows
                    .map(|rows| profile_steps(rows, "positive cap on rows"))
                    .transpose()?;
                let per_member = 1 + definition.model_features();
                let size = settings.group.size();
                if let Some(cap) = max_rows
                    && size > 1
                    && size.saturating_mul(per_member) > cap.get()
                {
                    return Err(ServerError::Invalid(format!(
                        "a group of {size} series takes {} rows with the covariates, past \
                         the model's {cap} per group",
                        size * per_member
                    )));
                }
                Ok(SeriesSpec {
                    step: settings.step,
                    patch: profile_steps(profile.patch, "patch")?,
                    context: profile_steps(profile.context_limit, "context limit")?,
                    max_horizon: horizon_limit(
                        profile_steps(profile.output_patch, "output patch")?,
                        profile
                            .pass_limit
                            .map(|steps| profile_steps(steps, "pass limit"))
                            .transpose()?,
                    ),
                    max_rows,
                    known: settings.known,
                    group: settings.group,
                    neighbors: Neighbors::Targets,
                    reusable: profile.reusable,
                    extendable: profile.extendable,
                    patch_bytes: profile.patch_bytes,
                })
            })
            .transpose()?;
        // A table of series plans its own units: no retrieval, regions or pages.
        let spec = TableSpec {
            model: ModelId(u32::try_from(model).expect("few models")),
            features: definition.features(),
            retrieval: columns::retrieval(&definition.columns),
            categorical: columns::categorical(&definition.columns),
            categories: config.categories,
            unlabeled_context,
            task: config.task,
            page_size: config.page_size,
            strategy: if series.is_some() {
                Strategy::Exact
            } else {
                config.strategy
            },
            time_bucket: config.time_bucket,
            refit: config.refit,
            fit_rows: config.fit_rows,
            context_rows: context_rows.and_then(NonZeroU64::new),
            past_cap: config.past_cap,
            regions: if series.is_some() {
                None
            } else {
                config.regions
            },
            scale_rows: config.scale_rows,
            summary: config.summary,
            sizes: EntrySizes {
                bytes_per_row,
                bytes_per_representation,
                bucket_minimum: self.bucket_minimum,
                state_bytes: if !own {
                    0
                } else if columns::categorical(&definition.columns).is_empty() {
                    state_bytes
                } else {
                    categorical_state_bytes
                },
                state_rows: usize::MAX,
                fused_rows: usize::MAX,
                fit_bound: fit_bound.map_or(usize::MAX, |rows| rows as usize),
            },
            series,
        };
        let outputs = if regression {
            outputs
        } else {
            u64::from(definition.classes)
        };
        // The level of each prediction column: a forecaster's own, a tabular
        // regressor's evenly spaced, and none for a classifier.
        let levels = if !regression {
            None
        } else if let Some(quantiles) = profile.quantiles {
            Some(quantiles)
        } else {
            Some(implied_levels(outputs))
        };
        Ok((spec, outputs, levels))
    }

    /// Stops serving a table on the node and the worker, as `serve` began.
    pub(super) async fn unserve(&mut self, id: TableId) {
        self.recent.remove(&id);
        self.lift_quarantine(id);
        if self.broken.remove(&id).is_some() {
            self.metrics.tables_broken.sub(1);
        }
        if let Ok(dropped) = self.node.remove_table(id) {
            self.send_drops(&dropped).await;
        }
        if self.down.is_some() {
            // The worker starts again with only the tables the shard still serves.
            return;
        }
        let reply = self
            .worker
            .pipeline
            .control(WorkerRequest::RemoveTable { table: id.0 });
        self.controls.push_back((Control::Remove(id), reply));
    }

    /// Serves a new table with a new log under these terms; returns its
    /// quantile levels, if it predicts values.
    pub(super) async fn create(
        &mut self,
        definition: &TableDefinition,
        terms: Terms,
    ) -> Result<Option<Vec<f64>>, ServerError> {
        let (id, outputs, levels) = self.serve(definition, true).await?;
        let path = self.log_path(&definition.name);
        let log = match Log::create(&path, &self.key, definition, terms) {
            Ok(log) => log,
            Err(error) => {
                self.unserve(id).await;
                return Err(error.into());
            }
        };
        self.add_entry(definition, id, outputs, levels.clone(), log, None);
        Ok(levels)
    }

    /// Serves a table from its log at startup, applying each record as it is
    /// read. A table that fails to restore is served nowhere, and its log
    /// stays, or is set aside if it is damaged.
    pub(super) async fn restore(
        &mut self,
        mut replay: Replay,
    ) -> Result<Option<Vec<f64>>, ServerError> {
        let mut definition = replay.definition().clone();
        let (id, outputs, levels) = self.serve(&definition, false).await?;
        let path = self.log_path(&definition.name);
        let mut vocabularies = vec![Vocabulary::default(); vocabularies_of(&definition)];
        let mut series = SeriesKeys::default();
        let mut replayed = Ok(());
        while let Some(record) = replay.next() {
            let record = match record {
                Ok(record) => record,
                Err(error) => {
                    replayed = Err(error);
                    break;
                }
            };
            if let Record::Strategy(strategy) = &record {
                definition.strategy = strategy.clone();
            }
            if let Some(strategy) = snapshot_strategy(&record) {
                take_strategy(&mut definition, strategy);
            }
            learn(
                &mut vocabularies,
                record.words(),
                matches!(record, Record::Snapshot(_)),
            );
            series.track(&record);
            // The worker starts empty, so a refit's drops have nothing to drop.
            if let Err(error) = self.apply_logged(id, record, replay.length()) {
                self.unserve(id).await;
                return Err(error);
            }
        }
        match replayed.and_then(|()| replay.finish()) {
            Ok(log) => {
                self.unsaved_heads.insert(definition.name.clone());
                self.add_entry(
                    &definition,
                    id,
                    outputs,
                    levels.clone(),
                    log,
                    Some(Learned {
                        vocabularies,
                        series,
                    }),
                );
                Ok(levels)
            }
            Err(error) => {
                self.unserve(id).await;
                if error.damaged() {
                    wal::set_aside_damaged(&path, &error);
                }
                Err(error.into())
            }
        }
    }

    pub(super) fn log_path(&self, table: &TableName) -> PathBuf {
        self.directory.join(format!("{table}.log"))
    }

    /// Records a table the node and worker serve, with its log.
    pub(super) fn add_entry(
        &mut self,
        definition: &TableDefinition,
        id: TableId,
        outputs: u64,
        levels: Option<Vec<f64>>,
        log: Log,
        learned: Option<Learned>,
    ) {
        let Learned {
            vocabularies,
            series,
        } = learned.unwrap_or_else(|| Learned {
            vocabularies: vec![Vocabulary::default(); vocabularies_of(definition)],
            series: SeriesKeys::default(),
        });
        self.names.insert(id, definition.name.clone());
        self.tables.insert(
            definition.name.clone(),
            Entry {
                id,
                log,
                definition: definition.clone(),
                outputs,
                levels,
                vocabularies,
                series,
                compact_after: 0,
                compact_wait: COMPACT_RECORDS,
                metrics: self.metrics.table(definition.name.as_str()),
            },
        );
    }

    /// Stops serving a table: its log goes first, so a failure changes
    /// nothing, then the node forgets it, then the worker.
    pub(super) async fn release(
        &mut self,
        table: &TableName,
        keep: bool,
    ) -> Result<(), ServerError> {
        if let Some(compacting) = self.compacting.remove(table) {
            let gone = ServerError::NotFound(format!("table {table:?} was dropped"));
            for queued in compacting.held {
                let _ = queued.reply.send(Err(gone.clone()));
            }
            for reply in compacting.replies {
                let _ = reply.send(Err(gone.clone()));
            }
        }
        let id = self.entry(table)?;
        self.settle().await;
        let term = self.tables[table].log.term().term;
        let path = self.log_path(table);
        let removed = if keep {
            // Not a .log, so restarts leave it alone.
            wal::set_aside(&path, &format!("diverged-{term}")).map(drop)
        } else {
            wal::remove(&path)
        };
        removed.map_err(|error| ServerError::Internal(format!("removing the log: {error}")))?;
        self.tables.remove(table);
        self.unsaved_heads.remove(table);
        self.names.remove(&id);
        self.unserve(id).await;
        Ok(())
    }

    /// Runs every pending and in-flight step to its end, so a table about to
    /// leave has no entries held and no requests waiting.
    pub(super) async fn settle(&mut self) {
        loop {
            while self.dispatch().await {}
            if self.in_flight.is_empty() {
                break;
            }
            self.drain().await;
        }
    }

    /// Stops serving a table whose log another shard will restore: its
    /// requests finish here first, and its log stays as it is.
    pub(super) async fn hand_off(&mut self, table: &TableName) -> Result<(), ServerError> {
        if self.compacting.contains_key(table) {
            return Err(ServerError::Busy(format!("table {table:?} is compacting")));
        }
        let id = self.entry(table)?;
        self.settle().await;
        let entry = self.tables.remove(table).expect("found above");
        if let Err(error) = entry.log.close() {
            tracing::error!(table = table.as_str(), "closing the log's head: {error}");
        }
        self.unsaved_heads.remove(table);
        self.names.remove(&id);
        self.unserve(id).await;
        Ok(())
    }
}
