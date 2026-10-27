// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The Flight service's calls: writes, exchanges and actions, each checked
//! where it arrives and reaching its table's shard under its route.

use super::frames::{
    Laps, answer, answered, batches, changes, creation_id, descriptor, encoded, exchange_path,
    forecasted, future_column, given_keys, groups_column, horizon_of, invalid, keys, lineage, path,
    prediction_batch, quantile_batch, quantile_levels, query_rows, results, series_column, times,
    write_id,
};
use super::*;

impl Service {
    pub(super) async fn put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<PutResult>>, Status> {
        let replicating = shipment(&request);
        let fence = self.fence(&request)?;
        let scope = Scope::of(&request);
        let id = write_id(&request)?;
        let lineage = lineage(&request)?;
        let missing_ok = request.metadata().contains_key(MISSING_OK);
        let mut stream = request.into_inner().peekable();
        let (command, table) = path(descriptor(&mut stream).await.as_ref())?;
        scope.check(&table)?;
        let command: PutCommand = command.parse()?;
        if command == PutCommand::Replicate {
            replicating?;
            if lineage.created.is_none() {
                return Err(invalid(format!(
                    "a shipment names its source copy's creation in {CREATED}"
                )));
            }
            let log_length = self.replicate(table, stream, lineage).await?;
            let result = PutResult {
                app_metadata: json!({"log_length": log_length}).to_string().into(),
            };
            return Ok(Response::new(Box::pin(stream::iter([Ok(result)]))));
        }
        // Everything that can refuse the call goes before its rows are read.
        self.leased()?;
        let mut laps = self.laps(Instant::now);
        let placement = self.placement(&table).await?;
        let mut route = laps.lap();
        let (batches, charge) = batches(stream, self.budget.charge()).await?;
        let write = match command {
            PutCommand::Insert => Write::Insert(changes(&batches, &placement, id, false)?),
            PutCommand::Update => Write::Update(changes(&batches, &placement, id, false)?),
            PutCommand::Upsert => Write::Upsert(changes(&batches, &placement, id, false)?),
            PutCommand::Label => Write::Label(changes(&batches, &placement, id, true)?),
            PutCommand::Delete => Write::Delete {
                keys: keys(&batches)?,
                missing_ok,
                id,
            },
            PutCommand::Replicate => unreachable!("shipments are applied above"),
        };
        let decode = laps.lap();
        // The shard command must not cross a move; the upload above may. The
        // table's counts are taken under its route, which a drop waits for,
        // so a drop never finds them made again behind it.
        let (placement, _routed) = self.routed(&table).await?;
        let counts = self.metrics.table(&table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let written = self
            .ask(placement.shard, |reply| Command::Write {
                table,
                write,
                fence,
                charge,
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.rows_written.add(written.rows as u64);
        let result = PutResult {
            app_metadata: answer(&written, &placement).to_string().into(),
        };
        let encode = laps.lap();
        laps.observe(&self.metrics.request_encode_seconds, encode);
        Ok(Response::new(Box::pin(stream::iter([Ok(result)]))))
    }

    pub(super) async fn exchange(
        &self,
        request: Request<Streaming<FlightData>>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let scope = Scope::of(&request);
        let levels = quantile_levels(&request)?;
        let horizon = horizon_of(&request)?;
        let internal = replication(&request);
        let mut stream = request.into_inner().peekable();
        let descriptor = descriptor(&mut stream).await;
        let (command, table, body) = exchange_path(descriptor.as_ref())?;
        scope.check(&table)?;
        match command.as_str() {
            "predict" => {}
            "forecast" => {
                return self
                    .forecast(&table, stream, horizon, levels.as_deref(), trace)
                    .await;
            }
            "contexts" => return self.contexts(&table, stream).await,
            "region_churn" => return self.region_churn(&table, stream).await,
            // Rows of a partitioned table travel between nodes only for routers.
            "candidates" => {
                internal?;
                return self.offer(&table, stream, &body).await;
            }
            "predict_given" => {
                internal?;
                return self
                    .predict_given(&table, stream, &body, levels.as_deref(), trace)
                    .await;
            }
            _ => return Err(invalid(format!("unknown exchange {command:?}"))),
        }
        let placement = self.placement(&table).await?;
        if placement.series.is_some() {
            return Err(invalid(
                "a table of series takes forecasts, not queries; use the forecast exchange",
            ));
        }
        if levels.is_some() && placement.task != Task::Values {
            return Err(invalid("quantiles are a regressor's predictions"));
        }
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let before = times(&batches, "before")?;
        let given = given_keys(&batches)?;
        if before.is_some() && given.is_some() {
            return Err(invalid(
                "a query reads either its given rows or rows before its cutoff, not both",
            ));
        }
        let decode = laps.lap();
        let (placement, _routed) = self.routed(&table).await?;
        let counts = self.metrics.table(&table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::Predict {
                table,
                features,
                strings,
                before,
                given,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        self.predicted(&predictions, levels.as_deref(), &placement, laps)
            .await
    }

    /// Forecasts the series a batch names, each from its `before` time if
    /// the batch has one, over the horizon the call or the table gives.
    pub(super) async fn forecast(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        horizon: Option<usize>,
        levels: Option<&[f64]>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let placement = self.placement(table).await?;
        let Some(series) = placement.series else {
            return Err(invalid(
                "the table holds rows to predict from, not series; use the predict exchange",
            ));
        };
        let horizon = horizon
            .or(series.horizon)
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| {
                invalid(format!(
                    "give a horizon of at least one step in the {HORIZON} header, or a \
                     default when creating the table"
                ))
            })?;
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge) = batches(stream, self.budget.charge()).await?;
        let keys = series_column(&batches)?;
        if keys.is_empty() {
            return Err(invalid("no series to forecast"));
        }
        let before = times(&batches, "before")?;
        let groups = groups_column(&batches)?;
        let future = future_column(&batches, &keys, &placement.columns)?;
        let decode = laps.lap();
        let (placement, _routed) = self.routed(table).await?;
        let counts = self.metrics.table(table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::Forecast {
                table: table.clone(),
                series: keys.clone(),
                before,
                horizon,
                groups,
                future,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        let batch = forecasted(
            &predictions,
            &keys,
            series.step,
            horizon,
            levels,
            &placement,
        )?;
        answered(batch, &predictions, laps, Arc::clone(&self.metrics))
    }

    /// The keys of each query's context, as a prediction of them would read
    /// it: one row per query, its keys a list of 16-byte values.
    pub(super) async fn contexts(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let placement = self.placement(table).await?;
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let before = times(&batches, "before")?;
        let (placement, _routed) = self.routed(table).await?;
        let contexts = self
            .ask(placement.shard, |reply| Command::Contexts {
                table: table.clone(),
                features,
                strings,
                before,
                reply,
            })
            .await?;
        let mut keys = ListBuilder::new(FixedSizeBinaryBuilder::new(16));
        for context in contexts {
            for key in context {
                keys.values()
                    .append_value(key.to_le_bytes())
                    .map_err(|error| Status::internal(error.to_string()))?;
            }
            keys.append(true);
        }
        let batch = RecordBatch::try_from_iter([("keys", Arc::new(keys.finish()) as ArrayRef)])
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// Each query's region churn, as `Rebuild::Churn` compares it: one row per
    /// query, NaN where its region has no built context.
    async fn region_churn(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let placement = self.placement(table).await?;
        let (_, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        let (placement, _routed) = self.routed(table).await?;
        let churn = self
            .ask(placement.shard, |reply| Command::RegionChurn {
                table: table.clone(),
                features,
                strings,
                reply,
            })
            .await?;
        let churn: Float64Array = churn
            .into_iter()
            .map(|churn| Some(churn.unwrap_or(f64::NAN)))
            .collect();
        let batch = RecordBatch::try_from_iter([("churn", Arc::new(churn) as ArrayRef)])
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// The rows a partition offers queries, and the queries, as rows travel
    /// between partitions; the body may name the partition's share of a sample.
    async fn offer(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        body: &[u8],
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        #[derive(Default, serde::Deserialize)]
        struct Share {
            share: Option<u64>,
            #[serde(default)]
            strategy: Option<StrategySpec>,
        }
        let share: Share = if body.is_empty() {
            Share::default()
        } else {
            serde_json::from_slice(body).map_err(|error| invalid(error.to_string()))?
        };
        let placement = self.placement(table).await?;
        let (batches, _charge, features, strings) =
            query_rows(stream, &placement.columns, self.budget.charge()).await?;
        // A router asks for one cutoff's queries at a time.
        let before = match times(&batches, "before")? {
            None => None,
            Some(cutoffs) if cutoffs.windows(2).all(|pair| pair[0] == pair[1]) => {
                cutoffs.first().copied()
            }
            Some(_) => return Err(invalid("a partition's candidates take one cutoff")),
        };
        let (placement, _routed) = self.routed(table).await?;
        let (offered, offers) = self
            .ask(placement.shard, |reply| Command::Candidates {
                table: table.clone(),
                features,
                strings,
                share: share.share,
                before,
                strategy: share.strategy.map(|strategy| strategy.strategy()),
                reply,
            })
            .await?;
        let batch = partition::rows_batch(
            &offered,
            &placement.columns,
            placement.task == Task::Values,
            Some(&offers),
        )
        .map_err(|error| Status::internal(error.to_string()))?;
        Ok(encoded(batch))
    }

    /// Predicts the rows without keys against contexts the body gives them
    /// from the rows with keys, gathered across a partitioned table.
    async fn predict_given(
        &self,
        table: &TableName,
        stream: Peekable<Streaming<FlightData>>,
        body: &[u8],
        levels: Option<&[f64]>,
        trace: &Trace,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let started = Instant::now();
        let given: partition::Given =
            serde_json::from_slice(body).map_err(|error| invalid(error.to_string()))?;
        let placement = self.placement(table).await?;
        if levels.is_some() && placement.task != Task::Values {
            return Err(invalid("quantiles are a regressor's predictions"));
        }
        let mut laps = self.laps(|| started);
        let mut route = laps.lap();
        let (batches, _charge) = batches(stream, self.budget.charge()).await?;
        let rows = partition::read_rows(&batches, &placement.columns).map_err(invalid)?;
        let decode = laps.lap();
        let (placement, _routed) = self.routed(table).await?;
        let counts = self.metrics.table(table);
        route += laps.lap();
        laps.observe(&self.metrics.request_route_seconds, route);
        laps.observe(&self.metrics.request_decode_seconds, decode);
        let predictions = self
            .ask(placement.shard, |reply| Command::PredictGiven {
                table: table.clone(),
                rows,
                given,
                trace: trace.id().to_string(),
                reply,
            })
            .await
            .inspect_err(|_| counts.requests_failed.add(1))?;
        let shard = laps.lap();
        laps.observe(&self.metrics.request_shard_seconds, shard);
        counts.predictions.add(predictions.rows as u64);
        counts.predict_seconds.observe(started.elapsed());
        self.predicted(&predictions, levels, &placement, laps).await
    }

    /// Predictions as a Flight response: class probabilities or quantiles,
    /// with their worker time in the schema's metadata.
    async fn predicted(
        &self,
        predictions: &Predictions,
        levels: Option<&[f64]>,
        placement: &Placement,
        laps: Laps,
    ) -> Result<Response<Boxed<FlightData>>, Status> {
        let batch = match levels {
            Some(levels) => quantile_batch(predictions, levels, placement)?,
            None => prediction_batch(predictions, placement.task)?,
        };
        answered(batch, predictions, laps, Arc::clone(&self.metrics))
    }

    pub(super) async fn act(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Boxed<arrow_flight::Result>>, Status> {
        let parsed = FlightAction::parse(request.get_ref())?;
        if parsed.needs_replication() {
            replication(&request)?;
        }
        if parsed.needs_lease() {
            self.leased()?;
        }
        let fence = self.fence(&request)?;
        let scope = Scope::of(&request);
        parsed.check_scope(&scope)?;
        let reply = match parsed {
            FlightAction::Health => results(vec![json!("ok")]),
            FlightAction::ListTables => {
                let mut names: Vec<TableName> = self
                    .placement
                    .read()
                    .await
                    .keys()
                    .filter(|name| scope.allows(name))
                    .cloned()
                    .collect();
                names.sort();
                results(vec![json!(names)])
            }
            FlightAction::ListModels => {
                let mut names = std::collections::BTreeSet::new();
                for shard in 0..self.shards.len() {
                    names.extend(self.ask(shard, |reply| Command::Models { reply }).await?);
                }
                results(vec![json!(names)])
            }
            FlightAction::Locate(_) => {
                return Err(Status::unimplemented("locate is a router's action"));
            }
            FlightAction::Placements => {
                let mut tables: Vec<(TableName, usize)> = self
                    .placement
                    .read()
                    .await
                    .iter()
                    .filter(|(name, _)| scope.allows(name))
                    .map(|(name, placed)| (name.clone(), placed.shard))
                    .collect();
                tables.sort();
                let loads = self.mover.as_ref().map(|mover| {
                    mover
                        .shard_loads
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clone()
                });
                results(vec![json!({
                    "shards": self.shards.len(),
                    "tables": tables.into_iter().collect::<BTreeMap<_, _>>(),
                    "shard_loads": loads,
                    "moves": self.metrics.moves.get(),
                })])
            }
            FlightAction::MoveTable { table, shard } => {
                let seconds = self.move_table(&table, shard).await?;
                results(vec![json!({"seconds": seconds})])
            }
            FlightAction::Rebalance => {
                let round = self.rebalance_round().await?;
                results(vec![json!(round)])
            }
            FlightAction::ProfileWorkers {
                seconds,
                directory,
                trace,
            } => {
                let Some(root) = &self.profiles else {
                    return Err(Status::failed_precondition(
                        "profile_workers needs [profiling] in the server's configuration",
                    ));
                };
                let directory = profile_directory(root, &directory)?;
                let mut files = Vec::new();
                let mut traces = Vec::new();
                for shard in 0..self.shards.len() {
                    let path = directory.join(format!("worker-{shard}.txt"));
                    let path = path.to_string_lossy().into_owned();
                    let trace = trace.then(|| {
                        directory
                            .join(format!("worker-{shard}"))
                            .to_string_lossy()
                            .into_owned()
                    });
                    self.ask(shard, |reply| Command::Record {
                        seconds,
                        path: path.clone(),
                        trace: trace.clone(),
                        reply,
                    })
                    .await?;
                    files.push(path);
                    traces.extend(trace);
                }
                results(vec![
                    json!({"seconds": seconds, "files": files, "traces": traces}),
                ])
            }
            FlightAction::CreateTable(definition) => {
                let split = serde_json::from_slice::<serde_json::Value>(&request.get_ref().body)
                    .ok()
                    .and_then(|body| body.get("partitions").and_then(serde_json::Value::as_u64))
                    .is_some_and(|partitions| partitions != 1);
                if split {
                    return Err(invalid(
                        "a table splits into partitions only through a router",
                    ));
                }
                if crate::engine::is_bound(&definition.name) {
                    return Err(invalid(format!(
                        "table names starting with {:?} are the Models API's own",
                        crate::engine::BOUND_PREFIX
                    )));
                }
                let mut definition = *definition;
                let created = creation_id()?;
                definition.created = Some(created.clone());
                let name = definition.name.clone();
                self.create(definition, Terms::default()).await?;
                let answer = Answer {
                    rows: None,
                    log_length: Some(1),
                    created: Some(created),
                };
                let mut answer = json!(answer);
                answer["table"] = json!(name);
                results(vec![answer])
            }
            FlightAction::Delete(DeleteAction {
                table,
                keys,
                missing_ok,
                write_id,
            }) => {
                // The keys are held until the shard answers, as a put's rows are.
                let mut charge = self.budget.charge();
                charge.grow((keys.len() * std::mem::size_of::<u128>()) as u64)?;
                let name = table.clone();
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Write {
                        table,
                        write: Write::Delete {
                            keys: keys.into_iter().map(|key| key as u128).collect(),
                            missing_ok,
                            id: write_id,
                        },
                        fence,
                        charge,
                        reply,
                    })
                    .await?;
                if let Some(counts) = self.metrics.existing_table(&name) {
                    counts.rows_written.add(written.rows as u64);
                }
                results(vec![answer(&written, &placement)])
            }
            FlightAction::Describe { table, words } => {
                let (
                    Description {
                        definition,
                        rows,
                        log_length,
                        outputs,
                        levels,
                        strategy,
                        context_rows,
                        effective_fit_rows,
                        regions,
                        vocabularies,
                        labels,
                        series,
                    },
                    _,
                ) = self
                    .at_table(table, |table, reply| Command::Describe { table, reply })
                    .await?;
                let words: serde_json::Map<String, serde_json::Value> = definition
                    .columns
                    .iter()
                    .zip(vocabularies)
                    .filter(|(column, _)| words && column.kind == ColumnKind::Categorical)
                    .map(|(column, words)| (column.name.clone(), json!(words)))
                    .collect();
                let mut described = serde_json::to_value(definition)
                    .map_err(|error| Status::internal(error.to_string()))?;
                described["vocabularies"] = json!(words);
                described["labels"] = json!(labels);
                described["rows"] = json!(rows);
                described["log_length"] = json!(log_length);
                described["outputs"] = json!(outputs);
                described["quantile_levels"] = json!(levels);
                described["context_rows"] = json!(context_rows);
                described["resolved_strategy"] = json!(strategy);
                if let Some(rows) = effective_fit_rows {
                    described["effective_fit_rows"] = json!(rows);
                }
                if let Some(count) = series {
                    described["series_count"] = json!(count);
                }
                described["region_stats"] = json!(regions.map(|stats| json!({
                    "regions": stats.regions,
                    "mean_rows": stats.mean_rows,
                    "levels": stats.levels,
                    "hits": stats.hits,
                    "stale": stats.stale,
                    "misses": stats.misses,
                    "fresh": stats.fresh,
                    "rebuilds": stats.rebuilds,
                    "rebuilt_rows": stats.rebuilt_rows,
                    "overdue_seconds": stats.overdue.as_secs_f64(),
                })));
                results(vec![described])
            }
            FlightAction::KnownSeries(table) => {
                let (keys, _) = self
                    .at_table(table, |table, reply| Command::KnownSeries { table, reply })
                    .await?;
                results(vec![json!(keys)])
            }
            FlightAction::Truncate(table) => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Write {
                        table,
                        write: Write::Truncate,
                        fence,
                        charge: Charge::default(),
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::Refit(table) => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::Refit {
                        table,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::SetStrategy { table, strategy } => {
                let (written, placement) = self
                    .at_table(table, |table, reply| Command::SetStrategy {
                        table,
                        strategy,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![answer(&written, &placement)])
            }
            FlightAction::LogLengths => {
                let mut lengths = LogLengths::new();
                for shard in 0..self.shards.len() {
                    for LogLength {
                        table,
                        length,
                        terms,
                        created,
                    } in self
                        .ask(shard, |reply| Command::LogLengths { reply })
                        .await?
                    {
                        if scope.allows(&table) {
                            let copy = CopyLength {
                                length,
                                terms,
                                created,
                            };
                            lengths.insert(table.into(), copy);
                        }
                    }
                }
                results(vec![json!(lengths)])
            }
            FlightAction::Compact(table) => {
                let (log_length, _) = self
                    .at_table(table, |table, reply| Command::Compact {
                        table,
                        fence,
                        reply,
                    })
                    .await?;
                results(vec![json!({"log_length": log_length})])
            }
            FlightAction::DropTable(table) => {
                self.release(table, false).await?;
                results(vec![json!({})])
            }
            FlightAction::Release { table, keep } => {
                self.release(table, keep).await?;
                results(vec![json!({})])
            }
            FlightAction::Promote {
                table,
                term,
                primary,
            } => {
                let (log_length, _) = self
                    .at_table(table, |table, reply| Command::Promote {
                        table,
                        term,
                        primary,
                        reply,
                    })
                    .await?;
                results(vec![json!({"log_length": log_length})])
            }
            FlightAction::Lease(request) => {
                let mut duration = Duration::from_secs_f64(request.seconds.clamp(0.0, 3600.0));
                if let Some(sent) = request.sent_unix_ms {
                    // The lease ends as long after the router sent it, by its clock.
                    let age = Duration::from_millis(unix_ms().saturating_sub(sent));
                    duration = duration.checked_sub(age).ok_or_else(|| {
                        Status::failed_precondition(
                            "a lease request arrived after the lease would have ended; \
                             the node was slow to read it, or its clock and the router's disagree",
                        )
                    })?;
                }
                let lease = self.lease.as_ref();
                let other = lease.and_then(|lease| lease.held_by_another(&request.holder));
                let renewed = match (lease, request.probe) {
                    (Some(lease), false) => lease.renew(&request.holder, duration),
                    _ => other.map_or(Ok(()), Err),
                };
                let answer = LeaseAnswer {
                    granted: renewed.is_ok(),
                    holder: renewed.err(),
                    enforced: lease.is_some(),
                };
                results(vec![json!(answer)])
            }
        };
        Ok(Response::new(reply))
    }
}
