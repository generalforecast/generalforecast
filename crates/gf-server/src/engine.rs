// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The engine: a server's shards, and how a validated request reaches one.
//!
//! A request's context becomes an ephemeral table named for its
//! fingerprint, on the shard the fingerprint routes to, so a repeated
//! context finds its table and cache; a snapshot predictor's rows bind once
//! into a durable table. The Models API and the in-process `_core` engine
//! both answer through it.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use gf_core::columns;
use gf_core::forecasting::{KNOWN, TARGET};
use gf_core::node::Targets;
use gf_core::predictor::{self, Encoder};
use gf_core::request::{self, AlignedGroup, Forecast, Label, Tabular};
use gf_core::route;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use crate::budget::{Charge, RequestBudget};
use crate::definition::TableDefinition;
use crate::error::ServerError;
use crate::http::ApiError;
use crate::name::TableName;
use crate::predictors::ServedPredictor;
use crate::routes::{Busy, Routes};
use crate::shard::{Asked, Command, Fence, Incoming, Inferred, RequestContext, Write};
use crate::wal::{Rows, Terms};

/// A snapshot predictor's rows, bound once into their durable table.
pub struct Bound {
    pub table: TableName,
    /// What names the table's shard.
    pub fingerprint: [u8; 32],
    /// The binding's encoder, which codes queries' words as the rows' were.
    pub encoder: Encoder,
    pub classes: Vec<Label>,
    pub rows: usize,
    pub levels: Option<Vec<f64>>,
}

/// The shards, once their workers are up, and the bounds a call answers within.
pub struct Engine {
    shards: Vec<mpsc::Sender<Command>>,
    routes: Arc<Routes>,
    budget: RequestBudget,
    timeout: Duration,
    /// The shard every new context goes to, if pinned; startup warming
    /// pins each shard in turn so every worker compiles the buckets.
    pinned: Option<usize>,
}

impl Engine {
    /// # Panics
    ///
    /// If `routes` routes over another count of shards.
    #[must_use]
    pub fn new(
        shards: Vec<mpsc::Sender<Command>>,
        routes: Arc<Routes>,
        budget: RequestBudget,
        timeout: Duration,
    ) -> Self {
        assert_eq!(routes.shards(), shards.len(), "routes over the shards");
        Self {
            shards,
            routes,
            budget,
            timeout,
            pinned: None,
        }
    }

    /// The same engine sending every new context to shard `shard`.
    ///
    /// # Panics
    ///
    /// If `shard` is not one of its shards.
    #[must_use]
    pub fn pinned(&self, shard: usize) -> Self {
        assert!(
            shard < self.shards.len(),
            "pinned to a shard the engine has"
        );
        Self {
            pinned: Some(shard),
            ..self.with_timeout(self.timeout)
        }
    }

    /// How many shards it routes over.
    #[must_use]
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// The same engine with another limit on how long one inference may
    /// take, as startup warming allows a compile that a request would not.
    #[must_use]
    pub fn with_timeout(&self, timeout: Duration) -> Self {
        Self {
            shards: self.shards.clone(),
            routes: self.routes.clone(),
            budget: self.budget.clone(),
            timeout,
            pinned: self.pinned,
        }
    }

    /// Predicts a validated tabular request from its context's ephemeral
    /// table, on the shard its fingerprint names.
    pub async fn tabular(
        &self,
        model: &str,
        request: &Tabular,
        charge: Charge,
    ) -> Result<Inferred, ApiError> {
        let context = rows_context(
            model,
            &request.fingerprint,
            request.features.len(),
            request.classes.len(),
            request.context.clone(),
            request.targets.clone(),
            request.usable_rows(),
            None,
        )?;
        let answered = self
            .ask(
                context,
                Asked::Rows(request.queries.clone()),
                &request.fingerprint,
            )
            .await;
        drop(charge);
        // A context the model cannot fit is refused by its class count, as
        // the catalog's engine refuses it.
        answered.map_err(|error| {
            if error.code == "InvalidContext" {
                ApiError::new("UnsupportedClassCount", error.message, 422)
            } else {
                error
            }
        })
    }

    /// Forecasts the members of one aligned group from its ephemeral table,
    /// named for the group's fingerprint and laid out as the local engine
    /// lays it: each target row a series on periods from zero, every other
    /// row a covariate column, and known covariates' future after the origin.
    pub async fn forecast(
        &self,
        model: &str,
        group: &AlignedGroup,
        fingerprint: &[u8; 32],
        members: &[usize],
        horizon: usize,
    ) -> Result<Inferred, ApiError> {
        let failed = || ApiError::new("InternalError", "request failed safely", 500);
        let horizon = NonZeroUsize::new(horizon).ok_or_else(failed)?;
        let targets: Vec<usize> = (0..group.rows)
            .filter(|&row| group.roles[row] == TARGET)
            .collect();
        let covariates: Vec<usize> = (0..group.rows)
            .filter(|&row| group.roles[row] != TARGET)
            .collect();
        let names: Vec<String> = (0..covariates.len())
            .map(|index| format!("c{index}"))
            .collect();
        let known: Vec<&String> = names
            .iter()
            .zip(&covariates)
            .filter(|&(_, &row)| group.roles[row] == KNOWN)
            .map(|(name, _)| name)
            .collect();
        let span = group.steps + if known.is_empty() { 0 } else { group.horizon };
        // Unobserved values are NaN; past the origin only known covariates have any.
        let value = |row: usize, time: usize| -> f32 {
            if time < group.steps {
                let at = row * group.steps + time;
                if group.observed[at] {
                    group.values[at]
                } else {
                    f32::NAN
                }
            } else if group.roles[row] == KNOWN {
                group.future[row * group.horizon + time - group.steps]
            } else {
                f32::NAN
            }
        };
        let definition: TableDefinition = serde_json::from_value(json!({
            "name": format!("ctx-{}", request::hex(fingerprint)),
            "model": model,
            "columns": names
                .iter()
                .map(|name| json!({"name": name, "kind": "numeric"}))
                .collect::<Vec<_>>(),
            "classes": 0,
            "durability": "ephemeral",
            "series": {
                "series": "series",
                "time": "time",
                "target": "target",
                "step": 1,
                "known": known,
                "group": {"kind": "given", "size": targets.len()},
                "horizon": horizon.get(),
            },
            "times": "integer",
        }))
        .map_err(|_| failed())?;
        let rows = targets.len() * span;
        let mut context = RequestContext {
            fingerprint: *fingerprint,
            definition,
            keys: Vec::with_capacity(rows),
            features: Vec::with_capacity(rows * covariates.len()),
            targets: Targets::Values(Vec::with_capacity(rows)),
            times: Some(Vec::with_capacity(rows)),
        };
        let mut values = Vec::with_capacity(rows);
        for (series, &target) in targets.iter().enumerate() {
            for time in 0..span {
                // A series row key holds the series in its high half and the time in its low.
                context.keys.push(((series as u128) << 64) | time as u128);
                context
                    .features
                    .extend(covariates.iter().map(|&row| value(row, time)));
                values.push(value(target, time));
                if let Some(times) = context.times.as_mut() {
                    times.push(Some(time as i64));
                }
            }
        }
        context.targets = Targets::Values(values);
        let count = members.len();
        let asked = Asked::Forecast {
            series: members.iter().map(|&member| member as u64).collect(),
            before: Some(vec![group.steps as i64; count]),
            horizon,
            groups: Some(vec![(0..targets.len() as u64).collect(); count]),
        };
        self.ask(context, asked, fingerprint).await
    }

    /// Asks the shard `Routes` gives a context for an answer against it.
    pub async fn ask(
        &self,
        context: RequestContext,
        asked: Asked,
        fingerprint: &[u8; 32],
    ) -> Result<Inferred, ApiError> {
        let shard = match self.pinned {
            Some(shard) => self.routes.shard(shard),
            None => self.routes.context(fingerprint),
        };
        let (reply, answer) = oneshot::channel();
        let command = Command::Infer {
            context: Box::new(context),
            asked,
            trace: String::new(),
            reply,
        };
        self.wait(shard, command, answer).await
    }

    /// Predicts encoded queries from a snapshot predictor's durable table.
    pub async fn predicted(&self, bound: &Bound, features: Vec<f32>) -> Result<Inferred, ApiError> {
        self.predict_bound(
            &bound.table,
            &bound.fingerprint,
            features,
            bound.levels.clone(),
        )
        .await
    }

    /// Predicts encoded queries from a durable table `bind` wrote, on the
    /// shard its fingerprint names.
    pub async fn predict_bound(
        &self,
        table: &TableName,
        fingerprint: &[u8; 32],
        features: Vec<f32>,
        levels: Option<Vec<f64>>,
    ) -> Result<Inferred, ApiError> {
        let shard = self.durable(fingerprint);
        let (reply, answer) = oneshot::channel();
        let command = Command::Predict {
            table: table.clone(),
            features,
            strings: Vec::new(),
            before: None,
            given: None,
            trace: String::new(),
            reply,
        };
        let predictions = self.wait(self.routes.shard(shard), command, answer).await?;
        Ok(Inferred {
            predictions,
            levels,
        })
    }

    /// Binds a snapshot predictor's rows through `gf_core::predictor` and
    /// writes them, once, into a durable table named for the predictor's
    /// digest, which every query of the predictor then reads.
    pub async fn bind_snapshot(
        &self,
        model: &str,
        predictor: &ServedPredictor,
    ) -> Result<Bound, ApiError> {
        let failed = || ApiError::new("InternalError", "request failed safely", 500);
        let rows = crate::http::snapshot_rows(predictor)?;
        let target = predictor.target.as_deref().ok_or_else(|| {
            ApiError::new(
                "InvalidPredictor",
                "a snapshot predictor names its target",
                422,
            )
        })?;
        let context = predictor::bind(&predictor.schema, predictor.task()?, target, &rows)?;
        let (table, fingerprint) = bound_table(&predictor.lock_digest).ok_or_else(failed)?;
        let levels = self
            .bind(
                model,
                &table,
                &fingerprint,
                context.encoder.width(),
                context.classes.len(),
                context.features,
                context.targets,
                context.rows,
                predictor.strategy.as_ref(),
            )
            .await?;
        Ok(Bound {
            table,
            fingerprint,
            encoder: context.encoder,
            classes: context.classes,
            rows: context.rows,
            levels,
        })
    }

    /// Writes labeled rows, once, into a durable table on the shard their
    /// fingerprint names; answers the table's quantile levels, if it
    /// predicts values. A table an earlier start bound and restored there
    /// is bound again, since rows the binder of then encoded need not be
    /// the rows this one encodes. `strategy` is the selection each query's
    /// context is drawn by, as a predictor's lock resolves it; exact when
    /// None.
    #[allow(clippy::too_many_arguments)]
    pub async fn bind(
        &self,
        model: &str,
        table: &TableName,
        fingerprint: &[u8; 32],
        width: usize,
        classes: usize,
        features: Vec<f32>,
        targets: Targets,
        rows: usize,
        strategy: Option<&Value>,
    ) -> Result<Option<Vec<f64>>, ApiError> {
        let definition: TableDefinition = serde_json::from_value(definition(
            table.as_str(),
            model,
            width,
            classes,
            strategy,
            None,
        ))
        .map_err(|_| ApiError::validation("invalid predictor selection"))?;
        let shard = self.durable(fingerprint);
        let create = |definition| {
            move |reply| Command::Create {
                definition,
                terms: Terms::default(),
                reply,
            }
        };
        let levels = match self.exchange(shard, create(definition.clone())).await? {
            Ok(levels) => levels,
            Err(ServerError::Exists(_)) => {
                let released = self
                    .exchange(shard, |reply| Command::Release {
                        table: table.clone(),
                        keep: false,
                        reply,
                    })
                    .await?;
                if let Err(error) = released {
                    return Err(ApiError::new(
                        "InternalError",
                        format!(
                            "table {table} is on disk from an earlier start but cannot be \
                             bound again: {error}"
                        ),
                        500,
                    ));
                }
                self.exchange(shard, create(definition)).await??
            }
            Err(error) => return Err(error.into()),
        };
        let written = Rows {
            keys: (0..rows as u128).collect(),
            features,
            targets,
            times: None,
            id: None,
            words: Vec::new(),
            series: Vec::new(),
            writes: Vec::new(),
            layout: None,
            table: None,
        };
        let charge = self.budget.charge();
        self.exchange(shard, |reply| Command::Write {
            table: table.clone(),
            write: Write::Insert(Incoming::from(written)),
            fence: Fence::default(),
            charge,
            reply,
        })
        .await??;
        Ok(levels)
    }

    /// Forecasts every query of a request: each aligned group from the
    /// ephemeral table its fingerprint names, the groups at once; answers
    /// each group's query indices and forecasts.
    pub async fn forecast_request(
        &self,
        model: &str,
        request: &Forecast,
    ) -> Result<Vec<(Vec<usize>, Inferred)>, ApiError> {
        let mut groups: Vec<(&[u8; 32], Vec<usize>)> = Vec::new();
        let mut held: HashMap<&[u8; 32], usize> = HashMap::new();
        for (index, query) in request.queries.iter().enumerate() {
            let at = *held.entry(&query.fingerprint).or_insert_with(|| {
                groups.push((&query.fingerprint, Vec::new()));
                groups.len() - 1
            });
            groups[at].1.push(index);
        }
        let answers = futures::future::try_join_all(groups.iter().map(|(fingerprint, indices)| {
            let members: Vec<usize> = indices
                .iter()
                .map(|&index| request.queries[index].member)
                .collect();
            let group = &request.queries[indices[0]].group;
            async move {
                self.forecast(model, group, fingerprint, &members, request.horizon)
                    .await
            }
        }))
        .await?;
        Ok(groups
            .into_iter()
            .map(|(_, indices)| indices)
            .zip(answers)
            .collect())
    }

    /// Drops a durable table `bind` wrote.
    pub async fn unbind(&self, table: &TableName, fingerprint: &[u8; 32]) -> Result<(), ApiError> {
        self.exchange(self.durable(fingerprint), |reply| Command::Release {
            table: table.clone(),
            keep: false,
            reply,
        })
        .await??;
        Ok(())
    }

    /// Has every shard's worker load a model under `name` from `spec`, as a
    /// worker started with it in `[workers.models]` would; a name already
    /// served is left as it is.
    pub async fn add_model(&self, name: &str, spec: &str) -> Result<(), ApiError> {
        futures::future::try_join_all((0..self.shards.len()).map(|shard| async move {
            let (reply, answer) = oneshot::channel();
            let command = Command::AddModel {
                name: name.to_string(),
                spec: spec.to_string(),
                reply,
            };
            self.wait(self.routes.shard(shard), command, answer).await
        }))
        .await?;
        Ok(())
    }

    /// A share of the request budget, held while a call's buffers are.
    #[must_use]
    pub fn charge(&self) -> Charge {
        self.budget.charge()
    }

    /// The shard a durable table's fingerprint names: its table stays there,
    /// however busy.
    fn durable(&self, fingerprint: &[u8; 32]) -> usize {
        bound_shard(fingerprint, self.shards.len())
    }

    /// Asks shard `shard` and waits for its answer, as `wait` does.
    async fn exchange<T>(
        &self,
        shard: usize,
        command: impl FnOnce(oneshot::Sender<Result<T, ServerError>>) -> Command,
    ) -> Result<Result<T, ServerError>, ApiError> {
        let (reply, answer) = oneshot::channel();
        self.answered(self.routes.shard(shard), command(reply), answer)
            .await
    }

    /// Sends a shard a command and waits, within the request timeout, for
    /// its answer, counted in the shard's load until then.
    ///
    /// The tables a call reaches are the engine's own, a request's context
    /// or a predictor's binding, which the shard may have just evicted,
    /// dropped with a restarted worker, or not yet restored; one it does not
    /// find is gone for now, not a resource the caller named, so the call is
    /// answered as one to retry rather than as not found.
    async fn wait<T>(
        &self,
        busy: Busy<'_>,
        command: Command,
        answer: oneshot::Receiver<Result<T, ServerError>>,
    ) -> Result<T, ApiError> {
        Ok(self
            .answered(busy, command, answer)
            .await?
            .map_err(|error| match error {
                ServerError::NotFound(message) => ServerError::Unavailable(message),
                error => error,
            })?)
    }

    /// The shard's answer to a command, or why there is none: it stopped, or
    /// it neither took the command nor answered within the request timeout,
    /// which a shard whose queue is full counts against too.
    async fn answered<T>(
        &self,
        busy: Busy<'_>,
        command: Command,
        answer: oneshot::Receiver<Result<T, ServerError>>,
    ) -> Result<Result<T, ServerError>, ApiError> {
        let stopped = || {
            ApiError::new(
                "ModelUnavailable",
                "the model is restarting; retry shortly",
                503,
            )
        };
        let timed_out = || ApiError::new("RequestTimeout", "inference timed out", 504);
        let deadline = tokio::time::Instant::now() + self.timeout;
        let shard = &self.shards[busy.shard];
        tokio::time::timeout_at(deadline, shard.send(command))
            .await
            .map_err(|_| timed_out())?
            .map_err(|_| stopped())?;
        tokio::time::timeout_at(deadline, answer)
            .await
            .map_err(|_| timed_out())?
            .map_err(|_| stopped())
    }
}

/// A bound context's table definition: numeric columns of the encoded
/// width, and the predictor's selection strategy when it has one.
fn definition(
    name: &str,
    model: &str,
    width: usize,
    classes: usize,
    strategy: Option<&Value>,
    durability: Option<&str>,
) -> Value {
    let mut definition = json!({
        "name": name,
        "model": model,
        "columns": columns::numeric(width),
        "classes": classes,
    });
    if let Some(strategy) = strategy {
        definition["strategy"] = strategy.clone();
    }
    if let Some(durability) = durability {
        definition["durability"] = json!(durability);
    }
    definition
}

/// The first part of the name of every table a snapshot predictor's rows
/// are bound into, which Flight clients may not create.
pub const BOUND_PREFIX: &str = "pkg-";

/// The durable table a snapshot predictor locked as `lock_digest` binds
/// into, and the fingerprint that places it.
#[must_use]
pub fn bound_table(lock_digest: &str) -> Option<(TableName, [u8; 32])> {
    let fingerprint: [u8; 32] = Sha256::digest(lock_digest.as_bytes()).into();
    let name = TableName::new(format!("{BOUND_PREFIX}{}", request::hex(&fingerprint))).ok()?;
    Some((name, fingerprint))
}

/// Whether `table` takes the prefix `bound_table` names tables with, in
/// any case, as a filesystem that ignores case would read it.
#[must_use]
pub fn is_bound(table: &TableName) -> bool {
    table
        .as_str()
        .get(..BOUND_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(BOUND_PREFIX))
}

/// The shard of `shards` a bound table with `fingerprint` lives on.
#[must_use]
pub fn bound_shard(fingerprint: &[u8; 32], shards: usize) -> usize {
    route::shard_of(fingerprint, shards)
}

/// A tabular context in the ephemeral table its fingerprint names, drawn
/// per query by `strategy` when a predictor's lock selects; exact when None.
#[allow(clippy::too_many_arguments)]
pub fn rows_context(
    model: &str,
    fingerprint: &[u8; 32],
    width: usize,
    classes: usize,
    features: Vec<f32>,
    targets: Targets,
    rows: usize,
    strategy: Option<&Value>,
) -> Result<RequestContext, ApiError> {
    let name = format!("ctx-{}", request::hex(fingerprint));
    let definition: TableDefinition = serde_json::from_value(definition(
        &name,
        model,
        width,
        classes,
        strategy,
        Some("ephemeral"),
    ))
    .map_err(|_| ApiError::validation("invalid predictor selection"))?;
    Ok(RequestContext {
        fingerprint: *fingerprint,
        definition,
        keys: (0..rows as u128).collect(),
        features,
        targets,
        times: None,
    })
}
