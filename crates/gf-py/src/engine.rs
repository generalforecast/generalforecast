// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The engine in this process: gf-server's shards, each a node and a worker
//! subprocess, on a runtime of their own.
//!
//! It is `gf_server::engine::Engine` itself, so a context routes to its
//! shard, becomes an ephemeral table named for its fingerprint and is
//! pinned and evicted exactly as `gf serve` does it; Python only hands it
//! validated requests and reads back the outputs.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use gf_core::node::Targets;
use gf_server::config::Config;
use gf_server::engine;
use gf_server::http::ApiError;
use gf_server::metrics::Metrics;
use gf_server::name::TableName;
use gf_server::private::{self, Private};
use gf_server::service::RequestBudget;
use gf_server::setup;
use gf_server::shard::{Command, Inferred};
use gf_server::wal::EncryptionKey;
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

use crate::{ForecastRequest, RequestError, TabularRequest};

/// An engine's refusal, with the code, message and status the Models API answers.
fn answered(error: ApiError) -> PyErr {
    RequestError::new_err((error.code, error.message, error.status))
}

fn failed(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// The shards and the runtime their tasks run on.
struct Running {
    /// gf-server's Flight `Service` on these shards, once `serve` starts it.
    served: Mutex<Option<Private>>,
    runtime: Runtime,
    engine: engine::Engine,
    shards: usize,
    senders: Vec<mpsc::Sender<Command>>,
    metrics: Arc<Metrics>,
    data: PathBuf,
}

/// gf-server's shards in this process, sized as `gf serve` sizes them.
///
/// One worker per GPU, or on CPU one per run of the configuration's
/// `cpu_threads` cores, pinned to them.
///
/// Calls release the GIL while they wait, so concurrent callers' queries
/// share the shards' steps.
#[pyclass(frozen)]
pub struct Engine {
    running: Mutex<Option<Arc<Running>>>,
}

/// A context bound as a durable table, which lives until `Engine.unbind`.
#[pyclass(frozen)]
pub struct BoundTable {
    table: TableName,
    fingerprint: [u8; 32],
    /// The values in each of its rows, which every query has too.
    width: usize,
    levels: Option<Vec<f64>>,
}

#[pymethods]
impl BoundTable {
    /// The table's name.
    #[getter]
    fn name(&self) -> &str {
        self.table.as_str()
    }

    /// The quantile levels a regression's outputs are; None for a classification.
    #[getter]
    fn levels(&self) -> Option<Vec<f64>> {
        self.levels.clone()
    }
}

impl Engine {
    fn running(&self) -> PyResult<Arc<Running>> {
        self.running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("the engine is closed"))
    }
}

/// Outputs as an array, `(rows, columns)` or `(rows, horizon, columns)`,
/// and the quantile levels its columns are, if any.
fn outputs(py: Python<'_>, inferred: Inferred) -> PyResult<(Py<PyAny>, Option<Vec<f64>>)> {
    let predictions = inferred.predictions;
    let values = predictions.values.into_pyarray(py);
    let array = match predictions.horizon {
        Some(horizon) => values
            .reshape([predictions.rows, horizon, predictions.columns])?
            .into_any(),
        None => values
            .reshape([predictions.rows, predictions.columns])?
            .into_any(),
    };
    Ok((array.unbind(), inferred.levels))
}

/// A row-major copy of a matrix's values, and its rows and columns.
fn rows(matrix: &PyReadonlyArray2<'_, f32>) -> (Vec<f32>, usize, usize) {
    let shape = matrix.shape();
    (
        matrix.as_array().iter().copied().collect(),
        shape[0],
        shape[1],
    )
}

/// What names a bound table's shard: random, so bound tables spread over them.
fn random_fingerprint() -> PyResult<[u8; 32]> {
    let hex = setup::random_hex(32).map_err(failed)?;
    let mut fingerprint = [0u8; 32];
    for (index, byte) in fingerprint.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * index..2 * index + 2], 16).map_err(failed)?;
    }
    Ok(fingerprint)
}

#[pymethods]
impl Engine {
    /// Starts the shards a gf-server configuration's TOML text describes.
    ///
    /// Its `data` directory holds bound tables' logs and the worker caches.
    #[new]
    fn new(py: Python<'_>, config: &str) -> PyResult<Self> {
        let config =
            Config::parse(Path::new("engine.toml"), config).map_err(PyValueError::new_err)?;
        let key =
            EncryptionKey::from_hex(&setup::random_hex(32).map_err(failed)?).map_err(failed)?;
        let tables = config.data.join("tables");
        let cache = config.data.join("cache");
        for directory in [&tables, &cache] {
            std::fs::create_dir_all(directory).map_err(failed)?;
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(failed)?;
        let metrics = Arc::new(match &config.timing_log {
            Some(path) => Metrics::with_timing_log(path)
                .map_err(|error| failed(format!("the timing log {}: {error}", path.display())))?,
            None => Metrics::default(),
        });
        let (shards, routes) = py
            .detach(|| {
                let seats = gf_server::runtime::seats(&config.workers);
                runtime.block_on(gf_server::runtime::start(
                    &config, &seats, &key, &tables, &cache, &metrics,
                ))
            })
            .map_err(failed)?;
        let count = shards.len();
        let senders = shards.clone();
        let engine = engine::Engine::new(
            shards,
            routes,
            RequestBudget::new(config.request_bytes_in_flight),
            Duration::from_secs_f64(config.request_timeout_seconds),
        );
        Ok(Self {
            running: Mutex::new(Some(Arc::new(Running {
                served: Mutex::new(None),
                runtime,
                engine,
                shards: count,
                senders,
                metrics,
                data: config.data.clone(),
            }))),
        })
    }

    /// Serves gf-server's Flight `Service` on these shards to this process alone.
    ///
    /// Returns `(location, token)`: a Unix socket in an owner-only directory
    /// under the data directory, or a loopback port where there are none, and
    /// the random token it accepts. The first call starts it; `close` stops it.
    fn serve(&self, py: Python<'_>) -> PyResult<(String, String)> {
        let running = self.running()?;
        py.detach(|| {
            let mut served = running
                .served
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(served) = &*served {
                return Ok((served.location.clone(), served.token.clone()));
            }
            let started = running.runtime.block_on(private::serve(
                running.senders.clone(),
                running.metrics.clone(),
                &running.data,
            ))?;
            let endpoint = (started.location.clone(), started.token.clone());
            *served = Some(started);
            Ok(endpoint)
        })
        .map_err(|error: Box<dyn std::error::Error + Send + Sync>| failed(error))
    }

    /// How many shards, and so worker processes, the engine runs.
    #[getter]
    fn shards(&self) -> PyResult<usize> {
        Ok(self.running()?.shards)
    }

    /// Has every worker load a model under `name` from `spec`.
    ///
    /// `spec` is a built-in model's repository id; a worker serving that
    /// name already keeps it.
    fn add_model(&self, py: Python<'_>, name: &str, spec: &str) -> PyResult<()> {
        let running = self.running()?;
        py.detach(|| {
            running
                .runtime
                .block_on(running.engine.add_model(name, spec))
        })
        .map_err(answered)
    }

    /// Scores a validated request's queries against its context.
    ///
    /// The context is the ephemeral table its fingerprint names. Returns
    /// `(queries, outputs)` and the quantile levels of a regression's outputs.
    fn tabular(
        &self,
        py: Python<'_>,
        model: &str,
        request: &TabularRequest,
    ) -> PyResult<(Py<PyAny>, Option<Vec<f64>>)> {
        let running = self.running()?;
        let parsed = request.parsed.clone();
        let inferred = py
            .detach(|| {
                running.runtime.block_on(async {
                    let charge = running.engine.charge();
                    running.engine.tabular(model, &parsed, charge).await
                })
            })
            .map_err(answered)?;
        outputs(py, inferred)
    }

    /// Forecasts every query of a validated request.
    ///
    /// Each aligned group reads the ephemeral table its fingerprint names.
    /// Returns `(queries, horizon, levels)` and the model's quantile levels.
    fn forecast(
        &self,
        py: Python<'_>,
        model: &str,
        request: &ForecastRequest,
    ) -> PyResult<(Py<PyAny>, Option<Vec<f64>>)> {
        let running = self.running()?;
        let parsed = request.parsed.clone();
        let groups = py
            .detach(|| {
                running
                    .runtime
                    .block_on(running.engine.forecast_request(model, &parsed))
            })
            .map_err(answered)?;
        let invalid = || {
            answered(ApiError::new(
                "ModelOutputError",
                "native forecaster returned invalid forecasts",
                503,
            ))
        };
        let horizon = parsed.horizon;
        let levels = groups
            .first()
            .and_then(|(_, inferred)| inferred.levels.clone());
        let columns = levels.as_ref().map_or(0, Vec::len);
        let width = horizon * columns;
        let mut values = vec![0f32; parsed.queries.len() * width];
        for (indices, inferred) in &groups {
            let forecast = &inferred.predictions;
            if (forecast.rows, forecast.horizon, forecast.columns)
                != (indices.len(), Some(horizon), columns)
                || inferred.levels != levels
            {
                return Err(invalid());
            }
            for (row, &index) in indices.iter().enumerate() {
                values[index * width..(index + 1) * width]
                    .copy_from_slice(&forecast.values[row * width..(row + 1) * width]);
            }
        }
        let array = values
            .into_pyarray(py)
            .reshape([parsed.queries.len(), horizon, columns])?;
        Ok((array.into_any().unbind(), levels))
    }

    /// Binds labeled rows once as a durable table.
    ///
    /// `features` is `(rows, columns)`; `targets` holds class codes (int32)
    /// when `classes` is positive, else values (float32). `strategy` is the
    /// selection each query's context is drawn by, as `Strategy.spec()`
    /// spells it; exact when None.
    #[pyo3(signature = (model, features, targets, classes, strategy=None))]
    fn bind(
        &self,
        py: Python<'_>,
        model: &str,
        features: PyReadonlyArray2<'_, f32>,
        targets: &Bound<'_, PyAny>,
        classes: usize,
        strategy: Option<&str>,
    ) -> PyResult<BoundTable> {
        let running = self.running()?;
        let strategy: Option<serde_json::Value> = strategy
            .map(serde_json::from_str)
            .transpose()
            .map_err(|_| PyValueError::new_err("strategy is not a strategy spec"))?;
        let (features, count, width) = rows(&features);
        let targets = if classes > 0 {
            Targets::Classes(
                targets
                    .extract::<PyReadonlyArray1<i32>>()?
                    .as_array()
                    .to_vec(),
            )
        } else {
            Targets::Values(
                targets
                    .extract::<PyReadonlyArray1<f32>>()?
                    .as_array()
                    .to_vec(),
            )
        };
        let fingerprint = random_fingerprint()?;
        let table = TableName::new(format!("bound-{}", gf_core::request::hex(&fingerprint)))
            .map_err(failed)?;
        let levels = py
            .detach(|| {
                running.runtime.block_on(running.engine.bind(
                    model,
                    &table,
                    &fingerprint,
                    width,
                    classes,
                    features,
                    targets,
                    count,
                    strategy.as_ref(),
                ))
            })
            .map_err(answered)?;
        Ok(BoundTable {
            table,
            fingerprint,
            width,
            levels,
        })
    }

    /// Scores query features `(queries, columns)` against a bound table.
    fn predict(
        &self,
        py: Python<'_>,
        bound: &BoundTable,
        features: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<(Py<PyAny>, Option<Vec<f64>>)> {
        let running = self.running()?;
        // Queries are read in rows of the table's width, so another width
        // would score other rows than the caller shaped.
        let (features, queries, width) = rows(&features);
        if queries > 0 && width != bound.width {
            return Err(PyValueError::new_err(format!(
                "queries are rows of {width} values, and the bound table's rows are {}",
                bound.width
            )));
        }
        let inferred = py
            .detach(|| {
                running.runtime.block_on(running.engine.predict_bound(
                    &bound.table,
                    &bound.fingerprint,
                    features,
                    bound.levels.clone(),
                ))
            })
            .map_err(answered)?;
        outputs(py, inferred)
    }

    /// Drops a bound table and its log.
    fn unbind(&self, py: Python<'_>, bound: &BoundTable) -> PyResult<()> {
        let running = self.running()?;
        py.detach(|| {
            running
                .runtime
                .block_on(running.engine.unbind(&bound.table, &bound.fingerprint))
        })
        .map_err(answered)
    }

    /// Stops the shards and their workers once calls in progress return.
    ///
    /// Later calls raise.
    fn close(&self, py: Python<'_>) {
        let running = self
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        py.detach(move || drop(running));
    }
}
