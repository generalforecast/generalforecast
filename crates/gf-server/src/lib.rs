// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! generalforecast's server: an Arrow Flight API over nodes, with a write-ahead log and Python workers.

pub mod action;
pub mod as_of;
pub mod auth;
pub mod budget;
pub mod config;
pub mod definition;
pub mod engine;
pub mod ephemeral;
pub mod error;
pub mod http;
pub mod metrics;
pub mod name;
pub mod parent;
pub mod partition;
pub mod predictors;
pub mod private;
pub mod progress;
pub mod protocol;
pub mod quantiles;
pub mod revisions;
pub mod routes;
pub mod runtime;
#[cfg(test)]
mod scratch;
pub mod service;
pub mod setup;
pub mod shard;
pub mod trace;
pub mod upstream;
pub mod wal;
pub mod worker;
