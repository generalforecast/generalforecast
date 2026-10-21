//! generalforecast's control plane: catalog, contexts, cache directory and scheduler.
//!
//! This crate is plain Rust with no Python types, so it builds and tests with
//! `cargo test` alone. `gf-py` exposes it to Python.

#![forbid(unsafe_code)]

pub mod cache;
pub mod catalog;
pub mod categories;
pub mod columns;
pub mod context;
pub mod forecasting;
pub mod node;
pub mod package;
pub mod pages;
pub mod planner;
pub mod regions;
pub mod request;
pub mod route;
pub mod rows;
pub mod series;
pub mod shapes;
pub mod step;
pub mod table;
pub mod time_index;
