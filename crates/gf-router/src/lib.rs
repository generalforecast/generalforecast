//! generalforecast's router: places tables on nodes by rendezvous hashing, keeps their replicas, and forwards clients to them.

pub mod config;
pub mod partitions;
pub mod placement;
pub mod quota;
pub mod replication;
pub mod service;
pub mod setup;
