//! Bearer tokens, the tables each may reach, and what routers mark node calls with.

use std::collections::HashMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use tonic::{Request, Status};

/// Metadata a router sets only on the log entries it ships between nodes,
/// never on calls it forwards, so no client can apply log entries through it.
pub const SHIPMENT: &str = "gf-shipment";

/// Metadata naming the term a router sent a write under; the node logs the
/// write only while its table is still at that term.
pub const TERM: &str = "gf-term";

/// Metadata on a shipment: the source copy's terms, as JSON, which the
/// target's must be a prefix of.
pub const TERMS: &str = "gf-terms";

/// Metadata on a shipment: the source copy's creation id, which the target's
/// must equal.
pub const CREATED: &str = "gf-created";

/// Metadata a client sets on a write: an id of its own, so a write sent twice
/// applies once.
pub const WRITE_ID: &str = "gf-write-id";

/// Metadata a client sets on a delete: keys the table lacks are skipped.
pub const MISSING_OK: &str = "gf-missing-ok";

/// Metadata a client sets on a regressor's predictions: the quantile levels
/// to keep, comma-separated, each strictly between 0 and 1.
pub const QUANTILES: &str = "gf-quantiles";

/// The tables a token may reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    All,
    /// Tables named by one of these patterns; a pattern ending in `*` names
    /// every table with that prefix.
    Tables(Vec<String>),
}

impl Scope {
    /// Whether the token may reach `table`.
    #[must_use]
    pub fn allows(&self, table: &str) -> bool {
        match self {
            Self::All => true,
            Self::Tables(patterns) => patterns.iter().any(|pattern| {
                pattern
                    .strip_suffix('*')
                    .map_or(pattern == table, |prefix| table.starts_with(prefix))
            }),
        }
    }

    /// Refuses a table the token may not reach, as if it did not exist, so a
    /// token learns nothing about tables outside its scope.
    ///
    /// # Errors
    ///
    /// `NOT_FOUND` if `table` is out of scope.
    pub fn check(&self, table: &str) -> Result<(), Status> {
        if self.allows(table) {
            Ok(())
        } else {
            Err(Status::not_found(format!("table {table:?} does not exist")))
        }
    }

    /// Refuses creating a table the token could not reach.
    ///
    /// # Errors
    ///
    /// `PERMISSION_DENIED` if `table` is out of scope.
    pub fn check_create(&self, table: &str) -> Result<(), Status> {
        if self.allows(table) {
            Ok(())
        } else {
            Err(Status::permission_denied(format!(
                "this token may not create table {table:?}"
            )))
        }
    }

    /// The scope an interceptor attached to a request; none if it attached nothing.
    #[must_use]
    pub fn of<T>(request: &Request<T>) -> Self {
        request
            .extensions()
            .get::<Self>()
            .cloned()
            .unwrap_or(Self::Tables(Vec::new()))
    }
}

/// A token in a configuration file: a string reaches every table, and a table
/// `{ token, tables }` only the tables its patterns name.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum TokenConfig {
    Any(String),
    Scoped { token: String, tables: Vec<String> },
}

impl TokenConfig {
    #[must_use]
    pub fn token(&self) -> &str {
        match self {
            Self::Any(token) | Self::Scoped { token, .. } => token,
        }
    }

    #[must_use]
    pub fn scope(&self) -> Scope {
        match self {
            Self::Any(_) => Scope::All,
            Self::Scoped { tables, .. } => Scope::Tables(tables.clone()),
        }
    }
}

/// Configured tokens and their scopes, kept by SHA-256 digest: a lookup
/// compares digests, so how long it takes says nothing about the tokens.
#[derive(Clone, Debug, Default)]
pub struct Tokens(HashMap<[u8; 32], Scope>);

impl Tokens {
    /// The tokens a configuration lists.
    #[must_use]
    pub fn new(configs: &[TokenConfig]) -> Self {
        Self(
            configs
                .iter()
                .map(|config| (digest(config.token()), config.scope()))
                .collect(),
        )
    }

    /// The scope of `token`, if it is one of these.
    #[must_use]
    pub fn scope(&self, token: &str) -> Option<&Scope> {
        self.0.get(&digest(token))
    }
}

/// A token's SHA-256 digest.
#[must_use]
pub fn digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// The bearer token a request carries, if any.
#[must_use]
pub fn bearer<T>(request: &Request<T>) -> Option<&str> {
    request
        .metadata()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}
