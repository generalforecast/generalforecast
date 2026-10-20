//! Per-token request quotas, as token buckets.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use tokio::time::Instant;

/// How fast a token may call: a sustained rate and a burst.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limit {
    pub per_second: f64,
    pub burst: u32,
}

/// Allows `burst` calls at once and refills at `per_second`.
#[derive(Clone, Debug)]
pub struct TokenBucket {
    limit: Limit,
    available: f64,
    updated: Instant,
}

impl TokenBucket {
    /// A full bucket.
    #[must_use]
    pub fn new(limit: Limit, now: Instant) -> Self {
        Self {
            limit,
            available: f64::from(limit.burst),
            updated: now,
        }
    }

    /// Takes one call's worth if available at `now`.
    pub fn take(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.updated).as_secs_f64();
        self.available =
            (self.available + elapsed * self.limit.per_second).min(f64::from(self.limit.burst));
        self.updated = now;
        if self.available >= 1.0 {
            self.available -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Buckets per token, by its digest; tokens without a limit are not limited.
#[derive(Debug)]
pub struct Quotas {
    limits: HashMap<[u8; 32], Limit>,
    buckets: Mutex<HashMap<[u8; 32], TokenBucket>>,
}

impl Quotas {
    /// Quotas for the tokens with limits, by digest.
    #[must_use]
    pub fn new(limits: HashMap<[u8; 32], Limit>) -> Self {
        Self {
            limits,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Counts a call against the quota of the token with this digest.
    ///
    /// # Errors
    ///
    /// If the token has used its quota.
    pub fn check(&self, token: &[u8; 32]) -> Result<(), String> {
        let Some(&limit) = self.limits.get(token) else {
            return Ok(());
        };
        let now = Instant::now();
        let mut buckets = self.buckets.lock().unwrap_or_else(PoisonError::into_inner);
        let bucket = buckets
            .entry(*token)
            .or_insert_with(|| TokenBucket::new(limit, now));
        if bucket.take(now) {
            Ok(())
        } else {
            Err(format!(
                "quota exceeded: {} calls per second, bursts of {}",
                limit.per_second, limit.burst
            ))
        }
    }
}
