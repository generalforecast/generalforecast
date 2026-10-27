// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The request budget: bytes of client calls held in memory at once, which
//! Flight calls and Models API requests alike charge as they read.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tonic::Status;

/// Bytes of client calls held in memory at once, across every connection.
///
/// Each call charges what it reads as it reads it, and gives it back when it
/// finishes, so a flood of large requests is refused rather than buffered.
#[derive(Clone, Debug)]
pub struct RequestBudget {
    held: Arc<AtomicU64>,
    limit: u64,
}

impl RequestBudget {
    /// A budget of `limit` bytes.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            held: Arc::default(),
            limit,
        }
    }

    /// The most bytes the budget holds at once.
    #[must_use]
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// An empty charge against the budget, to grow as a call reads.
    #[must_use]
    pub fn charge(&self) -> Charge {
        Charge {
            held: self.held.clone(),
            limit: self.limit,
            bytes: 0,
        }
    }
}

/// What one call holds of the budget; given back when dropped. The default
/// holds nothing, of a budget of its own.
#[derive(Debug)]
pub struct Charge {
    held: Arc<AtomicU64>,
    limit: u64,
    bytes: u64,
}

impl Charge {
    /// Takes `bytes` more of the budget.
    ///
    /// # Errors
    ///
    /// `RESOURCE_EXHAUSTED` if the budget is spent; nothing more is taken then.
    pub fn grow(&mut self, bytes: u64) -> Result<(), Status> {
        let before = self.held.fetch_add(bytes, Ordering::Relaxed);
        if before + bytes > self.limit {
            self.held.fetch_sub(bytes, Ordering::Relaxed);
            return Err(Status::resource_exhausted(format!(
                "the server holds {} MiB of requests already; retry shortly",
                self.limit >> 20
            )));
        }
        self.bytes += bytes;
        Ok(())
    }

    /// Another charge against the same budget, holding nothing yet; each
    /// gives back what it took when it is dropped.
    #[must_use]
    pub fn split(&self) -> Self {
        Self {
            held: self.held.clone(),
            limit: self.limit,
            bytes: 0,
        }
    }
}

impl Default for Charge {
    fn default() -> Self {
        RequestBudget::new(u64::MAX).charge()
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.held.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::RequestBudget;

    #[test]
    fn charges_share_one_budget_and_give_it_back_when_dropped() {
        let budget = RequestBudget::new(100);
        let mut first = budget.charge();
        let mut second = budget.charge();

        first.grow(60).unwrap();
        let refused = second.grow(50);
        second.grow(40).unwrap();
        drop(first);
        let after = second.grow(50);

        assert_eq!(refused.unwrap_err().code(), tonic::Code::ResourceExhausted);
        assert!(after.is_ok());
        drop(second);
        assert!(budget.charge().grow(100).is_ok());
    }
}
