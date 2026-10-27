// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! How far a start has got through compiling programs: the workers' recorded
//! programs, summed over them, and the warm shape buckets, for the terminal's
//! `Compiling` line.

use std::sync::Arc;

use tokio::sync::watch;

/// Programs compiled so far, and the programs to compile in all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

/// A count of programs compiled, shared by the workers' pipelines and the
/// warm-up, which whoever shows it watches for changes.
#[derive(Clone, Debug)]
pub struct Compiling(Arc<watch::Sender<Progress>>);

impl Default for Compiling {
    fn default() -> Self {
        Self(Arc::new(watch::channel(Progress::default()).0))
    }
}

impl Compiling {
    /// Adds `programs` to those to compile.
    pub fn expect(&self, programs: u64) {
        if programs > 0 {
            self.0.send_modify(|progress| progress.total += programs);
        }
    }

    /// Adds `programs` to those compiled.
    pub fn advance(&self, programs: u64) {
        if programs > 0 {
            self.0.send_modify(|progress| progress.done += programs);
        }
    }

    #[must_use]
    pub fn current(&self) -> Progress {
        *self.0.borrow()
    }

    /// Resolves with each change after the call.
    #[must_use]
    pub fn changes(&self) -> watch::Receiver<Progress> {
        self.0.subscribe()
    }
}
