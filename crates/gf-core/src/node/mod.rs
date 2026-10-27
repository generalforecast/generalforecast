// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A node: many tables, one request queue, one cache.
//!
//! The node owns each table's rows, so it is the single writer for them. A
//! step carries everything a worker needs, in order: tables to fit, rows to
//! embed since the fit, units to encode or reuse, and queries to decode. Small
//! tables share the node's cache and its steps, so many of them cost what one
//! large table does.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::{Duration, Instant};

use roaring::RoaringBitmap;

use crate::cache::{
    Admission, Cache, CacheError, CacheKey, Epoch, Eviction, Kind, ModelId, Movement, Placement,
    Tier, TierConfig,
};
use crate::categories::{CategoryEncoding, Lookup};
use crate::context::{Context, Fingerprint};
use crate::distance::squared;
use crate::forecasting::{self, Forecast, ForecastError, SeriesRequest, SeriesSpec, SeriesUnit};
use crate::pages::{Bucket, IndexError, IndexLayout, PageId, PageIndex, PageLayout, Vectors};
use crate::planner::{
    Core, CoreKind, Cores, Plan, PlanError, Ranking, Recency, Source, Strata, Strategy,
    StrategyError, UnitId, Windows, draw_core, plan_given, plan_with, recent, smallest,
};
use crate::regions::{Built, Rebuild, RegionSpec, RegionStats, Regions, RegionsError};
use crate::rows::{RowMemory, RowSnapshot, RowStore, RowsError, Sealer};
use crate::series::{SeriesIndex, SeriesKey};
use crate::shapes::bucket_size;
use crate::table::{Change, Keys, ROW_IDS, RowId, RowKey, Snapshot, Table, TableError, Version};
use crate::time_index::TimeIndex;

pub use crate::step::{Action, Placed, RequestId, StepPage, StepUnit};
pub use crate::table::TableId;

mod error;
mod layout;
mod placement;
mod plan;
mod release;
mod requests;
mod restore;
mod scaling;
mod schedule;
mod spec;
mod state;
mod step;
mod tables;
mod writes;

pub use error::*;
pub use layout::*;
use placement::*;
use plan::*;
use release::*;
use restore::*;
use scaling::*;
pub use spec::*;
pub use step::*;
use writes::*;

/// What embedding one feature of one row costs, in the units of a context's
/// cost (its rows squared). Measured on TabICL v2, where embedding 4,096 rows of
/// 10 features takes about a ninth of encoding them.
const EMBED_COST_PER_CELL: f64 = 40.0;

/// Bins a regression table's targets fall in for stratified and balanced cores.
const TARGET_BINS: usize = 8;

/// Seeds the sample a table larger than its `fit_rows` is fit on.
const FIT_SEED: u64 = 0x6669_745f_726f_7773;

/// A kept series unit a later one of the same group may extend: its cutoff,
/// grid length, the patches its entry holds and the entry.
#[derive(Clone, Copy, Debug)]
struct Recent {
    cutoff: i64,
    steps: usize,
    /// Patches the worker stored: the padded grid's for an encoding, the
    /// source's plus the new ones for an extension.
    patches: usize,
    key: CacheKey,
}

/// Groups whose latest kept units a table remembers before it prunes them:
/// past this many, and again each time they double, those whose entries are
/// gone are forgotten.
const RECENT_PRUNE: usize = 4096;

#[derive(Debug)]
struct TableState {
    spec: TableSpec,
    /// The feature count, which is positive.
    dimensions: NonZeroUsize,
    table: Table,
    index: PageIndex,
    epoch: Epoch,
    /// Every row id's features, target and key, live or since deleted.
    rows: RowStore,
    standardizer: Option<Standardizer>,
    /// Live rows when the standardizer was fit; it is fit again once they double.
    scaled_rows: u64,
    /// The epoch whose fit has been scheduled.
    fitted: Option<Epoch>,
    /// Each row's time; rows inserted without one are before every cutoff.
    times: TimeIndex,
    /// Each series' rows by time, for a table of series.
    series: Option<SeriesIndex>,
    /// Forecast requests, for a table of series.
    forecasts: Vec<(RequestId, SeriesRequest)>,
    /// The latest kept unit of each group of series, by the group's print:
    /// what a unit of the group at a later cutoff may extend. Only units a
    /// step has written are here, so an extension never reads an entry a
    /// step in flight is still encoding, nor one a failed step forgot.
    recent: HashMap<Fingerprint, Recent>,
    /// Kept units of groups a step in flight writes, which join `recent` as
    /// their step completes.
    writing: HashMap<CacheKey, (Fingerprint, Recent)>,
    /// How many groups `recent` held after it was last pruned.
    recent_pruned: usize,
    /// The fingerprint of the rows plans read at each time, rows without one first
    /// (None sorts first), so a window before a cutoff is fingerprinted by a
    /// sum over times rather than over rows.
    prints: BTreeMap<Option<i64>, Fingerprint>,
    /// The rows plans read ranked by the strategy's core seed, if it has a core.
    ranking: Option<Ranking>,
    /// The strategy the last plan resolved `spec.strategy` to; cores and
    /// regions are drawn for it, and dropped when it changes.
    resolved: Strategy,
    /// Live rows whose target is not known yet.
    unlabeled: RoaringBitmap,
    /// The quantiles a regression table's targets are binned by for
    /// stratified and balanced cores, set once per state, at its first plan.
    edges: Option<(Epoch, Vec<f32>)>,
    /// Diverse cores drawn for the state of an epoch.
    cores: (Epoch, Cores),
    /// Running per-column sums over live rows, for drift.
    columns: Columns,
    /// The columns' means and deviations when the state was last fit.
    fitted_columns: Option<Fitted>,
    /// Writes to rows plans read, before or after the write, since the state
    /// was last fit, so they count in `Fitted::rows`' units.
    changed: u64,
    /// Requests: their queries, and each query's cutoff if it has one.
    pending: Vec<(RequestId, Vec<f32>, Option<Vec<i64>>)>,
    /// Requests whose cutoffs need a state fit on other rows than `pending`'s,
    /// which wait until those are served.
    later: Vec<(RequestId, Vec<f32>, Option<Vec<i64>>)>,
    /// The rows the state was fit on: those before this time, or every row.
    horizon: Option<i64>,
    /// The latest time of any row the state was fit on, if any had one. Rows
    /// may since have gone or moved, so a cutoff at or before it refits.
    fitted_latest: Option<i64>,
    /// The context of each query of its one pending request, if they were given.
    given: Option<Vec<Given>>,
    /// Requests with given contexts waiting for the table's pending requests
    /// to be planned; they go before requests that wait for another state.
    givens: std::collections::VecDeque<(RequestId, Vec<f32>, Vec<Given>)>,
    /// Whether the last step served given requests, whose turn is then over.
    gave: bool,
    /// The most row ids the table assigns, live or since deleted, before it
    /// refuses inserts: `ROW_IDS` unless `Node::set_row_ids` lowered it.
    row_ids: u64,
    /// Counts changes to the table or its requests, so a plan made for them
    /// is known to still hold.
    generation: u64,
    /// The plan of requests the step budget left waiting, and the generation
    /// it was made at, so a later step need not plan them again.
    deferred: Option<(u64, Plan)>,
    /// A rebuild from a compacted log under way, if one is.
    restoring: Option<Restoring>,
    /// Whether a plan read a context past the model's trained length.
    past_cap: bool,
    /// The regions of the state queries last read, if the table has them.
    regions: Option<Regions>,
    region_stats: RegionStats,
    /// Regions drawn so far, which numbers each draw.
    region_draws: u64,
    /// Queries of the latest plan of the pending requests, counted into
    /// `region_stats` once it joins a step.
    planned_counts: RegionStats,
    /// The generation regions' contexts were last aged at.
    aged_at: Option<u64>,
    /// Request steps taken since a region context fell due, without a
    /// rebuild step.
    rebuild_waited: u32,
    /// Whether the table holds a context one request brought; see
    /// `Node::set_single_use`.
    single_use: bool,
    /// The table's requests in the steps in flight.
    scheduled: Scheduled,
    /// The program the worker compiles for the table's part of a step,
    /// which steps leave it out until.
    compiling: Option<String>,
}

/// Dead row ids past which, and past the live rows, a table is renumbered
/// before its next insert, so ids and the arrays they index stay near the
/// live rows however many rows come and go.
const SPARSE_ROWS: u64 = 1 << 12;

/// The order a step takes waiting tables in, while its budget lasts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scheduling {
    /// Each step starts after the table the last one started with, so a busy
    /// table cannot starve the rest.
    #[default]
    Turns,
    /// The table whose oldest request has waited longest goes first.
    Arrival,
}

impl Scheduling {
    /// Every order, for parsing names.
    pub const ALL: [Self; 2] = [Self::Turns, Self::Arrival];

    /// The order's name, as configurations spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Turns => "turns",
            Self::Arrival => "arrival",
        }
    }
}

/// Many tables behind one request queue and one cache.
///
/// Each step has a budget: the contexts it newly encodes, kept or not, take at
/// most `step_bytes`, the device tier's size by default, or no limit without a
/// device tier, since then nothing is kept. Requests past it wait for a later
/// step, in the order `scheduling` picks.
#[derive(Debug)]
pub struct Node {
    cache: Cache,
    tables: BTreeMap<TableId, TableState>,
    next_table: u32,
    next_request: u64,
    step_bytes: Option<u64>,
    scheduling: Scheduling,
    /// Memory the tables' rows share.
    rows: RowMemory,
    /// The table the next step starts from.
    turn: TableId,
    /// What the steps in flight hold.
    flight: Flight,
    /// Drops from purges, for the executor to apply.
    drops: Vec<Movement>,
    /// Requests planning refused, for the executor to answer.
    refused: Vec<(RequestId, NodeError)>,
    /// Tables that started an epoch since the executor last asked.
    advanced: Vec<TableId>,
    /// Tables whose contexts first passed their model's trained length.
    past_cap: Vec<TableId>,
    /// Tables waiting for the worker to measure how many rows their fit
    /// takes, and those the executor was not yet told of.
    unmeasured: BTreeSet<TableId>,
    untold: Vec<TableId>,
    /// Region contexts a step in flight builds again, by their entries, and
    /// the regions they are for.
    rebuilding: HashMap<CacheKey, Vec<Rebuilding>>,
    rebuilt_regions: HashSet<(TableId, usize)>,
    /// Programs the worker reported compiled beside its steps, by key.
    compiled: HashSet<String>,
    /// Where the last `schedule` spent its time.
    parts: PlanParts,
}

/// A given context, by the keys of its rows: a renumbering or a restore
/// gives rows new ids, so the rows are found when the context is planned.
type Given = Vec<RowKey>;

/// A table's query requests in the steps in flight, kept until their steps
/// complete, so those of a part the worker defers queue again; see
/// `Node::complete_deferring`.
#[derive(Debug, Default)]
struct Scheduled {
    queries: Vec<(RequestId, Vec<f32>, Option<Vec<i64>>)>,
    /// Requests with given contexts, which queue again with them.
    given: Vec<(RequestId, Vec<f32>, Vec<Given>)>,
}

impl Scheduled {
    /// Takes out the requests `of` names.
    fn take(&mut self, of: impl Fn(RequestId) -> bool) -> Self {
        let mut taken = Self::default();
        (taken.queries, self.queries) = std::mem::take(&mut self.queries)
            .into_iter()
            .partition(|(request, _, _)| of(*request));
        (taken.given, self.given) = std::mem::take(&mut self.given)
            .into_iter()
            .partition(|(request, _, _)| of(*request));
        taken
    }
}

/// What the steps in flight hold, which later steps plan around.
#[derive(Debug, Default)]
struct Flight {
    /// Entries a step in flight moves between tiers: later steps read them
    /// only once it completes, since until then their slots may not hold them.
    moving: HashSet<CacheKey>,
    /// Stale entries a running step still reads, dropped once every step
    /// reading them completes.
    doomed: HashSet<CacheKey>,
    /// Entries a step in flight encodes, and tables it fits. Requests that
    /// would read them wait for it: if it fails, the worker never holds them.
    encoding: HashSet<CacheKey>,
    fitting: HashSet<TableId>,
}

impl Flight {
    /// Whether a new step must not read the entry: a step in flight moves it,
    /// so its slots may not hold it yet, or it is stale and still read. A unit
    /// or page on such an entry is computed again, transiently.
    fn busy(&self, key: &CacheKey) -> bool {
        self.moving.contains(key) || self.doomed.contains(key)
    }

    /// Whether a step may copy rows from the entry: one a step in flight
    /// computes is held by the worker only once it completes.
    fn source(&self, key: &CacheKey) -> bool {
        !self.busy(key) && !self.encoding.contains(key)
    }

    /// Whether the cache holds the entry for later steps: it is cached and
    /// no change has made it stale. A step in flight may still move it.
    fn holds(&self, cache: &Cache, key: &CacheKey) -> bool {
        cache.contains(key) && !self.doomed.contains(key)
    }

    /// Whether a new step reads the entry as it is, encoding nothing.
    fn reuses(&self, cache: &Cache, key: &CacheKey) -> bool {
        cache.contains(key) && !self.busy(key)
    }

    /// Records what a step about to run holds: the entries it moves and
    /// encodes, and the tables it fits.
    fn hold(&mut self, step: &Step) {
        self.moving.extend(
            step.movements
                .iter()
                .filter(|movement| movement.to.is_some())
                .map(|movement| movement.key),
        );
        self.encoding.extend(encoded(step));
        self.fitting.extend(fitted_tables(step));
    }

    /// Forgets what a finished step held.
    fn release(&mut self, step: &Step) {
        for movement in &step.movements {
            self.moving.remove(&movement.key);
        }
        for key in encoded(step) {
            self.encoding.remove(&key);
        }
        for table in fitted_tables(step) {
            self.fitting.remove(&table);
        }
    }
}

/// What planning a step has taken so far: the cache pins it holds, the
/// entries it admitted, the fits it marked and the region rebuilds it
/// claimed. A step that plans whole hands them to the step, whose release
/// gives them back; one that fails partway gives every one back at once, see
/// `Node::unwind`, so nothing it took outlives it.
#[derive(Debug, Default)]
struct Ledger {
    pins: Vec<CacheKey>,
    admitted: Vec<CacheKey>,
    fits: Vec<FitMark>,
    /// Each rebuild's entry, table and region.
    rebuilds: Vec<(CacheKey, TableId, usize)>,
}

/// A table's fit as it stood before a step planned its own, which an
/// unwound plan puts back: the state fit, and what drift and refits are
/// measured against. The running column sums are not kept: a fit recounts
/// them from the live rows, which they already sum, so they stand as they
/// were.
#[derive(Debug)]
struct FitMark {
    table: TableId,
    fitted: Option<Epoch>,
    horizon: Option<i64>,
    fitted_latest: Option<i64>,
    fitted_columns: Option<Fitted>,
    changed: u64,
}

/// A region's context a rebuild step encodes, to replace its built one.
#[derive(Debug)]
struct Rebuilding {
    table: TableId,
    region: usize,
    /// The state and draw of the regions it was planned in.
    epoch: Epoch,
    generation: u64,
    rows: Context,
    /// Changes the region had counted when it was planned.
    seen: u64,
    /// Since when the region was due.
    due: Instant,
}
