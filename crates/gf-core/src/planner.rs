//! Context strategies: which row sets to encode, and what each query decodes against.
//!
//! A plan lists units, each a set of rows encoded together once and cached
//! under a key that is stable across batches, and the unit each query reads.
//!
//! A retrieved context is a core every query of a window shares plus each
//! query's nearest rows outside it, encoded together. The core is a seeded
//! sample: random, stratified by label, balanced across labels, or spread out
//! by farthest-point sampling. Seeded cores change little as rows arrive: a
//! new row enters only if it hashes below the members of its stratum. Rows
//! hash by key, not by row id, and rows at equal distances tie to the lower
//! key, so which rows a plan picks does not depend on the order they were
//! written in. The retrieval scaling that distances are measured in still
//! does; see `regions`.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use crate::context::{Context, Fingerprint, mix};
use crate::pages::{IndexError, PageIndex, Vectors};
use crate::table::{Keys, RowId, Snapshot};
use crate::time_index::TimeIndex;

/// How a core's rows are drawn from a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreKind {
    /// The rows with the smallest seeded hashes.
    Random,
    /// Each stratum's smallest hashes, in proportion to the stratum's rows.
    /// A regression table's strata are 8 bins of its targets, set from its
    /// live rows at each state's first plan, also for a core drawn inside a
    /// cutoff's or a recency window.
    Stratified,
    /// Each stratum's smallest hashes, an equal share each; a stratum with
    /// fewer rows than its share gives the rest to the others.
    Balanced,
    /// Farthest-point sampling in the index's space, from the row with the
    /// smallest hash, so the core spreads over the table.
    Diverse,
}

impl CoreKind {
    /// Every kind, in order.
    pub const ALL: [Self; 4] = [
        Self::Random,
        Self::Stratified,
        Self::Balanced,
        Self::Diverse,
    ];

    /// The kind's name, as clients and logs spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Stratified => "stratified",
            Self::Balanced => "balanced",
            Self::Diverse => "diverse",
        }
    }
}

impl std::str::FromStr for CoreKind {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.name() == name)
            .ok_or_else(|| format!("unknown core kind {name:?}"))
    }
}

/// Rows every query of a window shares, drawn by `kind` under `seed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Core {
    pub rows: u64,
    pub seed: u64,
    pub kind: CoreKind,
}

/// How to choose each query's context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Every live row, for every query.
    Exact,
    /// A core shared by every query of a window, plus each query's
    /// `neighbors` nearest rows outside it, encoded together. Without
    /// neighbors it is a subsample; without a core, each query's nearest rows.
    Retrieved {
        core: Option<Core>,
        neighbors: usize,
        /// Neighbors kept as the `neighbors` latest of this many nearest,
        /// rather than the nearest alone.
        among: Option<usize>,
        /// The recent rows of its window the whole context is drawn from.
        recency: Option<Recency>,
    },
}

/// The recent rows of a window a context keeps, counted back from the
/// window's newest row, so a replay plans as it ran and a live table needs
/// no clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Recency {
    /// The latest rows by time, untimed rows the oldest.
    Last(u64),
    /// Rows at most this long before the newest row's time; untimed rows
    /// are left out.
    Span(u64),
}

/// Seeds the sample that stands in for every row past a model's trained length.
const CAP_SEED: u64 = 0x6361_705f_726f_7773;

/// The most rows a strategy's count may name, since no table holds more.
pub const MAX_STRATEGY_ROWS: u64 = u32::MAX as u64;

/// Why a strategy cannot be used: a core of no rows, a context of no rows,
/// or a count past any table's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyError {
    ZeroRows,
    Empty,
    /// Neighbors kept among no nearest rows.
    AmongZero,
    /// Neighbors kept among fewer nearest rows than they are.
    FewerAmong,
    /// A window of the latest rows holding none.
    LastZero,
    TooLarge,
}

impl fmt::Display for StrategyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRows => write!(formatter, "a core's rows must be positive"),
            Self::Empty => write!(formatter, "a strategy needs a core or neighbors"),
            Self::AmongZero => write!(formatter, "among must be positive"),
            Self::FewerAmong => write!(
                formatter,
                "neighbors are kept among at least as many nearest rows"
            ),
            Self::LastZero => write!(formatter, "a window of the latest rows must hold some"),
            Self::TooLarge => write!(
                formatter,
                "a strategy's counts are at most {MAX_STRATEGY_ROWS} rows"
            ),
        }
    }
}

impl std::error::Error for StrategyError {}

impl Strategy {
    /// Checks that a core has rows, a retrieved context has a core or
    /// neighbors, and every count is at most `MAX_STRATEGY_ROWS`.
    ///
    /// # Errors
    ///
    /// Naming the first problem.
    pub fn check(&self) -> Result<(), StrategyError> {
        let Self::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = *self
        else {
            return Ok(());
        };
        let counts = [
            core.map_or(0, |core| core.rows),
            neighbors as u64,
            among.unwrap_or(0) as u64,
            match recency {
                Some(Recency::Last(rows)) => rows,
                _ => 0,
            },
        ];
        match core {
            Some(core) if core.rows == 0 => Err(StrategyError::ZeroRows),
            None if neighbors == 0 => Err(StrategyError::Empty),
            _ if among == Some(0) => Err(StrategyError::AmongZero),
            _ if among.is_some_and(|among| among < neighbors) => Err(StrategyError::FewerAmong),
            _ if recency == Some(Recency::Last(0)) => Err(StrategyError::LastZero),
            _ if counts.iter().any(|&count| count > MAX_STRATEGY_ROWS) => {
                Err(StrategyError::TooLarge)
            }
            _ => Ok(()),
        }
    }

    /// A core alone: one sample every query of a window shares.
    #[must_use]
    pub const fn subsample(rows: u64, seed: u64, kind: CoreKind) -> Self {
        Self::Retrieved {
            core: Some(Core { rows, seed, kind }),
            neighbors: 0,
            among: None,
            recency: None,
        }
    }

    /// Each query's `rows` nearest rows, without a core.
    #[must_use]
    pub const fn neighbors(rows: usize) -> Self {
        Self::Retrieved {
            core: None,
            neighbors: rows,
            among: None,
            recency: None,
        }
    }

    /// The strategy cut to contexts of at most `rows` rows: every row becomes
    /// a random sample of `rows`, and a retrieved context keeps at most `rows`
    /// of its core and fills what is left with neighbors.
    #[must_use]
    pub fn capped(self, rows: u64) -> Self {
        match self {
            Self::Exact => Self::subsample(rows, CAP_SEED, CoreKind::Random),
            Self::Retrieved {
                core,
                neighbors,
                among,
                recency,
            } => {
                let core = core.map(|core| Core {
                    rows: core.rows.min(rows),
                    ..core
                });
                let left = rows - core.map_or(0, |core| core.rows);
                Self::Retrieved {
                    core,
                    neighbors: neighbors.min(usize::try_from(left).unwrap_or(usize::MAX)),
                    among,
                    recency,
                }
            }
        }
    }

    /// The core's seed, which the live rows are ranked by, if it has a core.
    #[must_use]
    pub fn seed(&self) -> Option<u64> {
        match self {
            Self::Retrieved {
                core: Some(core), ..
            } => Some(core.seed),
            _ => None,
        }
    }
}

/// Why a batch could not be planned. Nothing is changed then.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    Index(IndexError),
    /// The windows name a different number of queries than were given.
    Windows {
        queries: usize,
        windows: usize,
    },
    /// A query names a window that was not given.
    UnknownWindow(usize),
    /// A given context with no live rows.
    Given,
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(error) => error.fmt(formatter),
            Self::Windows { queries, windows } => {
                write!(formatter, "{queries} queries but {windows} windows")
            }
            Self::UnknownWindow(window) => write!(formatter, "window {window} does not exist"),
            Self::Given => write!(formatter, "a given context has no live rows"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<IndexError> for PlanError {
    fn from(error: IndexError) -> Self {
        Self::Index(error)
    }
}

/// A unit's position in its plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(pub usize);

/// A set of rows encoded together once, and their fingerprint, which keys
/// its cache entry within one model and table state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub rows: Context,
    pub key: Fingerprint,
}

/// The units to encode for a batch of queries, and the unit each query reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub(crate) units: Vec<Unit>,
    pub(crate) queries: Vec<UnitId>,
}

impl Plan {
    /// The units to encode, each once.
    #[must_use]
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// The unit each query reads.
    #[must_use]
    pub fn queries(&self) -> &[UnitId] {
        &self.queries
    }
}

/// Rows ranked by seeded hash, smallest first, kept as rows come and go, so a
/// sample is a scan of the ranking rather than a sort of every row.
///
/// A new row enters a sample only if it hashes below the current members, so
/// the sample, and the cache keyed on it, changes little as rows arrive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ranking {
    seed: u64,
    order: BTreeSet<(u64, u32)>,
}

impl Ranking {
    /// The ranking of `rows`, whose keys `keys` holds, under `seed`.
    #[must_use]
    pub fn new(seed: u64, rows: &Context, keys: &dyn Keys) -> Self {
        Self {
            seed,
            order: rows
                .iter()
                .map(|row| (hash(seed, keys.key(row)), row.0))
                .collect(),
        }
    }

    /// A ranking of no rows under `seed`.
    #[must_use]
    pub fn empty(seed: u64) -> Self {
        Self {
            seed,
            order: BTreeSet::new(),
        }
    }

    /// The seed the rows are ranked by.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Ranks a new row of this key.
    pub fn insert(&mut self, row: RowId, key: u128) {
        self.order.insert((hash(self.seed, key), row.0));
    }

    /// Forgets a row of this key.
    pub fn remove(&mut self, row: RowId, key: u128) {
        self.order.remove(&(hash(self.seed, key), row.0));
    }

    fn rows(&self) -> impl Iterator<Item = RowId> + '_ {
        self.order.iter().map(|&(_, row)| RowId(row))
    }
}

/// A seeded hash of a row's key, or of a stratum.
pub(crate) fn hash(seed: u64, key: u128) -> u64 {
    mix(seed ^ mix(key as u64 ^ mix((key >> 64) as u64)))
}

/// Each row's stratum for stratified and balanced cores: its class, or the
/// bin its target falls in.
pub trait Strata {
    fn stratum(&self, row: RowId) -> u32;
}

/// What planning reads of a table besides its queries.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub snapshot: &'a Snapshot,
    pub index: &'a PageIndex,
    /// Each row's vector in the index's space.
    pub vectors: &'a dyn Vectors,
    pub strata: &'a dyn Strata,
    /// Each row's key, which seeded samples hash.
    pub keys: &'a dyn Keys,
    /// The caller's ranking of the live rows, if it keeps one for the core's seed.
    pub ranking: Option<&'a Ranking>,
    /// Each row's time, for recency and among.
    pub times: &'a TimeIndex,
}

/// Diverse cores most recently drawn, by window and size, kept until the
/// table's state is fit again: farthest-point sampling reads every row of
/// its window, so a core is drawn once and loses only the rows that go. The
/// whole table is the window None, so it keeps its core as rows arrive. Only
/// the `KEPT_CORES` latest are kept, so a replay over many cutoffs holds few.
#[derive(Clone, Debug, Default)]
pub struct Cores(Vec<(CoreWindow, Context)>);

/// A window's fingerprint, or None for every live row, and a core's size.
type CoreWindow = (Option<Fingerprint>, u64);

/// Diverse cores kept at most.
const KEPT_CORES: usize = 8;

impl Cores {
    /// Forgets every core, as a new state does.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// The core kept for `window`, or one `draw` makes, kept as the latest.
    fn get_or_draw(
        &mut self,
        window: CoreWindow,
        draw: impl FnOnce() -> Result<Context, IndexError>,
    ) -> Result<&Context, IndexError> {
        let entry = match self.0.iter().position(|(kept, _)| *kept == window) {
            Some(position) => self.0.remove(position),
            None => (window, draw()?),
        };
        if self.0.len() == KEPT_CORES {
            self.0.remove(0);
        }
        self.0.push(entry);
        Ok(&self.0.last().expect("pushed above").1)
    }
}

/// The windows a batch's queries read.
#[derive(Clone, Copy, Debug)]
pub enum Windows<'a> {
    /// Each window's rows, given whole.
    Given(&'a [Window]),
    /// The live rows before each of ascending cutoffs, with the fingerprint of
    /// each window's live rows. Planning grows one window into the next, so it
    /// holds only one at a time however many cutoffs there are.
    Before {
        times: &'a TimeIndex,
        cutoffs: &'a [i64],
        prints: &'a [Fingerprint],
    },
}

impl Windows<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Given(windows) => windows.len(),
            Self::Before { cutoffs, .. } => cutoffs.len(),
        }
    }
}

/// The rows before one cutoff after another, ascending, grown rather than
/// gathered again for each.
#[derive(Default)]
struct Growing {
    rows: Context,
    /// The cutoff `rows` reaches, once grown.
    reached: Option<i64>,
}

impl Growing {
    /// Grows the rows to the `live` ones before `cutoff`; returns the rows added.
    fn grow(&mut self, times: &TimeIndex, cutoff: i64, live: &Context) -> Context {
        let added = &times.added_before(self.reached, cutoff) & live;
        self.reached = Some(cutoff);
        self.rows |= &added;
        added
    }
}

/// Samples of growing windows, by seed and count, kept as the window grows.
#[derive(Default)]
struct Samples(HashMap<(u64, u64), Sample>);

impl Samples {
    fn add(&mut self, rows: &Context, keys: &dyn Keys) {
        for sample in self.0.values_mut() {
            sample.add(rows, keys);
        }
    }

    /// The `count` rows of `window` with the smallest hashes under `seed`:
    /// the same rows as a ranking's first `count` within the window.
    fn of(&mut self, window: &Context, count: u64, seed: u64, keys: &dyn Keys) -> Context {
        self.0
            .entry((seed, count))
            .or_insert_with(|| {
                let mut sample = Sample {
                    seed,
                    count: usize::try_from(count).unwrap_or(usize::MAX),
                    best: BTreeSet::new(),
                };
                sample.add(window, keys);
                sample
            })
            .rows()
    }
}

/// The `count` rows of `rows` with the smallest hashes of their keys under
/// `seed`: a sample that changes little as rows come and go.
#[must_use]
pub fn smallest(seed: u64, rows: &Context, count: usize, keys: &dyn Keys) -> Context {
    let mut sample = Sample {
        seed,
        count,
        best: BTreeSet::new(),
    };
    sample.add(rows, keys);
    sample.rows()
}

/// The rows with the smallest seeded hashes among those added.
struct Sample {
    seed: u64,
    count: usize,
    best: BTreeSet<(u64, u32)>,
}

impl Sample {
    fn add(&mut self, rows: &Context, keys: &dyn Keys) {
        for row in rows.iter() {
            self.best.insert((hash(self.seed, keys.key(row)), row.0));
            if self.best.len() > self.count {
                self.best.pop_last();
            }
        }
    }

    fn rows(&self) -> Context {
        self.best.iter().map(|&(_, row)| RowId(row)).collect()
    }
}

/// Rows queries may read, such as those before a time cutoff.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub rows: Context,
    /// The fingerprint of `rows`, if the caller keeps it, which saves hashing
    /// each row; then `rows` must be live in the snapshot planned against.
    pub fingerprint: Option<Fingerprint>,
}

/// Builds a plan, encoding each distinct unit once.
struct Builder<'a> {
    snapshot: &'a Snapshot,
    plan: Plan,
    ids: HashMap<Fingerprint, UnitId>,
    /// The caller's ranking of the live rows, if it keeps one.
    ranking: Option<&'a Ranking>,
    /// Rankings made for this plan, for seeds the caller keeps none of.
    ranked: HashMap<u64, Ranking>,
}

impl Builder<'_> {
    /// The live rows ranked under `seed`: the caller's, or one made once.
    fn ranking(&mut self, live: &Context, seed: u64, keys: &dyn Keys) -> &Ranking {
        match self.ranking {
            Some(ranking) if ranking.seed == seed => ranking,
            _ => self
                .ranked
                .entry(seed)
                .or_insert_with(|| Ranking::new(seed, live, keys)),
        }
    }

    /// The `count` rows of `allowed` with the smallest seeded hashes.
    ///
    /// Rows are ranked over every live row, so a window's sample is the first
    /// `count` ranked rows the window holds.
    fn sample(
        &mut self,
        live: &Context,
        allowed: &Context,
        count: u64,
        seed: u64,
        keys: &dyn Keys,
    ) -> Context {
        let whole = std::ptr::eq(allowed, live) || allowed == live;
        self.ranking(live, seed, keys)
            .rows()
            .filter(|&row| whole || allowed.contains(row))
            .take(usize::try_from(count).unwrap_or(usize::MAX))
            .collect()
    }

    /// The core of `allowed`; a diverse one is kept in `cores` under `key`,
    /// the window's fingerprint, or None for every live row.
    fn core(
        &mut self,
        core: Core,
        live: &Context,
        allowed: &Context,
        key: Option<Fingerprint>,
        cores: &mut Cores,
        source: &Source<'_>,
    ) -> Result<Context, IndexError> {
        Ok(match core.kind {
            CoreKind::Random => self.sample(live, allowed, core.rows, core.seed, source.keys),
            CoreKind::Stratified | CoreKind::Balanced => {
                self.stratified(live, allowed, core, source)
            }
            CoreKind::Diverse => {
                let drawn = cores.get_or_draw((key, core.rows), || {
                    farthest(allowed, core.rows, core.seed, source)
                })?;
                drawn & allowed
            }
        })
    }

    /// Each stratum's rows of `allowed` with the smallest seeded hashes, as
    /// many as `quotas` gives it from the strata's sizes.
    ///
    /// One ranking serves every stratum: restricted to a stratum it is that
    /// stratum's own ranking, so a new row enters only below its members.
    fn stratified(
        &mut self,
        live: &Context,
        allowed: &Context,
        core: Core,
        source: &Source<'_>,
    ) -> Context {
        let strata = source.strata;
        let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
        for row in allowed.iter() {
            *sizes.entry(strata.stratum(row)).or_default() += 1;
        }
        let mut quotas = if core.kind == CoreKind::Balanced {
            balanced(&sizes, core.rows, core.seed)
        } else {
            proportional(&sizes, core.rows)
        };
        let mut left: u64 = quotas.values().sum();
        let whole = std::ptr::eq(allowed, live) || allowed == live;
        let mut rows = Vec::new();
        for row in self.ranking(live, core.seed, source.keys).rows() {
            if left == 0 {
                break;
            }
            if !whole && !allowed.contains(row) {
                continue;
            }
            let quota = quotas.entry(strata.stratum(row)).or_default();
            if *quota > 0 {
                *quota -= 1;
                left -= 1;
                rows.push(row);
            }
        }
        rows.into_iter().collect()
    }

    fn unit(&mut self, rows: Context) -> UnitId {
        let key = self.snapshot.fingerprint(&rows);
        self.known_unit(rows, key)
    }

    /// A unit of rows whose fingerprint is known.
    fn known_unit(&mut self, rows: Context, key: Fingerprint) -> UnitId {
        *self.ids.entry(key).or_insert_with(|| {
            self.plan.units.push(Unit { rows, key });
            UnitId(self.plan.units.len() - 1)
        })
    }
}

/// Shares of `count` rows in proportion to the strata's sizes, by largest
/// remainder, ties to the lower stratum.
fn proportional(sizes: &BTreeMap<u32, u64>, count: u64) -> BTreeMap<u32, u64> {
    let total: u64 = sizes.values().sum();
    let count = count.min(total);
    if total == 0 {
        return BTreeMap::new();
    }
    let mut quotas: BTreeMap<u32, u64> = BTreeMap::new();
    let mut remainders: Vec<(u128, u32)> = Vec::with_capacity(sizes.len());
    for (&stratum, &size) in sizes {
        let exact = u128::from(count) * u128::from(size);
        quotas.insert(stratum, (exact / u128::from(total)) as u64);
        remainders.push((exact % u128::from(total), stratum));
    }
    let given: u64 = quotas.values().sum();
    remainders.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    for &(_, stratum) in remainders.iter().take((count - given) as usize) {
        *quotas.get_mut(&stratum).expect("every stratum has a quota") += 1;
    }
    quotas
}

/// Equal shares of `count` rows, each at most its stratum's size; what a
/// small stratum cannot take goes to the others. Rows that do not divide
/// evenly go to strata in the order of their hashes under `seed`.
fn balanced(sizes: &BTreeMap<u32, u64>, count: u64, seed: u64) -> BTreeMap<u32, u64> {
    let mut quotas: BTreeMap<u32, u64> = sizes.keys().map(|&stratum| (stratum, 0)).collect();
    let mut left = count.min(sizes.values().sum());
    let mut open: Vec<u32> = sizes.keys().copied().collect();
    open.sort_by_key(|&stratum| (hash(seed, u128::from(stratum)), stratum));
    while left > 0 && !open.is_empty() {
        let share = (left / open.len() as u64).max(1);
        open.retain(|stratum| {
            let quota = quotas.get_mut(stratum).expect("every stratum has a quota");
            let take = share.min(sizes[stratum] - *quota).min(left);
            *quota += take;
            left -= take;
            *quota < sizes[stratum]
        });
    }
    quotas
}

/// `count` rows of `allowed` by farthest-point sampling over their vectors:
/// the row whose key hashes smallest under `seed`, then each time the row
/// farthest from those chosen, ties to the lower key.
///
/// # Errors
///
/// If a row's vector cannot be read.
fn farthest(
    allowed: &Context,
    count: u64,
    seed: u64,
    source: &Source<'_>,
) -> Result<Context, IndexError> {
    let rows: Vec<RowId> = allowed.iter().collect();
    let keys: Vec<u128> = rows.iter().map(|&row| source.keys.key(row)).collect();
    let Some(first) = (0..rows.len()).min_by_key(|&index| (hash(seed, keys[index]), keys[index]))
    else {
        return Ok(Context::new());
    };
    let points: Vec<Vec<f32>> = rows
        .iter()
        .map(|&row| source.vectors.vector(row))
        .collect::<Result<_, _>>()?;
    let mut nearest = vec![f64::INFINITY; rows.len()];
    let mut chosen = Vec::new();
    let mut next = first;
    for _ in 0..count.min(rows.len() as u64) {
        chosen.push(rows[next]);
        nearest[next] = f64::NEG_INFINITY;
        let mut farthest = (f64::NEG_INFINITY, next);
        for (index, point) in points.iter().enumerate() {
            if nearest[index] == f64::NEG_INFINITY {
                continue;
            }
            let distance: f64 = point
                .iter()
                .zip(&points[next])
                .map(|(&value, &other)| (f64::from(value) - f64::from(other)).powi(2))
                .sum();
            nearest[index] = nearest[index].min(distance);
            let ahead = nearest[index]
                .total_cmp(&farthest.0)
                .then(keys[farthest.1].cmp(&keys[index]));
            if ahead.is_gt() {
                farthest = (nearest[index], index);
            }
        }
        next = farthest.1;
    }
    Ok(chosen.into_iter().collect())
}

/// Plans a batch of queries against one snapshot, each reading any live row.
///
/// Rows the index still holds but the snapshot no longer has are left out.
///
/// # Errors
///
/// If a query vector has the wrong length for the index.
pub fn plan(
    strategy: &Strategy,
    source: &Source<'_>,
    queries: &[impl AsRef<[f32]>],
) -> Result<Plan, PlanError> {
    plan_within(strategy, source, queries, &[], &vec![None; queries.len()])
}

/// Plans a batch of queries, each reading only the live rows of its window.
///
/// `windows` are the distinct sets of rows queries may read, such as the rows
/// before a time cutoff, and `window_of` names each query's window, or None for
/// every live row. Units every query shares (the exact context, a core) are
/// shared within a window, and neighbors are found among its rows.
///
/// # Errors
///
/// If a query vector has the wrong length for the index, `window_of` has a
/// different length than `queries`, or a query names a window that was not given.
pub fn plan_within(
    strategy: &Strategy,
    source: &Source<'_>,
    queries: &[impl AsRef<[f32]>],
    windows: &[Context],
    window_of: &[Option<usize>],
) -> Result<Plan, PlanError> {
    let windows: Vec<Window> = windows
        .iter()
        .map(|rows| Window {
            rows: rows.clone(),
            fingerprint: None,
        })
        .collect();
    plan_with(
        strategy,
        source,
        &mut Cores::default(),
        queries,
        Windows::Given(&windows),
        window_of,
    )
}

/// Plans as `plan_within` does, with windows whose fingerprints the caller
/// may keep, and diverse cores it keeps between plans.
///
/// # Errors
///
/// As `plan_within`, or if a row's vector cannot be read for a diverse core.
pub fn plan_with(
    strategy: &Strategy,
    source: &Source<'_>,
    cores: &mut Cores,
    queries: &[impl AsRef<[f32]>],
    windows: Windows<'_>,
    window_of: &[Option<usize>],
) -> Result<Plan, PlanError> {
    if window_of.len() != queries.len() {
        return Err(PlanError::Windows {
            queries: queries.len(),
            windows: window_of.len(),
        });
    }
    if let Some(&window) = window_of
        .iter()
        .flatten()
        .find(|&&window| window >= windows.len())
    {
        return Err(PlanError::UnknownWindow(window));
    }
    let snapshot = source.snapshot;
    let live = snapshot.rows();
    let mut builder = Builder {
        snapshot,
        plan: Plan::default(),
        ids: HashMap::new(),
        ranking: source.ranking,
        ranked: HashMap::new(),
    };
    // Queries by window: those reading every live row, then each window in
    // turn, so nested windows grow one into the next.
    let mut groups: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for (query, &window) in window_of.iter().enumerate() {
        groups.entry(window).or_default().push(query);
    }
    let mut planned: Vec<UnitId> = vec![UnitId(0); queries.len()];
    let mut growing = Growing::default();
    let mut samples = Samples::default();
    for (window, members) in groups {
        let nested = matches!(windows, Windows::Before { .. }) && window.is_some();
        // A window with a fingerprint is live already, so it needs no intersection.
        let (allowed, known): (Cow<Context>, Option<Fingerprint>) = match (window, windows) {
            (None, _) => (Cow::Borrowed(&live), None),
            (Some(window), Windows::Given(given)) => match given[window].fingerprint {
                Some(fingerprint) => (Cow::Borrowed(&given[window].rows), Some(fingerprint)),
                None => (Cow::Owned(&given[window].rows & &live), None),
            },
            (
                Some(window),
                Windows::Before {
                    times,
                    cutoffs,
                    prints,
                },
            ) => {
                samples.add(&growing.grow(times, cutoffs[window], &live), source.keys);
                (Cow::Borrowed(&growing.rows), Some(prints[window]))
            }
        };
        let Strategy::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = *strategy
        else {
            let allowed = allowed.as_ref();
            let unit = match known {
                Some(fingerprint) => builder.known_unit(allowed.clone(), fingerprint),
                None => builder.unit(allowed.clone()),
            };
            for query in members {
                planned[query] = unit;
            }
            continue;
        };
        // The whole context comes from the window's recent rows, if it keeps
        // only those, which change as rows arrive, so a diverse core is kept
        // by their fingerprint.
        let (allowed, known) = match recency {
            None => (allowed, known),
            Some(recency) => {
                let before = match (window, windows) {
                    (Some(window), Windows::Before { cutoffs, .. }) => Some(cutoffs[window]),
                    _ => None,
                };
                let recent = recent(source.times, allowed.as_ref(), recency, before);
                let fingerprint = snapshot.fingerprint(&recent);
                (Cow::Owned(recent), Some(fingerprint))
            }
        };
        let allowed = allowed.as_ref();
        let keyed = window.is_some() || recency.is_some();
        // The window's core, and the rows outside it neighbors come from.
        let core = match core {
            None => Context::new(),
            Some(core) if nested && recency.is_none() && core.kind == CoreKind::Random => {
                samples.of(allowed, core.rows, core.seed, source.keys)
            }
            Some(core) => {
                let key = (core.kind == CoreKind::Diverse && keyed)
                    .then(|| known.unwrap_or_else(|| snapshot.fingerprint(allowed)));
                builder.core(core, &live, allowed, key, cores, source)?
            }
        };
        if neighbors == 0 {
            let unit = builder.unit(core);
            for query in members {
                planned[query] = unit;
            }
            continue;
        }
        let outside = if core.is_empty() {
            Cow::Borrowed(allowed)
        } else {
            Cow::Owned(allowed - &core)
        };
        for query in members {
            let vector = queries[query].as_ref();
            let nearest = source.index.nearest_rows(
                vector,
                among.unwrap_or(neighbors),
                &outside,
                source.vectors,
                source.keys,
            )?;
            let nearest = match among {
                Some(_) => source.times.latest_of(&nearest, neighbors as u64),
                None => nearest,
            };
            planned[query] = builder.unit(&core | &nearest);
        }
    }
    builder.plan.queries = planned;
    Ok(builder.plan)
}

/// The recent rows of `window` that `recency` keeps; `before`, if given, is
/// a cutoff every row of the window comes before.
#[must_use]
pub fn recent(
    times: &TimeIndex,
    window: &Context,
    recency: Recency,
    before: Option<i64>,
) -> Context {
    match recency {
        Recency::Last(rows) => times.latest(window, rows, before),
        Recency::Span(span) => times.recent(window, span, before),
    }
}

/// The core a plan would draw from `allowed`, a window of live rows, as for
/// a table's partition offering its share of a core.
///
/// # Errors
///
/// If a row's vector cannot be read for a diverse core.
pub fn draw_core(
    core: Core,
    source: &Source<'_>,
    cores: &mut Cores,
    allowed: &Context,
) -> Result<Context, PlanError> {
    let live = source.snapshot.rows();
    let key = (allowed != &live).then(|| source.snapshot.fingerprint(allowed));
    let mut builder = Builder {
        snapshot: source.snapshot,
        plan: Plan::default(),
        ids: HashMap::new(),
        ranking: source.ranking,
        ranked: HashMap::new(),
    };
    Ok(builder.core(core, &live, allowed, key, cores, source)?)
}

/// Plans queries whose contexts were chosen elsewhere, such as from rows
/// gathered across a table's partitions: each query reads its given context,
/// encoded once however many queries share it. Rows the snapshot no longer
/// has are left out.
///
/// # Errors
///
/// `PlanError::Given` if a query's context holds no live rows.
pub fn plan_given(snapshot: &Snapshot, given: &[Context]) -> Result<Plan, PlanError> {
    let live = snapshot.rows();
    let mut builder = Builder {
        snapshot,
        plan: Plan::default(),
        ids: HashMap::new(),
        ranking: None,
        ranked: HashMap::new(),
    };
    let mut planned = Vec::with_capacity(given.len());
    for rows in given {
        let rows = rows & &live;
        if rows.is_empty() {
            return Err(PlanError::Given);
        }
        planned.push(builder.unit(rows));
    }
    builder.plan.queries = planned;
    Ok(builder.plan)
}
