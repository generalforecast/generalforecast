//! Forecasting: groups of series planned as units, laid onto grids.
//!
//! A series table's forecast request names series and the cutoff each is
//! forecast from. The planner puts each series in a group with the others
//! its strategy picks, lays every member's rows before the cutoff onto one
//! grid, and keys the unit by its members, their rows, the cutoff and the
//! grid length, so queries naming the same group at the same cutoff share
//! one encoding. A member contributes a row for its target and a row per
//! covariate column; the columns the table declares known over the horizon
//! are read from the rows after the cutoff as well, so a forecast carries
//! them to the model. A member whose grid sits at another offset than the
//! cutoff is read at its own last grid point before it.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

use crate::cache::{CacheKey, Epoch};
use crate::context::{Context, Fingerprint, mix};
use crate::rows::Row;
use crate::series::{SeriesError, SeriesIndex, SeriesKey, Window, with_window};
use crate::step::{Placed, RequestId, StepUnit};
use crate::table::{RowId, Snapshot, TableId};

/// The roles of a group's rows, as the models read them.
pub const TARGET: i32 = 0;
pub const PAST: i32 = 1;
pub const KNOWN: i32 = 2;

/// Most passes of a forecaster's network one forecast runs: the cap on a
/// horizon, in steps one pass predicts, so one call cannot hold a shard for
/// minutes.
pub const HORIZON_PASSES: usize = 64;

/// The largest horizon, in steps, that one forecast may request.
#[must_use]
pub fn horizon_limit(output_patch: NonZeroUsize, pass_limit: Option<NonZeroUsize>) -> NonZeroUsize {
    NonZeroUsize::new(HORIZON_PASSES.saturating_mul(pass_limit.unwrap_or(output_patch).get()))
        .expect("a horizon limit is nonzero")
}

/// Which other series join a series in its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupStrategy {
    /// None: each series with its own covariate columns.
    Alone,
    /// `size - 1` others by seeded hash with the series, so its group changes
    /// little as series come and go; a table holding fewer refuses the
    /// request rather than drawing a smaller group.
    Random { size: NonZeroUsize, seed: u64 },
    /// The members each forecast names, up to `size` series in all, such as
    /// the nearest series an index outside the table chose; the table draws
    /// none, and refuses a forecast that names no group.
    Given { size: NonZeroUsize },
}

impl GroupStrategy {
    /// The most series in a group, the forecast one included: 1 alone.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Alone => 1,
            Self::Random { size, .. } | Self::Given { size } => size.get(),
        }
    }
}

/// The role the other series of a group take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Neighbors {
    /// Targets forecast jointly, which every member's query shares.
    #[default]
    Targets,
    /// Covariates the target reads and that do not read it; then each
    /// member's query has a group of its own.
    Covariates,
}

/// How a table's series are forecast.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesSpec {
    /// The grid's step, in the table's time unit.
    pub step: NonZeroU64,
    /// Steps per patch of the model; grids are whole patches.
    pub patch: NonZeroUsize,
    /// Most context steps a grid holds; fewer than a patch holds one patch.
    pub context: NonZeroUsize,
    /// Most steps one forecast covers: HORIZON_PASSES passes of the model's output.
    pub max_horizon: NonZeroUsize,
    /// Most rows in a group, from the model, or None for any number.
    pub max_rows: Option<NonZeroUsize>,
    /// Feature columns known over the horizon, read from rows after the cutoff.
    pub known: Vec<usize>,
    pub group: GroupStrategy,
    pub neighbors: Neighbors,
    /// Whether the model reuses a context encoding across horizons, so a unit
    /// without a known covariate is kept rather than forecast whole.
    pub reusable: bool,
    /// Whether the model grows an encoding by whole patches of new steps
    /// exactly (`extend`), so a kept unit whose cutoff advances by whole
    /// patches is extended from the cached one rather than encoded again.
    pub extendable: bool,
    /// Cache bytes per patch beside those per row, which `EntrySizes` holds.
    pub patch_bytes: u64,
}

/// A request to forecast series, each from its cutoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesRequest {
    pub series: Vec<SeriesKey>,
    /// Each series' cutoff, or None for the step after its latest observed value.
    pub cutoffs: Vec<Option<i64>>,
    pub horizon: NonZeroUsize,
    /// Each series' group, the series itself in it or not, for a table of
    /// given groups; None for a table that draws its own.
    pub groups: Option<Vec<Vec<SeriesKey>>>,
}

/// Why a request could not be planned, or a row placed in a series table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastError {
    /// A forecast asked of a table that holds no series.
    NotSeries,
    /// Query rows sent to a table of series, which takes forecasts.
    OnlyForecasts,
    /// A forecast horizon longer than the model's maximum.
    HorizonTooLong { horizon: usize, limit: usize },
    /// A row of a series table without a time.
    Untimed,
    /// Not one cutoff per series.
    Cutoffs { series: usize, cutoffs: usize },
    /// Not one group per series.
    Groups { series: usize, groups: usize },
    /// A series the table does not hold, forecast without a group.
    UnknownSeries(SeriesKey),
    /// A given group names a series the table does not hold.
    UnknownMember(SeriesKey),
    /// A series forecast without its group on a table of given groups.
    NoGroup(SeriesKey),
    /// Groups given to a table that draws its own.
    UngivenGroups,
    /// A given group of more series than the table's size.
    GroupPastSize { given: usize, size: usize },
    /// A series without an observed value has no time to forecast from.
    NoRows(SeriesKey),
    /// A cutoff that no observation of the series' group comes before: an
    /// empty context.
    NothingBefore { series: SeriesKey, cutoff: i64 },
    /// One series' rows alone pass the model's cap on a group.
    TooManyRows { rows: usize, cap: usize },
    /// A group of the strategy's size would pass the model's cap on rows, so
    /// it is refused rather than drawn smaller than asked.
    GroupTooLarge {
        size: usize,
        rows: usize,
        cap: usize,
    },
    /// The table holds too few other series for a group of the strategy's
    /// size, so it is refused rather than drawn smaller than asked.
    TooFewSeries { size: usize, held: usize },
    /// A known column the table does not have.
    KnownColumn { column: usize, features: usize },
    /// A time span past `i64`.
    Series(SeriesError),
}

impl fmt::Display for ForecastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSeries => write!(formatter, "the table holds no series to forecast"),
            Self::OnlyForecasts => {
                write!(
                    formatter,
                    "a table of series takes forecasts, not query rows"
                )
            }
            Self::HorizonTooLong { horizon, limit } => write!(
                formatter,
                "a horizon of {horizon} steps is past the {limit} the model forecasts in \
                 {HORIZON_PASSES} passes"
            ),
            Self::Untimed => write!(formatter, "a row of a series table needs a time"),
            Self::Cutoffs { series, cutoffs } => {
                write!(formatter, "{series} series were given {cutoffs} cutoffs")
            }
            Self::Groups { series, groups } => write!(
                formatter,
                "expected {series} groups, one per series, got {groups}"
            ),
            Self::UnknownSeries(series) => write!(
                formatter,
                "the table holds no series {}, and alone it has no group to forecast it from",
                series.0
            ),
            Self::UnknownMember(series) => {
                write!(formatter, "series {} does not exist", series.0)
            }
            Self::NoGroup(series) => write!(
                formatter,
                "the table takes each series' group with the forecast, and series {} came \
                 without one; use forecast_given",
                series.0
            ),
            Self::UngivenGroups => write!(
                formatter,
                "the table draws its own groups; create it with a given-groups strategy to \
                 name them"
            ),
            Self::GroupPastSize { given, size } => write!(
                formatter,
                "a group of {given} series was given, past the table's {size} per group"
            ),
            Self::NoRows(series) => write!(
                formatter,
                "series {} has no observed value to forecast from",
                series.0
            ),
            Self::NothingBefore { series, cutoff } => write!(
                formatter,
                "neither series {} nor its group has an observation before cutoff {cutoff}",
                series.0
            ),
            Self::TooManyRows { rows, cap } => write!(
                formatter,
                "one series takes {rows} rows, past the model's {cap} per group"
            ),
            Self::GroupTooLarge { size, rows, cap } => write!(
                formatter,
                "a group of {size} series takes {rows} rows with the covariates, past \
                 the model's {cap} per group; ask for a smaller group"
            ),
            Self::TooFewSeries { size, held } => write!(
                formatter,
                "a group of {size} series needs {} others, but the table holds {held} other \
                 series; ask for a smaller group",
                size - 1
            ),
            Self::KnownColumn { column, features } => write!(
                formatter,
                "known column {column} is not one of the table's {features} covariate columns"
            ),
            Self::Series(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for ForecastError {}

impl From<SeriesError> for ForecastError {
    fn from(error: SeriesError) -> Self {
        Self::Series(error)
    }
}

/// A group of series laid onto one grid before a cutoff, encoded once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedUnit {
    /// The group's series, in the order their rows take.
    pub members: Vec<SeriesKey>,
    /// The series whose query the unit serves, when the group's roles depend
    /// on it; None when every member's query shares it.
    pub target: Option<SeriesKey>,
    pub cutoff: i64,
    /// Grid points before the cutoff, whole patches.
    pub steps: usize,
    /// Each member's rows on the grid, read at the member's own last grid
    /// point before the cutoff.
    pub windows: Vec<Window>,
    /// The unit's identity: its members, their rows at their versions, the
    /// cutoff, the grid length and, with a target, the target.
    pub key: Fingerprint,
    /// The group's identity alone: its members and, with a target, the
    /// target, whatever the rows or cutoff; what an extension looks up.
    pub group: Fingerprint,
    /// Whether the unit is forecast whole: the model cannot reuse its
    /// encoding, or a known column makes the context read the horizon.
    pub whole: bool,
}

impl PlannedUnit {
    /// Every row on the grid.
    #[must_use]
    pub fn rows(&self) -> Context {
        self.windows
            .iter()
            .fold(Context::new(), |rows, window| &rows | &window.context())
    }
}

/// One series' forecast: which unit it reads, where it sits, and its horizon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedForecast {
    pub unit: usize,
    /// The target's member index in the unit.
    pub target: usize,
    pub horizon: usize,
    /// Each member's rows from the cutoff on, over the horizon, for the known
    /// columns; empty when the table has none.
    pub future: Vec<Window>,
}

/// The units a batch of requests encodes, and each request's forecasts in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeriesPlan {
    pub units: Vec<PlannedUnit>,
    pub forecasts: Vec<Vec<PlannedForecast>>,
}

/// A unit of a step: a group's grid to encode, reuse, extend or forecast whole.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesUnit {
    pub table: TableId,
    pub key: CacheKey,
    pub members: Vec<SeriesKey>,
    pub cutoff: i64,
    /// The group's context, which `Step::take_rows` moves out.
    pub grid: Grid,
    /// What to do with the unit, and the device slots for its context; a
    /// unit forecast whole is transient.
    pub placed: Placed,
    /// For an extension, the entry the grid's last `grown` steps grow.
    pub source: Option<CacheKey>,
    pub grown: usize,
}

/// One series' forecast of a step: the unit it reads and its horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    pub request: RequestId,
    pub table: TableId,
    pub epoch: Epoch,
    /// The step's series unit the forecast reads.
    pub unit: StepUnit,
    /// The target's member index in the unit.
    pub target: usize,
    pub horizon: usize,
    /// The group's known future values over the horizon, which
    /// `Step::take_rows` moves out.
    pub future: Grid,
}

/// A group's rows laid out densely for the model, row-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grid {
    pub rows: usize,
    pub steps: usize,
    /// `rows` by `steps`, 0 where unobserved.
    pub values: Vec<f32>,
    pub observed: Vec<bool>,
    /// Each row's role.
    pub roles: Vec<i32>,
}

impl SeriesSpec {
    /// Rows a member contributes: its target and its covariate columns.
    fn rows_per_member(&self, features: usize) -> usize {
        1 + features
    }

    /// The role of a member's row: its target, or one of its columns.
    fn role(&self, member: usize, target: Option<usize>, row: usize) -> i32 {
        if row == 0 {
            match (self.neighbors, target) {
                (Neighbors::Covariates, Some(target)) if member != target => PAST,
                _ => TARGET,
            }
        } else if self.known.contains(&(row - 1)) {
            KNOWN
        } else {
            PAST
        }
    }

    /// Whether a group with `features` columns holds a known covariate row.
    fn has_known(&self, features: usize) -> bool {
        self.known.iter().any(|&column| column < features)
    }

    /// Whether a group of the strategy's size fits the model's cap on rows,
    /// with `features` covariate rows per member; `check_fit` refuses one
    /// that does not, so a group is never drawn smaller than asked.
    pub fn check_fit(&self, features: usize) -> Result<(), ForecastError> {
        let per_member = self.rows_per_member(features);
        let Some(cap) = self.max_rows else {
            return Ok(());
        };
        if per_member > cap.get() {
            return Err(ForecastError::TooManyRows {
                rows: per_member,
                cap: cap.get(),
            });
        }
        let size = self.group.size();
        if size > 1 && size.saturating_mul(per_member) > cap.get() {
            return Err(ForecastError::GroupTooLarge {
                size,
                rows: size * per_member,
                cap: cap.get(),
            });
        }
        Ok(())
    }

    /// The other series of a group, by the strategy: `size - 1` of them, or
    /// fewer when the table holds fewer, which `plan` refuses.
    fn others(&self, series: SeriesKey, all: &[SeriesKey]) -> Vec<SeriesKey> {
        match self.group {
            GroupStrategy::Alone | GroupStrategy::Given { .. } => Vec::new(),
            GroupStrategy::Random { size, seed } => {
                let salt = mix(seed ^ mix(series.0));
                let mut ranked: BTreeSet<(u64, SeriesKey)> = all
                    .iter()
                    .filter(|&&other| other != series)
                    .map(|&other| (mix(salt ^ other.0), other))
                    .collect();
                let wanted = size.get() - 1;
                let mut chosen = Vec::with_capacity(wanted);
                while chosen.len() < wanted {
                    match ranked.pop_first() {
                        Some((_, other)) => chosen.push(other),
                        None => break,
                    }
                }
                chosen
            }
        }
    }
}

/// Plans requests against a snapshot of the table's rows.
///
/// Units a model may keep are shared across requests and queries that name
/// the same group at the same cutoff; units forecast whole read their
/// horizon too, so they are shared only among forecasts of one horizon.
/// Each is laid onto the shortest grid of whole patches that holds every
/// member's rows, up to the spec's context. `features` is the table's
/// covariate column count, which sets a member's rows. A series without a
/// cutoff is forecast from the step after its latest observed value. A
/// series with no history before its cutoff, or none at all, is forecast
/// through the observed members of its group, as a cold start.
///
/// # Errors
///
/// If a request has not one cutoff per series, a table of given groups gets
/// a series without its group or one naming a series the table lacks or
/// more series than the table's size, a table that draws its groups gets
/// them given, a group of the strategy's
/// size would pass the model's cap on rows (`SeriesSpec::check_fit`), a
/// series without a cutoff has no observed value, or no member of a series'
/// group has an observation before its cutoff.
pub fn plan(
    spec: &SeriesSpec,
    index: &SeriesIndex,
    snapshot: &Snapshot,
    features: usize,
    requests: &[SeriesRequest],
) -> Result<SeriesPlan, ForecastError> {
    spec.check_fit(features)?;
    let mut all: Vec<SeriesKey> = index.series().collect();
    all.sort_unstable();
    let step = index.step();
    let whole = !spec.reusable || spec.has_known(features);
    let mut plan = SeriesPlan::default();
    let mut ids: HashMap<(Fingerprint, Option<usize>), usize> = HashMap::new();
    for request in requests {
        if request.series.len() != request.cutoffs.len() {
            return Err(ForecastError::Cutoffs {
                series: request.series.len(),
                cutoffs: request.cutoffs.len(),
            });
        }
        match (&spec.group, &request.groups) {
            (GroupStrategy::Given { .. }, None) => {
                return Err(ForecastError::NoGroup(request.series[0]));
            }
            (GroupStrategy::Given { .. }, Some(groups)) if groups.len() != request.series.len() => {
                return Err(ForecastError::Groups {
                    series: request.series.len(),
                    groups: groups.len(),
                });
            }
            (GroupStrategy::Alone | GroupStrategy::Random { .. }, Some(_)) => {
                return Err(ForecastError::UngivenGroups);
            }
            _ => {}
        }
        let mut forecasts = Vec::with_capacity(request.series.len());
        for (at, (&series, &cutoff)) in request.series.iter().zip(&request.cutoffs).enumerate() {
            let cutoff = match cutoff {
                Some(cutoff) => cutoff,
                None => index
                    .latest_observed(series)
                    .and_then(|latest| latest.checked_add(step))
                    .ok_or(ForecastError::NoRows(series))?,
            };
            let mut members = match (&spec.group, &request.groups) {
                (GroupStrategy::Given { size }, Some(groups)) => {
                    given_members(index, series, &groups[at], size.get())?
                }
                _ => spec.others(series, &all),
            };
            if members.is_empty() && !index.has(series) {
                return Err(ForecastError::UnknownSeries(series));
            }
            if let GroupStrategy::Random { size, .. } = spec.group
                && members.len() + 1 < size.get()
            {
                return Err(ForecastError::TooFewSeries {
                    size: size.get(),
                    held: members.len(),
                });
            }
            members.push(series);
            members.sort_unstable();
            if !members
                .iter()
                .any(|&member| index.observed_before(member, cutoff))
            {
                return Err(ForecastError::NothingBefore { series, cutoff });
            }
            let named = matches!(spec.neighbors, Neighbors::Covariates).then_some(series);
            let steps = grid_steps(spec, index, &members, cutoff, step);
            let windows: Vec<Window> = members
                .iter()
                .map(|&member| index.window(member, index.align(member, cutoff)?, steps))
                .collect::<Result<_, _>>()?;
            let rows = windows
                .iter()
                .fold(Context::new(), |rows, window| &rows | &window.context());
            let key = unit_key(snapshot.fingerprint(&rows), cutoff, steps, &members, named);
            let planned = PlannedUnit {
                group: unit_key(Fingerprint::default(), 0, 0, &members, named),
                members: members.clone(),
                target: named,
                cutoff,
                steps,
                windows,
                key,
                whole,
            };
            let shared = (key, whole.then_some(request.horizon.get()));
            let unit = *ids.entry(shared).or_insert_with(|| {
                plan.units.push(planned);
                plan.units.len() - 1
            });
            let target = plan.units[unit]
                .members
                .iter()
                .position(|&member| member == series)
                .expect("the series is a member of its unit");
            let future = if spec.has_known(features) {
                plan.units[unit]
                    .members
                    .iter()
                    .map(|&member| {
                        index.after(member, index.align(member, cutoff)?, request.horizon.get())
                    })
                    .collect::<Result<_, _>>()?
            } else {
                Vec::new()
            };
            forecasts.push(PlannedForecast {
                unit,
                target,
                horizon: request.horizon.get(),
                future,
            });
        }
        plan.forecasts.push(forecasts);
    }
    Ok(plan)
}

/// A unit's identity: its rows' fingerprint with its window folded in, then
/// its members and, with a target, the target.
#[must_use]
pub fn unit_key(
    rows: Fingerprint,
    cutoff: i64,
    steps: usize,
    members: &[SeriesKey],
    target: Option<SeriesKey>,
) -> Fingerprint {
    let mut key = with_window(rows, cutoff, steps);
    for member in members {
        key = key.salted(member.0);
    }
    if let Some(series) = target {
        key = key.salted(series.0);
    }
    key
}

/// The rows on a unit's first `steps` grid points: the grid of the same group
/// at the cutoff `steps` points after the grid's start, which an extension
/// grows.
#[must_use]
pub fn prefix_rows(unit: &PlannedUnit, steps: usize) -> Context {
    unit.windows
        .iter()
        .flat_map(|window| window.rows.iter().take(steps).flatten().copied())
        .collect()
}

/// A given group's other members: those named beside the series, each held by
/// the table and once, at most `size` with the series.
fn given_members(
    index: &SeriesIndex,
    series: SeriesKey,
    given: &[SeriesKey],
    size: usize,
) -> Result<Vec<SeriesKey>, ForecastError> {
    let mut others: Vec<SeriesKey> = given.iter().copied().filter(|&m| m != series).collect();
    others.sort_unstable();
    others.dedup();
    if others.len() + 1 > size {
        return Err(ForecastError::GroupPastSize {
            given: others.len() + 1,
            size,
        });
    }
    if let Some(&unknown) = others.iter().find(|&&member| !index.has(member)) {
        return Err(ForecastError::UnknownMember(unknown));
    }
    Ok(others)
}

/// The grid points before `cutoff` that hold every member's rows, whole
/// patches, at least one and at most the spec's context.
fn grid_steps(
    spec: &SeriesSpec,
    index: &SeriesIndex,
    members: &[SeriesKey],
    cutoff: i64,
    step: i64,
) -> usize {
    let patch = spec.patch.get();
    let limit = (spec.context.get() / patch * patch).max(patch);
    let earliest = members
        .iter()
        .filter_map(|&member| index.earliest(member))
        .min();
    let span = match earliest {
        Some(earliest) if earliest < cutoff => {
            let span = i128::from(cutoff) - i128::from(earliest);
            let steps = (span + i128::from(step) - 1) / i128::from(step);
            usize::try_from(steps).unwrap_or(usize::MAX)
        }
        _ => 0,
    };
    span.max(1)
        .div_ceil(patch)
        .saturating_mul(patch)
        .clamp(patch, limit)
}

/// Lays a unit's rows out densely, from the rows as the store holds them.
///
/// A member's target row takes each row's target, unobserved where it is NaN
/// or the grid point holds no row; its column rows take the row's `features`
/// columns. `fetch` reads rows from the store, in order.
///
/// # Errors
///
/// Whatever `fetch` returns.
pub fn grid<E>(
    spec: &SeriesSpec,
    unit: &PlannedUnit,
    features: usize,
    fetch: impl FnOnce(&[RowId]) -> Result<Vec<Row>, E>,
) -> Result<Grid, E> {
    let target = unit
        .target
        .and_then(|series| unit.members.iter().position(|&member| member == series));
    lay(spec, &unit.windows, unit.steps, features, target, fetch)
}

/// Lays known future rows out densely, as `grid` lays a unit: only the known
/// columns' rows are observed, since nothing else after a cutoff may be read.
/// A forecast without known columns has no future rows at all.
///
/// # Errors
///
/// Whatever `fetch` returns.
pub fn future<E>(
    spec: &SeriesSpec,
    unit: &PlannedUnit,
    forecast: &PlannedForecast,
    features: usize,
    fetch: impl FnOnce(&[RowId]) -> Result<Vec<Row>, E>,
) -> Result<Grid, E> {
    let target = unit
        .target
        .and_then(|series| unit.members.iter().position(|&member| member == series));
    if forecast.future.is_empty() {
        return Ok(Grid {
            rows: 0,
            steps: forecast.horizon,
            ..Grid::default()
        });
    }
    let mut laid = lay(
        spec,
        &forecast.future,
        forecast.horizon,
        features,
        target,
        fetch,
    )?;
    for (row, &role) in laid.roles.iter().enumerate() {
        if role != KNOWN {
            let start = row * laid.steps;
            laid.values[start..start + laid.steps].fill(0.0);
            laid.observed[start..start + laid.steps].fill(false);
        }
    }
    Ok(laid)
}

fn lay<E>(
    spec: &SeriesSpec,
    windows: &[Window],
    steps: usize,
    features: usize,
    target: Option<usize>,
    fetch: impl FnOnce(&[RowId]) -> Result<Vec<Row>, E>,
) -> Result<Grid, E> {
    let per_member = spec.rows_per_member(features);
    let rows = windows.len() * per_member;
    let wanted: Vec<RowId> = windows
        .iter()
        .flat_map(|window| window.rows.iter().flatten().copied())
        .collect();
    let fetched = fetch(&wanted)?;
    let mut fetched = fetched.into_iter();
    let mut values = vec![0.0; rows * steps];
    let mut observed = vec![false; rows * steps];
    let mut roles = Vec::with_capacity(rows);
    for (member, window) in windows.iter().enumerate() {
        let first = member * per_member;
        for row in 0..per_member {
            roles.push(spec.role(member, target, row));
        }
        for (point, held) in window.rows.iter().enumerate() {
            if held.is_none() {
                continue;
            }
            let row = fetched.next().expect("one fetched row per grid row");
            let value = f32::from_bits(row.target);
            if value.is_finite() {
                values[first * steps + point] = value;
                observed[first * steps + point] = true;
            }
            for (column, &feature) in row.features.iter().enumerate().take(features) {
                let at = (first + 1 + column) * steps + point;
                if feature.is_finite() {
                    values[at] = feature;
                    observed[at] = true;
                }
            }
        }
    }
    Ok(Grid {
        rows,
        steps,
        values,
        observed,
        roles,
    })
}
