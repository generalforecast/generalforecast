// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A table's definition: its name, model, columns and how it is served, as
//! a client creates it and its log records it first.

use std::collections::HashSet;
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

use gf_core::categories::CategoryEncoding;
use gf_core::columns::Column;
use gf_core::forecasting::GroupStrategy;
use gf_core::node::{CutoffRefit, DEFAULT_FIT_ROWS, PastCap, RefitPolicy, SummaryScope, Task};
use gf_core::planner::Strategy;
use gf_core::regions::{Rebuild, RebuildWait, RegionSpec};
use gf_core::table::ROW_IDS;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::name::TableName;
use crate::protocol::{StrategySpec, SummarySpec};

/// How a table is served, as recorded at the head of its log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableDefinition {
    pub name: TableName,
    pub model: String,
    /// The feature columns, in order.
    pub columns: Vec<Column>,
    /// Number of classes, or 0 for regression.
    pub classes: u32,
    #[serde(default)]
    pub strategy: StrategySpec,
    #[serde(default = "page_size")]
    pub page_size: usize,
    /// A window of time pages never straddle, if any.
    #[serde(default)]
    pub time_bucket: Option<u64>,
    /// Refit once this fraction of the fitted rows has changed, if set.
    #[serde(default)]
    pub refit_after: Option<f64>,
    /// Refit once a column's mean moves this many deviations, if set.
    #[serde(default)]
    pub refit_shift: Option<f64>,
    /// The most rows the table's state is fit on; `DEFAULT_FIT_ROWS` if unset.
    #[serde(default)]
    pub fit_rows: Option<u64>,
    /// What a context larger than the model was trained on does.
    #[serde(default)]
    pub past_cap: PastCapSpec,
    /// When a query's cutoff calls for a new state: `each`, `changed` or `never`.
    #[serde(default = "each")]
    pub refit_cutoffs: String,
    /// Rows the retrieval scaling stays fixed after, once fit on at least
    /// that many, if set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_rows: Option<u64>,
    /// How categorical columns place rows for retrieval: `codes`, `one_hot`
    /// or `target`.
    #[serde(default = "codes")]
    pub retrieval_categoricals: String,
    /// Regions whose queries share a context, if the table has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regions: Option<RegionsSpec>,
    /// What the column summary is fit on: the table, or each context.
    #[serde(default)]
    pub summary: SummarySpec,
    /// How the table's series are forecast, for a table of series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub series: Option<Box<SeriesDefinition>>,
    /// The columns a client's frames hold around the feature columns, and
    /// the target's class values, kept so a client can open the table again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roles: Option<Box<Roles>>,
    /// An id the server gives each creation, so copies of an earlier table of
    /// the same name never mix with this one's; None until it is created.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "CreationId::optional"
    )]
    pub created: Option<CreationId>,
    /// What row times are, as clients convert them; the node reads only numbers.
    #[serde(default)]
    pub times: TimeKind,
    /// The column whose value a router sends rows and queries to one
    /// partition by, for a table it splits; a node serves such a partition
    /// as any table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_by: Option<RouteBy>,
    /// Whether the table's rows outlive the server: `durable` tables keep a
    /// log and replicate, `ephemeral` ones hold a request's context in memory
    /// alone, under the name `ctx-<fingerprint>`, until the shard evicts them.
    #[serde(default, skip_serializing_if = "Durability::is_durable")]
    pub durability: Durability,
    /// The most row ids the table assigns, live or since deleted, before it
    /// refuses inserts: from 1 to `ROW_IDS`, the most any table assigns,
    /// which it may assign if unset. Its log records it, so a restart keeps it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_ids: Option<u64>,
}

/// The frame columns a client's rows carry besides the feature columns, as
/// `gf.engine.Table` reads them: the target, and the id and time if the
/// frame has them. A classifier's table also keeps its class values, in code
/// order, since clients write codes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Roles {
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<Value>,
}

impl Roles {
    /// Checks the roles name distinct columns and the classes match `classes`.
    ///
    /// # Errors
    ///
    /// If the target is unnamed, two roles share a name, or there are more
    /// class values than classes.
    pub fn check(&self, classes: u32) -> Result<(), String> {
        if self.target.is_empty() {
            return Err("roles need a target column name".into());
        }
        let named: Vec<&String> = [Some(&self.target), self.id.as_ref(), self.time.as_ref()]
            .into_iter()
            .flatten()
            .collect();
        if named.iter().collect::<HashSet<_>>().len() != named.len() {
            return Err("the target, id and time roles need columns of their own".into());
        }
        let count = self.classes.len();
        if count > classes as usize {
            return Err(format!(
                "roles name {count} class values for a table of {classes} classes"
            ));
        }
        Ok(())
    }
}

/// Whether a table's rows are logged, or held in memory for requests alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Every change is logged before it applies, and replicas copy the log.
    #[default]
    Durable,
    /// Rows live in memory only: never logged, never replicated, and gone
    /// once the shard evicts the table or the server stops.
    Ephemeral,
}

impl Durability {
    /// Whether this is the default, which definitions leave unwritten.
    #[must_use]
    pub fn is_durable(&self) -> bool {
        *self == Self::Durable
    }
}

/// The id a server gives a table when it creates it: 16 hex digits, the
/// milliseconds since 1970 by its clock and then random, so a later creation
/// of a name sorts after an earlier one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CreationId(String);

impl CreationId {
    /// The id of a creation at `millis` since 1970, with two random bytes.
    #[must_use]
    pub fn new(millis: u128, random: [u8; 2]) -> Self {
        Self(format!(
            "{:012x}{:02x}{:02x}",
            millis & 0xffff_ffff_ffff,
            random[0],
            random[1]
        ))
    }

    /// An id as another node or a router sent it; an empty one names no creation.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        (!text.is_empty()).then(|| Self(text.to_string()))
    }

    /// The id's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Reads an optional id, taking an empty string as none, as older
    /// definitions and heads hold.
    pub(crate) fn optional<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Self>, D::Error> {
        let text = Option::<String>::deserialize(deserializer)?;
        Ok(text.as_deref().and_then(Self::parse))
    }
}

impl fmt::Display for CreationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// What a table's row times and cutoffs are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeKind {
    /// Integers, in a unit of the client's.
    #[default]
    Integer,
    /// Dates and times, as microseconds since 1970-01-01.
    Datetime,
}

impl TableDefinition {
    /// The series settings, checked, for a table of series.
    ///
    /// # Errors
    ///
    /// If the step, a group's size or the horizon is 0, a known column is
    /// not a covariate column or is named twice, the series, time and target
    /// columns share a name with each other or a covariate, or the table has
    /// retrieval columns, regions, a strategy or a time bucket, which a table
    /// of series never reads: it plans its own units.
    pub fn series_settings(&self) -> Result<Option<SeriesSettings>, String> {
        let Some(series) = &self.series else {
            return Ok(None);
        };
        if self.regions.is_some() || self.time_bucket.is_some() {
            return Err("a table of series takes no regions or time bucket".into());
        }
        if self.strategy != StrategySpec::default() {
            return Err("a table of series takes no strategy: its groups are its contexts".into());
        }
        if series.known.iter().collect::<HashSet<_>>().len() != series.known.len() {
            return Err("a known column is named twice".into());
        }
        let step = NonZeroU64::new(series.step).ok_or("a series table's step must be positive")?;
        let fixed = [&series.series, &series.time, &series.target];
        if fixed.iter().collect::<HashSet<_>>().len() != fixed.len() {
            return Err("the series, time and target columns need names of their own".into());
        }
        if self
            .columns
            .iter()
            .any(|column| fixed.contains(&&column.name))
        {
            return Err(
                "covariate columns need names other than the series, time and target columns"
                    .into(),
            );
        }
        if gf_core::columns::retrieval(&self.columns) > 0 {
            return Err("a table of series takes no retrieval columns".into());
        }
        let starts = gf_core::columns::offsets(&self.columns);
        let mut known = Vec::new();
        for name in &series.known {
            let index = self
                .columns
                .iter()
                .position(|column| column.name == *name)
                .ok_or_else(|| format!("known column {name:?} is not a covariate column"))?;
            let width = self.columns[index].kind.width();
            known.extend(starts[index]..starts[index] + width);
        }
        let sized = |size: u64| {
            usize::try_from(size)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or("a group's size must be positive")
        };
        let group = match series.group {
            GroupSpec::Alone => GroupStrategy::Alone,
            GroupSpec::Random { size, seed } => GroupStrategy::Random {
                size: sized(size)?,
                seed,
            },
            GroupSpec::Given { size } => GroupStrategy::Given { size: sized(size)? },
        };
        if series.horizon == Some(0) {
            return Err("a horizon is at least one step".into());
        }
        Ok(Some(SeriesSettings { step, known, group }))
    }

    /// Number of values its columns become in a row: one per column, and
    /// more for a datetime column.
    #[must_use]
    pub fn features(&self) -> usize {
        gf_core::columns::width(&self.columns)
    }

    /// Number of those values the model reads: all but the retrieval values.
    #[must_use]
    pub fn model_features(&self) -> usize {
        self.features() - gf_core::columns::retrieval(&self.columns)
    }

    /// The definition's settings, checked once, as the node serves the table.
    ///
    /// # Errors
    ///
    /// If a setting is out of range: a page size, time bucket or fit row
    /// count of 0, or a refit limit that is not positive and finite.
    pub fn config(&self) -> Result<TableConfig, String> {
        let cutoffs: CutoffRefit = self.refit_cutoffs.parse()?;
        Ok(TableConfig {
            task: Task::of_classes(self.classes),
            strategy: self.strategy.strategy(),
            page_size: NonZeroUsize::new(self.page_size).ok_or("page_size must be positive")?,
            time_bucket: self
                .time_bucket
                .map(|width| NonZeroU64::new(width).ok_or("time_bucket must be positive"))
                .transpose()?,
            refit: RefitPolicy::new(self.refit_after, self.refit_shift)
                .and_then(|policy| policy.with_cutoffs(cutoffs))
                .map_err(|error| error.to_string())?,
            scale_rows: self
                .scale_rows
                .map(|rows| NonZeroU64::new(rows).ok_or("scale_rows must be positive"))
                .transpose()?,
            categories: self.retrieval_categoricals.parse()?,
            fit_rows: self
                .fit_rows
                .map_or(Some(DEFAULT_FIT_ROWS), |rows| usize::try_from(rows).ok())
                .and_then(NonZeroUsize::new)
                .ok_or("fit_rows must be positive")?,
            past_cap: match self.past_cap {
                PastCapSpec::Warn => PastCap::Warn,
                PastCapSpec::Clamp => PastCap::Clamp,
            },
            regions: self
                .regions
                .map(|regions| -> Result<RegionSpec, String> {
                    let rebuild = match regions.rebuild {
                        RebuildSpec::Always => Rebuild::Always,
                        RebuildSpec::Never => Rebuild::Never,
                        RebuildSpec::Churn(share) if share > 0.0 && share.is_finite() => {
                            Rebuild::Churn(share)
                        }
                        RebuildSpec::Every(seconds) if seconds >= 0.0 && seconds.is_finite() => {
                            Rebuild::Every(std::time::Duration::from_secs_f64(seconds))
                        }
                        _ => return Err("a rebuild's churn share or period is out of range".into()),
                    };
                    if regions.rows == 0 {
                        return Err("a region's rows must be positive".into());
                    }
                    let time = regions
                        .wait
                        .seconds
                        .map(std::time::Duration::try_from_secs_f64)
                        .transpose()
                        .map_err(|_| "a rebuild's wait in seconds is out of range")?;
                    Ok(RegionSpec {
                        rows: regions.rows,
                        rebuild,
                        wait: RebuildWait {
                            steps: regions.wait.steps,
                            time,
                        },
                    })
                })
                .transpose()?,
            summary: self.summary.scope(),
            row_ids: self
                .row_ids
                .map(|row_ids| {
                    NonZeroU64::new(row_ids)
                        .filter(|row_ids| row_ids.get() <= ROW_IDS)
                        .ok_or(format!("row_ids is 1 to {ROW_IDS}"))
                })
                .transpose()?,
        })
    }
}

/// What a table does with a context larger than its model was trained on;
/// see `PastCap`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PastCapSpec {
    #[default]
    Warn,
    Clamp,
}

/// How a table's queries share contexts by region; see `RegionSpec`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionsSpec {
    /// Live rows per region.
    pub rows: u64,
    #[serde(default)]
    pub rebuild: RebuildSpec,
    #[serde(default, skip_serializing_if = "RebuildWaitSpec::is_default")]
    pub wait: RebuildWaitSpec,
}

/// How a router sends a split table's rows and queries to partitions by one
/// column's value, so each partition answers its own queries. The value
/// places a row, so it cannot change for a key once the row is written. A
/// row whose value changes is deleted and inserted again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteBy {
    /// The numeric column routed by.
    pub column: String,
    #[serde(default)]
    pub kind: RouteKind,
    /// The increasing values that cut the column's range into partitions,
    /// one fewer than the partitions, for `ordered` routes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bounds: Vec<f64>,
}

/// How a route maps a value to the key space its partitions slice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    /// By a hash of the value, so partitions hold equal shares of the values.
    #[default]
    Hashed,
    /// In order of the value, so a partition holds one range of it.
    Ordered,
}

impl RouteBy {
    /// Checks the route against the table's columns.
    ///
    /// # Errors
    ///
    /// If the column is not a numeric column of the table, or a hashed route
    /// has bounds, or an ordered route's bounds are not finite and increasing.
    pub fn check(&self, columns: &[Column]) -> Result<(), String> {
        let column = columns
            .iter()
            .find(|column| column.name == self.column)
            .ok_or_else(|| format!("route_by column {:?} is not a column", self.column))?;
        if column.kind != gf_core::columns::ColumnKind::Numeric {
            return Err(format!("route_by column {:?} must be numeric", self.column));
        }
        match self.kind {
            RouteKind::Hashed if !self.bounds.is_empty() => {
                Err("a hashed route takes no bounds".into())
            }
            RouteKind::Hashed => Ok(()),
            RouteKind::Ordered => {
                let finite = self.bounds.iter().all(|bound| bound.is_finite());
                let increasing = self.bounds.windows(2).all(|pair| pair[0] < pair[1]);
                (finite && increasing)
                    .then_some(())
                    .ok_or_else(|| "an ordered route's bounds must be finite and increasing".into())
            }
        }
    }

    /// Where a value sits in the route's key space. Negative zero sits where
    /// zero does.
    #[must_use]
    pub fn key(&self, value: f64) -> gf_assign::Key {
        let value = value + 0.0;
        match self.kind {
            RouteKind::Hashed => gf_assign::key::hashed(&value.to_bits().to_le_bytes()),
            RouteKind::Ordered => gf_assign::key::ordered(value),
        }
    }

    /// The partitions' slices of the key space: equal shares of it for a
    /// hashed route, or the ranges between the bounds for an ordered one,
    /// each held by the partition of its index. An ordered route gives one
    /// slice more than it has bounds, whatever `count` says.
    #[must_use]
    pub fn assignment(&self, count: usize) -> gf_assign::Assignment {
        match self.kind {
            RouteKind::Hashed => gf_assign::Assignment::even(count, count, true),
            RouteKind::Ordered => gf_assign::Assignment {
                generation: 0,
                slices: std::iter::once(0)
                    .chain(self.bounds.iter().map(|&bound| self.key(bound)))
                    .enumerate()
                    .map(|(index, start)| gf_assign::Slice {
                        start,
                        holders: vec![index],
                    })
                    .collect(),
            },
        }
    }
}

/// How long a context due a rebuild is read stale while requests queue; see
/// `RebuildWait`. Without `seconds`, only request steps count.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RebuildWaitSpec {
    pub steps: u32,
    pub seconds: Option<f64>,
}

impl Default for RebuildWaitSpec {
    fn default() -> Self {
        let wait = RebuildWait::default();
        Self {
            steps: wait.steps,
            seconds: wait.time.map(|time| time.as_secs_f64()),
        }
    }
}

impl RebuildWaitSpec {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// How a table of series is forecast; see `SeriesSpec`.
///
/// The series, time and target columns name what clients' frames hold. The
/// server reads observations as keyed rows: the client packs the series'
/// digest and the time into each row key, as `generalforecast.keys.series_keys` does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesDefinition {
    /// The frame column naming each observation's series.
    pub series: String,
    /// The frame column holding each observation's time.
    pub time: String,
    /// The frame column holding the value forecast.
    pub target: String,
    /// The grid's step, in the table's time unit.
    pub step: u64,
    /// The covariate columns known over the horizon, by name.
    #[serde(default)]
    pub known: Vec<String>,
    #[serde(default)]
    pub group: GroupSpec,
    /// Steps a forecast covers when its call names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon: Option<u64>,
}

/// Which other series join a series in its group; see `GroupStrategy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupSpec {
    #[default]
    Alone,
    /// `size - 1` others by seeded hash with the series; a table holding fewer refuses the request.
    Random { size: u64, seed: u64 },
    /// The members each forecast names, up to `size` series; the table draws none.
    Given { size: u64 },
}

/// A series definition checked: what the node's `SeriesSpec` takes from it,
/// beside what the model's profile gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesSettings {
    pub step: NonZeroU64,
    /// The model features of the known columns.
    pub known: Vec<usize>,
    pub group: GroupStrategy,
}

/// When a region's context is built again; see `Rebuild`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RebuildSpec {
    #[default]
    Always,
    /// Once changed rows reach this share of the context's.
    Churn(f64),
    /// Once this many seconds passed since the build.
    Every(f64),
    Never,
}

/// A table's definition, checked: the settings its node serves it with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableConfig {
    pub task: Task,
    pub strategy: Strategy,
    pub page_size: NonZeroUsize,
    pub time_bucket: Option<NonZeroU64>,
    pub refit: RefitPolicy,
    pub scale_rows: Option<NonZeroU64>,
    pub categories: CategoryEncoding,
    pub fit_rows: NonZeroUsize,
    pub past_cap: PastCap,
    pub regions: Option<RegionSpec>,
    pub summary: SummaryScope,
    /// The most row ids the table assigns, if capped below `ROW_IDS`.
    pub row_ids: Option<NonZeroU64>,
}

fn each() -> String {
    "each".to_string()
}

fn codes() -> String {
    "codes".to_string()
}

fn page_size() -> usize {
    128
}

#[cfg(test)]
mod tests {
    use super::*;
    use gf_core::columns::ColumnKind;
    use serde_json::json;

    fn route(kind: RouteKind, bounds: &[f64]) -> RouteBy {
        RouteBy {
            column: "amount".to_string(),
            kind,
            bounds: bounds.to_vec(),
        }
    }

    #[test]
    fn a_hashed_route_spreads_values_over_partitions_and_never_moves() {
        let route = route(RouteKind::Hashed, &[]);
        let assignment = route.assignment(4);
        let mut counts = [0usize; 4];
        for value in 0..4_000 {
            let partition = assignment.slice_of(route.key(f64::from(value)));
            counts[partition] += 1;
            assert_eq!(partition, assignment.slice_of(route.key(f64::from(value))));
        }
        assert!(
            counts.iter().all(|&count| (800..1_200).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn an_ordered_route_sends_each_value_past_its_bounds() {
        let route = route(RouteKind::Ordered, &[-1.0, 5.0]);
        let assignment = route.assignment(3);
        assert_eq!(assignment.slices.len(), 3);
        let partitions: Vec<usize> = [-1e9, -1.5, -1.0, 0.0, 4.9, 5.0, 7e12]
            .into_iter()
            .map(|value| assignment.slice_of(route.key(value)))
            .collect();
        assert_eq!(partitions, [0, 0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn a_route_keys_negative_zero_as_zero() {
        for kind in [RouteKind::Hashed, RouteKind::Ordered] {
            let route = route(kind, &[]);
            assert_eq!(route.key(-0.0), route.key(0.0));
            assert_ne!(route.key(-1.0), route.key(1.0));
        }
        // A bound of zero puts negative zero past it, with zero.
        let route = route(RouteKind::Ordered, &[0.0]);
        let assignment = route.assignment(2);
        assert_eq!(assignment.slice_of(route.key(-0.0)), 1);
        assert_eq!(assignment.slice_of(route.key(-1e-300)), 0);
    }

    #[test]
    fn a_route_names_a_numeric_column_and_bounds_in_order() {
        let columns = vec![
            Column {
                name: "amount".to_string(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "merchant".to_string(),
                kind: ColumnKind::Categorical,
            },
        ];
        assert_eq!(route(RouteKind::Hashed, &[]).check(&columns), Ok(()));
        assert_eq!(
            route(RouteKind::Ordered, &[0.0, 1.0]).check(&columns),
            Ok(())
        );
        assert!(route(RouteKind::Hashed, &[1.0]).check(&columns).is_err());
        assert!(
            route(RouteKind::Ordered, &[1.0, 1.0])
                .check(&columns)
                .is_err()
        );
        assert!(
            route(RouteKind::Ordered, &[f64::INFINITY])
                .check(&columns)
                .is_err()
        );
        let mut other = route(RouteKind::Hashed, &[]);
        other.column = "merchant".to_string();
        assert!(other.check(&columns).is_err());
        other.column = "missing".to_string();
        assert!(other.check(&columns).is_err());
        // A definition without the fields reads as a table without a route or roles.
        let plain: TableDefinition = serde_json::from_value(json!({
            "name": "payments", "model": "tiny", "columns": [], "classes": 0
        }))
        .unwrap();
        assert_eq!(plain.route_by, None);
        assert_eq!(plain.roles, None);
    }

    fn roles(classes: &[Value]) -> Roles {
        Roles {
            target: "y".to_string(),
            id: Some("k".to_string()),
            time: None,
            classes: classes.to_vec(),
        }
    }

    #[test]
    fn roles_need_distinct_columns_and_at_most_one_value_per_class() {
        assert_eq!(roles(&[]).check(2), Ok(()));
        assert_eq!(roles(&[Value::from(0)]).check(2), Ok(()));
        assert_eq!(roles(&[Value::from(0), Value::from(1)]).check(2), Ok(()));
        assert_eq!(
            roles(&[Value::from(0), Value::from(1), Value::from(2)]).check(2),
            Err("roles name 3 class values for a table of 2 classes".to_string())
        );
        let mut same = roles(&[]);
        same.time = Some("y".to_string());
        assert_eq!(
            same.check(2),
            Err("the target, id and time roles need columns of their own".to_string())
        );
        let mut unnamed = roles(&[]);
        unnamed.target = String::new();
        assert_eq!(
            unnamed.check(2),
            Err("roles need a target column name".to_string())
        );
    }
}
