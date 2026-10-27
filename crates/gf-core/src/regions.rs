// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Regions: queries near each other sharing one context.
//!
//! A table with regions divides its retrieval space by k-means centers, drawn
//! once per state from a sample seeded by the rows' keys. Two tables of the
//! same rows and the same retrieval scaling draw the same regions, whatever
//! order the rows came in. The scaling itself follows the writes: it is fit
//! on the first labeled rows and again each time they double, so tables whose
//! labels arrived in another order may scale, and so draw, differently until
//! `scale_rows` or `fit_rows` stops the refits.
//!
//! A query reads its region's context: the strategy's core plus the rows
//! nearest the region's center, built once and served until the table's
//! rebuild policy finds enough of the region's rows changed since. Rebuilds
//! run in steps of their own, so queries keep reading the old context, and
//! wait for an encode only in a region that has none.
//!
//! Up to `FLAT_REGIONS` regions are one level of k-means: k-means++ seeding,
//! then Lloyd rounds over a sample of at least `SAMPLE_PER_REGION` rows a
//! region, but no more than `MOST_SAMPLE` rows unless there are more regions,
//! when each region has one. Past that, regions are the leaves of two levels: about sqrt(k)
//! top centers, then each top cluster's share of k within it, so a row or
//! query goes to its nearest top center and then its nearest leaf there.

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::time::{Duration, Instant};

use crate::cache::Epoch;
use crate::context::{Context, Fingerprint, mix};
use crate::distance::squared32;
use crate::pages::Vectors;
use crate::planner::{hash, smallest};
use crate::rows::RowsError;
use crate::table::Keys;

/// Seeds the sample regions are drawn from, and their k-means++ draws.
const REGION_SEED: u64 = 0x7265_6769_6f6e_7321;

/// Sample rows each region is drawn from, at least where the table has them.
const SAMPLE_PER_REGION: u64 = 32;

/// Sample rows regions are drawn from at most, unless there are more regions.
const MOST_SAMPLE: u64 = 1 << 18;

/// Lloyd rounds after seeding, at each level.
const LLOYD_ROUNDS: usize = 6;

/// Regions drawn as one level of k-means; past this, two levels.
pub const FLAT_REGIONS: u64 = 1 << 10;

/// The most regions two levels draw.
pub const MOST_REGIONS: u64 = 1 << 20;

/// When a region's context is built again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rebuild {
    /// After any change to the region's rows.
    Always,
    /// Once rows changed since the build reach this share of the context's rows.
    Churn(f64),
    /// Once this long has passed since the build, if any row changed.
    Every(Duration),
    /// Only when the cache no longer holds the context.
    Never,
}

/// How long a region context due a rebuild is read stale while requests
/// queue, before its rebuild takes a step ahead of them; once no request
/// waits, it takes the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebuildWait {
    /// Request steps it waits through at most.
    pub steps: u32,
    /// How long it waits at most, if wall time counts too; without it, two
    /// runs of the same requests rebuild at the same points.
    pub time: Option<Duration>,
}

impl Default for RebuildWait {
    fn default() -> Self {
        Self {
            steps: 8,
            time: Some(Duration::from_secs(1)),
        }
    }
}

impl RebuildWait {
    /// Whether a rebuild due since `due` and waited `steps` request steps
    /// has waited long enough by `now`.
    pub fn over(&self, steps: u32, due: Instant, now: Instant) -> bool {
        steps >= self.steps
            || self
                .time
                .is_some_and(|time| now.duration_since(due) >= time)
    }
}

/// How a table groups queries into regions that share a context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionSpec {
    /// Live rows per region: the table has about this many fewer regions
    /// than rows.
    pub rows: u64,
    pub rebuild: Rebuild,
    pub wait: RebuildWait,
}

/// What a table's regions are, what they served, and what rebuilding them
/// cost. Every query a step answers counts once, in one of `hits`, `stale`,
/// `misses` or `fresh`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RegionStats {
    /// Regions the current state's queries map to, their mean live rows when
    /// drawn, and the levels of k-means that drew them.
    pub regions: u64,
    pub mean_rows: f64,
    pub levels: u32,
    /// Queries read from a built context that was not due a rebuild.
    pub hits: u64,
    /// Queries read from a built context that was due a rebuild.
    pub stale: u64,
    /// Queries that waited for their region's context to be encoded.
    pub misses: u64,
    /// Queries with a cutoff, which read a context built for their window.
    pub fresh: u64,
    /// Contexts built again in rebuild steps, and their rows.
    pub rebuilds: u64,
    pub rebuilt_rows: u64,
    /// Time rebuilt contexts spent due before their rebuild finished.
    pub overdue: Duration,
}

impl RegionStats {
    /// Adds the query counts of `other`.
    pub(crate) fn count(&mut self, other: &Self) {
        self.hits += other.hits;
        self.stale += other.stale;
        self.misses += other.misses;
        self.fresh += other.fresh;
    }
}

/// Why regions could not be drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionsError {
    Rows(RowsError),
    /// More regions than two levels draw, at this many rows a region.
    TooMany {
        regions: u64,
        rows: u64,
    },
}

impl fmt::Display for RegionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rows(error) => error.fmt(formatter),
            Self::TooMany { regions, rows } => write!(
                formatter,
                "regions of {rows} rows make {regions} regions, past the {MOST_REGIONS} \
                 that can be drawn; give regions more rows"
            ),
        }
    }
}

impl std::error::Error for RegionsError {}

impl From<RowsError> for RegionsError {
    fn from(error: RowsError) -> Self {
        Self::Rows(error)
    }
}

/// A region's context as last built.
#[derive(Clone, Debug)]
pub(crate) struct Built {
    pub rows: Context,
    pub key: Fingerprint,
    /// The rows' times, ascending, untimed rows first as the oldest.
    pub times: Vec<i64>,
    /// Rows inserted, updated or deleted in the region since the build.
    pub changes: u64,
    /// The context's rows that a recency window has left behind since.
    pub aged: u64,
    pub at: Instant,
    /// Since when the policy has asked for a rebuild, if it has.
    pub due: Option<Instant>,
}

/// A table's regions for one state.
#[derive(Clone, Debug)]
pub(crate) struct Regions {
    /// The state they were drawn for, the scaling of the vectors, and which
    /// of the table's draws they are.
    pub epoch: Epoch,
    pub scaling: u64,
    pub generation: u64,
    /// For two levels, the top centers and each one's range of regions;
    /// both empty for one.
    top: Vec<Vec<f32>>,
    within: Vec<Range<usize>>,
    /// Each region's center: a leaf's, for two levels.
    pub centers: Vec<Vec<f32>>,
    pub built: Vec<Option<Built>>,
    /// Regions whose built context the rebuild policy has asked to build again.
    due: BTreeSet<usize>,
    /// For a periodic rebuild, when changed regions' periods end, if they
    /// were not due yet; some may since have been built again.
    timers: BTreeSet<(Instant, usize)>,
}

impl Regions {
    /// Regions of about `per_region` of the `live` rows each, drawn from a
    /// sample of `SAMPLE_PER_REGION` rows a region, seeded by keys, but of
    /// at most `MOST_SAMPLE` rows or one a region.
    ///
    /// # Errors
    ///
    /// If a row's vector cannot be read, or there would be more regions than
    /// `MOST_REGIONS`.
    pub fn draw(
        (epoch, scaling, generation): (Epoch, u64, u64),
        live: &Context,
        per_region: u64,
        dimensions: usize,
        vectors: &dyn Vectors,
        keys: &dyn Keys,
    ) -> Result<Self, RegionsError> {
        let count = live.len().div_ceil(per_region.max(1)).max(1);
        // Two levels round each top cluster's share up, so may draw one more
        // region for each of their top centers.
        let most = if count <= FLAT_REGIONS {
            count
        } else {
            count + count.isqrt() + 1
        };
        if most > MOST_REGIONS {
            return Err(RegionsError::TooMany {
                regions: most,
                rows: per_region,
            });
        }
        let wanted = count
            .saturating_mul(SAMPLE_PER_REGION)
            .min(MOST_SAMPLE.max(count));
        let wanted = usize::try_from(wanted).unwrap_or(usize::MAX);
        let sample = smallest(REGION_SEED, live, wanted, keys);
        let mut ranked: Vec<(u64, Vec<f32>)> = Vec::with_capacity(sample.len() as usize);
        for row in sample.iter() {
            ranked.push((hash(REGION_SEED, keys.key(row)), vectors.vector(row)?));
        }
        ranked.sort_by_key(|(hash, _)| *hash);
        let points: Vec<Vec<f32>> = ranked.into_iter().map(|(_, vector)| vector).collect();
        let count = (count as usize).min(points.len()).max(1);
        let (top, within, centers) = if count as u64 <= FLAT_REGIONS {
            let centers = k_means(&points, count, REGION_SEED, dimensions);
            (Vec::new(), Vec::new(), centers)
        } else {
            two_levels(&points, count, dimensions)
        };
        Ok(Self {
            epoch,
            scaling,
            generation,
            top,
            within,
            built: vec![None; centers.len()],
            centers,
            due: BTreeSet::new(),
            timers: BTreeSet::new(),
        })
    }

    /// Levels of k-means the regions were drawn by.
    pub fn levels(&self) -> u32 {
        if self.top.is_empty() { 1 } else { 2 }
    }

    /// The region `vector` belongs to: its nearest center, or for two levels
    /// the nearest region of its nearest top center; ties to the lower.
    pub fn of(&self, vector: &[f32]) -> usize {
        if self.top.is_empty() {
            return nearest(&self.centers, vector);
        }
        let regions = self.within[nearest(&self.top, vector)].clone();
        regions.start + nearest(&self.centers[regions], vector)
    }

    /// The built context of the region `vector` belongs to, if it has one.
    pub fn built_of(&self, vector: &[f32]) -> Option<&Built> {
        self.built[self.of(vector)].as_ref()
    }

    /// Counts a change to a row of the region `vector` belongs to, and notes
    /// when its context falls due under `rebuild`.
    pub fn change(&mut self, vector: &[f32], rebuild: Rebuild, now: Instant) {
        let region = self.of(vector);
        if let Some(built) = &mut self.built[region] {
            built.changes += 1;
            self.note(region, rebuild, now);
        }
    }

    /// Counts each built context's rows from before `oldest`, the time a
    /// recency window now starts at, as aged out, and notes the contexts that
    /// fall due under `rebuild` for it.
    pub fn age(&mut self, oldest: i64, rebuild: Rebuild, now: Instant) {
        for built in self.built.iter_mut().flatten() {
            built.aged = built.times.partition_point(|&time| time < oldest) as u64;
        }
        for region in 0..self.built.len() {
            self.note(region, rebuild, now);
        }
    }

    /// Marks a region due once `rebuild` asks for its context to be built
    /// again, or for a periodic rebuild, sets a timer for its period's end.
    fn note(&mut self, region: usize, rebuild: Rebuild, now: Instant) {
        let Some(built) = &mut self.built[region] else {
            return;
        };
        if built.due.is_some() || built.changes + built.aged == 0 {
            return;
        }
        if built.wants(rebuild, now) {
            built.due = Some(now);
            self.due.insert(region);
        } else if let Rebuild::Every(period) = rebuild {
            self.timers.insert((built.at + period, region));
        }
    }

    /// The regions due a rebuild by `now`, and since when each has been.
    pub fn due(&mut self, rebuild: Rebuild, now: Instant) -> Vec<(usize, Instant)> {
        while let Some(&(end, region)) = self.timers.first().filter(|(end, _)| *end <= now) {
            self.timers.remove(&(end, region));
            // A timer for a context built again since is spent.
            if let Some(built) = self.built[region].as_mut()
                && built.due.is_none()
                && built.wants(rebuild, now)
            {
                built.due = Some(end);
                self.due.insert(region);
            }
        }
        self.due
            .iter()
            .filter_map(|&region| Some((region, self.built[region].as_ref()?.due?)))
            .collect()
    }

    /// Makes `built` a region's context, built just now for a query.
    pub fn build(&mut self, region: usize, built: Built) {
        self.built[region] = Some(built);
        self.due.remove(&region);
    }

    /// Makes `rows` a region's built context, keeping the changes counted
    /// after the first `seen`, which the rows were planned without, toward
    /// its next rebuild under `rebuild`.
    pub fn rebuilt(
        &mut self,
        region: usize,
        (rows, key, times): (Context, Fingerprint, Vec<i64>),
        seen: u64,
        rebuild: Rebuild,
        now: Instant,
    ) {
        let later = self.built[region]
            .as_ref()
            .map_or(0, |built| built.changes.saturating_sub(seen));
        let mut built = Built::new(rows, key, times, now);
        built.changes = later;
        self.build(region, built);
        self.note(region, rebuild, now);
    }
}

impl Built {
    /// A context just built from `rows`, whose times are `times`.
    pub fn new(rows: Context, key: Fingerprint, times: Vec<i64>, now: Instant) -> Self {
        Self {
            rows,
            key,
            times,
            changes: 0,
            aged: 0,
            at: now,
            due: None,
        }
    }

    /// Rows changed in the region since the build, and the context's rows
    /// aged out of its window, as a share of the context's rows.
    pub fn churn(&self) -> f64 {
        (self.changes + self.aged) as f64 / self.rows.len().max(1) as f64
    }

    /// Whether `rebuild` asks for this context to be built again by `now`:
    /// its changed rows and those aged out of its window count alike.
    pub fn wants(&self, rebuild: Rebuild, now: Instant) -> bool {
        self.changes + self.aged > 0
            && match rebuild {
                Rebuild::Always => true,
                Rebuild::Churn(share) => self.churn() >= share,
                Rebuild::Every(period) => now.duration_since(self.at) >= period,
                Rebuild::Never => false,
            }
    }
}

/// Top centers, each one's range of regions, and the regions' centers.
type Levels = (Vec<Vec<f32>>, Vec<Range<usize>>, Vec<Vec<f32>>);

/// Two levels of centers: about sqrt(`count`) top centers over `points`, then
/// within each top cluster its share of `count` regions; returns the top
/// centers, each one's range of regions, and the regions' centers.
fn two_levels(points: &[Vec<f32>], count: usize, dimensions: usize) -> Levels {
    let tops = (count as f64).sqrt().ceil() as usize;
    let top = k_means(points, tops, REGION_SEED, dimensions);
    let mut members: Vec<Vec<Vec<f32>>> = vec![Vec::new(); top.len()];
    for point in points {
        members[nearest(&top, point)].push(point.clone());
    }
    let mut within = Vec::with_capacity(top.len());
    let mut centers = Vec::with_capacity(count);
    for (cluster, rows) in members.iter().enumerate() {
        let share = (count * rows.len())
            .div_ceil(points.len())
            .clamp(1, rows.len().max(1));
        let start = centers.len();
        centers.extend(k_means(
            rows,
            share,
            mix(REGION_SEED ^ cluster as u64),
            dimensions,
        ));
        within.push(start..centers.len());
    }
    (top, within, centers)
}

/// `count` centers over `points` by k-means: k-means++ seeding from the first
/// point, each later center drawn in proportion to its squared distance from
/// those before under `seed`, then `LLOYD_ROUNDS` rounds of moving each center
/// to the mean of the points nearest it. With no points, one center at zero.
fn k_means(points: &[Vec<f32>], count: usize, seed: u64, dimensions: usize) -> Vec<Vec<f32>> {
    let Some(first) = points.first() else {
        return vec![vec![0.0; dimensions]];
    };
    let mut centers = vec![first.clone()];
    let mut nearest_distance: Vec<f32> =
        points.iter().map(|point| squared32(point, first)).collect();
    while centers.len() < count.min(points.len()) {
        let total: f64 = nearest_distance
            .iter()
            .map(|&distance| f64::from(distance))
            .sum();
        if total <= 0.0 {
            break;
        }
        // A uniform draw from the seed's hash, as a share of the total.
        let draw = (mix(seed ^ centers.len() as u64) >> 11) as f64 / (1_u64 << 53) as f64 * total;
        let mut reached = 0.0;
        let chosen = nearest_distance
            .iter()
            .position(|&distance| {
                reached += f64::from(distance);
                reached > draw
            })
            .unwrap_or(points.len() - 1);
        let center = points[chosen].clone();
        for (held, point) in nearest_distance.iter_mut().zip(points) {
            *held = held.min(squared32(point, &center));
        }
        centers.push(center);
    }
    for _ in 0..LLOYD_ROUNDS {
        let mut sums = vec![vec![0.0_f64; dimensions]; centers.len()];
        let mut members = vec![0_u64; centers.len()];
        for point in points {
            let nearest = nearest(&centers, point);
            members[nearest] += 1;
            for (sum, &value) in sums[nearest].iter_mut().zip(point) {
                *sum += f64::from(value);
            }
        }
        for ((center, sum), &members) in centers.iter_mut().zip(&sums).zip(&members) {
            // A center no point is nearest stays where it is.
            if members > 0 {
                *center = sum
                    .iter()
                    .map(|&sum| (sum / members as f64) as f32)
                    .collect();
            }
        }
    }
    centers
}

/// The index of the center nearest `vector`, ties to the lower.
fn nearest(centers: &[Vec<f32>], vector: &[f32]) -> usize {
    let mut best = (f32::INFINITY, 0);
    for (index, center) in centers.iter().enumerate() {
        let distance = squared32(center, vector);
        if distance < best.0 {
            best = (distance, index);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{MOST_SAMPLE, Regions};
    use crate::cache::Epoch;
    use crate::context::Context;
    use crate::pages::Vectors;
    use crate::rows::RowsError;
    use crate::table::{Keys, RowId};

    /// One vector for every row, so k-means has nothing to seed past its
    /// first center, counting how many are read.
    struct Counted(Cell<u64>);

    impl Vectors for Counted {
        fn vector(&self, _: RowId) -> Result<Vec<f32>, RowsError> {
            self.0.set(self.0.get() + 1);
            Ok(vec![1.0, 2.0])
        }
    }

    struct Ids;

    impl Keys for Ids {
        fn key(&self, row: RowId) -> u128 {
            u128::from(row.0)
        }
    }

    /// Vectors read to draw regions of `per_region` of `rows` live rows.
    fn read(rows: u32, per_region: u64) -> u64 {
        let live: Context = (0..rows).map(RowId).collect();
        let vectors = Counted(Cell::new(0));
        Regions::draw((Epoch(0), 0, 0), &live, per_region, 2, &vectors, &Ids).unwrap();
        vectors.0.get()
    }

    #[test]
    fn regions_are_drawn_from_a_sample_of_at_most_its_cap_or_one_row_a_region() {
        assert_eq!(read(1_000, 4), 1_000);
        assert_eq!(read(300_000, 1_024), 293 * 32);
        assert_eq!(read(300_000, 16), MOST_SAMPLE);
        assert_eq!(read(300_000, 1), 300_000);
    }
}
