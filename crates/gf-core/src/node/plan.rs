// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Planning a table's pending requests into units, with or without regions.

use super::*;

/// The unit of `rows` in `plan`, added once.
pub(super) fn add_unit(
    plan: &mut Plan,
    ids: &mut HashMap<Fingerprint, UnitId>,
    rows: &Context,
    key: Fingerprint,
) -> UnitId {
    *ids.entry(key).or_insert_with(|| {
        plan.units.push(crate::planner::Unit {
            rows: rows.clone(),
            key,
        });
        UnitId(plan.units.len() - 1)
    })
}

/// `rows` in the order of their keys.
pub(super) fn by_key(table: &Table, rows: &[RowId]) -> Vec<RowId> {
    let mut keyed: Vec<(u128, RowId)> = rows
        .iter()
        .map(|&row| (Keys::key(table, row), row))
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, row)| row).collect()
}

impl TableState {
    /// Plans the pending requests for the state of `epoch`.
    ///
    /// With regions, a query without a cutoff reads its region's built
    /// context while `cached` says its entry is held, stale or not, and one
    /// is built for it otherwise, as when the entry is doomed by a write; a
    /// query with a cutoff reads a context built around its region's center
    /// within its window. A region's context
    /// keeps to the strategy's recency window as it stood when built, and
    /// its rows the window has since left behind count toward a rebuild.
    pub(super) fn plan_pending(
        &mut self,
        epoch: Epoch,
        snapshot: &Snapshot,
        cached: &dyn Fn(Fingerprint) -> bool,
        now: Instant,
        parts: &mut PlanParts,
    ) -> Result<Plan, NodeError> {
        let since = parts.start();
        self.prepare_cores(epoch);
        let width = self.spec.features;
        let vectors: Vec<Vec<f32>> = self
            .pending
            .iter()
            .flat_map(|(_, queries, _)| queries.chunks(width))
            .map(|row| self.vector(row))
            .collect::<Result<_, _>>()?;
        let (cutoffs, prints, window_of) = self.windows();
        let Some(spec) = self.spec.regions else {
            let planned = self.plan_around(snapshot, &vectors, &cutoffs, &prints, &window_of);
            parts.add("retrieve", since);
            return planned;
        };
        self.prepare_regions(epoch, snapshot, spec)?;
        self.age_regions(now);
        let mut regions = self.regions.take().expect("prepared above");
        // Periods that have ended mark their regions due.
        regions.due(spec.rebuild, now);
        // Counted once the plan joins a step, so a plan made again counts once.
        let mut counts = RegionStats::default();
        let mut plan = Plan::default();
        let mut ids: HashMap<Fingerprint, UnitId> = HashMap::new();
        let mut planned = vec![UnitId(0); vectors.len()];
        let mut fresh: Vec<(usize, usize)> = Vec::new();
        for (query, vector) in vectors.iter().enumerate() {
            let region = regions.of(vector);
            match &mut regions.built[region] {
                Some(built) if window_of[query].is_none() && cached(built.key) => {
                    if built.due.is_some() {
                        counts.stale += 1;
                    } else {
                        counts.hits += 1;
                    }
                    planned[query] = add_unit(&mut plan, &mut ids, &built.rows, built.key);
                }
                _ => fresh.push((query, region)),
            }
        }
        // One retrieval per region to build, not per query: its queries
        // without a window share the context. A query with a window keeps
        // a retrieval of its own, since its rows end at its cutoff.
        let mut first_of: HashMap<usize, usize> = HashMap::new();
        let mut distinct: Vec<(usize, Option<usize>)> = Vec::new();
        let mut slot_of = Vec::with_capacity(fresh.len());
        for &(query, region) in &fresh {
            let slot = match window_of[query] {
                None => *first_of.entry(region).or_insert_with(|| {
                    distinct.push((region, None));
                    distinct.len() - 1
                }),
                window => {
                    distinct.push((region, window));
                    distinct.len() - 1
                }
            };
            slot_of.push(slot);
        }
        let centers: Vec<Vec<f32>> = distinct
            .iter()
            .map(|&(region, _)| regions.centers[region].clone())
            .collect();
        let windows: Vec<Option<usize>> = distinct.iter().map(|&(_, window)| window).collect();
        parts.add("regions", since);
        let since = parts.start();
        let around = if distinct.is_empty() {
            Ok(Plan::default())
        } else {
            self.plan_around(snapshot, &centers, &cutoffs, &prints, &windows)
        };
        parts.add("retrieve", since);
        let since = parts.start();
        let around = match around {
            Ok(around) => around,
            Err(error) => {
                self.regions = Some(regions);
                return Err(error);
            }
        };
        let mut built_now: HashSet<usize> = HashSet::new();
        for (&(query, region), &slot) in fresh.iter().zip(&slot_of) {
            let unit = &around.units[around.queries[slot].0];
            if window_of[query].is_none() {
                if built_now.insert(region) {
                    let times = self.times.times_of(&unit.rows);
                    regions.build(region, Built::new(unit.rows.clone(), unit.key, times, now));
                }
                counts.misses += 1;
            } else {
                counts.fresh += 1;
            }
            planned[query] = add_unit(&mut plan, &mut ids, &unit.rows, unit.key);
        }
        self.regions = Some(regions);
        self.planned_counts = counts;
        plan.queries = planned;
        parts.add("regions", since);
        Ok(plan)
    }

    /// Draws the regions of the state of `epoch` if they are not drawn yet,
    /// or the retrieval scaling changed since. A strategy without neighbors
    /// gives every center the same context, so it has one region.
    pub(super) fn prepare_regions(
        &mut self,
        epoch: Epoch,
        snapshot: &Snapshot,
        spec: RegionSpec,
    ) -> Result<(), NodeError> {
        let drawn = self
            .regions
            .as_ref()
            .is_some_and(|regions| regions.epoch == epoch && regions.scaling == self.scaled_rows);
        if !drawn {
            let per_region = match self.strategy() {
                Strategy::Retrieved { neighbors, .. } if neighbors > 0 => spec.rows,
                _ => u64::MAX,
            };
            self.region_draws += 1;
            let live = snapshot.rows();
            let regions = Regions::draw(
                (epoch, self.scaled_rows, self.region_draws),
                &live,
                per_region,
                self.dimensions.get(),
                &self.scaled(),
                &self.table,
            )
            .map_err(NodeError::Regions)?;
            let stats = &mut self.region_stats;
            stats.regions = regions.centers.len() as u64;
            stats.mean_rows = live.len() as f64 / stats.regions as f64;
            stats.levels = regions.levels();
            self.regions = Some(regions);
        }
        Ok(())
    }

    /// Plans queries at `vectors`, each within its window among the rows
    /// before `cutoffs`, or every live row.
    pub(super) fn plan_around(
        &mut self,
        snapshot: &Snapshot,
        vectors: &[Vec<f32>],
        cutoffs: &[i64],
        prints: &[Fingerprint],
        window_of: &[Option<usize>],
    ) -> Result<Plan, NodeError> {
        let strategy = self.strategy();
        let windows = Windows::Before {
            times: &self.times,
            cutoffs,
            prints,
        };
        let mut cores = std::mem::take(&mut self.cores.1);
        let source = Source {
            snapshot,
            index: &self.index,
            vectors: &self.scaled(),
            strata: &self.labels(),
            keys: &self.table,
            ranking: self.ranking.as_ref(),
            times: &self.times,
        };
        let planned = plan_with(&strategy, &source, &mut cores, vectors, windows, window_of);
        self.cores.1 = cores;
        Ok(planned?)
    }

    /// Counts the rows of built region contexts that a recency window has
    /// left behind since they were built, once after each change to the table.
    pub(super) fn age_regions(&mut self, now: Instant) {
        let (
            Some(spec),
            Strategy::Retrieved {
                recency: Some(recency),
                ..
            },
        ) = (self.spec.regions, self.resolved)
        else {
            return;
        };
        if self.regions.is_none() || self.aged_at == Some(self.generation) {
            return;
        }
        self.aged_at = Some(self.generation);
        let live = self.served().rows();
        let oldest = match recency {
            Recency::Last(rows) => Some(self.times.latest_from(&live, rows)),
            Recency::Span(span) => self
                .times
                .newest(&live)
                .map(|newest| newest.saturating_sub(i64::try_from(span).unwrap_or(i64::MAX))),
        };
        if let (Some(oldest), Some(regions)) = (oldest, self.regions.as_mut()) {
            regions.age(oldest, spec.rebuild, now);
        }
    }

    /// The distinct cutoffs pending queries read, ascending, the fingerprint
    /// of the rows before each, and each query's window among them, or None
    /// for every row. Planning grows each window from the one before.
    pub(super) fn windows(&self) -> (Vec<i64>, Vec<Fingerprint>, Vec<Option<usize>>) {
        let mut distinct: Vec<i64> = self
            .pending
            .iter()
            .filter_map(|(_, _, cutoffs)| cutoffs.as_deref())
            .flatten()
            .copied()
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        // The fingerprint of the rows before each cutoff, in one pass over times.
        let mut prints = Vec::with_capacity(distinct.len());
        let mut sum = Fingerprint::default();
        let mut times = self.prints.iter().peekable();
        for &cutoff in &distinct {
            while let Some((_, print)) = times.next_if(|(time, _)| **time < Some(cutoff)) {
                sum = sum + *print;
            }
            prints.push(sum);
        }
        let width = self.spec.features;
        let mut window_of = Vec::new();
        for (_, queries, cutoffs) in &self.pending {
            let count = queries.len() / width;
            match cutoffs {
                None => window_of.extend(std::iter::repeat_n(None, count)),
                Some(cutoffs) => window_of.extend(cutoffs.iter().map(|cutoff| {
                    Some(
                        distinct
                            .binary_search(cutoff)
                            .expect("every cutoff is listed"),
                    )
                })),
            }
        }
        (distinct, prints, window_of)
    }
}

impl Node {
    /// What a table's regions served and what rebuilding them cost, if it has regions.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn region_stats(&self, table: TableId) -> Result<Option<RegionStats>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(state.spec.regions.map(|_| state.region_stats))
    }

    /// Each query's region churn: rows changed in its region since its
    /// context was built, and the context's rows aged out of its window, as
    /// a share of the context's rows, which `Rebuild::Churn` compares. Aged
    /// rows count as of the table's latest plan or schedule. None for a query
    /// whose region has no built context or whose context is no longer
    /// cached, as the next read builds it again, and for every query until a
    /// plan draws the table's regions.
    ///
    /// # Errors
    ///
    /// If the table does not exist, or the query rows are of the wrong width.
    pub fn region_churn(
        &self,
        table: TableId,
        queries: &[f32],
    ) -> Result<Vec<Option<f64>>, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.check_queries(queries, None)?;
        let rows = queries.chunks(state.spec.features);
        let Some(regions) = state
            .regions
            .as_ref()
            .filter(|regions| regions.epoch == state.epoch && regions.scaling == state.scaled_rows)
        else {
            return Ok(vec![None; rows.len()]);
        };
        // A context a write doomed, or one still being encoded, is not built.
        let cached = |built: &&Built| {
            let key = CacheKey {
                model: state.spec.model,
                table,
                epoch: state.epoch,
                kind: Kind::Context,
                rows: built.key,
            };
            self.cache.contains(&key)
                && !self.doomed.contains(&key)
                && !self.encoding.contains(&key)
        };
        rows.map(|row| {
            state
                .vector(row)
                .map(|vector| regions.built_of(&vector).filter(cached).map(Built::churn))
        })
        .collect::<Result<_, _>>()
    }

    /// The rows a table offers each query under `strategy`, or its own when
    /// None, for contexts assembled across partitions: every live row for
    /// `Exact`; for `Retrieved`, its core of `share` rows (the strategy's own
    /// count without one), drawn as a plan draws it, and each query's nearest
    /// rows outside the core. The router that assembles gives the strategy
    /// when the table's is adaptive, resolved against every partition's rows
    /// rather than this one's. Shared rows come first, each once. A diverse core is
    /// spread over this partition's rows alone, so the cores partitions offer
    /// together are not one farthest-point sample of the whole table. With a
    /// cutoff, only rows from before it are offered, as a query with that
    /// cutoff reads.
    ///
    /// # Errors
    ///
    /// As `submit`, for the queries, or if the table's rows cannot be read.
    pub fn candidates(
        &mut self,
        table: TableId,
        queries: &[f32],
        share: Option<u64>,
        cutoff: Option<i64>,
        strategy: Option<Strategy>,
    ) -> Result<Vec<Candidate>, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let width = state.spec.features;
        if queries.is_empty() {
            return Err(NodeError::NoQueries);
        }
        if !queries.len().is_multiple_of(width) {
            return Err(NodeError::Width {
                expected: width,
                got: queries.len(),
            });
        }
        // The bins and diverse cores of the state a prediction would read.
        let wanted = state.horizon_of(cutoff.as_ref().map(std::slice::from_ref));
        let refit = state.refit_for(wanted);
        state.prepare_cores(Epoch(state.epoch.0 + u64::from(refit)));
        let strategy = match strategy {
            Some(given) => given,
            None => state.strategy(),
        };
        let live = state.served().rows();
        let live = match cutoff {
            Some(cutoff) => &state.times.before(cutoff) & &live,
            None => live,
        };
        let vectors: Vec<Vec<f32>> = queries
            .chunks(width)
            .map(|row| state.vector(row))
            .collect::<Result<_, _>>()?;
        // Its fields alone, so the table's cores can be drawn meanwhile.
        let scaled = Scaled {
            rows: &state.rows,
            standardizer: state.standardizer.as_ref(),
        };
        let mut offered = Vec::new();
        let shared = |offered: &mut Vec<Candidate>, rows: &Context| {
            offered.extend(rows.iter().map(|row| Candidate {
                query: None,
                row,
                score: 0.0,
            }));
        };
        let Strategy::Retrieved {
            core,
            neighbors,
            among,
            recency,
        } = strategy
        else {
            shared(&mut offered, &live);
            return Ok(offered);
        };
        let live = match recency {
            Some(recency) => recent(&state.times, &live, recency, cutoff),
            None => live,
        };
        let core = match core {
            Some(core) => {
                let core = Core {
                    rows: share.unwrap_or(core.rows),
                    ..core
                };
                let mut cores = std::mem::take(&mut state.cores.1);
                let source = Source {
                    snapshot: &state.served(),
                    index: &state.index,
                    vectors: &scaled,
                    strata: &state.labels(),
                    keys: &state.table,
                    ranking: state.ranking.as_ref(),
                    times: &state.times,
                };
                let drawn = draw_core(core, &source, &mut cores, &live);
                state.cores.1 = cores;
                drawn?
            }
            None => Context::new(),
        };
        shared(&mut offered, &core);
        let outside = &live - &core;
        for (query, vector) in vectors.iter().enumerate() {
            let nearest = state.index.nearest_rows(
                vector,
                among.unwrap_or(neighbors),
                &outside,
                &scaled,
                &state.table,
            )?;
            let nearest = match among {
                Some(_) => state.times.latest_of(&nearest, neighbors as u64),
                None => nearest,
            };
            for row in nearest.iter() {
                let score = scaled
                    .vector(row)?
                    .iter()
                    .zip(vector)
                    .map(|(&value, &target)| (f64::from(value) - f64::from(target)).powi(2))
                    .sum();
                offered.push(Candidate {
                    query: Some(query as u32),
                    row,
                    score,
                });
            }
        }
        Ok(offered)
    }

    /// The keys of the rows each query's context holds, in the order the
    /// model reads them, ascending, planned as a request of these queries
    /// alone would be, with its refit, but neither queued nor
    /// encoded, and changing nothing a later plan reads. A query whose cutoff
    /// precedes every row has none.
    ///
    /// # Errors
    ///
    /// As `submit_before`, or if a row cannot be read.
    pub fn contexts(
        &mut self,
        table: TableId,
        queries: &[f32],
        cutoffs: Option<&[i64]>,
    ) -> Result<Vec<Vec<u128>>, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        if state.spec.series.is_some() {
            return Err(ForecastError::OnlyForecasts.into());
        }
        state.check_queries(queries, cutoffs)?;
        if let Some(error) = state.empty(table, cutoffs) {
            return Err(error);
        }
        let alone = (
            RequestId(u64::MAX),
            queries.to_vec(),
            cutoffs.map(<[i64]>::to_vec),
        );
        let pending = std::mem::replace(&mut state.pending, vec![alone]);
        let refit = state.refit_for(state.horizon_of(cutoffs));
        let epoch = Epoch(state.epoch.0 + u64::from(refit));
        // Planned against copies of what plans keep, so a refit this plan
        // foresees drops no core, bins or region the current state reads,
        // and counted as no region's hit or miss.
        let kept = (
            state.cores.clone(),
            state.edges.clone(),
            state.regions.clone(),
            state.region_stats,
            state.planned_counts,
        );
        let model = state.spec.model;
        let cached = |rows| {
            let key = CacheKey {
                model,
                table,
                epoch,
                kind: Kind::Context,
                rows,
            };
            self.cache.contains(&key) && !self.doomed.contains(&key)
        };
        let planned = state.plan_pending(
            epoch,
            &state.served(),
            &cached,
            Instant::now(),
            &mut PlanParts::default(),
        );
        (
            state.cores,
            state.edges,
            state.regions,
            state.region_stats,
            state.planned_counts,
        ) = kept;
        state.pending = pending;
        let planned = planned?;
        let alone = vec![(
            RequestId(u64::MAX),
            queries.to_vec(),
            cutoffs.map(<[i64]>::to_vec),
        )];
        let pending = std::mem::replace(&mut state.pending, alone);
        let blind = !state.blind(&planned).is_empty();
        state.pending = pending;
        if blind {
            return Err(NodeError::EmptyContext(table));
        }
        planned
            .queries
            .iter()
            .map(|unit| {
                let rows: Vec<RowId> = planned.units[unit.0].rows.iter().collect();
                let mut keys: Vec<u128> =
                    state.rows.rows(&rows)?.iter().map(|row| row.key).collect();
                keys.sort_unstable();
                Ok(keys)
            })
            .collect()
    }

    /// Changes how a table's contexts are chosen from here on, as a table
    /// created with `strategy` would choose them: under `PastCap::Clamp` it
    /// is cut to the model's cap first. The ranking, cores and regions drawn
    /// for the old strategy are dropped and drawn anew by the next plan;
    /// cached encodings stay.
    ///
    /// # Errors
    ///
    /// If the table does not exist or the strategy is not valid.
    pub fn set_strategy(&mut self, table: TableId, strategy: Strategy) -> Result<(), NodeError> {
        strategy.check()?;
        let state = self.state(table)?;
        let strategy = match (state.spec.past_cap, state.spec.context_rows) {
            (PastCap::Clamp, Some(rows)) => strategy.capped(rows.get()),
            _ => strategy,
        };
        state.set_strategy(strategy);
        Ok(())
    }

    /// The strategy a table's next plan chooses contexts with: its adaptive
    /// strategy resolved against its live rows, or the one it was given.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn strategy(&self, table: TableId) -> Result<Strategy, NodeError> {
        Ok(self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .resolved())
    }
}
