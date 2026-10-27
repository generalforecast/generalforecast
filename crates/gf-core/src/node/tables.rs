// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Making, configuring and removing a node's tables.

use super::*;

impl Node {
    /// An empty node over a device and a host pool.
    ///
    /// # Errors
    ///
    /// If the pools' slots differ in size.
    pub fn new(device: TierConfig, host: TierConfig) -> Result<Self, NodeError> {
        Ok(Self {
            cache: Cache::new(device, host)?,
            tables: BTreeMap::new(),
            next_table: 0,
            next_request: 0,
            step_bytes: None,
            scheduling: Scheduling::Turns,
            rows: RowMemory::unbounded(),
            turn: TableId(0),
            flight: Flight::default(),
            drops: Vec::new(),
            refused: Vec::new(),
            advanced: Vec::new(),
            past_cap: Vec::new(),
            unmeasured: BTreeSet::new(),
            untold: Vec::new(),
            rebuilding: HashMap::new(),
            rebuilt_regions: HashSet::new(),
            compiled: HashSet::new(),
            parts: PlanParts::default(),
        })
    }

    /// Where the last `schedule` spent its time, by part.
    #[must_use]
    pub fn plan_parts(&self) -> &PlanParts {
        &self.parts
    }

    /// Times each part of planning a step from the next `schedule` on, if `timed`.
    pub fn set_timers(&mut self, timed: bool) {
        self.parts = PlanParts::new(timed);
    }

    /// Drops of cached entries that held rows since changed or deleted, for the
    /// executor to apply now: no cache keeps a row's old values once it changes.
    pub fn take_drops(&mut self) -> Vec<Movement> {
        std::mem::take(&mut self.drops)
    }

    /// Requests planning refused since the last call, with why: their queries
    /// would read no rows. Every other request goes into a step.
    pub fn take_refused(&mut self) -> Vec<(RequestId, NodeError)> {
        std::mem::take(&mut self.refused)
    }

    /// Tables that started an epoch, by a refit asked for or due, since the
    /// last call, once per epoch.
    pub fn take_advanced(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.advanced)
    }

    /// Tables that, since the last call, first read a context of more rows
    /// than their model was trained on, under `PastCap::Warn`; each table once.
    pub fn take_past_cap(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.past_cap)
    }

    /// Tables that, since the last call, came to wait for the worker to
    /// measure how many rows their fit takes; each table once.
    ///
    /// A table waits once a fit would read more than its `EntrySizes::fit_bound`
    /// rows, and its requests stay pending until `set_fit_limits`.
    pub fn take_unmeasured(&mut self) -> Vec<TableId> {
        std::mem::take(&mut self.untold)
    }

    /// Sets the most rows a table's state alone is fit on and the most a fit
    /// embeds with its state, as the worker measured them, so it fits again.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_fit_limits(
        &mut self,
        table: TableId,
        state_rows: usize,
        fused_rows: usize,
    ) -> Result<(), NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.spec.sizes.state_rows = state_rows.max(1);
        state.spec.sizes.fused_rows = fused_rows;
        state.spec.sizes.fit_bound = usize::MAX;
        self.unmeasured.remove(&table);
        self.untold.retain(|&untold| untold != table);
        Ok(())
    }

    /// How a table is served: its strategy as cut to the model's trained
    /// length under `PastCap::Clamp`.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn spec(&self, table: TableId) -> Result<&TableSpec, NodeError> {
        Ok(&self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .spec)
    }

    /// The node with a disk tier that contexts spill to from the host.
    ///
    /// # Errors
    ///
    /// If the disk's slots differ in size from the other pools'.
    pub fn with_disk(mut self, disk: TierConfig) -> Result<Self, NodeError> {
        self.cache = self.cache.with_disk(disk)?;
        Ok(self)
    }

    /// The node with another eviction policy; set before anything is cached.
    #[must_use]
    pub fn with_eviction(self, eviction: Eviction) -> Self {
        Self {
            cache: self.cache.with_eviction(eviction),
            ..self
        }
    }

    /// Bytes of a cache tier's slots that entries hold.
    #[must_use]
    pub fn cache_bytes(&self, tier: Tier) -> u64 {
        self.cache.used_bytes(tier)
    }

    /// Limits the context bytes one step may newly encode.
    pub fn set_step_bytes(&mut self, bytes: u64) {
        self.step_bytes = Some(bytes);
    }

    /// Whether every table with requests waiting is a single-use context
    /// not yet fitted, whose requests a step would fit, encode and decode
    /// in one call.
    #[must_use]
    pub fn only_new_contexts_wait(&self) -> bool {
        let mut waiting = false;
        for state in self.tables.values() {
            if (state.pending.is_empty()
                && state.later.is_empty()
                && state.givens.is_empty()
                && state.deferred.is_none())
                || state.compiling.is_some()
            {
                continue;
            }
            if !state.single_use || state.fitted == Some(state.epoch) || state.restoring.is_some() {
                return false;
            }
            waiting = true;
        }
        waiting && self.drops.is_empty() && self.rebuilding.is_empty()
    }

    /// Marks a table as holding a context one request brought, which may
    /// never be read again. Its first step, when the queries waiting read one
    /// context of every row, fits its state, encodes the context and decodes
    /// them in one call, writing no pages and keeping no context; a later
    /// step that reads it again caches it as any table's.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_single_use(&mut self, table: TableId) -> Result<(), NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.single_use = true;
        Ok(())
    }

    /// Caps the row ids `table` assigns, live or since deleted, at
    /// `row_ids`, past which its inserts are refused as `TableError::Full`.
    /// Every table may assign `ROW_IDS` unless capped; a cap above that is
    /// `ROW_IDS`, and one below the ids taken already refuses every insert
    /// until a renumbering frees dead ids.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_row_ids(&mut self, table: TableId, row_ids: NonZeroU64) -> Result<(), NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.row_ids = row_ids.get().min(ROW_IDS);
        Ok(())
    }

    /// The node with tables' rows in `memory`, which bounds what tables with
    /// files keep in memory; set before any table is made.
    #[must_use]
    pub fn with_row_memory(self, memory: RowMemory) -> Self {
        Self {
            rows: memory,
            ..self
        }
    }

    /// Bytes of rows the node's tables keep in memory.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        self.rows.bytes()
    }

    /// Bytes a table's page index holds, about, and its rows in memory.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn index_bytes(&self, table: TableId) -> Result<u64, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        Ok(state.index.bytes() + state.rows.memory_bytes())
    }

    /// Sets the order a step takes waiting tables in.
    pub fn set_scheduling(&mut self, scheduling: Scheduling) {
        self.scheduling = scheduling;
    }

    /// Adds an empty table.
    ///
    /// # Errors
    ///
    /// If the table has no feature columns, or its strategy a zero count.
    pub fn create_table(&mut self, spec: TableSpec) -> Result<TableId, NodeError> {
        let rows = RowStore::new(&self.rows, spec.features);
        self.add_table(spec, rows)
    }

    /// Adds an empty table whose rows spill to a file at `path` when the
    /// node's row memory runs short, sealed by `sealer`.
    ///
    /// # Errors
    ///
    /// As `create_table`, or if the file cannot be made.
    pub fn create_table_on_disk(
        &mut self,
        spec: TableSpec,
        path: &std::path::Path,
        sealer: std::sync::Arc<dyn Sealer>,
    ) -> Result<TableId, NodeError> {
        let rows = RowStore::on_disk(&self.rows, spec.features, path, sealer)?;
        self.add_table(spec, rows)
    }

    pub(super) fn add_table(
        &mut self,
        spec: TableSpec,
        rows: RowStore,
    ) -> Result<TableId, NodeError> {
        if spec.retrieval >= spec.features && (spec.retrieval > 0 || spec.series.is_none()) {
            return Err(NodeError::NoFeatures);
        }
        let ordered = spec.categorical.windows(2).all(|pair| pair[0] < pair[1]);
        if !ordered
            || spec
                .categorical
                .iter()
                .any(|&feature| feature >= spec.model_features())
        {
            return Err(NodeError::Categories(
                "categorical features are distinct model features, in order",
            ));
        }
        if spec.retrieval > 0 && spec.categories != CategoryEncoding::Codes {
            return Err(NodeError::Categories(
                "retrieval columns place rows themselves, so categoricals keep their codes",
            ));
        }
        // A table of series may hold targets alone: it places no rows in its
        // page index, which stays empty.
        let space = spec.dimensions().max(usize::from(spec.series.is_some()));
        let dimensions = NonZeroUsize::new(space).ok_or(NodeError::NoFeatures)?;
        spec.strategy.check()?;
        if let Some(series) = &spec.series {
            if let Some(&column) = series.known.iter().find(|&&column| column >= spec.features) {
                return Err(ForecastError::KnownColumn {
                    column,
                    features: spec.features,
                }
                .into());
            }
            // A group that would pass the model's cap on rows is refused
            // here, never drawn smaller than the table asked.
            series.check_fit(spec.features)?;
        }
        let spec = TableSpec {
            strategy: spec.clamped(spec.strategy),
            ..spec
        };
        let series = spec
            .series
            .as_ref()
            .map(|series| SeriesIndex::new(series.step).map_err(ForecastError::from))
            .transpose()?;
        let (model_features, page_size) = (spec.model_features(), spec.page_size);
        let ranking = spec.strategy.seed().map(Ranking::empty);
        let resolved = spec.strategy.resolve(0, spec.context_rows, false);
        let id = TableId(self.next_table);
        self.next_table += 1;
        self.tables.insert(
            id,
            TableState {
                spec,
                dimensions,
                table: Table::new(),
                index: PageIndex::new(dimensions, page_size),
                epoch: Epoch(0),
                rows,
                standardizer: None,
                scaled_rows: 0,
                prints: BTreeMap::new(),
                ranking,
                resolved,
                unlabeled: RoaringBitmap::new(),
                edges: None,
                cores: (Epoch(0), Cores::default()),
                fitted: None,
                times: TimeIndex::new(),
                series,
                forecasts: Vec::new(),
                recent: HashMap::new(),
                writing: HashMap::new(),
                recent_pruned: 0,
                columns: Columns::new(model_features),
                fitted_columns: None,
                changed: 0,
                pending: Vec::new(),
                later: Vec::new(),
                horizon: None,
                fitted_latest: None,
                given: None,
                givens: std::collections::VecDeque::new(),
                gave: false,
                row_ids: ROW_IDS,
                generation: 0,
                deferred: None,
                restoring: None,
                past_cap: false,
                regions: None,
                region_stats: RegionStats::default(),
                region_draws: 0,
                planned_counts: RegionStats::default(),
                aged_at: None,
                rebuild_waited: 0,
                single_use: false,
                scheduled: Scheduled::default(),
                compiling: None,
            },
        );
        Ok(id)
    }

    /// Removes a table and returns drops for its cache entries.
    ///
    /// Requests still queued for the table are refused; see `take_refused`.
    /// Entries a step in flight reads are dropped once it completes.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn remove_table(&mut self, table: TableId) -> Result<Vec<Movement>, NodeError> {
        let state = self
            .tables
            .remove(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        self.unmeasured.remove(&table);
        self.untold.retain(|&untold| untold != table);
        let queued = state
            .pending
            .iter()
            .chain(&state.later)
            .map(|(request, _, _)| *request);
        let given = state.givens.iter().map(|(request, _, _)| *request);
        self.refused.extend(
            queued
                .chain(given)
                .chain(state.forecasts.iter().map(|(request, _)| *request))
                .map(|request| (request, NodeError::UnknownTable(table))),
        );
        let after = Epoch(state.epoch.0 + 1);
        Ok(self
            .cache
            .drop_epochs_before(state.spec.model, table, after))
    }

    /// A table to change, or whose requests change.
    pub(super) fn state(&mut self, table: TableId) -> Result<&mut TableState, NodeError> {
        let state = self
            .tables
            .get_mut(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        state.generation += 1;
        Ok(state)
    }

    /// Number of live rows in a table.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn len(&self, table: TableId) -> Result<u64, NodeError> {
        Ok(self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .table
            .len())
    }

    /// The series of a table of series that hold rows now: a series whose
    /// rows were all deleted is none of them. None for a table of rows.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn held_series(&self, table: TableId) -> Result<Option<HashSet<SeriesKey>>, NodeError> {
        Ok(self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .series
            .as_ref()
            .map(|index| index.series().collect()))
    }

    /// How many more rows `table` can take before its row ids run out,
    /// counting the dead ids a sparse table frees when it is renumbered
    /// before its next insert, as the write checks count them; a caller
    /// checking several writes before any applies counts what each takes.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn ids_left(&self, table: TableId) -> Result<u64, NodeError> {
        Ok(self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?
            .ids_left())
    }
}
