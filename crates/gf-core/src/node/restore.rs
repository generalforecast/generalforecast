// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Compactions of a table, and rebuilding one from a compacted log.

use super::*;

/// A table being rebuilt from a compacted log, which takes its requests only
/// once every row is back.
#[derive(Debug)]
pub(super) struct Restoring {
    layout: TableLayout,
    pages: HashMap<PageId, PageLayout>,
    members: BTreeMap<PageId, Vec<RowId>>,
    restored: u64,
}

impl TableState {
    /// Empties the table to restore it from a compacted log's layout: every
    /// row id it took, none live, and its scaling and refit state as recorded.
    pub(super) fn begin_restore(&mut self, layout: TableLayout) -> Result<(), NodeError> {
        self.table = Table::with_ids(layout.ids);
        self.index = PageIndex::new(self.dimensions, self.spec.page_size);
        self.rows.clear()?;
        self.rows.reserve(layout.ids)?;
        self.times = TimeIndex::new();
        self.reindex_series();
        self.prints = BTreeMap::new();
        self.ranking = self
            .ranking
            .as_ref()
            .map(|ranking| Ranking::empty(ranking.seed()));
        if let Some(scaling) = &layout.scaling {
            let encoded =
                self.spec.retrieval == 0 && self.spec.categories != CategoryEncoding::Codes;
            let lookups = if encoded {
                self.spec.categorical.len()
            } else {
                0
            };
            let width = if self.spec.retrieval > 0 {
                self.spec.retrieval
            } else {
                self.spec.features
            };
            let block = self.spec.categories.width(self.spec.task);
            let shaped = scaling.lookups.iter().all(|(codes, vectors)| {
                vectors.len() == (codes.len() + 1) * block
                    && codes.iter().all(|code| code.is_finite() && *code >= 0.0)
            });
            if scaling.center.len() != width
                || scaling.scale.len() != width
                || scaling.lookups.len() != lookups
                || !shaped
            {
                return Err(NodeError::Restore(
                    "a layout's scaling differs from the table's retrieval space",
                ));
            }
        }
        self.standardizer = layout.scaling.as_ref().map(|scaling| {
            let lookups = self
                .spec
                .categorical
                .iter()
                .zip(&scaling.lookups)
                .map(|(&feature, (codes, vectors))| {
                    let lookup = Lookup::new(self.spec.categories, codes.clone(), vectors.clone());
                    (feature, lookup)
                })
                .collect();
            Standardizer::of(
                &self.spec,
                scaling.center.clone(),
                scaling.scale.clone(),
                lookups,
            )
        });
        self.scaled_rows = layout.scaling.as_ref().map_or(0, |scaling| scaling.rows);
        self.columns = Columns {
            sums: layout.column_sums.clone(),
            counts: layout.column_counts.clone(),
        };
        self.fitted_columns = layout.fitted.as_ref().map(|fitted| Fitted {
            rows: fitted.rows,
            means: fitted.means.clone(),
            deviations: fitted.deviations.clone(),
        });
        self.changed = layout.changed;
        self.fitted = None;
        self.deferred = None;
        self.restoring = Some(Restoring {
            layout,
            pages: HashMap::new(),
            members: BTreeMap::new(),
            restored: 0,
        });
        Ok(())
    }

    /// Builds the restored table's pages, once its last row is back.
    pub(super) fn finish_restore(&mut self) -> Result<(), NodeError> {
        let Restoring {
            layout,
            mut pages,
            members,
            ..
        } = self.restoring.take().expect("a restore is under way");
        if self.spec.series.is_some() {
            // Its rows came back on one page without a layout; the series
            // index places them.
            self.reprint();
            return Ok(());
        }
        let mut paged = Vec::with_capacity(members.len());
        for (page, rows) in members {
            let page = pages
                .remove(&page)
                .ok_or(NodeError::Restore("a row's page has no layout"))?;
            paged.push((page, rows));
        }
        if !pages.is_empty() {
            return Err(NodeError::Restore("a page's layout has no rows"));
        }
        if self.standardizer.is_none() && !paged.is_empty() {
            return Err(NodeError::Restore("rows come back without a scaling"));
        }
        self.index = PageIndex::restore(self.dimensions, self.spec.page_size, layout.index, paged)?;
        self.reprint();
        Ok(())
    }
}

impl Node {
    /// Every live row of a table, in key order: keys, row-major features, targets.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn rows(&self, table: TableId) -> Result<(Vec<u128>, Vec<f32>, Targets), NodeError> {
        let rows = self.rows_with_times(table)?;
        Ok((rows.keys, rows.features, rows.targets))
    }

    /// Rows of a table by id, with their keys and times, in the order given.
    ///
    /// # Errors
    ///
    /// If the table does not exist or a row cannot be read.
    pub fn rows_of(&self, table: TableId, rows: &[RowId]) -> Result<TableRows, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let keys = state.rows.rows(rows)?.iter().map(|row| row.key).collect();
        let times = rows.iter().map(|&row| state.times.time(row)).collect();
        let (features, targets) = state.gather(rows)?;
        Ok(TableRows {
            keys,
            features,
            targets,
            times,
        })
    }

    /// Every live row of a table, in key order, with each row's time.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn rows_with_times(&self, table: TableId) -> Result<TableRows, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let mut keyed: Vec<(u128, RowId)> =
            state.table.keys().map(|(key, row)| (key.0, row)).collect();
        keyed.sort_unstable();
        let rows: Vec<RowId> = keyed.iter().map(|&(_, row)| row).collect();
        let keys = keyed.into_iter().map(|(key, _)| key).collect();
        let times = rows.iter().map(|&row| state.times.time(row)).collect();
        let (features, targets) = state.gather(&rows)?;
        Ok(TableRows {
            keys,
            features,
            targets,
            times,
        })
    }

    /// Every live row of a table in page order, with each row's id and page,
    /// and what the table holds besides: what a compacted log records, so that
    /// `restore` builds the same table again without searching or renumbering.
    ///
    /// # Errors
    ///
    /// If there is no such table.
    pub fn compaction(&self, table: TableId) -> Result<Compaction, NodeError> {
        let source = self.compaction_source(table)?;
        let (rows, layout) = source.part(0, source.layout.ids.len())?;
        Ok(Compaction {
            table: source.table,
            rows,
            layout,
        })
    }

    /// What a compaction of a table reads, as `compaction` would write it,
    /// with its rows left to read part by part, as the table was now.
    ///
    /// # Errors
    ///
    /// If there is no such table.
    pub fn compaction_source(&self, table: TableId) -> Result<CompactionSource, NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let (index, pages) = state.index.layout();
        let mut layout = RowLayout::default();
        if state.spec.series.is_some() {
            // A table of series places no rows in pages: its rows go in key
            // order, as one page without a layout.
            let mut keyed: Vec<(RowKey, RowId)> = state.table.keys().collect();
            keyed.sort_unstable();
            layout.ids.extend(keyed.iter().map(|&(_, row)| row));
            layout
                .pages
                .extend(std::iter::repeat_n(PageId(0), keyed.len()));
        }
        for (page, rows) in pages {
            layout.ids.extend_from_slice(rows);
            layout
                .pages
                .extend(std::iter::repeat_n(page.id, rows.len()));
            layout.new_pages.push(page);
        }
        let times = layout
            .ids
            .iter()
            .map(|&row| state.times.time(row))
            .collect();
        let fitted = state.fitted_columns.as_ref().map(|fitted| FittedColumns {
            rows: fitted.rows,
            means: fitted.means.clone(),
            deviations: fitted.deviations.clone(),
        });
        let table = TableLayout {
            rows: layout.ids.len() as u64,
            ids: u32::try_from(state.table.ids_used()).expect("row ids are u32"),
            index,
            scaling: state.standardizer.as_ref().map(|standardizer| Scaling {
                center: standardizer.center.clone(),
                scale: standardizer.scale.clone(),
                lookups: standardizer
                    .lookups
                    .iter()
                    .map(|(_, lookup)| (lookup.codes.clone(), lookup.vectors.clone()))
                    .collect(),
                rows: state.scaled_rows,
            }),
            column_sums: state.columns.sums.clone(),
            column_counts: state.columns.counts.clone(),
            fitted,
            changed: state.changed,
            strategy: Some(state.spec.strategy),
        };
        Ok(CompactionSource {
            table,
            layout,
            times,
            rows: state.rows.snapshot(),
            task: state.spec.task,
        })
    }

    /// Rebuilds a table from a compacted log's records, as `compaction` wrote
    /// them: the first with the table's layout, then any that continue it. The
    /// table takes no requests until its last row is back; then it holds the
    /// same rows under the same ids, on the same pages, as the table the log
    /// was compacted from, and fits its state again on its next request.
    ///
    /// Drops of the cached entries a table being restored over held are left
    /// for `take_drops`.
    ///
    /// # Errors
    ///
    /// If the rows or their layout are malformed, or continue no restore; the
    /// table is then left as the records found it.
    pub fn restore(
        &mut self,
        table: TableId,
        rows: &TableRows,
        layout: &RowLayout,
        table_layout: Option<&TableLayout>,
    ) -> Result<(), NodeError> {
        let state = self
            .tables
            .get(&table)
            .ok_or(NodeError::UnknownTable(table))?;
        let times: Vec<Option<i64>> = rows.times.clone();
        state.check(&rows.keys, &rows.features, &rows.targets)?;
        check_times(&rows.keys, Some(&times))?;
        state.check_series(&rows.keys, Some(&times), true)?;
        if layout.ids.len() != rows.keys.len() || layout.pages.len() != rows.keys.len() {
            return Err(NodeError::Restore(
                "a record's rows and layout differ in length",
            ));
        }
        if let Some(layout) = table_layout {
            let width = state.spec.model_features();
            let fitted = layout.fitted.as_ref().map_or([width; 2], |fitted| {
                [fitted.means.len(), fitted.deviations.len()]
            });
            if layout.column_sums.len() != width
                || layout.column_counts.len() != width
                || fitted != [width; 2]
            {
                return Err(NodeError::Restore(
                    "a layout's columns differ from the table's",
                ));
            }
            for key in self.cache.keys_of(table) {
                let dropped = self.drop_or_doom(&key);
                self.drops.extend(dropped);
            }
            let state = self.state(table)?;
            state.begin_restore(layout.clone())?;
            if let Some(strategy) = layout.strategy {
                let strategy = match (state.spec.past_cap, state.spec.context_rows) {
                    (PastCap::Clamp, Some(rows)) => strategy.capped(rows.get()),
                    _ => strategy,
                };
                state.set_strategy(strategy);
            }
        }
        let state = self.state(table)?;
        let Some(restoring) = &mut state.restoring else {
            return Err(NodeError::Restore("rows with a layout continue no restore"));
        };
        for page in &layout.new_pages {
            restoring.pages.insert(page.id, page.clone());
        }
        if layout.ids.iter().any(|row| row.0 >= state.rows.len()) {
            return Err(NodeError::Restore("a row id past those the layout took"));
        }
        state.rows.set(
            &layout.ids,
            &rows.features,
            &rows.targets.bits(),
            Some(&rows.keys),
        )?;
        for (index, (&row, &page)) in layout.ids.iter().zip(&layout.pages).enumerate() {
            state.table.restore(RowKey(rows.keys[index]), row)?;
            state.times.set(row, times[index]);
            if let Some(ranking) = &mut state.ranking {
                ranking.insert(row, rows.keys[index]);
            }
            restoring.members.entry(page).or_default().push(row);
        }
        restoring.restored += layout.ids.len() as u64;
        if restoring.restored == restoring.layout.rows {
            state.finish_restore()?;
        }
        Ok(())
    }
}
