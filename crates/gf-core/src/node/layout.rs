// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Rows and layouts the node hands out: a table's rows, the candidates it
//! offers, and what a compacted log records.

use super::*;

/// Every live row of a table, in key order.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRows {
    pub keys: Vec<u128>,
    /// Row-major, one row per key.
    pub features: Vec<f32>,
    pub targets: Targets,
    /// Each row's time, or None for a row without one.
    pub times: Vec<Option<i64>>,
}

/// A row a table offers for a query's context, and how it ranks there, so
/// contexts can be assembled from the rows of several partitions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The query it is offered for, or None for a row every query shares,
    /// such as one of a sample's.
    pub query: Option<u32>,
    pub row: RowId,
    /// Its squared distance from the query in the table's standardized
    /// space; 0 for a shared row.
    pub score: f64,
}

/// What a table holds besides its rows and pages, as a compacted log records
/// it, so a copy rebuilt from the log serves the table as the original did:
/// rows added later land on the same pages, and refits come at the same writes.
#[derive(Clone, Debug, PartialEq)]
pub struct TableLayout {
    /// Live rows the compaction's records hold, in all.
    pub rows: u64,
    /// Row ids taken, live or since deleted.
    pub ids: u32,
    pub index: IndexLayout,
    /// The retrieval space's scaling, if the table has had rows.
    pub scaling: Option<Scaling>,
    /// Running sums and counts of each column's finite values over live rows.
    pub column_sums: Vec<f64>,
    pub column_counts: Vec<u64>,
    /// The columns' moments when the state was last fit, if it has been.
    pub fitted: Option<FittedColumns>,
    /// Rows inserted, updated or deleted since the state was last fit.
    pub changed: u64,
    /// The table's strategy as the compaction began, as the table was given
    /// it rather than as a clamp cuts it, so a copy that takes the snapshot
    /// in place of the records before it keeps a change those records
    /// carried. None in layouts written before strategies changed.
    pub strategy: Option<Strategy>,
}

/// The center and scale of each column in the retrieval space, and the live
/// rows they were fit on.
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling {
    pub center: Vec<f32>,
    pub scale: Vec<f32>,
    /// Each categorical feature's codes and vectors, if they are encoded.
    pub lookups: Vec<(Vec<f32>, Vec<f32>)>,
    pub rows: u64,
}

/// The columns' means and deviations when a table's state was fit, and the
/// live rows it then had.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedColumns {
    pub rows: u64,
    pub means: Vec<f64>,
    pub deviations: Vec<f64>,
}

/// Rows of a compacted table, in page order: each row's id and page, and the
/// layout of each page that begins among them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RowLayout {
    pub ids: Vec<RowId>,
    pub pages: Vec<PageId>,
    pub new_pages: Vec<PageLayout>,
}

/// What a compaction reads, taken at once and cheaply: the table's layout,
/// each live row's id, page and time in page order, and a snapshot of its
/// rows, which reads them as they were while the table goes on changing.
#[derive(Clone, Debug)]
pub struct CompactionSource {
    pub table: TableLayout,
    pub layout: RowLayout,
    pub times: Vec<Option<i64>>,
    pub rows: RowSnapshot,
    pub task: Task,
}

impl CompactionSource {
    /// The rows of positions `start..end` in page order, with their layout:
    /// the pages that begin among them, as a log splits a compaction into records.
    ///
    /// # Errors
    ///
    /// If a row cannot be read.
    pub fn part(&self, start: usize, end: usize) -> Result<(TableRows, RowLayout), RowsError> {
        let ids = &self.layout.ids[start..end];
        let stored = self.rows.rows(ids)?;
        let bits: Vec<u32> = stored.iter().map(|row| row.target).collect();
        let rows = TableRows {
            keys: stored.iter().map(|row| row.key).collect(),
            features: stored
                .iter()
                .flat_map(|row| row.features.iter().copied())
                .collect(),
            targets: Targets::from_bits(self.task, &bits),
            times: self.times[start..end].to_vec(),
        };
        let pages = &self.layout.pages[start..end];
        let begun: HashSet<PageId> = (start..end)
            .filter(|&position| {
                position == 0 || self.layout.pages[position - 1] != self.layout.pages[position]
            })
            .map(|position| self.layout.pages[position])
            .collect();
        let layout = RowLayout {
            ids: ids.to_vec(),
            pages: pages.to_vec(),
            new_pages: self
                .layout
                .new_pages
                .iter()
                .filter(|page| begun.contains(&page.id))
                .cloned()
                .collect(),
        };
        Ok((rows, layout))
    }
}

/// A table as a compaction writes it: every live row in page order, with
/// each row's id and page, and what the table holds besides.
#[derive(Clone, Debug, PartialEq)]
pub struct Compaction {
    pub table: TableLayout,
    pub rows: TableRows,
    pub layout: RowLayout,
}
