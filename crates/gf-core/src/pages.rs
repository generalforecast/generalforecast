//! Pages: groups of similar rows, the unit of retrieval, caching and invalidation.
//!
//! Rows are grouped per bucket (a time window, for example), so rows from
//! different buckets never share a page. Within a bucket a new row joins the
//! page with the nearest centroid, and a page that outgrows the page size splits
//! in two along its widest dimension. Retrieval returns whole pages, so nearby
//! queries share one context and its cache.
//!
//! Searches are exact but do not scan every page. Pages are grouped into about
//! the square root of their number of cells, each with a fixed center and a
//! radius that bounds its pages' centroids. A search visits cells and pages
//! best first, a cell at the least distance any of its pages could have, so it
//! returns pages in exactly the order a full scan would, opening few cells.
//! Each page also bounds its rows' distance from its centroid, so the nearest
//! rows are found by opening pages best first until no unopened page could
//! hold a nearer row. A page's bound is measured again when a row leaves it, and
//! cells are regrouped once their bounds have doubled, so searches stay tight
//! as rows change. A cell keeps its pages by bucket, so a search within one
//! bucket opens only the cells that hold it.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Index;

use crate::context::Context;
use crate::rows::RowsError;
use crate::table::{Keys, RowId};

/// A page's id. A split keeps the id for one half and gives the other a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

/// Rows in different buckets never share a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bucket(pub u64);

/// Why a row or query was rejected. The index is left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexError {
    Dimensions {
        expected: usize,
        got: usize,
    },
    /// A vector with a NaN or infinite coordinate, which no distance can place.
    NotFinite,
    Duplicate(RowId),
    Unknown(RowId),
    /// A restored layout naming a page it does not hold, or leaving one out of every cell.
    UnknownPage(PageId),
    /// A row's values that could not be read to place it.
    Rows(RowsError),
}

impl From<RowsError> for IndexError {
    fn from(error: RowsError) -> Self {
        Self::Rows(error)
    }
}

/// Where an index reads its rows' vectors, of which it keeps none: they come
/// from the rows, so the index holds only its pages and cells.
pub trait Vectors {
    /// A row's vector.
    ///
    /// # Errors
    ///
    /// If the row's values cannot be read.
    fn vector(&self, row: RowId) -> Result<Vec<f32>, RowsError>;
}

impl fmt::Display for IndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimensions { expected, got } => {
                write!(
                    formatter,
                    "expected a {expected}-dimensional vector, got {got}"
                )
            }
            Self::NotFinite => write!(formatter, "vectors must be finite"),
            Self::Duplicate(row) => write!(formatter, "row {} is already indexed", row.0),
            Self::Unknown(row) => write!(formatter, "row {} is not indexed", row.0),
            Self::UnknownPage(page) => {
                write!(
                    formatter,
                    "a restored layout leaves page {} unplaced",
                    page.0
                )
            }
            Self::Rows(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for IndexError {}

#[derive(Debug)]
struct Page {
    bucket: Bucket,
    rows: Vec<RowId>,
    /// Sum of the rows' vectors, so the centroid updates in O(dimensions).
    sum: Vec<f64>,
    /// At least the distance from the centroid to any row. It grows by as much
    /// as the centroid moves, so it stays a bound without a rescan.
    radius: f64,
    /// The centroid in float32, kept as rows come and go, for placing rows fast.
    near: Vec<f32>,
}

impl Page {
    fn new(bucket: Bucket, dimensions: usize) -> Self {
        Self {
            bucket,
            rows: Vec::new(),
            sum: vec![0.0; dimensions],
            radius: 0.0,
            near: vec![0.0; dimensions],
        }
    }

    /// The float32 centroid of `sum` over `count` rows.
    fn near_of(sum: &[f64], count: usize) -> Vec<f32> {
        let count = count.max(1) as f64;
        sum.iter().map(|total| (total / count) as f32).collect()
    }

    /// Adds a row, widening the radius by how far the centroid moves and to
    /// cover the row, in one pass over the sum.
    fn add(&mut self, row: RowId, vector: &[f32]) {
        let before = self.rows.len() as f64;
        let after = before + 1.0;
        let (mut shift, mut reach) = (0.0, 0.0);
        for ((total, near), &value) in self.sum.iter_mut().zip(&mut self.near).zip(vector) {
            let old = *total / before;
            *total += f64::from(value);
            let new = *total / after;
            *near = new as f32;
            shift += (old - new).powi(2);
            reach += (new - f64::from(value)).powi(2);
        }
        if !self.rows.is_empty() {
            self.radius = slack(self.radius + shift.sqrt());
        }
        self.rows.push(row);
        self.radius = self.radius.max(slack(reach.sqrt()));
    }

    fn take(&mut self, row: RowId, vector: &[f32]) {
        let before = self.centroid();
        if let Some(position) = self.rows.iter().position(|&other| other == row) {
            self.rows.remove(position);
        }
        for (total, &value) in self.sum.iter_mut().zip(vector) {
            *total -= f64::from(value);
        }
        self.near = Self::near_of(&self.sum, self.rows.len());
        if !self.rows.is_empty() {
            self.moved(Some(before));
        }
    }

    /// Widens the radius by how far the centroid moved from `before`.
    fn moved(&mut self, before: Option<Vec<f64>>) {
        if let Some(before) = before {
            let shift = before
                .iter()
                .zip(self.centroid())
                .map(|(old, new)| (old - new).powi(2))
                .sum::<f64>()
                .sqrt();
            self.radius = slack(self.radius + shift);
        }
    }

    /// The least squared distance from `vector` any row could have.
    fn row_bound(&self, vector: &[f32]) -> f64 {
        (self.distance(vector).sqrt() - self.radius)
            .max(0.0)
            .powi(2)
    }

    fn distance(&self, vector: &[f32]) -> f64 {
        let count = self.rows.len() as f64;
        self.sum
            .iter()
            .zip(vector)
            .map(|(total, &value)| (total / count - f64::from(value)).powi(2))
            .sum()
    }

    fn centroid(&self) -> Vec<f64> {
        let count = self.rows.len() as f64;
        self.sum.iter().map(|total| total / count).collect()
    }
}

/// Pages by id, in a vector, since ids are dense: a split's new id is the next.
#[derive(Debug, Default)]
struct Pages {
    slots: Vec<Option<Page>>,
    count: usize,
}

impl Pages {
    fn get(&self, id: &PageId) -> Option<&Page> {
        self.slots.get(id.0 as usize).and_then(Option::as_ref)
    }

    fn get_mut(&mut self, id: &PageId) -> Option<&mut Page> {
        self.slots.get_mut(id.0 as usize).and_then(Option::as_mut)
    }

    fn contains_key(&self, id: &PageId) -> bool {
        self.get(id).is_some()
    }

    fn insert(&mut self, id: PageId, page: Page) {
        let index = id.0 as usize;
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        if self.slots[index].replace(page).is_none() {
            self.count += 1;
        }
    }

    fn remove(&mut self, id: &PageId) -> Option<Page> {
        let page = self.slots.get_mut(id.0 as usize)?.take();
        if page.is_some() {
            self.count -= 1;
        }
        page
    }

    fn len(&self) -> usize {
        self.count
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Index<&PageId> for Pages {
    type Output = Page;

    fn index(&self, id: &PageId) -> &Page {
        self.get(id).expect("the page exists")
    }
}

/// The page of each row, by row id: four bytes a row, since row ids are dense.
#[derive(Debug, Default)]
struct Assignment(Vec<u32>);

impl Assignment {
    const NONE: u32 = u32::MAX;

    fn get(&self, row: RowId) -> Option<PageId> {
        self.0
            .get(row.0 as usize)
            .filter(|&&page| page != Self::NONE)
            .map(|&page| PageId(page))
    }

    fn contains_key(&self, row: &RowId) -> bool {
        self.get(*row).is_some()
    }

    fn insert(&mut self, row: RowId, page: PageId) -> Option<PageId> {
        let index = row.0 as usize;
        if index >= self.0.len() {
            self.0.resize(index + 1, Self::NONE);
        }
        let before = std::mem::replace(&mut self.0[index], page.0);
        (before != Self::NONE).then_some(PageId(before))
    }

    fn remove(&mut self, row: &RowId) {
        if let Some(page) = self.0.get_mut(row.0 as usize) {
            *page = Self::NONE;
        }
    }
}

/// Cells a new row's page is chosen from: those with the nearest centers.
const PROBED_CELLS: usize = 2;

/// Rows `insert_many` finds pages for at once, against one state of the index.
const PROBED_TOGETHER: usize = 1024;

/// A group of pages near a fixed center, for searches to skip as a whole.
#[derive(Debug)]
struct Cell {
    center: Vec<f64>,
    /// The center in float32, for placing rows fast.
    near: Vec<f32>,
    /// At least the distance from the center to any member's centroid. It only
    /// grows, so it stays a bound as centroids move.
    radius: f64,
    /// At least any member page's radius, kept as pages are placed, so a
    /// search for rows bounds the cell without visiting its pages.
    reach: f64,
    /// Its pages by bucket, so a search within one bucket reads no others.
    pages: BTreeMap<Bucket, Vec<PageId>>,
}

impl Cell {
    /// The least squared distance from `vector` any member's centroid could have.
    fn bound(&self, vector: &[f32]) -> f64 {
        let distance = squared(&self.center, vector).sqrt();
        (distance - self.radius).max(0.0).powi(2)
    }

    /// Every page of the cell.
    fn members(&self) -> impl Iterator<Item = PageId> + '_ {
        self.pages.values().flatten().copied()
    }

    /// The cell's pages in `bucket`, or every one.
    fn within(&self, bucket: Option<Bucket>) -> impl Iterator<Item = PageId> + '_ {
        let pages: Box<dyn Iterator<Item = &Vec<PageId>>> = match bucket {
            Some(bucket) => Box::new(self.pages.get(&bucket).into_iter()),
            None => Box::new(self.pages.values()),
        };
        pages.flatten().copied()
    }

    fn holds(&self, bucket: Option<Bucket>) -> bool {
        bucket.is_none_or(|bucket| self.pages.contains_key(&bucket))
    }
}

/// A page as a compacted log records it: all but its rows, which the log
/// lists page by page, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct PageLayout {
    pub id: PageId,
    pub bucket: Bucket,
    /// The sum of its rows' vectors, as added and taken away, so the centroid
    /// comes back to the bit.
    pub sum: Vec<f64>,
    pub radius: f64,
}

/// A cell as a compacted log records it.
#[derive(Clone, Debug, PartialEq)]
pub struct CellLayout {
    pub center: Vec<f64>,
    pub radius: f64,
    pub reach: f64,
    /// Its pages by bucket, each list in order.
    pub pages: Vec<(Bucket, Vec<PageId>)>,
}

/// The index apart from its pages: its cells and counters. With each page's
/// layout and rows, it builds the same index again without a search, so rows
/// added later land on the same pages as in the index it was taken from.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexLayout {
    pub cells: Vec<CellLayout>,
    pub next_page: u32,
    pub reach: f64,
    pub built_reach: f64,
}

/// A distance bound widened a little, so rounding never leaves a point outside it.
fn slack(distance: f64) -> f64 {
    distance * (1.0 + 1e-9) + 1e-9
}

/// The squared distance between two float32 points, eight lanes at a time.
fn squared32(point: &[f32], vector: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut pairs = point.chunks_exact(8).zip(vector.chunks_exact(8));
    for (left, right) in &mut pairs {
        for lane in 0..8 {
            let difference = left[lane] - right[lane];
            lanes[lane] += difference * difference;
        }
    }
    let whole = point.len() / 8 * 8;
    let tail: f32 = point[whole..]
        .iter()
        .zip(&vector[whole..])
        .map(|(left, right)| (left - right) * (left - right))
        .sum();
    lanes.iter().sum::<f32>() + tail
}

fn squared(point: &[f64], vector: &[f32]) -> f64 {
    point
        .iter()
        .zip(vector)
        .map(|(coordinate, &value)| (coordinate - f64::from(value)).powi(2))
        .sum()
}

/// A step of a best-first search: a cell to open, or a page to yield.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Candidate {
    /// Cells sort before pages at the same distance, so ties among pages
    /// resolve by id, as a full scan does.
    Cell(usize),
    Page(PageId),
}

/// Groups rows into pages of similar rows, per bucket, from their vectors.
///
/// Vectors live in a retrieval space chosen by the model layer, such as
/// standardized features. A page that shrinks below a quarter of the page size
/// merges into its nearest neighbor if they fit together, and splits never leave
/// fewer than an eighth of the page size on a side, so outliers join full pages.
#[derive(Debug)]
pub struct PageIndex {
    dimensions: usize,
    page_size: usize,
    pages: Pages,
    /// Page ids in order, so listings and rebuilds are deterministic.
    ids: BTreeSet<PageId>,
    assignment: Assignment,
    next_page: u32,
    cells: Vec<Cell>,
    cell_of: HashMap<PageId, usize>,
    /// The cells' radii, summed now and when they were last regrouped.
    reach: f64,
    built_reach: f64,
}

impl PageIndex {
    /// An empty index for `dimensions`-dimensional vectors.
    #[must_use]
    pub fn new(dimensions: NonZeroUsize, page_size: NonZeroUsize) -> Self {
        Self {
            dimensions: dimensions.get(),
            page_size: page_size.get(),
            pages: Pages::default(),
            ids: BTreeSet::new(),
            assignment: Assignment::default(),
            next_page: 0,
            cells: Vec::new(),
            cell_of: HashMap::new(),
            reach: 0.0,
            built_reach: 0.0,
        }
    }

    /// Bytes the index holds, about, counted from its sizes rather than a
    /// walk: each page's centroid sums and row list, each cell's center, the
    /// row assignment and the maps between them.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let vector = self.dimensions * (size_of::<f64>() + size_of::<f32>());
        let rows = self.assignment.0.capacity() * size_of::<u32>();
        let pages = self.pages.slots.capacity() * size_of::<Option<Page>>()
            + self.pages.len() * (vector + 3 * size_of::<PageId>() + size_of::<usize>());
        let cells = self.cells.capacity() * size_of::<Cell>() + self.cells.len() * vector;
        // Each indexed row is in its page's list as well as the assignment.
        (2 * rows + pages + cells) as u64
    }

    fn check(&self, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dimensions {
            return Err(IndexError::Dimensions {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(IndexError::NotFinite);
        }
        Ok(())
    }

    /// Adds a row to the nearest page in its bucket, splitting the page if it outgrows the page size.
    ///
    /// # Errors
    ///
    /// If the row is already indexed or the vector has the wrong length or is not finite.
    pub fn insert(
        &mut self,
        row: RowId,
        bucket: Bucket,
        vector: &[f32],
        vectors: &dyn Vectors,
    ) -> Result<PageId, IndexError> {
        self.check(vector)?;
        if self.assignment.contains_key(&row) {
            return Err(IndexError::Duplicate(row));
        }
        let nearest = self.nearby(vector, bucket);
        self.insert_near(row, bucket, vector, vectors, nearest)
    }

    /// Adds rows as `insert` does, in order, finding their pages on several
    /// threads. Each run of `PROBED_TOGETHER` rows looks for its pages in the
    /// index as the run found it, and a row whose page has gone since looks
    /// again, so rows land on close pages, if not always the ones inserting
    /// them one by one would choose. Where they land depends only on the
    /// rows and their order, never on the threads, so copies given the same
    /// rows agree.
    ///
    /// # Errors
    ///
    /// As `insert`, for the first row it refuses; the rows before it stay.
    pub fn insert_many(
        &mut self,
        rows: &[(RowId, Bucket, Vec<f32>)],
        vectors: &dyn Vectors,
    ) -> Result<Vec<PageId>, IndexError> {
        let mut placed = Vec::with_capacity(rows.len());
        for run in rows.chunks(PROBED_TOGETHER) {
            for (_, _, vector) in run {
                self.check(vector)?;
            }
            let probed: Vec<Option<PageId>> = if self.pages.len() < PROBED_TOGETHER {
                vec![None; run.len()]
            } else {
                self.probe(run)
            };
            for ((row, bucket, vector), probe) in run.iter().zip(probed) {
                if self.assignment.contains_key(row) {
                    return Err(IndexError::Duplicate(*row));
                }
                let nearest = probe
                    .filter(|page| {
                        self.pages
                            .get(page)
                            .is_some_and(|page| page.bucket == *bucket)
                    })
                    .or_else(|| self.nearby(vector, *bucket));
                placed.push(self.insert_near(*row, *bucket, vector, vectors, nearest)?);
            }
        }
        Ok(placed)
    }

    /// Each row's page as `nearby` finds it, on as many threads as there are cores.
    fn probe(&self, run: &[(RowId, Bucket, Vec<f32>)]) -> Vec<Option<PageId>> {
        let threads = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let size = run.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let parts: Vec<_> = run
                .chunks(size)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|(_, bucket, vector)| self.nearby(vector, *bucket))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| part.join().expect("probing never panics"))
                .collect()
        })
    }

    /// Adds a checked row to `nearest`, or to a new page without one.
    fn insert_near(
        &mut self,
        row: RowId,
        bucket: Bucket,
        vector: &[f32],
        vectors: &dyn Vectors,
        nearest: Option<PageId>,
    ) -> Result<PageId, IndexError> {
        // A page the row fills past its size splits by its rows' vectors, read
        // before anything changes, so a failed read leaves the index as it was.
        let mut known = HashMap::new();
        if let Some(id) = nearest
            && self.pages[&id].rows.len() >= self.page_size
        {
            for &other in &self.pages[&id].rows {
                known.insert(other, vectors.vector(other)?);
            }
            known.insert(row, vector.to_vec());
        }
        let id = nearest.unwrap_or_else(|| {
            let id = self.new_page_id();
            self.pages.insert(id, Page::new(bucket, self.dimensions));
            self.ids.insert(id);
            id
        });
        let page = self
            .pages
            .get_mut(&id)
            .expect("the page was just found or made");
        page.add(row, vector);
        let full = page.rows.len() > self.page_size;
        self.assignment.insert(row, id);
        if full {
            self.split(id, &known);
        } else {
            self.place(id);
        }
        self.regroup_if_loose();
        Ok(self.assignment.get(row).expect("the row was just placed"))
    }

    /// The index's cells and counters, and each page with its rows in order,
    /// in id order.
    #[must_use]
    pub fn layout(&self) -> (IndexLayout, Vec<(PageLayout, &[RowId])>) {
        let cells = self
            .cells
            .iter()
            .map(|cell| CellLayout {
                center: cell.center.clone(),
                radius: cell.radius,
                reach: cell.reach,
                pages: cell
                    .pages
                    .iter()
                    .map(|(&bucket, pages)| (bucket, pages.clone()))
                    .collect(),
            })
            .collect();
        let pages = self
            .ids
            .iter()
            .map(|id| {
                let page = &self.pages[id];
                let layout = PageLayout {
                    id: *id,
                    bucket: page.bucket,
                    sum: page.sum.clone(),
                    radius: page.radius,
                };
                (layout, page.rows.as_slice())
            })
            .collect();
        let index = IndexLayout {
            cells,
            next_page: self.next_page,
            reach: self.reach,
            built_reach: self.built_reach,
        };
        (index, pages)
    }

    /// Builds an index as `layout` took it, from each page's layout and rows
    /// in order.
    ///
    /// # Errors
    ///
    /// If a row is in two pages, a page's sum has the wrong length, or the
    /// layout names a page it does not hold.
    pub fn restore(
        dimensions: NonZeroUsize,
        page_size: NonZeroUsize,
        layout: IndexLayout,
        pages: Vec<(PageLayout, Vec<RowId>)>,
    ) -> Result<Self, IndexError> {
        let mut index = Self::new(dimensions, page_size);
        for (page, rows) in pages {
            if page.sum.len() != index.dimensions {
                return Err(IndexError::Dimensions {
                    expected: index.dimensions,
                    got: page.sum.len(),
                });
            }
            for &row in &rows {
                if index.assignment.insert(row, page.id).is_some() {
                    return Err(IndexError::Duplicate(row));
                }
            }
            index.ids.insert(page.id);
            let restored = Page {
                bucket: page.bucket,
                near: Page::near_of(&page.sum, rows.len()),
                rows,
                sum: page.sum,
                radius: page.radius,
            };
            index.pages.insert(page.id, restored);
        }
        for (position, cell) in layout.cells.into_iter().enumerate() {
            for (_, members) in &cell.pages {
                for &page in members {
                    if !index.pages.contains_key(&page) {
                        return Err(IndexError::UnknownPage(page));
                    }
                    index.cell_of.insert(page, position);
                }
            }
            index.cells.push(Cell {
                near: cell.center.iter().map(|&value| value as f32).collect(),
                center: cell.center,
                radius: cell.radius,
                reach: cell.reach,
                pages: cell.pages.into_iter().collect(),
            });
        }
        if let Some(&page) = index.ids.iter().find(|id| !index.cell_of.contains_key(id)) {
            return Err(IndexError::UnknownPage(page));
        }
        index.next_page = layout.next_page;
        index.reach = layout.reach;
        index.built_reach = layout.built_reach;
        Ok(index)
    }

    fn new_page_id(&mut self) -> PageId {
        let id = PageId(self.next_page);
        self.next_page += 1;
        id
    }

    /// Splits a page in two along its widest dimension, where the halves are tightest.
    fn split(&mut self, id: PageId, vectors: &HashMap<RowId, Vec<f32>>) {
        let page = self.pages.remove(&id).expect("only existing pages split");
        let coordinate = |row: &RowId, dimension: usize| vectors[row][dimension];
        let widest = (0..self.dimensions)
            .map(|dimension| {
                let values: Vec<f64> = page
                    .rows
                    .iter()
                    .map(|row| f64::from(coordinate(row, dimension)))
                    .collect();
                (variance(&values), dimension)
            })
            .max_by(|left, right| left.0.total_cmp(&right.0).then(right.1.cmp(&left.1)))
            .map_or(0, |(_, dimension)| dimension);

        let mut rows = page.rows;
        rows.sort_by(|left, right| {
            coordinate(left, widest)
                .total_cmp(&coordinate(right, widest))
                .then(left.cmp(right))
        });
        let values: Vec<f64> = rows
            .iter()
            .map(|row| f64::from(coordinate(row, widest)))
            .collect();
        let cut = tightest_cut(&values, (self.page_size / 8).max(1));

        let upper = self.new_page_id();
        for (half, half_rows) in [(id, &rows[..cut]), (upper, &rows[cut..])] {
            let mut new = Page::new(page.bucket, self.dimensions);
            for row in half_rows {
                new.add(*row, &vectors[row]);
                self.assignment.insert(*row, half);
            }
            self.pages.insert(half, new);
            self.ids.insert(half);
            self.place(half);
        }
        if self.cells.len().pow(2) * 2 < self.pages.len() {
            self.rebuild_cells();
        }
    }

    /// A page in `bucket` near `vector`, for a new row to join.
    ///
    /// The nearest page among the cells with the nearest centers: a row needs a
    /// close page, not provably the closest, and searching every cell costs as
    /// much as scanning when vectors have many noisy dimensions. With few cells
    /// this is the nearest page.
    fn nearby(&self, vector: &[f32], bucket: Bucket) -> Option<PageId> {
        // The nearest cells, nearest first, ties by index, in one pass.
        let mut cells = [(f32::INFINITY, usize::MAX); PROBED_CELLS];
        for (index, cell) in self.cells.iter().enumerate() {
            let mut candidate = (squared32(&cell.near, vector), index);
            for held in &mut cells {
                if candidate.0 < held.0 {
                    std::mem::swap(held, &mut candidate);
                }
            }
        }
        let mut probed: Option<(f32, PageId)> = None;
        for &(_, index) in cells.iter().filter(|(_, index)| *index != usize::MAX) {
            if let Some(pages) = self.cells[index].pages.get(&bucket) {
                for &id in pages {
                    let candidate = (squared32(&self.pages[&id].near, vector), id);
                    let nearer = probed.is_none_or(|best| {
                        candidate
                            .0
                            .total_cmp(&best.0)
                            .then(candidate.1.cmp(&best.1))
                            == Ordering::Less
                    });
                    if nearer {
                        probed = Some(candidate);
                    }
                }
            }
        }
        probed
            .map(|(_, id)| id)
            .or_else(|| self.search(vector, Some(bucket)).next())
    }

    /// Puts a page in the cell with the nearest center, or widens its cell to
    /// cover its centroid if it already has one.
    fn place(&mut self, id: PageId) {
        let centroid = self.pages[&id].centroid();
        let vector: Vec<f32> = centroid.iter().map(|&value| value as f32).collect();
        let cell = match self.cell_of.get(&id) {
            Some(&cell) => cell,
            None => {
                let nearest = (0..self.cells.len()).min_by(|&left, &right| {
                    squared(&self.cells[left].center, &vector)
                        .total_cmp(&squared(&self.cells[right].center, &vector))
                        .then(left.cmp(&right))
                });
                let cell = nearest.unwrap_or_else(|| {
                    self.cells.push(Cell {
                        near: vector.clone(),
                        center: centroid.clone(),
                        radius: 0.0,
                        reach: 0.0,
                        pages: BTreeMap::new(),
                    });
                    self.cells.len() - 1
                });
                let bucket = self.pages[&id].bucket;
                self.cells[cell].pages.entry(bucket).or_default().push(id);
                self.cell_of.insert(id, cell);
                cell
            }
        };
        let distance = self.cells[cell]
            .center
            .iter()
            .zip(&centroid)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        // A little slack, so rounding never leaves a centroid outside its bound.
        let page_radius = self.pages[&id].radius;
        let reach = &mut self.cells[cell].reach;
        *reach = reach.max(page_radius);
        let radius = &mut self.cells[cell].radius;
        let widened = radius.max(distance * (1.0 + 1e-9) + 1e-9);
        self.reach += widened - *radius;
        *radius = widened;
    }

    /// Regroups the cells once their radii have doubled since they were last
    /// regrouped, as pages that drift and churn leave them, so searches that
    /// skip cells by their radii keep skipping them.
    fn regroup_if_loose(&mut self) {
        if self.cells.len() > 1 && self.reach > 2.0 * self.built_reach.max(f64::MIN_POSITIVE) {
            self.rebuild_cells();
        }
    }

    /// Regroups pages into about the square root of their number of cells,
    /// centered on every so many pages' centroids, in id order.
    fn rebuild_cells(&mut self) {
        if self.pages.is_empty() {
            self.cells.clear();
            self.cell_of.clear();
            self.reach = 0.0;
            self.built_reach = 0.0;
            return;
        }
        let count = (self.pages.len() as f64).sqrt().ceil() as usize;
        let step = self.pages.len().div_ceil(count).max(1);
        self.cells = self
            .ids
            .iter()
            .step_by(step)
            .map(|id| Cell {
                near: self.pages[id].near.clone(),
                center: self.pages[id].centroid(),
                radius: 0.0,
                reach: 0.0,
                pages: BTreeMap::new(),
            })
            .collect();
        self.cell_of.clear();
        self.reach = 0.0;
        let ids: Vec<PageId> = self.ids.iter().copied().collect();
        for id in ids {
            self.place(id);
        }
        self.built_reach = self.reach;
    }

    /// Removes a row from its page, dropping the page if it empties.
    ///
    /// # Errors
    ///
    /// If the row is not indexed.
    pub fn remove(&mut self, row: RowId, vectors: &dyn Vectors) -> Result<PageId, IndexError> {
        let id = self.assignment.get(row).ok_or(IndexError::Unknown(row))?;
        // Every vector the removal reads, before anything changes.
        let known: HashMap<RowId, Vec<f32>> = self.pages[&id]
            .rows
            .iter()
            .map(|&other| Ok((other, vectors.vector(other)?)))
            .collect::<Result<_, RowsError>>()?;
        self.assignment.remove(&row);
        let page = self.pages.get_mut(&id).expect("assigned pages exist");
        page.take(row, &known[&row]);
        let remaining = page.rows.len();
        if remaining == 0 {
            self.drop_page(id);
        } else if remaining < self.page_size / 4 {
            self.merge(id, &known);
        } else {
            self.tighten(id, &known);
            self.place(id);
        }
        self.regroup_if_loose();
        Ok(id)
    }

    /// Measures a page's radius again over its rows, so it stops growing with
    /// rows that have left it; a page holds at most `page_size` rows.
    fn tighten(&mut self, id: PageId, vectors: &HashMap<RowId, Vec<f32>>) {
        let page = self.pages.get_mut(&id).expect("pages to tighten exist");
        let centroid = page.centroid();
        let farthest = page
            .rows
            .iter()
            .map(|row| squared(&centroid, &vectors[row]).sqrt())
            .fold(0.0, f64::max);
        page.radius = slack(farthest);
    }

    fn drop_page(&mut self, id: PageId) {
        let bucket = self.pages.remove(&id).map(|page| page.bucket);
        self.ids.remove(&id);
        if let (Some(cell), Some(bucket)) = (self.cell_of.remove(&id), bucket)
            && let Some(pages) = self.cells[cell].pages.get_mut(&bucket)
        {
            pages.retain(|&other| other != id);
            if pages.is_empty() {
                self.cells[cell].pages.remove(&bucket);
            }
        }
    }

    /// Moves an underfull page's rows into the nearest other page in its bucket,
    /// if they fit there together; otherwise keeps it.
    fn merge(&mut self, id: PageId, vectors: &HashMap<RowId, Vec<f32>>) {
        let page = &self.pages[&id];
        let centroid: Vec<f32> = page.centroid().iter().map(|&value| value as f32).collect();
        let bucket = page.bucket;
        let room = self.page_size - page.rows.len();
        let target = self
            .search(&centroid, Some(bucket))
            .find(|&other| other != id)
            .filter(|other| self.pages[other].rows.len() <= room);
        let Some(target) = target else {
            self.place(id);
            return;
        };
        let moved = self.pages[&id].rows.clone();
        self.drop_page(id);
        for row in moved {
            let page = self.pages.get_mut(&target).expect("found above");
            page.add(row, &vectors[&row]);
            self.assignment.insert(row, target);
        }
        self.place(target);
    }

    /// The page a row is in, if it is indexed.
    #[must_use]
    pub fn page_of(&self, row: RowId) -> Option<PageId> {
        self.assignment.get(row)
    }

    /// A page's rows, or None if there is no such page.
    #[must_use]
    pub fn rows(&self, page: PageId) -> Option<Context> {
        self.pages
            .get(&page)
            .map(|page| page.rows.iter().copied().collect())
    }

    /// Every page, in id order.
    pub fn pages(&self) -> impl Iterator<Item = PageId> + '_ {
        self.ids.iter().copied()
    }

    /// The pages holding any of `rows`, in id order.
    #[must_use]
    pub fn pages_holding(&self, rows: &Context) -> Vec<PageId> {
        let mut held = vec![false; self.next_page as usize];
        for page in rows.iter().filter_map(|row| self.assignment.get(row)) {
            held[page.0 as usize] = true;
        }
        (0..self.next_page)
            .filter(|&page| held[page as usize])
            .map(PageId)
            .collect()
    }

    /// The `count` rows within `allowed` nearest `query`, by exact distance,
    /// ties to the lower key of `keys`, so which rows tie in does not depend
    /// on the order they were written in.
    ///
    /// Pages open best first, each at the least distance any of its rows could
    /// have, until none unopened could hold a row nearer than the `count`th
    /// found, so a query reads the rows near it rather than every row.
    ///
    /// # Errors
    ///
    /// If the query has the wrong length.
    pub fn nearest_rows(
        &self,
        query: &[f32],
        count: usize,
        allowed: &Context,
        vectors: &dyn Vectors,
        keys: &dyn Keys,
    ) -> Result<Context, IndexError> {
        self.check(query)?;
        if count == 0 {
            return Ok(Context::new());
        }
        let reachable = usize::try_from(allowed.len()).unwrap_or(usize::MAX);
        let mut best: BinaryHeap<Near> =
            BinaryHeap::with_capacity(count.min(reachable).saturating_add(1));
        let mut frontier: BinaryHeap<Reverse<Ranked>> = self
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let distance = squared(&cell.center, query).sqrt();
                let bound = (distance - cell.radius - cell.reach).max(0.0).powi(2);
                Reverse(Ranked(bound, Candidate::Cell(index)))
            })
            .collect();
        while let Some(Reverse(Ranked(bound, candidate))) = frontier.pop() {
            if best.len() == count && best.peek().is_some_and(|far| bound > slack(far.0)) {
                break;
            }
            match candidate {
                Candidate::Cell(index) => {
                    for id in self.cells[index].members() {
                        let page = &self.pages[&id];
                        if page.rows.iter().any(|&row| allowed.contains(row)) {
                            let ranked = Ranked(page.row_bound(query), Candidate::Page(id));
                            frontier.push(Reverse(ranked));
                        }
                    }
                }
                Candidate::Page(id) => {
                    for &row in &self.pages[&id].rows {
                        if !allowed.contains(row) {
                            continue;
                        }
                        let distance = vectors
                            .vector(row)?
                            .iter()
                            .zip(query)
                            .map(|(&value, &target)| (f64::from(value) - f64::from(target)).powi(2))
                            .sum();
                        best.push(Near(distance, keys.key(row), row));
                        if best.len() > count {
                            best.pop();
                        }
                    }
                }
            }
        }
        Ok(best.into_iter().map(|Near(_, _, row)| row).collect())
    }

    /// Pages nearest `vector` first, ties by id, optionally only those in `bucket`.
    fn search<'a>(
        &'a self,
        vector: &'a [f32],
        bucket: Option<Bucket>,
    ) -> impl Iterator<Item = PageId> + 'a {
        // Cells without a page in the bucket are never opened.
        let mut frontier: BinaryHeap<Reverse<Ranked>> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| cell.holds(bucket))
            .map(|(index, cell)| Reverse(Ranked(cell.bound(vector), Candidate::Cell(index))))
            .collect();
        std::iter::from_fn(move || {
            while let Some(Reverse(Ranked(_, candidate))) = frontier.pop() {
                match candidate {
                    Candidate::Page(id) => return Some(id),
                    Candidate::Cell(index) => {
                        for id in self.cells[index].within(bucket) {
                            let distance = self.pages[&id].distance(vector);
                            frontier.push(Reverse(Ranked(distance, Candidate::Page(id))));
                        }
                    }
                }
            }
            None
        })
    }
}

/// A row, its distance and its key, ordered by distance, then key.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Near(f64, u128, RowId);

impl Eq for Near {}

impl PartialOrd for Near {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Near {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// A candidate and its distance, ordered by distance, then cells before pages, then id.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ranked(f64, Candidate);

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| match (self.1, other.1) {
                (Candidate::Cell(left), Candidate::Cell(right)) => left.cmp(&right),
                (Candidate::Page(left), Candidate::Page(right)) => left.cmp(&right),
                (Candidate::Cell(_), Candidate::Page(_)) => Ordering::Less,
                (Candidate::Page(_), Candidate::Cell(_)) => Ordering::Greater,
            })
    }
}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

/// The cut of sorted values into two halves of at least `minimum` values each
/// with the least total squared deviation.
///
/// Evenly spread values cut near the middle; values with a gap cut at the gap,
/// unless that would leave fewer than `minimum` values on a side.
fn tightest_cut(values: &[f64], minimum: usize) -> usize {
    let minimum = minimum.min(values.len() / 2).max(1);
    let total: f64 = values.iter().sum();
    let total_squares: f64 = values.iter().map(|value| value * value).sum();
    let (mut sum, mut squares) = (0.0, 0.0);
    let mut best = (f64::INFINITY, values.len() / 2);
    for (index, value) in values.iter().enumerate().take(values.len() - 1) {
        sum += value;
        squares += value * value;
        let (left, right) = (index + 1, values.len() - index - 1);
        if left < minimum || right < minimum {
            continue;
        }
        let deviation = (squares - sum * sum / left as f64)
            + (total_squares - squares - (total - sum).powi(2) / right as f64);
        if deviation < best.0 {
            best = (deviation, index + 1);
        }
    }
    best.1
}
