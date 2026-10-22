// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The page index (similar rows grouped into pages, per bucket), the time
//! index and the series index.

use std::collections::HashSet;
use std::num::{NonZeroU64, NonZeroUsize};

mod support;

use gf_core::context::{Context, Fingerprint};
use gf_core::pages::{Bucket, IndexError, PageId, PageIndex};
use gf_core::series::{SeriesError, SeriesIndex, SeriesKey, with_window};
use gf_core::table::{RowId, Version};
use gf_core::time_index::TimeIndex;
use support::{Keeping, Kept};

/// A deterministic stream of numbers in [-1, 1).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn near(&mut self, center: [f32; 2]) -> [f32; 2] {
        [center[0] + self.next(), center[1] + self.next()]
    }
}

const A: [f32; 2] = [0.0, 0.0];
const B: [f32; 2] = [100.0, 100.0];
const PAGE: usize = 16;

fn index() -> PageIndex {
    PageIndex::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(PAGE).unwrap(),
    )
}

/// 64 rows around A (even ids) and 64 around B (odd ids), inserted interleaved.
fn two_clusters() -> PageIndex {
    let mut index = index();
    let mut noise = Noise(7);
    for row in 0..128 {
        let center = if row % 2 == 0 { A } else { B };
        index
            .put(RowId(row), Bucket(0), &noise.near(center))
            .unwrap();
    }
    index
}

fn ids(context: &Context) -> HashSet<u32> {
    context.iter().map(|row| row.0).collect()
}

fn page_rows(index: &PageIndex) -> Vec<HashSet<u32>> {
    index
        .pages()
        .map(|page| ids(&index.rows(page).unwrap()))
        .collect()
}

#[test]
fn pages_partition_the_rows_within_the_page_size() {
    let mut index = index();
    let mut noise = Noise(1);
    for row in 0..200 {
        let vector = [10.0 * noise.next(), 10.0 * noise.next()];
        index.put(RowId(row), Bucket(0), &vector).unwrap();
    }

    let pages = page_rows(&index);
    let total: usize = pages.iter().map(HashSet::len).sum();
    let union: HashSet<u32> = pages.iter().flatten().copied().collect();
    assert!(
        pages
            .iter()
            .all(|rows| !rows.is_empty() && rows.len() <= PAGE)
    );
    assert_eq!((total, union.len()), (200, 200));
    for row in 0..200 {
        let page = index.page_of(RowId(row)).unwrap();
        assert!(index.rows(page).unwrap().contains(RowId(row)));
    }
}

#[test]
fn pages_do_not_mix_separated_clusters() {
    let index = two_clusters();

    for rows in page_rows(&index) {
        let parities: HashSet<u32> = rows.iter().map(|row| row % 2).collect();
        assert_eq!(parities.len(), 1, "a page mixes both clusters: {rows:?}");
    }
}

#[test]
fn pages_never_mix_buckets() {
    let mut index = index();
    let mut noise = Noise(3);
    for row in 0..60 {
        // Same region, different buckets: like the same kind of row in two time windows.
        index
            .put(RowId(row), Bucket(u64::from(row % 3)), &noise.near(A))
            .unwrap();
    }

    for rows in page_rows(&index) {
        let buckets: HashSet<u32> = rows.iter().map(|row| row % 3).collect();
        assert_eq!(buckets.len(), 1);
    }
}

#[test]
fn removing_rows_shrinks_and_then_drops_pages() {
    let mut index = index();
    for row in 0..3 {
        index
            .put(RowId(row), Bucket(0), &[row as f32, 0.0])
            .unwrap();
    }
    let page = index.page_of(RowId(0)).unwrap();

    index.take(RowId(0)).unwrap();
    assert_eq!(index.rows(page).unwrap().len(), 2);
    index.take(RowId(1)).unwrap();
    index.take(RowId(2)).unwrap();

    assert_eq!(index.rows(page), None);
    assert_eq!(index.pages().count(), 0);
}

#[test]
fn the_same_inserts_build_the_same_pages() {
    let first: Vec<(PageId, HashSet<u32>)> = {
        let index = two_clusters();
        index
            .pages()
            .map(|page| (page, ids(&index.rows(page).unwrap())))
            .collect()
    };
    let index = two_clusters();
    let second: Vec<(PageId, HashSet<u32>)> = index
        .pages()
        .map(|page| (page, ids(&index.rows(page).unwrap())))
        .collect();

    assert_eq!(first, second);
}

#[test]
fn invalid_rows_and_vectors_are_rejected() {
    assert_eq!(
        index().put(RowId(9), Bucket(0), &[f32::NAN, 0.0]),
        Err(IndexError::NotFinite)
    );
    let mut index = index();
    index.put(RowId(0), Bucket(0), &[0.0, 0.0]).unwrap();

    assert_eq!(
        index.put(RowId(1), Bucket(0), &[0.0]),
        Err(IndexError::Dimensions {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(
        index.put(RowId(0), Bucket(0), &[1.0, 1.0]),
        Err(IndexError::Duplicate(RowId(0)))
    );
    assert_eq!(index.take(RowId(9)), Err(IndexError::Unknown(RowId(9))));
    assert_eq!(
        index.nearest_rows(&[0.0, 0.0, 0.0], 1, &Context::new(), &Kept, &Kept),
        Err(IndexError::Dimensions {
            expected: 2,
            got: 3
        })
    );
}

#[test]
fn time_index_selects_rows_by_time() {
    let mut times = TimeIndex::new();
    for (row, time) in [(0, 30), (1, 10), (2, 20), (3, 10)] {
        times.insert(RowId(row), time);
    }

    assert_eq!(ids(&times.before(20)), HashSet::from([1, 3]));
    assert_eq!(ids(&times.between(10, 30)), HashSet::from([1, 2, 3]));
    assert!(times.before(10).is_empty());
}

#[test]
fn time_index_finds_a_sets_latest_and_recent_rows() {
    let mut times = TimeIndex::new();
    for (row, time) in [(0, 30), (1, 10), (2, 20), (3, 10), (4, 40)] {
        times.insert(RowId(row), time);
    }
    times.insert_untimed(RowId(5));
    let within: Context = [0, 1, 2, 3, 5].into_iter().map(RowId).collect();

    // Ties at one time go to the higher row; untimed rows are the oldest.
    assert_eq!(
        ids(&times.latest(&within, 3, None)),
        HashSet::from([0, 2, 3])
    );
    assert_eq!(ids(&times.latest_of(&within, 3)), HashSet::from([0, 2, 3]));
    assert_eq!(
        ids(&times.latest(&within, 5, None)),
        HashSet::from([0, 1, 2, 3, 5])
    );
    // A walk from a cutoff reads only what comes before it.
    let early: Context = [1, 2, 3, 5].into_iter().map(RowId).collect();
    assert_eq!(
        ids(&times.latest(&early, 2, Some(25))),
        HashSet::from([2, 3])
    );
    // Counted back from the newest time among them, 30, not the table's.
    assert_eq!(ids(&times.recent(&within, 10, None)), HashSet::from([0, 2]));
    assert!(
        times
            .recent(&[5].into_iter().map(RowId).collect(), 10, None)
            .is_empty()
    );
}

#[test]
fn time_index_moves_and_removes_rows() {
    let mut times = TimeIndex::new();
    times.insert(RowId(0), 10);
    times.insert(RowId(1), 20);

    times.insert(RowId(0), 40);
    let removed = times.remove(RowId(1));

    assert_eq!(removed, Some(20));
    assert!(times.before(30).is_empty());
    assert_eq!(ids(&times.between(40, 41)), HashSet::from([0]));
}

#[test]
fn windows_before_ascending_cutoffs_grow_one_into_the_next() {
    let mut times = TimeIndex::new();
    for row in 0..10 {
        times.insert(RowId(row), i64::from(9 - row));
    }
    times.insert_untimed(RowId(10));

    let first = times.added_before(None, 5);
    let added = times.added_before(Some(5), 9);

    assert_eq!(first, times.before(5));
    assert!(first.contains(RowId(10)));
    assert_eq!(&first | &added, times.before(9));
    assert!(times.added_before(Some(9), 9).is_empty());
}

#[test]
fn indices_combine_with_set_algebra() {
    let index = two_clusters();
    let mut times = TimeIndex::new();
    for row in 0..128 {
        times.insert(RowId(row), i64::from(row));
    }

    // Rows like the query, from before time 32.
    let every: Context = (0..128).map(RowId).collect();
    let near = index.nearest_rows(&A, 64, &every, &Kept, &Kept).unwrap();
    let context = &near & &times.before(32);

    assert_eq!(ids(&context), (0..32).step_by(2).collect());
}

#[test]
fn underfull_pages_merge_into_their_nearest_neighbor() {
    let mut index = two_clusters();
    let before = index.pages().count();
    let page = index.page_of(RowId(0)).unwrap();
    let mut doomed: Vec<RowId> = index.rows(page).unwrap().iter().collect();
    let survivor = doomed.pop().unwrap();

    for row in doomed {
        index.take(row).unwrap();
    }

    assert_eq!(index.pages().count(), before - 1);
    let home = index.page_of(survivor).unwrap();
    assert_ne!(home, page);
    // The survivor is near A, so it joins a page of rows near A.
    assert!(
        ids(&index.rows(home).unwrap())
            .iter()
            .all(|row| row % 2 == 0)
    );
    assert!(page_rows(&index).iter().all(|rows| rows.len() <= PAGE));
}

#[test]
fn splits_keep_outliers_in_full_pages() {
    let mut index = index();
    let mut noise = Noise(3);
    for row in 0..PAGE as u32 {
        index.put(RowId(row), Bucket(0), &noise.near(A)).unwrap();
    }

    index
        .put(RowId(PAGE as u32), Bucket(0), &[1_000.0, 1_000.0])
        .unwrap();

    assert!(page_rows(&index).iter().all(|rows| rows.len() >= PAGE / 8));
}

#[test]
fn nearest_rows_stay_within_their_window() {
    let index = two_clusters();
    let window: Context = (0..64).map(RowId).collect();

    let nearest = index.nearest_rows(&A, 3, &window, &Kept, &Kept).unwrap();
    assert_eq!(nearest.len(), 3);
    assert!(nearest.iter().all(|row| row.0 < 64));
}

#[test]
fn nearest_rows_match_a_full_scan_as_rows_come_and_go() {
    let mut noise = Noise(7);
    let mut index = index();
    let mut vectors = std::collections::HashMap::new();
    for row in 0..600 {
        let vector = [noise.next() * 50.0, noise.next() * 50.0];
        index
            .put(RowId(row), Bucket(u64::from(row % 3)), &vector)
            .unwrap();
        vectors.insert(row, vector);
    }
    // Removing rows moves centroids, which the pages' bounds must follow.
    for row in (0..600).step_by(3) {
        index.take(RowId(row)).unwrap();
        vectors.remove(&row);
    }
    let allowed: Context = (0..600).filter(|row| row % 5 != 0).map(RowId).collect();

    for _ in 0..20 {
        let query = [noise.next() * 60.0, noise.next() * 60.0];
        let mut scanned: Vec<(f64, u32)> = vectors
            .iter()
            .filter(|(row, _)| allowed.contains(RowId(**row)))
            .map(|(&row, vector)| {
                let distance = vector
                    .iter()
                    .zip(&query)
                    .map(|(&value, &target)| (f64::from(value) - f64::from(target)).powi(2))
                    .sum();
                (distance, row)
            })
            .collect();
        scanned.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let expected: Context = scanned
            .iter()
            .take(25)
            .map(|&(_, row)| RowId(row))
            .collect();

        assert_eq!(
            index
                .nearest_rows(&query, 25, &allowed, &Kept, &Kept)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn rows_put_in_many_at_once_land_alike_every_time() {
    let mut noise = Noise(7);
    // Enough rows for over 1,024 pages, so later runs probe on threads.
    let rows: Vec<(RowId, Bucket, Vec<f32>)> = (0..30_000)
        .map(|row| {
            let center = if row % 2 == 0 { A } else { B };
            (
                RowId(row),
                Bucket(u64::from(row % 3 == 0)),
                noise.near(center).to_vec(),
            )
        })
        .collect();

    let mut first = index();
    let mut second = index();
    first.put_many(&rows).unwrap();
    second.put_many(&rows).unwrap();

    for (row, _, _) in &rows {
        let page = first.page_of(*row).unwrap();
        assert_eq!(second.page_of(*row), Some(page));
        assert!(first.rows(page).unwrap().contains(*row));
    }
    assert!(first.pages().count() > 1_024);
    assert!(page_rows(&first).iter().all(|rows| rows.len() <= PAGE));
    for rows in page_rows(&first) {
        let parities: HashSet<u32> = rows.iter().map(|row| row % 2).collect();
        let buckets: HashSet<bool> = rows.iter().map(|row| row % 3 == 0).collect();
        assert_eq!((parities.len(), buckets.len()), (1, 1), "{rows:?}");
    }
}

const SERIES_A: SeriesKey = SeriesKey(1);
const SERIES_B: SeriesKey = SeriesKey(2);

fn series_index(step: u64) -> SeriesIndex {
    SeriesIndex::new(NonZeroU64::new(step).unwrap()).unwrap()
}

/// Series A at times 100, 110 and 130, and B at 105, on a step of 10.
fn two_series() -> SeriesIndex {
    let mut index = series_index(10);
    for (row, time) in [(0, 100), (1, 110), (2, 130)] {
        index
            .insert(RowId(row), SERIES_A.row_key(time), time, true)
            .unwrap();
    }
    index
        .insert(RowId(3), SERIES_B.row_key(105), 105, true)
        .unwrap();
    index
}

fn some_rows(rows: &[Option<u32>]) -> Vec<Option<RowId>> {
    rows.iter().map(|row| row.map(RowId)).collect()
}

#[test]
fn row_keys_pack_the_series_and_the_time() {
    let key = SERIES_A.row_key(-30);

    assert_eq!(SeriesKey::unpack(key), (SERIES_A, -30));
    assert_ne!(key, SERIES_A.row_key(30));
    assert_ne!(key, SERIES_B.row_key(-30));
    assert_eq!(
        SeriesKey::unpack(SeriesKey(u64::MAX).row_key(i64::MIN)),
        (SeriesKey(u64::MAX), i64::MIN)
    );
}

#[test]
fn windows_lay_rows_on_the_grid_with_gaps_unobserved() {
    let index = two_series();

    let window = index.window(SERIES_A, 140, 5).unwrap();

    assert_eq!(
        window.rows,
        some_rows(&[None, Some(0), Some(1), None, Some(2)])
    );
    assert_eq!(window.observed(), [false, true, true, false, true]);
    assert_eq!(ids(&window.context()), HashSet::from([0, 1, 2]));
    assert_eq!(index.window(SERIES_A, 130, 3).unwrap().count(), 2);
    assert_eq!(index.window(SERIES_A, 130, 0).unwrap().rows, []);
    assert_eq!(
        index.after(SERIES_A, 120, 2).unwrap().rows,
        some_rows(&[None, Some(2)])
    );
    assert_eq!(index.after(SERIES_A, 120, 0).unwrap().rows, []);
    assert_eq!(index.latest(SERIES_A), Some(130));
    assert_eq!(index.count(SERIES_A), 3);
    assert_eq!(index.row(SERIES_A.row_key(110)), Some(RowId(1)));
    assert_eq!(index.row(SERIES_A.row_key(120)), None);
}

#[test]
fn a_new_series_lies_on_any_grid() {
    let index = two_series();

    let window = index.window(SeriesKey(9), 7, 3).unwrap();

    assert_eq!(window.count(), 0);
    assert_eq!(window.rows.len(), 3);
}

#[test]
fn misplaced_rows_and_cutoffs_are_refused() {
    let mut index = two_series();

    assert_eq!(
        index.insert(RowId(4), SERIES_A.row_key(125), 125, true),
        Err(SeriesError::Irregular {
            series: SERIES_A,
            time: 125
        })
    );
    assert_eq!(
        index.insert(RowId(4), SERIES_A.row_key(110), 110, true),
        Err(SeriesError::Taken {
            series: SERIES_A,
            time: 110
        })
    );
    assert_eq!(
        index.insert(RowId(4), SERIES_A.row_key(120), 110, true),
        Err(SeriesError::Mismatch {
            key: SERIES_A.row_key(120),
            time: 110
        })
    );
    assert_eq!(
        index.window(SERIES_A, 125, 2),
        Err(SeriesError::OffGrid {
            series: SERIES_A,
            cutoff: 125
        })
    );
    // B's grid is offset from A's.
    assert_eq!(
        index.window(SERIES_B, 115, 1).unwrap().rows,
        some_rows(&[Some(3)])
    );
    assert!(index.window(SERIES_B, 110, 1).is_err());
    assert_eq!(index.count(SERIES_A), 3);
    assert!(SeriesIndex::new(NonZeroU64::new(u64::MAX).unwrap()).is_err());
}

#[test]
fn rows_leave_and_an_emptied_series_takes_a_new_grid() {
    let mut index = two_series();

    assert_eq!(index.remove(SERIES_B.row_key(105)), Some(RowId(3)));
    assert_eq!(index.remove(SERIES_B.row_key(105)), None);
    assert_eq!(index.count(SERIES_B), 0);
    assert_eq!(index.series().collect::<Vec<_>>(), [SERIES_A]);
    // Off the old grid, which the series forgot with its last row.
    index
        .insert(RowId(3), SERIES_B.row_key(102), 102, true)
        .unwrap();
    assert_eq!(
        index.window(SERIES_B, 112, 1).unwrap().rows,
        some_rows(&[Some(3)])
    );
    assert!(index.window(SERIES_B, 115, 1).is_err());
    for time in [100, 110, 130] {
        index.remove(SERIES_A.row_key(time));
    }
    index.remove(SERIES_B.row_key(102));
    assert!(index.is_empty());
}

#[test]
fn windows_at_the_time_limits_stay_in_their_half() {
    let step = 1u64 << 20;
    let mut index = series_index(step);
    let late = i64::MAX - (step as i64) + 1;
    index
        .insert(RowId(0), SERIES_A.row_key(late), late, true)
        .unwrap();
    index
        .insert(RowId(1), SERIES_A.row_key(i64::MIN), i64::MIN, true)
        .unwrap();

    // Both times share a grid; neither reaches the other's window.
    let early = index.window(SERIES_A, i64::MIN + (step as i64), 2).unwrap();
    assert_eq!(early.rows, some_rows(&[None, Some(1)]));
    // The latest grid point: nothing is before it within a step, and a grid
    // from it reaches past the last time.
    assert_eq!(
        index.window(SERIES_A, late, 2).unwrap().rows,
        some_rows(&[None, None])
    );
    assert_eq!(
        index.after(SERIES_A, late, 3).unwrap().rows,
        some_rows(&[Some(0), None, None])
    );
    assert_eq!(
        index.after(SERIES_A, i64::MIN, 2).unwrap().rows,
        some_rows(&[Some(1), None])
    );
}

#[test]
fn the_window_tells_units_apart() {
    let print = Fingerprint::of(RowId(0), Version(0)) + Fingerprint::of(RowId(1), Version(0));

    assert_ne!(with_window(print, 100, 5), with_window(print, 110, 5));
    assert_ne!(with_window(print, 110, 5), with_window(print, 110, 6));
    assert_eq!(with_window(print, 100, 5), with_window(print, 100, 5));
    assert_ne!(with_window(print, 100, 5), print);
}

#[test]
fn series_windows_match_a_naive_list_through_random_changes() {
    let mut index = series_index(3);
    let mut naive: Vec<(SeriesKey, i64, RowId)> = Vec::new();
    let mut noise = Noise(11);
    let mut next_row = 0;
    for _ in 0..600 {
        let series = SeriesKey(u64::from(noise.next() > 0.0));
        let time = (noise.next() * 30.0) as i64 * 3 + series.0 as i64;
        let key = series.row_key(time);
        if noise.next() > -0.3 {
            let row = RowId(next_row);
            let placed = index.insert(row, key, time, true);
            let taken = naive.iter().any(|&(s, t, _)| s == series && t == time);
            assert_eq!(placed.is_ok(), !taken, "insert of {series:?} at {time}");
            if placed.is_ok() {
                naive.push((series, time, row));
                next_row += 1;
            }
        } else {
            let removed = index.remove(key);
            let position = naive.iter().position(|&(s, t, _)| s == series && t == time);
            assert_eq!(removed, position.map(|at| naive[at].2));
            if let Some(at) = position {
                naive.swap_remove(at);
            }
        }
        for series in [SeriesKey(0), SeriesKey(1)] {
            let cutoff = (noise.next() * 40.0) as i64 * 3 + series.0 as i64;
            let steps = (noise.next().abs() * 20.0) as usize;
            let window = index.window(series, cutoff, steps).unwrap();
            let expected: Vec<Option<RowId>> = (0..steps)
                .map(|k| {
                    let time = cutoff - (steps - k) as i64 * 3;
                    naive
                        .iter()
                        .find(|&&(s, t, _)| s == series && t == time)
                        .map(|&(_, _, row)| row)
                })
                .collect();
            assert_eq!(
                window.rows, expected,
                "window of {series:?} before {cutoff}"
            );
            let after = index.after(series, cutoff, steps).unwrap();
            let expected: Vec<Option<RowId>> = (0..steps)
                .map(|k| {
                    let time = cutoff + k as i64 * 3;
                    naive
                        .iter()
                        .find(|&&(s, t, _)| s == series && t == time)
                        .map(|&(_, _, row)| row)
                })
                .collect();
            assert_eq!(after.rows, expected, "after of {series:?} from {cutoff}");
        }
    }
}
