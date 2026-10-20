//! Context strategies: which row sets to encode, and what each query reads.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::LazyLock;

mod support;

use gf_core::context::Context;
use gf_core::pages::{Bucket, PageIndex};
use gf_core::planner::{
    Core, CoreKind, Cores, Plan, PlanError, Ranking, Source, Strata, Strategy, StrategyError,
    Window, Windows, plan, plan_with, plan_within,
};
use gf_core::table::{Change, RowId, RowKey, Snapshot, Table};
use gf_core::time_index::TimeIndex;
use support::{Keeping, Kept};

const PAGE: usize = 8;

/// 32 rows around (0, 0) (even ids) and 32 around (100, 100) (odd ids).
fn world() -> (Table, PageIndex) {
    let mut table = Table::new();
    let mut index = PageIndex::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(PAGE).unwrap(),
    );
    for row in 0..64u32 {
        let id = table
            .apply(Change::Insert(RowKey(u128::from(row))))
            .unwrap();
        let offset = (row / 2) as f32 * 0.03;
        let vector = if row % 2 == 0 {
            [offset, -offset]
        } else {
            [100.0 + offset, 100.0 - offset]
        };
        index.put(id, Bucket(0), &vector).unwrap();
    }
    (table, index)
}

const NEAR_A: [[f32; 2]; 2] = [[0.1, 0.0], [0.2, -0.1]];
const NEAR_B: [f32; 2] = [100.2, 99.9];

fn queries() -> Vec<Vec<f32>> {
    vec![NEAR_A[0].to_vec(), NEAR_A[1].to_vec(), NEAR_B.to_vec()]
}

fn ids(context: &Context) -> HashSet<u32> {
    context.iter().map(|row| row.0).collect()
}

/// The rows a query reads.
fn read(plan: &Plan, query: usize) -> HashSet<u32> {
    ids(&plan.units()[plan.queries()[query].0].rows)
}

/// Every row in one stratum.
struct Unstratified;

impl Strata for Unstratified {
    fn stratum(&self, _: RowId) -> u32 {
        0
    }
}

/// No row has a time.
static NO_TIMES: LazyLock<TimeIndex> = LazyLock::new(TimeIndex::new);

fn source<'a>(snapshot: &'a Snapshot, index: &'a PageIndex) -> Source<'a> {
    Source {
        snapshot,
        index,
        vectors: &Kept,
        strata: &Unstratified,
        keys: &Kept,
        ranking: None,
        times: &NO_TIMES,
    }
}

fn make(strategy: &Strategy, snapshot: &Snapshot, index: &PageIndex) -> Plan {
    plan(strategy, &source(snapshot, index), &queries()).unwrap()
}

#[test]
fn exact_encodes_every_row_once_for_all_queries() {
    let (table, index) = world();

    let plan = make(&Strategy::Exact, &table.snapshot(), &index);

    assert_eq!(plan.units().len(), 1);
    assert_eq!(plan.units()[0].rows.len(), 64);
    assert!(plan.queries().iter().all(|unit| unit.0 == 0));
}

#[test]
fn subsamples_are_deterministic_and_shared() {
    let (table, index) = world();
    let snapshot = table.snapshot();
    let sample = |seed| {
        make(
            &Strategy::subsample(16, seed, CoreKind::Random),
            &snapshot,
            &index,
        )
    };

    let (first, again, other) = (sample(1), sample(1), sample(2));

    assert_eq!(first.units().len(), 1);
    assert_eq!(first.units()[0].rows.len(), 16);
    assert_eq!(first.units()[0].rows, again.units()[0].rows);
    assert_ne!(first.units()[0].rows, other.units()[0].rows);
}

#[test]
fn subsamples_barely_change_as_rows_arrive() {
    let (mut table, index) = world();
    let strategy = Strategy::subsample(16, 3, CoreKind::Random);
    let before = make(&strategy, &table.snapshot(), &index);

    for key in 64..74 {
        table.apply(Change::Insert(RowKey(key))).unwrap();
    }
    let after = make(&strategy, &table.snapshot(), &index);

    // Ten new rows can displace at most ten sampled rows, so most of the cache key survives.
    let kept = (&before.units()[0].rows & &after.units()[0].rows).len();
    assert!(kept >= 6, "only {kept} of 16 sampled rows survived");
}

#[test]
fn row_neighbors_give_each_query_its_own_context() {
    let (table, index) = world();

    let plan = make(&Strategy::neighbors(5), &table.snapshot(), &index);

    // The per-query kNN baseline: exact neighbors, but contexts are rarely shared.
    assert_eq!(plan.units().len(), 3);
    assert_eq!(read(&plan, 0), HashSet::from([0, 2, 4, 6, 8]));
    assert!(plan.units().iter().all(|unit| unit.rows.len() == 5));
}

#[test]
fn deleted_rows_never_reach_a_plan() {
    let (mut table, index) = world();
    for key in [0, 2, 4] {
        table.apply(Change::Delete(RowKey(key))).unwrap();
    }
    let snapshot = table.snapshot();
    let strategies = [
        Strategy::Exact,
        Strategy::subsample(8, 1, CoreKind::Random),
        Strategy::neighbors(5),
    ];

    for strategy in &strategies {
        let plan = make(strategy, &snapshot, &index);
        for unit in plan.units() {
            assert!(!unit.rows.contains(RowId(0)) && !unit.rows.contains(RowId(2)));
        }
    }
}

#[test]
fn unit_keys_are_stable_across_batches_and_follow_versions() {
    let (mut table, index) = world();
    let strategy = Strategy::neighbors(5);
    let first = make(&strategy, &table.snapshot(), &index);
    let second = plan(
        &strategy,
        &source(&table.snapshot(), &index),
        &[NEAR_A[0].to_vec()],
    )
    .unwrap();

    table.apply(Change::Update(RowKey(0))).unwrap();
    let updated = plan(
        &strategy,
        &source(&table.snapshot(), &index),
        &[NEAR_A[0].to_vec()],
    )
    .unwrap();

    // A later batch with the same context hits the same cache entry; an update misses it.
    assert_eq!(
        second.units()[0].key,
        first.units()[first.queries()[0].0].key
    );
    assert_ne!(updated.units()[0].key, second.units()[0].key);
}

#[test]
fn queries_read_only_their_window_and_share_units_within_it() {
    let (table, index) = world();
    let early: Context = (0..32).map(RowId).collect();
    let queries = [NEAR_A[0].to_vec(), NEAR_A[0].to_vec(), NEAR_A[0].to_vec()];

    let plan = plan_within(
        &Strategy::Exact,
        &source(&table.snapshot(), &index),
        &queries,
        std::slice::from_ref(&early),
        &[Some(0), Some(0), None],
    )
    .unwrap();

    assert_eq!(plan.units().len(), 2);
    assert_eq!(plan.queries()[0], plan.queries()[1]);
    assert_eq!(plan.units()[plan.queries()[0].0].rows, early);
    assert_eq!(plan.units()[plan.queries()[2].0].rows.len(), 64);
}

#[test]
fn neighbors_are_found_among_rows_in_the_window() {
    let (table, index) = world();
    // Every fourth row near A: pages near A hold few rows in the window.
    let window: Context = (0..64).step_by(4).map(RowId).collect();

    let plan = plan_within(
        &Strategy::neighbors(12),
        &source(&table.snapshot(), &index),
        &[NEAR_A[0].to_vec()],
        std::slice::from_ref(&window),
        &[Some(0)],
    )
    .unwrap();

    let rows = &plan.units()[0].rows;
    assert_eq!(rows.len(), 12);
    assert!(rows.iter().all(|row| window.contains(row)));
}

#[test]
fn windows_grown_before_cutoffs_plan_as_windows_given_whole() {
    let (table, index) = world();
    let snapshot = table.snapshot();
    let mut times = TimeIndex::new();
    for row in 0..64u32 {
        // Times out of row order, and a row without one, before every cutoff.
        match row {
            7 => times.insert_untimed(RowId(row)),
            _ => times.insert(RowId(row), i64::from((row * 37) % 64)),
        }
    }
    let cutoffs = [3, 10, 30, 64];
    let prints: Vec<_> = cutoffs
        .iter()
        .map(|&cutoff| {
            let rows = times.added_before(None, cutoff);
            snapshot.fingerprint(&rows)
        })
        .collect();
    let given: Vec<Window> = cutoffs
        .iter()
        .zip(&prints)
        .map(|(&cutoff, &fingerprint)| Window {
            rows: times.added_before(None, cutoff),
            fingerprint: Some(fingerprint),
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..8).map(|index| NEAR_A[index % 2].to_vec()).collect();
    // Out of order, so planning groups queries by window and keeps their order.
    let window_of = vec![
        Some(3),
        Some(0),
        Some(2),
        Some(1),
        Some(0),
        Some(3),
        None,
        Some(2),
    ];
    let strategies = [
        Strategy::Exact,
        Strategy::subsample(5, 11, CoreKind::Random),
        Strategy::neighbors(4),
    ];

    for strategy in &strategies {
        let whole = plan_with(
            strategy,
            &source(&snapshot, &index),
            &mut Cores::default(),
            &queries,
            Windows::Given(&given),
            &window_of,
        )
        .unwrap();
        let grown = plan_with(
            strategy,
            &source(&snapshot, &index),
            &mut Cores::default(),
            &queries,
            Windows::Before {
                times: &times,
                cutoffs: &cutoffs,
                prints: &prints,
            },
            &window_of,
        )
        .unwrap();
        for query in 0..queries.len() {
            assert_eq!(read(&grown, query), read(&whole, query), "{strategy:?}");
        }
        assert_eq!(grown.units().len(), whole.units().len(), "{strategy:?}");
    }
}

#[test]
fn windows_must_match_their_queries() {
    let (table, index) = world();
    let snapshot = table.snapshot();
    let window = snapshot.rows();

    let short = plan_within(
        &Strategy::Exact,
        &source(&snapshot, &index),
        &queries(),
        std::slice::from_ref(&window),
        &[None],
    );
    let unknown = plan_within(
        &Strategy::Exact,
        &source(&snapshot, &index),
        &queries(),
        &[window],
        &[Some(1), None, None],
    );

    assert_eq!(
        short,
        Err(PlanError::Windows {
            queries: 3,
            windows: 1
        })
    );
    assert_eq!(unknown, Err(PlanError::UnknownWindow(1)));
}

#[test]
fn strategies_with_a_zero_count_are_refused() {
    assert!(Strategy::Exact.check().is_ok());
    assert_eq!(
        Strategy::subsample(0, 1, CoreKind::Random).check(),
        Err(StrategyError::ZeroRows)
    );
    assert_eq!(Strategy::neighbors(0).check(), Err(StrategyError::Empty));
    assert!(Strategy::neighbors(4).check().is_ok());
}

#[test]
fn retrieved_contexts_hold_the_core_and_each_querys_nearest_rows_outside_it() {
    let (table, index) = world();
    let snapshot = table.snapshot();
    let core = Core {
        rows: 8,
        seed: 3,
        kind: CoreKind::Random,
    };
    let strategy = Strategy::Retrieved {
        core: Some(core),
        neighbors: 5,
        among: None,
        recency: None,
    };

    let retrieved = make(&strategy, &snapshot, &index);

    let sampled = read(
        &make(
            &Strategy::subsample(8, 3, CoreKind::Random),
            &snapshot,
            &index,
        ),
        0,
    );
    for query in 0..3 {
        let rows = read(&retrieved, query);
        assert!(sampled.is_subset(&rows));
        let nearest: HashSet<u32> = rows.difference(&sampled).copied().collect();
        assert_eq!(nearest.len(), 5);
        // Neighbors near A are even rows, near B odd ones.
        assert!(nearest.iter().all(|row| row % 2 == u32::from(query == 2)));
    }
}

/// Rows below 48 in one stratum and the rest in another.
struct Halves;

impl Strata for Halves {
    fn stratum(&self, row: RowId) -> u32 {
        u32::from(row.0 >= 48)
    }
}

#[test]
fn stratified_cores_follow_their_strata_and_balanced_ones_even_them() {
    let (table, index) = world();
    let snapshot = table.snapshot();
    let source = Source {
        strata: &Halves,
        ..source(&snapshot, &index)
    };
    let core = |rows, kind| {
        let plan = plan(&Strategy::subsample(rows, 5, kind), &source, &queries()).unwrap();
        let rows = read(&plan, 0);
        let high = rows.iter().filter(|&&row| row >= 48).count();
        (rows.len() - high, high)
    };

    assert_eq!(core(16, CoreKind::Stratified), (12, 4));
    assert_eq!(core(16, CoreKind::Balanced), (8, 8));
    // The small stratum gives what it cannot take to the other.
    assert_eq!(core(40, CoreKind::Balanced), (24, 16));
}

#[test]
fn diverse_cores_spread_out_and_are_kept_until_cleared() {
    let (mut table, mut index) = world();
    let strategy = Strategy::subsample(2, 0, CoreKind::Diverse);
    let mut cores = Cores::default();
    let drawn = |table: &Table, index: &PageIndex, cores: &mut Cores| {
        let snapshot = table.snapshot();
        let windows = Windows::Given(&[]);
        let plan = plan_with(
            &strategy,
            &source(&snapshot, index),
            cores,
            &queries(),
            windows,
            &[None; 3],
        )
        .unwrap();
        read(&plan, 0)
    };

    let first = drawn(&table, &index, &mut cores);
    // A third cluster far from both.
    for key in 64..80 {
        let id = table.apply(Change::Insert(RowKey(key))).unwrap();
        index.put(id, Bucket(0), &[300.0, 300.0]).unwrap();
    }
    let kept = drawn(&table, &index, &mut cores);
    cores.clear();
    let redrawn = drawn(&table, &index, &mut cores);

    // One row of each cluster, the farthest apart.
    assert_eq!(first.iter().map(|row| row % 2).sum::<u32>(), 1);
    assert_eq!(kept, first);
    assert!(redrawn.iter().any(|&row| row >= 64), "{redrawn:?}");
}

#[test]
fn a_ranking_kept_through_changes_matches_one_made_afresh() {
    let rows = |ids: &[u32]| -> Context { ids.iter().map(|&id| RowId(id)).collect() };
    let mut kept = Ranking::new(7, &rows(&[0, 1, 2, 3, 4]), &Kept);
    kept.remove(RowId(1), 1);
    kept.remove(RowId(3), 3);
    kept.insert(RowId(9), 9);
    kept.insert(RowId(5), 5);

    assert_eq!(kept, Ranking::new(7, &rows(&[0, 2, 4, 5, 9]), &Kept));
    assert_ne!(kept, Ranking::new(8, &rows(&[0, 2, 4, 5, 9]), &Kept));
}
