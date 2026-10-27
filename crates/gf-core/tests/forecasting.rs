// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Series grouped into units before a cutoff, and laid onto grids.

use std::collections::HashSet;
use std::num::{NonZeroU64, NonZeroUsize};

use gf_core::forecasting::{
    ForecastError, FutureRow, GroupStrategy, HORIZON_PASSES, KNOWN, Neighbors, PAST, SeriesPlan,
    SeriesRequest, SeriesSpec, TARGET, future, grid, horizon_limit, plan,
};
use gf_core::rows::Row;
use gf_core::series::{SeriesError, SeriesIndex, SeriesKey};
use gf_core::table::{Change, RowId, Table};

const A: SeriesKey = SeriesKey(1);
const B: SeriesKey = SeriesKey(2);
const C: SeriesKey = SeriesKey(3);
const D: SeriesKey = SeriesKey(4);

fn spec(group: GroupStrategy, known: Vec<usize>) -> SeriesSpec {
    SeriesSpec {
        step: NonZeroU64::new(10).unwrap(),
        patch: NonZeroUsize::new(4).unwrap(),
        context: NonZeroUsize::new(16).unwrap(),
        max_horizon: NonZeroUsize::new(256).unwrap(),
        max_rows: None,
        known,
        group,
        neighbors: Neighbors::Targets,
        reusable: true,
        extendable: false,
        patch_bytes: 0,
    }
}

fn random(size: usize, seed: u64) -> GroupStrategy {
    GroupStrategy::Random {
        size: NonZeroUsize::new(size).unwrap(),
        seed,
    }
}

#[test]
fn horizon_limit_uses_pass_limit_and_saturates() {
    let output_patch = NonZeroUsize::new(4).unwrap();
    assert_eq!(horizon_limit(output_patch, None).get(), HORIZON_PASSES * 4);
    assert_eq!(
        horizon_limit(output_patch, NonZeroUsize::new(7)).get(),
        HORIZON_PASSES * 7
    );
    assert_eq!(horizon_limit(NonZeroUsize::MAX, None), NonZeroUsize::MAX);
}

/// A table of series on a step of 10, each row's target its time (NaN at
/// 130) and its one feature the time plus a half.
fn placed(placed: &[(SeriesKey, i64)]) -> (Table, SeriesIndex, Vec<Row>) {
    let mut table = Table::new();
    let mut index = SeriesIndex::new(NonZeroU64::new(10).unwrap()).unwrap();
    let mut rows = Vec::new();
    for &(series, time) in placed {
        let key = series.row_key(time);
        let row = table.apply(Change::Insert(key)).unwrap();
        index.insert(row, key, time, true).unwrap();
        let target = if time == 130 { f32::NAN } else { time as f32 };
        rows.push(Row {
            features: vec![time as f32 + 0.5],
            target: target.to_bits(),
            key: key.0,
        });
    }
    (table, index, rows)
}

/// A at 100..=150, B at 120..=170 with a gap at 140, and C with one row at 200.
fn table() -> (Table, SeriesIndex, Vec<Row>) {
    placed(&[
        (A, 100),
        (A, 110),
        (A, 120),
        (A, 130),
        (A, 140),
        (A, 150),
        (B, 120),
        (B, 130),
        (B, 150),
        (B, 160),
        (B, 170),
        (C, 200),
    ])
}

fn request(series: &[SeriesKey], cutoffs: &[Option<i64>], horizon: usize) -> SeriesRequest {
    SeriesRequest {
        series: series.to_vec(),
        cutoffs: cutoffs.to_vec(),
        horizon: NonZeroUsize::new(horizon).unwrap(),
        groups: None,
        future: Vec::new(),
    }
}

fn fetch(rows: &[Row]) -> impl Fn(&[RowId]) -> Result<Vec<Row>, ()> + '_ {
    move |wanted| {
        Ok(wanted
            .iter()
            .map(|row| rows[row.0 as usize].clone())
            .collect())
    }
}

fn groups(planned: &SeriesPlan) -> HashSet<Vec<SeriesKey>> {
    planned
        .units
        .iter()
        .map(|unit| unit.members.clone())
        .collect()
}

#[test]
fn each_series_alone_takes_the_shortest_grid_of_whole_patches() {
    let (table, index, rows) = table();
    let spec = spec(GroupStrategy::Alone, vec![]);
    let requests = [request(&[A, B], &[Some(160), Some(160)], 3)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    assert_eq!(planned.units.len(), 2);
    // A spans 100..160: six steps, padded to eight; B spans 120..160: four.
    assert_eq!(planned.units[0].steps, 8);
    assert_eq!(planned.units[1].steps, 4);
    assert_eq!(planned.units[0].members, [A]);
    assert!(!planned.units[0].whole);
    let laid = grid(&spec, &planned.units[0], 1, fetch(&rows)).unwrap();
    assert_eq!((laid.rows, laid.steps), (2, 8));
    assert_eq!(laid.roles, [TARGET, PAST]);
    assert_eq!(
        laid.observed[..8],
        [false, false, true, true, true, false, true, true]
    );
    // The NaN target at 130 is unobserved, while its feature is not.
    assert_eq!(
        laid.observed[8..],
        [false, false, true, true, true, true, true, true]
    );
    assert_eq!(laid.values[4], 120.0);
    assert_eq!(laid.values[8 + 5], 130.5);
    let forecasts = &planned.forecasts[0];
    assert_eq!((forecasts[0].unit, forecasts[0].target), (0, 0));
    assert_eq!((forecasts[1].unit, forecasts[1].target), (1, 0));
    // Without known columns, nothing after the cutoff is read: no future rows.
    assert!(forecasts[0].future.is_empty());
    let ahead = future(&spec, &planned.units[0], &forecasts[0], 1, fetch(&rows)).unwrap();
    assert_eq!((ahead.rows, ahead.steps), (0, 3));
    assert!(ahead.observed.is_empty());
}

#[test]
fn groups_share_one_unit_and_each_member_knows_its_row() {
    let (table, index, _) = table();
    let spec = spec(random(3, 7), vec![]);
    let requests = [
        request(&[A], &[Some(160)], 3),
        request(&[B], &[Some(160)], 3),
    ];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    // Three series, groups of three: every query names the same group.
    assert_eq!(planned.units.len(), 1);
    assert_eq!(planned.units[0].members, [A, B, C]);
    assert_eq!(planned.units[0].target, None);
    let targets: Vec<usize> = planned
        .forecasts
        .iter()
        .flatten()
        .map(|forecast| forecast.target)
        .collect();
    assert_eq!(targets, [0, 1]);
    // C's row at 200 is after the cutoff, so the grid is A's span.
    assert_eq!(planned.units[0].steps, 8);
    assert_eq!(planned.units[0].rows().len(), 9);
}

#[test]
fn series_without_history_are_forecast_through_their_group_or_refused() {
    let (table, index, _) = table();
    let snapshot = table.snapshot();
    let planned = |group, series, cutoff| {
        let spec = spec(group, vec![]);
        plan(
            &spec,
            &index,
            &snapshot,
            1,
            &[request(&[series], &[cutoff], 3)],
        )
    };

    // A series the table never saw is a cold start among observed siblings.
    let cold = planned(random(2, 7), SeriesKey(9), Some(160)).unwrap();
    assert_eq!(cold.units[0].members.len(), 2);
    assert_eq!(
        cold.units[0].members[cold.forecasts[0][0].target],
        SeriesKey(9)
    );
    assert_eq!(
        planned(GroupStrategy::Alone, SeriesKey(9), Some(160)),
        Err(ForecastError::UnknownSeries(SeriesKey(9)))
    );
    // A's rows start at 100 and B's at 120: nothing is observed before 100.
    assert_eq!(
        planned(GroupStrategy::Alone, A, Some(100)),
        Err(ForecastError::NothingBefore {
            series: A,
            cutoff: 100
        })
    );
    assert_eq!(
        planned(random(3, 7), A, Some(100)),
        Err(ForecastError::NothingBefore {
            series: A,
            cutoff: 100
        })
    );
    assert_eq!(
        planned(GroupStrategy::Alone, A, Some(i64::MIN)),
        Err(ForecastError::NothingBefore {
            series: A,
            cutoff: i64::MIN
        })
    );
    assert!(planned(GroupStrategy::Alone, A, Some(110)).is_ok());
    // C's only row is at 200; with A and B before 160 it is forecast cold.
    assert!(planned(random(3, 7), C, Some(160)).is_ok());
}

#[test]
fn random_groups_differ_across_targets_and_seeds() {
    let rows: Vec<(SeriesKey, i64)> = (1..=12).map(|series| (SeriesKey(series), 100)).collect();
    let (table, index, _) = placed(&rows);
    let snapshot = table.snapshot();
    let series: Vec<SeriesKey> = (1..=12).map(SeriesKey).collect();
    let cutoffs = vec![Some(110); 12];

    let first = plan(
        &spec(random(3, 7), vec![]),
        &index,
        &snapshot,
        1,
        &[request(&series, &cutoffs, 2)],
    )
    .unwrap();
    let second = plan(
        &spec(random(3, 8), vec![]),
        &index,
        &snapshot,
        1,
        &[request(&series, &cutoffs, 2)],
    )
    .unwrap();

    // Not every series picks the same two others.
    assert!(groups(&first).len() > 3, "{:?}", groups(&first));
    assert_ne!(groups(&first), groups(&second));
    for (forecast, &target) in first.forecasts[0].iter().zip(&series) {
        assert_eq!(first.units[forecast.unit].members[forecast.target], target);
    }
}

#[test]
fn every_forecast_reads_its_own_series_through_random_tables() {
    let mut state = 11u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state >> 33
    };
    for round in 0..40 {
        let count = 2 + (next() % 9) as usize;
        let mut rows = Vec::new();
        let (mut series, mut cutoffs) = (Vec::new(), Vec::new());
        for key in 1..=count as u64 {
            let start = 100 + (next() % 5) as i64 * 10;
            let length = (next() % 7) as i64;
            for time in (start..start + length * 10).step_by(10) {
                rows.push((SeriesKey(key), time));
            }
            // A series is forecast from after its first row, if it has one.
            if length > 0 {
                series.push(SeriesKey(key));
                cutoffs.push(Some(start + 10 + (next() % 7) as i64 * 10));
            }
        }
        let (table, index, _) = placed(&rows);
        if series.is_empty() {
            continue;
        }
        // A group no larger than the series the table holds, which it needs.
        let size = (1 + (next() % 4) as usize).min(series.len());
        let spec = spec(random(size, round), vec![]);

        let planned = plan(
            &spec,
            &index,
            &table.snapshot(),
            1,
            &[request(&series, &cutoffs, 2)],
        )
        .unwrap();

        for (forecast, &target) in planned.forecasts[0].iter().zip(&series) {
            let unit = &planned.units[forecast.unit];
            assert_eq!(unit.members[forecast.target], target);
            assert!(unit.members.len() <= size);
            // Every member's rows on the grid are its own and before the cutoff.
            for (member, window) in unit.members.iter().zip(&unit.windows) {
                for row in window.rows.iter().flatten() {
                    let (series, time) = rows[row.0 as usize];
                    assert_eq!(series, *member);
                    assert!(time < unit.cutoff);
                }
            }
        }
    }
}

#[test]
fn covariate_neighbors_give_each_query_its_own_unit() {
    let (table, index, rows) = table();
    let mut spec = spec(random(2, 7), vec![]);
    spec.neighbors = Neighbors::Covariates;
    let requests = [request(&[A, B], &[Some(160), Some(160)], 3)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    assert_eq!(planned.units.len(), 2);
    for (unit, series) in planned.units.iter().zip([A, B]) {
        assert_eq!(unit.target, Some(series));
        let laid = grid(&spec, unit, 1, fetch(&rows)).unwrap();
        let own = unit
            .members
            .iter()
            .position(|&member| member == series)
            .unwrap();
        for member in 0..unit.members.len() {
            let expected = if member == own { TARGET } else { PAST };
            assert_eq!(laid.roles[member * 2], expected);
        }
    }
    assert_ne!(planned.units[0].key, planned.units[1].key);
}

#[test]
fn known_columns_read_rows_after_the_cutoff_and_forecast_whole() {
    let (table, index, rows) = table();
    let spec = spec(GroupStrategy::Alone, vec![0]);
    let requests = [request(&[A], &[Some(130)], 3)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    assert!(planned.units[0].whole);
    let forecast = &planned.forecasts[0][0];
    let ahead = future(&spec, &planned.units[0], forecast, 1, fetch(&rows)).unwrap();
    assert_eq!((ahead.rows, ahead.steps), (2, 3));
    assert_eq!(ahead.roles, [TARGET, KNOWN]);
    // The target's future is never read; the known column's is.
    assert_eq!(ahead.observed[..3], [false, false, false]);
    assert_eq!(ahead.observed[3..], [true, true, true]);
    assert_eq!(ahead.values[3..], [130.5, 140.5, 150.5]);
    // A known column the table lacks makes no row known.
    let unknown_column = SeriesSpec {
        known: vec![3],
        ..spec
    };
    let planned = plan(&unknown_column, &index, &table.snapshot(), 1, &requests).unwrap();
    assert!(!planned.units[0].whole);
    // Forecasts of one group, cutoff and horizon share a whole unit; another
    // horizon reads another.
    let requests = [
        request(&[A], &[Some(130)], 3),
        request(&[A], &[Some(130)], 3),
        request(&[A], &[Some(130)], 4),
    ];
    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();
    assert_eq!(planned.units.len(), 2);
    assert_eq!(planned.forecasts[0][0].unit, planned.forecasts[1][0].unit);
    assert_ne!(planned.forecasts[0][0].unit, planned.forecasts[2][0].unit);
}

fn ahead(series: SeriesKey, time: i64, value: f32) -> FutureRow {
    FutureRow {
        series,
        time,
        features: vec![value],
    }
}

#[test]
fn a_future_the_request_brings_is_read_in_place_of_the_tables_rows() {
    let (table, index, rows) = table();
    let spec = spec(GroupStrategy::Alone, vec![0]);
    // A's latest observation is at 150, so its forecast starts at 160.
    let mut asked = request(&[A], &[None], 3);
    asked.future = vec![
        ahead(A, 160, 7.0),
        ahead(A, 170, 8.0),
        ahead(A, 180, f32::NAN),
    ];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &[asked.clone()]).unwrap();

    let laid = future(
        &spec,
        &planned.units[0],
        &planned.forecasts[0][0],
        1,
        fetch(&rows),
    )
    .unwrap();
    assert_eq!(laid.roles, [TARGET, KNOWN]);
    // A value not known stays unobserved, and the target's future is never read.
    assert_eq!(laid.values[3..5], [7.0, 8.0]);
    assert_eq!(laid.observed[3..], [true, true, false]);
    assert_eq!(laid.observed[..3], [false, false, false]);
    // Another request's other future reads another unit; the same future shares one.
    let mut other = asked.clone();
    other.future[0].features = vec![9.0];
    let planned = plan(
        &spec,
        &index,
        &table.snapshot(),
        1,
        &[asked.clone(), other, asked],
    )
    .unwrap();
    assert_eq!(planned.units.len(), 2);
    assert_ne!(planned.forecasts[0][0].unit, planned.forecasts[1][0].unit);
    assert_eq!(planned.forecasts[0][0].unit, planned.forecasts[2][0].unit);
}

#[test]
fn a_future_is_refused_unless_after_the_series_latest_observation_on_its_grid() {
    let (table, index, _) = table();
    let known = spec(GroupStrategy::Alone, vec![0]);
    let planned = |spec: &SeriesSpec, future: Vec<FutureRow>| {
        let mut asked = request(&[A], &[None], 3);
        asked.future = future;
        plan(spec, &index, &table.snapshot(), 1, &[asked]).map(|_| ())
    };

    assert_eq!(
        planned(&known, vec![ahead(A, 150, 1.0)]),
        Err(ForecastError::FutureNotAfter {
            series: A,
            time: 150
        })
    );
    assert_eq!(
        planned(&known, vec![ahead(A, 165, 1.0)]),
        Err(ForecastError::FutureOffGrid {
            series: A,
            time: 165
        })
    );
    assert_eq!(
        planned(&known, vec![ahead(A, 190, 1.0)]),
        Err(ForecastError::FuturePastHorizon {
            series: A,
            time: 190
        })
    );
    assert_eq!(
        planned(&known, vec![ahead(A, 160, f32::INFINITY)]),
        Err(ForecastError::FutureNotFinite {
            series: A,
            time: 160
        })
    );
    assert_eq!(
        planned(&known, vec![ahead(B, 180, 1.0)]),
        Err(ForecastError::FutureUnasked(B))
    );
    assert_eq!(
        planned(
            &spec(GroupStrategy::Alone, vec![]),
            vec![ahead(A, 160, 1.0)]
        ),
        Err(ForecastError::FutureWithoutKnown)
    );
}

#[test]
fn members_on_other_grid_offsets_are_read_at_their_own_grid() {
    // A on multiples of 10, D offset by 5, grouped and read at A's cutoff.
    let (table, index, rows) = placed(&[
        (A, 100),
        (A, 110),
        (A, 120),
        (D, 105),
        (D, 115),
        (D, 125),
        (D, 135),
    ]);
    let spec = spec(random(2, 7), vec![0]);
    let requests = [request(&[A], &[Some(130)], 2), request(&[D], &[None], 2)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    let unit = &planned.units[planned.forecasts[0][0].unit];
    assert_eq!(unit.members, [A, D]);
    // D's rows before 130 all land on its grid, read before 135, which ends
    // the grid as A's cutoff ends A's.
    let laid = grid(&spec, unit, 1, fetch(&rows)).unwrap();
    assert_eq!(laid.steps, 4);
    assert_eq!(laid.observed[2 * 4..3 * 4], [false, true, true, true]);
    assert_eq!(laid.values[2 * 4..3 * 4], [0.0, 105.0, 115.0, 125.0]);
    // Its known future starts at the first grid point from the cutoff on.
    let ahead = future(&spec, unit, &planned.forecasts[0][0], 1, fetch(&rows)).unwrap();
    assert_eq!(ahead.observed[3 * 2..3 * 2 + 2], [true, false]);
    assert_eq!(ahead.values[3 * 2], 135.5);
    // The default cutoff, the latest time plus a step, is off A's grid.
    let unit = &planned.units[planned.forecasts[1][0].unit];
    assert_eq!(unit.cutoff, 145);
    assert_eq!(unit.members, [A, D]);
}

#[test]
fn a_group_past_the_cap_on_rows_is_refused_rather_than_drawn_smaller() {
    let (table, index, _) = table();
    let mut spec = spec(random(3, 7), vec![]);
    spec.max_rows = NonZeroUsize::new(4);
    let requests = [request(&[A], &[Some(160)], 3)];

    // Two rows per member, four rows at most: a group of three does not fit.
    assert_eq!(
        plan(&spec, &index, &table.snapshot(), 1, &requests),
        Err(ForecastError::GroupTooLarge {
            size: 3,
            rows: 6,
            cap: 4
        })
    );
    assert_eq!(
        plan(&spec, &index, &table.snapshot(), 4, &requests),
        Err(ForecastError::TooManyRows { rows: 5, cap: 4 })
    );
    spec.max_rows = NonZeroUsize::new(6);
    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();
    assert_eq!(planned.units[0].members.len(), 3);
    // Three series in the table: a group of four has too few others.
    let wide = SeriesSpec {
        group: random(4, 7),
        max_rows: None,
        ..spec.clone()
    };
    assert_eq!(
        plan(&wide, &index, &table.snapshot(), 1, &requests),
        Err(ForecastError::TooFewSeries { size: 4, held: 2 })
    );
}

#[test]
fn cutoffs_default_to_the_series_latest_observed_time_plus_a_step() {
    let (table, index, _) = table();
    let spec = spec(GroupStrategy::Alone, vec![]);
    let requests = [request(&[B], &[None], 2)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    // B's rows end at 170, whatever the other series hold.
    assert_eq!(planned.units[0].cutoff, 180);
    // B spans 120..180: six steps, padded to eight.
    assert_eq!(planned.units[0].steps, 8);
    assert_eq!(
        plan(
            &spec,
            &SeriesIndex::new(NonZeroU64::new(10).unwrap()).unwrap(),
            &table.snapshot(),
            1,
            &requests
        ),
        Err(ForecastError::NoRows(B))
    );
}

#[test]
fn the_context_caps_the_grid() {
    let (table, index, _) = table();
    let mut spec = spec(GroupStrategy::Alone, vec![]);
    let requests = [request(&[A], &[Some(400)], 3)];

    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();

    assert_eq!(planned.units[0].steps, 16);
    // A context shorter than a patch is one patch.
    spec.context = NonZeroUsize::new(2).unwrap();
    let planned = plan(&spec, &index, &table.snapshot(), 1, &requests).unwrap();
    assert_eq!(planned.units[0].steps, 4);
}

#[test]
fn malformed_requests_are_refused() {
    let (table, index, _) = table();
    let spec = spec(GroupStrategy::Alone, vec![]);

    assert_eq!(
        plan(
            &spec,
            &index,
            &table.snapshot(),
            1,
            &[request(&[A, B], &[Some(160)], 3)]
        ),
        Err(ForecastError::Cutoffs {
            series: 2,
            cutoffs: 1
        })
    );
    // A cutoff so late that no grid point of the series follows it.
    assert!(matches!(
        plan(
            &spec,
            &index,
            &table.snapshot(),
            1,
            &[request(&[A], &[Some(i64::MAX - 3)], 3)]
        ),
        Err(ForecastError::Series(SeriesError::OffGrid { .. }))
    ));
}
