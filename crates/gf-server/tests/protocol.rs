// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Frames between the server and its workers, and strategy specs.

mod common;

use gf_core::planner::{Core, CoreKind, Recency, Strategy};
use gf_server::protocol::{Frame, ModelProfile, StrategySpec, WorkerReply, outcome_from};
use serde_json::json;

#[test]
fn frames_round_trip_with_their_buffers() {
    let frame = Frame {
        header: json!({"kind": "step", "units": [1, 2]}),
        buffers: vec![vec![1, 2, 3], Vec::new(), 1.5f32.to_le_bytes().to_vec()],
    };

    let again = Frame::from_bytes(&frame.to_bytes()).unwrap();

    assert_eq!(again, frame);
}

#[test]
fn truncated_frames_are_refused() {
    let bytes = Frame {
        header: json!({"kind": "x"}),
        buffers: vec![vec![0; 16]],
    }
    .to_bytes();

    assert!(Frame::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    assert!(Frame::from_bytes(&bytes[..3]).is_err());
}

#[test]
fn strategy_specs_map_to_planner_strategies() {
    let cases = [
        (json!({"kind": "exact"}), Strategy::Exact),
        (
            json!({"kind": "retrieved", "core": {"rows": 64, "seed": 3}}),
            Strategy::subsample(64, 3, CoreKind::Random),
        ),
        (
            json!({"kind": "retrieved", "neighbors": 8}),
            Strategy::neighbors(8),
        ),
        (
            json!({"kind": "retrieved", "neighbors": 8, "among": 32, "recency": {"span": 60}}),
            Strategy::Retrieved {
                core: None,
                neighbors: 8,
                among: Some(32),
                recency: Some(Recency::Span(60)),
            },
        ),
        // As earlier clients spell a core alone and neighbors alone.
        (
            json!({"kind": "subsample", "rows": 64, "seed": 3}),
            Strategy::subsample(64, 3, CoreKind::Random),
        ),
        (
            json!({"kind": "neighbors", "rows": 8}),
            Strategy::neighbors(8),
        ),
        (
            json!({"kind": "retrieved", "core": {"rows": 16, "kind": "diverse"}, "neighbors": 8}),
            Strategy::Retrieved {
                core: Some(Core {
                    rows: 16,
                    seed: 0,
                    kind: CoreKind::Diverse,
                }),
                neighbors: 8,
                among: None,
                recency: None,
            },
        ),
    ];

    for (spec, strategy) in cases {
        let spec: StrategySpec = serde_json::from_value(spec).unwrap();
        assert_eq!(spec.strategy(), strategy);
    }
    for unknown in [
        json!({"kind": "pages", "budget": 8}),
        json!({"kind": "retrieved", "core": {"rows": 8, "kind": "nearest"}}),
    ] {
        assert!(serde_json::from_value::<StrategySpec>(unknown).is_err());
    }
}

#[test]
fn shared_frames_carry_their_buffers_in_a_file() {
    let frame = Frame {
        header: json!({"kind": "step"}),
        buffers: vec![vec![1, 2, 3], Vec::new(), vec![4, 5]],
    };
    let path = common::scratch("frame").join("shared.frame");

    let shared = frame.shared(&path).unwrap();

    assert!(shared.buffers.is_empty());
    assert_eq!(shared.header["kind"], "step");
    assert_eq!(shared.header["shared"]["sizes"], json!([3, 0, 2]));
    assert_eq!(shared.header["shared"]["path"], path.to_str().unwrap());
    assert_eq!(std::fs::read(&path).unwrap(), [1, 2, 3, 4, 5]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn worker_replies_missing_or_renaming_a_field_are_refused() {
    let profile = json!({
        "kind": "profile",
        "bytes_per_row": 64,
        "representation_bytes": 16,
        "regression": false,
        "context_rows": 60000,
        "class_slots": 10,
        "outputs": 10,
        "state_bytes": 32,
        "categorical_state_bytes": 64,
        "column_summary": true,
        "unlabeled_context": false,
    });
    assert!(matches!(
        WorkerReply::of(&Frame::header(profile.clone())),
        Ok(WorkerReply::Profile(ModelProfile {
            class_slots: 10,
            ..
        }))
    ));
    // A field the worker stopped sending used to read as false.
    let mut missing = profile.clone();
    missing.as_object_mut().unwrap().remove("regression");
    assert!(WorkerReply::of(&Frame::header(missing)).is_err());
    let mut renamed = profile;
    let fields = renamed.as_object_mut().unwrap();
    let bytes = fields.remove("representation_bytes").unwrap();
    fields.insert("bytes_per_representation".to_string(), bytes);
    assert!(WorkerReply::of(&Frame::header(renamed)).is_err());
}

#[test]
fn predictions_whose_shape_overflows_are_refused() {
    let header = json!({
        "kind": "predictions",
        "results": [{"request": 1, "rows": usize::MAX, "columns": 2, "buffer": 0}],
        "failures": [],
    });
    let mut frame = Frame::header(header);
    frame.buffers.push(vec![0; 8]);

    let refused = outcome_from(&frame).unwrap_err();

    assert!(refused.0.contains("wrong size"));
}

#[test]
fn forecasts_carry_their_horizon_and_the_shape_is_checked() {
    let header = json!({
        "kind": "predictions",
        "results": [{"request": 1, "rows": 1, "horizon": 2, "columns": 3, "buffer": 0}],
        "failures": [],
    });
    let mut frame = Frame::header(header.clone());
    frame.buffers.push(vec![0; 24]);

    let outcome = outcome_from(&frame).unwrap();

    assert_eq!(outcome.predictions[0].1.horizon, Some(2));
    assert_eq!(outcome.predictions[0].1.values.len(), 6);
    let mut short = Frame::header(header);
    short.buffers.push(vec![0; 12]);
    assert!(outcome_from(&short).unwrap_err().0.contains("wrong size"));
}

#[test]
fn a_forecasters_profile_describes_it_and_a_tabular_one_reads_as_before() {
    let forecaster = json!({
        "kind": "profile",
        "bytes_per_row": 64,
        "representation_bytes": 0,
        "regression": true,
        "class_slots": 0,
        "outputs": 3,
        "context_rows": null,
        "state_bytes": 0,
        "column_summary": false,
        "unlabeled_context": false,
        "series": true,
        "patch_bytes": 8,
        "patch": 32,
        "output_patch": 32,
        "context_limit": 8192,
        "pass_limit": null,
        "quantiles": [0.1, 0.5, 0.9],
        "reusable": true,
        "max_rows": 32,
    });
    let WorkerReply::Profile(profile) = WorkerReply::of(&Frame::header(forecaster)).unwrap() else {
        panic!("a profile");
    };
    assert!(profile.series && profile.reusable);
    assert_eq!((profile.patch, profile.patch_bytes), (32, 8));
    assert_eq!(profile.quantiles, Some(vec![0.1, 0.5, 0.9]));
    assert_eq!(profile.max_rows, Some(32));
    let tabular = json!({
        "kind": "profile",
        "bytes_per_row": 64,
        "representation_bytes": 16,
        "regression": true,
        "class_slots": 0,
        "outputs": 9,
        "context_rows": null,
        "state_bytes": 0,
        "column_summary": false,
        "unlabeled_context": false,
    });
    let WorkerReply::Profile(profile) = WorkerReply::of(&Frame::header(tabular)).unwrap() else {
        panic!("a profile");
    };
    assert!(!profile.series && profile.quantiles.is_none());
    // A worker that leaves out the categorical state reads as keeping none.
    assert_eq!(profile.categorical_state_bytes, 0);
}

#[test]
fn a_steps_series_units_and_forecasts_travel_as_grids() {
    use std::num::{NonZeroU64, NonZeroUsize};

    use gf_core::cache::{ModelId, TierConfig};
    use gf_core::categories::CategoryEncoding;
    use gf_core::forecasting::{GroupStrategy, Neighbors, SeriesSpec};
    use gf_core::node::{
        DEFAULT_FIT_ROWS, EntrySizes, Node, PastCap, RefitPolicy, SummaryScope, TableSpec, Targets,
        Task,
    };
    use gf_server::protocol::step_frame;

    let tier = |slots| TierConfig {
        slot_bytes: NonZeroU64::new(4096).unwrap(),
        slots,
    };
    let mut node = Node::new(tier(64), tier(64)).unwrap();
    let table = node
        .create_table(TableSpec {
            model: ModelId(0),
            features: 0,
            unlabeled_context: false,
            task: Task::Values,
            page_size: NonZeroUsize::new(128).unwrap(),
            strategy: Strategy::Exact,
            sizes: EntrySizes {
                bytes_per_row: 4,
                bytes_per_representation: 0,
                state_bytes: 0,
                bucket_minimum: NonZeroUsize::new(1).unwrap(),
                state_rows: usize::MAX,
                fused_rows: usize::MAX,
                fit_bound: usize::MAX,
            },
            time_bucket: None,
            refit: RefitPolicy::default(),
            fit_rows: NonZeroUsize::new(DEFAULT_FIT_ROWS).unwrap(),
            context_rows: None,
            past_cap: PastCap::Warn,
            regions: None,
            scale_rows: None,
            retrieval: 0,
            categorical: Vec::new(),
            categories: CategoryEncoding::Codes,
            summary: SummaryScope::Table,
            series: Some(SeriesSpec {
                step: NonZeroU64::new(10).unwrap(),
                patch: NonZeroUsize::new(4).unwrap(),
                context: NonZeroUsize::new(16).unwrap(),
                max_horizon: NonZeroUsize::new(256).unwrap(),
                max_rows: None,
                known: Vec::new(),
                group: GroupStrategy::Alone,
                neighbors: Neighbors::Targets,
                reusable: true,
                extendable: false,
                patch_bytes: 0,
            }),
        })
        .unwrap();
    // Series 7 at 100, 110 and 120: its row keys pack the series and the time.
    let times: Vec<i64> = vec![100, 110, 120];
    let keys: Vec<u128> = times
        .iter()
        .map(|&time| (7u128 << 64) | time as u128)
        .collect();
    let timed: Vec<Option<i64>> = times.iter().copied().map(Some).collect();
    node.insert_at(
        table,
        &keys,
        &[],
        Targets::Values(vec![1.0, 2.0, 3.0]),
        Some(&timed),
    )
    .unwrap();
    let request = node
        .submit_forecast(
            table,
            &[7],
            Some(&[130]),
            NonZeroUsize::new(2).unwrap(),
            None,
        )
        .unwrap();
    let step = node.schedule().unwrap().expect("a step");

    let frame = step_frame(&step);

    let unit = &frame.header["series_units"][0];
    assert_eq!(unit["members"], json!(["7"]));
    assert_eq!(
        (unit["rows"].as_u64(), unit["steps"].as_u64()),
        (Some(1), Some(4))
    );
    assert_eq!(unit["cutoff"], 130);
    let values = &frame.buffers[unit["values"].as_u64().unwrap() as usize];
    assert_eq!(values.len(), 4 * 4);
    let observed = &frame.buffers[unit["observed"].as_u64().unwrap() as usize];
    assert_eq!(observed, &[0, 1, 1, 1]);
    let roles = &frame.buffers[unit["roles"].as_u64().unwrap() as usize];
    assert_eq!(roles.len(), 4);
    let forecast = &frame.header["forecasts"][0];
    assert_eq!(forecast["request"], request.0);
    assert_eq!(forecast["unit"], 0);
    assert_eq!(forecast["target"], 0);
    assert_eq!(forecast["horizon"], 2);
    // Without known columns, the future has no rows to carry.
    assert_eq!(forecast["rows"], 0);
    let future = &frame.buffers[forecast["values"].as_u64().unwrap() as usize];
    assert!(future.is_empty());
}
