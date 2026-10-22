// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics, rendered in the text exposition format.

use std::time::Duration;

use gf_server::metrics::Metrics;

#[test]
fn counters_and_gauges_render_with_help_and_type() {
    let metrics = Metrics::default();
    metrics.rows_written.add(5);
    metrics.rows_written.add(2);
    metrics.tables.set(3);
    metrics.cache_bytes[0].set(1024);

    let text = metrics.render();

    assert!(text.contains("# TYPE gf_rows_written_total counter\ngf_rows_written_total 7\n"));
    assert!(text.contains("# TYPE gf_tables gauge\ngf_tables 3\n"));
    assert!(text.contains("# HELP gf_rows_written_total "));
    assert!(
        text.contains("gf_cache_bytes{tier=\"device\"} 1024\ngf_cache_bytes{tier=\"host\"} 0\n")
    );
}

#[test]
fn step_times_render_as_a_cumulative_histogram() {
    let metrics = Metrics::default();
    metrics.step_seconds.observe(Duration::from_micros(8));
    metrics.step_seconds.observe(Duration::from_millis(3));
    metrics.step_seconds.observe(Duration::from_millis(40));
    metrics.step_seconds.observe(Duration::from_secs(20));

    let text = metrics.render();

    assert!(text.contains("gf_step_seconds_bucket{le=\"0.00001\"} 1\n"));
    assert!(text.contains("gf_step_seconds_bucket{le=\"0.005\"} 2\n"));
    assert!(text.contains("gf_step_seconds_bucket{le=\"0.05\"} 3\n"));
    assert!(text.contains("gf_step_seconds_bucket{le=\"10\"} 3\n"));
    assert!(text.contains("gf_step_seconds_bucket{le=\"+Inf\"} 4\n"));
    assert!(text.contains("gf_step_seconds_count 4\n"));
    assert!(text.contains("gf_step_seconds_sum 20.043008\n"));
}

#[test]
fn each_table_reports_its_own_series() {
    let metrics = Metrics::default();
    metrics.table("orders").rows_written.add(4);
    metrics.table("clinic-1").predictions.add(2);
    metrics
        .table("clinic-1")
        .predict_seconds
        .observe(Duration::from_millis(3));
    metrics.table("clinic-1").gpu_seconds.add(0.000_25);
    metrics.table("clinic-1").index_bytes.set(4096);
    metrics.table("gone").rows_written.add(1);
    metrics.forget("gone");

    let text = metrics.render();

    assert!(text.contains("gf_table_rows_written_total{table=\"orders\"} 4\n"));
    assert!(text.contains("gf_table_predictions_total{table=\"clinic-1\"} 2\n"));
    assert!(text.contains("gf_table_predict_seconds_bucket{table=\"clinic-1\",le=\"0.005\"} 1\n"));
    assert!(text.contains("gf_table_predict_seconds_count{table=\"clinic-1\"} 1\n"));
    assert!(text.contains("gf_table_gpu_seconds_total{table=\"clinic-1\"} 0.00025\n"));
    assert!(text.contains("# TYPE gf_table_index_bytes gauge\n"));
    assert!(text.contains("gf_table_index_bytes{table=\"clinic-1\"} 4096\n"));
    assert!(!text.contains("gone"));
}

#[test]
fn worker_stages_report_their_calls_and_seconds() {
    let metrics = Metrics::default();
    metrics.stage("decode").calls.add(3);
    metrics.stage("decode").seconds.add(0.5);

    let text = metrics.render();

    assert!(text.contains("gf_worker_stage_calls_total{stage=\"decode\"} 3\n"));
    assert!(text.contains("gf_worker_stage_seconds_total{stage=\"decode\"} 0.5\n"));
}

#[test]
fn step_parts_and_request_times_render_beside_the_stages() {
    let metrics = Metrics::default();
    metrics.part("pad").calls.add(2);
    metrics.part("pad").seconds.add(0.25);
    metrics.plan_part("retrieve").calls.add(3);
    metrics.plan_part("retrieve").seconds.add(0.5);
    metrics
        .request_decode_seconds
        .observe(Duration::from_millis(3));
    metrics
        .request_shard_seconds
        .observe(Duration::from_millis(40));

    let text = metrics.render();

    assert!(text.contains("gf_worker_part_calls_total{part=\"pad\"} 2\n"));
    assert!(text.contains("gf_worker_part_seconds_total{part=\"pad\"} 0.25\n"));
    assert!(text.contains("gf_plan_part_calls_total{part=\"retrieve\"} 3\n"));
    assert!(text.contains("gf_plan_part_seconds_total{part=\"retrieve\"} 0.5\n"));
    assert!(text.contains("gf_request_decode_seconds_bucket{le=\"0.005\"} 1\n"));
    assert!(text.contains("gf_request_decode_seconds_count 1\n"));
    assert!(text.contains("gf_request_shard_seconds_sum 0.04\n"));
    assert!(text.contains("gf_request_encode_seconds_count 0\n"));
}

#[test]
fn worker_memory_renders_the_most_any_step_held() {
    let metrics = Metrics::default();
    metrics.memory_peak("host_peak", 3 << 20);
    metrics.memory_peak("host_peak", 1 << 20);

    let text = metrics.render();

    assert!(text.contains("# TYPE gf_worker_memory_peak_bytes gauge\n"));
    assert!(text.contains(&format!(
        "gf_worker_memory_peak_bytes{{kind=\"host_peak\"}} {}\n",
        3 << 20
    )));
}
