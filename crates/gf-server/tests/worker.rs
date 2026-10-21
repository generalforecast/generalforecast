//! The worker pipeline: frames are sent without waiting, and each reply reaches its request.

use std::time::Duration;

use gf_server::protocol::{Frame, WorkerReply};
use gf_server::worker::Pipeline;
use serde_json::json;
use tokio::net::UnixStream;

/// A fake worker that answers frames in order; the first answer is slow.
fn echo(mut stream: UnixStream) {
    tokio::spawn(async move {
        let mut index = 0;
        while let Ok(frame) = Frame::read_from(&mut stream).await {
            if index == 0 {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            index += 1;
            let n = frame.header["n"].to_string();
            let reply = WorkerReply::Invalid { message: n }.frame();
            reply.write_to(&mut stream).await.unwrap();
        }
    });
}

fn frame(kind: &str, n: u64) -> Frame {
    Frame::header(json!({"kind": kind, "n": n}))
}

#[tokio::test]
async fn frames_are_pipelined_and_replies_reach_their_requests() {
    let (ours, theirs) = UnixStream::pair().unwrap();
    echo(theirs);
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir());

    let first = pipeline.send(&frame("step", 0)).await.unwrap();
    let second = pipeline.send(&frame("step", 1)).await.unwrap();
    let third = pipeline.request(&frame("apply", 2)).await.unwrap();

    let n = |frame: Frame| WorkerReply::of(&frame).unwrap();
    let echoed = |message: &str| WorkerReply::Invalid {
        message: message.to_string(),
    };
    assert_eq!(n(third), echoed("2"));
    assert_eq!(n(first.await.unwrap().unwrap()), echoed("0"));
    assert_eq!(n(second.await.unwrap().unwrap()), echoed("1"));
}

#[tokio::test]
async fn worker_errors_and_disconnects_reach_waiting_requests() {
    let (ours, mut theirs) = UnixStream::pair().unwrap();
    tokio::spawn(async move {
        Frame::read_from(&mut theirs).await.unwrap();
        Frame::read_from(&mut theirs).await.unwrap();
        let error = WorkerReply::Error {
            message: "out of memory".to_string(),
        }
        .frame();
        error.write_to(&mut theirs).await.unwrap();
    });
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir());

    let first = pipeline.send(&frame("step", 0)).await.unwrap();
    let second = pipeline.send(&frame("step", 1)).await.unwrap();

    assert_eq!(first.await.unwrap(), Err("out of memory".to_string()));
    assert!(
        second
            .await
            .unwrap()
            .unwrap_err()
            .contains("worker connection")
    );
}

#[tokio::test]
async fn a_stopped_pipeline_refuses_frames_at_once() {
    let (ours, mut theirs) = UnixStream::pair().unwrap();
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir());
    let stopped = pipeline.stopped();

    // A frame nothing asked for means the worker and the pipeline disagree.
    frame("echo", 0).write_to(&mut theirs).await.unwrap();
    stopped.await;

    let refused = pipeline.send(&frame("step", 1)).await.unwrap_err();
    assert!(refused.contains("nothing asked for"));
}

#[tokio::test]
async fn a_worker_that_stops_reading_never_blocks_sends() {
    let (ours, _theirs) = UnixStream::pair().unwrap();
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir());
    // Far past what the socket buffers, and under the size that goes through files.
    let mut big = frame("apply", 0);
    big.buffers.push(vec![0; 512 << 10]);

    let sent = tokio::time::timeout(Duration::from_secs(5), async {
        for _ in 0..64 {
            pipeline.send(&big).await.unwrap();
        }
    })
    .await;

    assert!(
        sent.is_ok(),
        "sending waited on a worker that reads nothing"
    );
}

#[test]
fn cpu_workers_each_take_a_run_of_cores() {
    use gf_server::worker::cpu_groups;
    let cpus: Vec<usize> = (0..8).collect();
    assert_eq!(
        cpu_groups(&cpus, 2),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    // Cores left over go to the last worker; a machine short of a run still gets one.
    assert_eq!(
        cpu_groups(&cpus[..7], 3),
        vec![vec![0, 1, 2], vec![3, 4, 5, 6]]
    );
    assert_eq!(cpu_groups(&cpus[..1], 2), vec![vec![0]]);
    assert_eq!(cpu_groups(&[4, 6], 0), vec![vec![4], vec![6]]);
}

#[test]
fn cpu_workers_take_the_default_run_of_cores_when_none_is_given() {
    use gf_server::config::WorkerConfig;
    use gf_server::worker::{CPU_WORKER_CORES, allowed_cpus, cpu_groups, visible_gpus};
    if !visible_gpus().is_empty() {
        return;
    }
    let cpus = allowed_cpus().expect("Linux lists the allowed cores");
    let worker = |cpu_threads| WorkerConfig {
        command: Vec::new(),
        per_gpu: true,
        cpu_threads,
        models: std::collections::BTreeMap::new(),
    };
    let cores = |cpu_threads| {
        gf_server::runtime::seats(&[worker(cpu_threads)])
            .into_iter()
            .map(|seat| seat.cpus.expect("a CPU seat is pinned"))
            .collect::<Vec<_>>()
    };
    assert_eq!(cores(None), cpu_groups(&cpus, CPU_WORKER_CORES));
    assert_eq!(cores(Some(1)), cpu_groups(&cpus, 1));
}

#[test]
fn the_allowed_cores_are_read_from_the_process_affinity() {
    let cpus = gf_server::worker::allowed_cpus().expect("Linux lists the allowed cores");
    assert_eq!(
        cpus.len(),
        std::thread::available_parallelism().unwrap().get()
    );
}
