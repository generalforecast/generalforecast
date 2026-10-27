// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The worker pipeline: frames are sent without waiting, and each reply
//! reaches its request; and the workers a shard starts again.

mod common;

use std::time::Duration;

use gf_server::progress::Compiling;
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
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir(), Compiling::default());

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
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir(), Compiling::default());

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
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir(), Compiling::default());
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
    let mut pipeline = Pipeline::new(ours, std::env::temp_dir(), Compiling::default());
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

// The affinity a CPU worker is pinned by is read from /proc, which Linux alone has.
#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
#[test]
fn the_allowed_cores_are_read_from_the_process_affinity() {
    let cpus = gf_server::worker::allowed_cpus().expect("Linux lists the allowed cores");
    assert_eq!(
        cpus.len(),
        std::thread::available_parallelism().unwrap().get()
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn without_a_process_affinity_a_cpu_worker_takes_one_unpinned_seat() {
    use gf_server::config::WorkerConfig;
    use gf_server::worker::{allowed_cpus, visible_gpus};
    if !visible_gpus().is_empty() {
        return;
    }
    assert_eq!(allowed_cpus(), None);
    let worker = WorkerConfig {
        command: Vec::new(),
        per_gpu: true,
        cpu_threads: None,
        models: std::collections::BTreeMap::new(),
    };
    let seats = gf_server::runtime::seats(&[worker]);
    assert_eq!(seats.len(), 1);
    assert_eq!(
        (seats[0].gpu.as_ref(), seats[0].cpus.as_ref()),
        (None, None)
    );
}

/// The argument after `flag`, as the server passes a worker its options.
fn option(arguments: &[String], flag: &str) -> Option<String> {
    let at = arguments.iter().position(|argument| argument == flag)?;
    arguments.get(at + 1).cloned()
}

/// Whether process `pid` is running, or a zombie no one has waited for.
fn running(pid: i32) -> bool {
    // SAFETY: signal 0 checks the process exists and touches no memory.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Not a test of its own: `a_worker_that_cannot_take_its_pools_is_killed_before_the_next_start`
/// starts this test binary as its worker, filtered down to this function,
/// which then answers the server's socket as a worker that loads and
/// reports ready but refuses its pools. In the file the server names as
/// its disk tier it notes its process id, and each worker started before it
/// that is still running as it starts, and it stays up, holding what a real
/// worker would hold, until it is killed or a minute has passed.
#[test]
fn stand_in_worker_that_refuses_its_pools() {
    use std::io::{Read, Write};
    let arguments: Vec<String> = std::env::args().collect();
    let (Some(socket), Some(noted)) = (
        option(&arguments, "--socket"),
        option(&arguments, "--disk-path"),
    ) else {
        return;
    };
    let earlier = std::fs::read_to_string(&noted).unwrap_or_default();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(noted)
        .unwrap();
    for pid in earlier.lines().filter_map(|line| line.parse::<i32>().ok()) {
        if running(pid) {
            writeln!(file, "held {pid}").unwrap();
        }
    }
    writeln!(file, "{}", std::process::id()).unwrap();
    let mut stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
    let send = |stream: &mut std::os::unix::net::UnixStream, reply: WorkerReply| {
        let bytes = reply.frame().to_bytes();
        stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
        stream.write_all(&bytes)
    };
    let ready = WorkerReply::Ready {
        protocol: gf_server::protocol::PROTOCOL_VERSION,
        models: vec!["tiny".to_string()],
        device_memory: 1 << 30,
        device_bytes: 1 << 29,
        host_bytes: 1 << 28,
        row_memory_bytes: 1 << 28,
    };
    send(&mut stream, ready).unwrap();
    let mut length = [0u8; 8];
    while stream.read_exact(&mut length).is_ok() {
        let mut frame = vec![0u8; usize::try_from(u64::from_le_bytes(length)).unwrap()];
        if stream.read_exact(&mut frame).is_err() {
            break;
        }
        let refused = WorkerReply::Error {
            message: "out of device memory".to_string(),
        };
        if send(&mut stream, refused).is_err() {
            break;
        }
    }
    std::thread::sleep(Duration::from_secs(60));
}

#[tokio::test]
async fn a_worker_that_cannot_take_its_pools_is_killed_before_the_next_start() {
    use gf_server::config::WorkerConfig;
    use gf_server::worker::{Launch, WorkerConnection};

    let noted = common::scratch("worker").join("pids");
    let _ = std::fs::remove_file(&noted);
    let (ours, theirs) = UnixStream::pair().unwrap();
    // A worker that answers until it is told to stop, which starts the restarts.
    let stop = std::sync::Arc::new(tokio::sync::Notify::new());
    let stopping = stop.clone();
    tokio::spawn(async move {
        let mut theirs = theirs;
        loop {
            tokio::select! {
                frame = Frame::read_from(&mut theirs) => {
                    if frame.is_err() {
                        break;
                    }
                    WorkerReply::Ok.frame().write_to(&mut theirs).await.unwrap();
                }
                () = stopping.notified() => break,
            }
        }
    });
    let mut connection = WorkerConnection::fake(ours, vec!["tiny".to_string()]);
    let exe = std::env::current_exe().unwrap();
    connection.launch = Launch {
        config: WorkerConfig {
            command: vec![
                exe.to_string_lossy().into_owned(),
                "--exact".to_string(),
                "stand_in_worker_that_refuses_its_pools".to_string(),
                "--nocapture".to_string(),
                "--".to_string(),
            ],
            per_gpu: false,
            cpu_threads: None,
            models: std::collections::BTreeMap::new(),
        },
        disk: Some(noted.clone()),
        ..connection.launch.clone()
    };
    let shard = common::shard_over(connection, common::fresh_directory()).await;
    stop.notify_one();

    let noted_lines = || std::fs::read_to_string(&noted).unwrap_or_default();
    let pids =
        |text: &str| -> Vec<i32> { text.lines().filter_map(|line| line.parse().ok()).collect() };
    let started = tokio::time::timeout(Duration::from_secs(60), async {
        while pids(&noted_lines()).len() < 2 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    let text = noted_lines();
    // Stopped as a server stops it, the shard's last worker goes, and its directory.
    let (reply, stopped) = tokio::sync::oneshot::channel();
    shard
        .send(gf_server::shard::Command::Shutdown { reply })
        .await
        .unwrap();
    let _ = stopped.await;
    drop(shard);
    for pid in pids(&text) {
        // SAFETY: kill(2) takes a process id and a signal, and touches no memory.
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    assert!(started.is_ok(), "the shard started its worker again twice");
    assert!(
        !text.contains("held"),
        "a worker that refused its pools still ran when the next one started: {text}"
    );
}

#[test]
fn stale_worker_directories_go_once_their_server_has_exited() {
    use gf_server::worker::Private;
    let parent = common::fresh_directory();
    std::fs::create_dir_all(&parent).unwrap();
    let mut exited = std::process::Command::new("true").spawn().unwrap();
    let gone = i32::try_from(exited.id()).unwrap();
    exited.wait().unwrap();
    let mut living = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let alive = living.id();
    let own = std::process::id();
    for name in [
        format!("gf-{gone}-0123"),
        format!("gf-{alive}-4567"),
        format!("gf-{own}-89ab"),
        "gf-notapid-cdef".to_string(),
        "other-1-2".to_string(),
    ] {
        std::fs::create_dir_all(parent.join(name).join("flight")).unwrap();
    }
    Private::remove_stale(&parent);
    let _ = living.kill();
    let _ = living.wait();
    let mut left: Vec<String> = std::fs::read_dir(&parent)
        .unwrap()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .collect();
    left.sort();
    let mut kept = vec![
        format!("gf-{alive}-4567"),
        format!("gf-{own}-89ab"),
        "gf-notapid-cdef".to_string(),
        "other-1-2".to_string(),
    ];
    kept.sort();
    assert_eq!(left, kept, "{gone} exited");
}

#[tokio::test]
async fn a_stopped_worker_leaves_no_directory_behind() {
    use gf_server::config::WorkerConfig;
    use gf_server::worker::{Launch, WorkerConnection};

    let noted = common::scratch("worker").join("pids");
    let (ours, _theirs) = UnixStream::pair().unwrap();
    let template = WorkerConnection::fake(ours, vec!["tiny".to_string()]);
    let exe = std::env::current_exe().unwrap();
    let launch = Launch {
        config: WorkerConfig {
            command: vec![
                exe.to_string_lossy().into_owned(),
                "--exact".to_string(),
                "stand_in_worker_that_refuses_its_pools".to_string(),
                "--nocapture".to_string(),
                "--".to_string(),
            ],
            per_gpu: false,
            cpu_threads: None,
            models: std::collections::BTreeMap::new(),
        },
        disk: Some(noted.clone()),
        ..template.launch.clone()
    };
    for kill in [false, true] {
        let mut worker = WorkerConnection::spawn(launch.clone()).await.unwrap();
        let directory = worker.directory().unwrap().to_path_buf();
        assert!(directory.join("worker-0.sock").exists());
        if kill {
            worker.kill().await;
        } else {
            worker.terminate().await;
        }
        assert!(
            !directory.exists(),
            "{} is left behind",
            directory.display()
        );
        assert!(worker.directory().is_none());
    }
    let _ = std::fs::remove_file(noted);
}
