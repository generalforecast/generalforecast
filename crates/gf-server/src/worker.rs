// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A worker process, spawned by the server and reached over a Unix socket.
//!
//! Frames are pipelined: the server sends a frame without waiting for the
//! replies to earlier ones, and the worker answers in order. So a shard can plan
//! and send the next step while the worker computes the current one.
//!
//! Large frames, such as a fit's rows, travel through shared memory: their
//! buffers go to a file of their own, which the worker maps and reads in place.
//! Those files and the worker's socket live in a directory only this user can
//! open, and the server talks only to the process it started.
//!
//! Frames are written on a task of their own, so a worker that stops reading
//! never blocks its shard: the frames wait in memory, and the shard's step
//! deadline finds the worker hung.
//!
//! A pipeline stops for good once the worker exits, its connection fails, or
//! it sends a frame nothing asked for: waiting requests fail, later ones fail at
//! once, and `stopped` resolves so the shard can start the worker again.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::process::Command;
use tokio::sync::{Notify, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::config::WorkerConfig;
use crate::progress::Compiling;
use crate::protocol::{Frame, PROTOCOL_VERSION, WorkerReply, WorkerRequest};

/// How long a worker may take to load its models and connect.
const STARTUP: Duration = Duration::from_secs(600);
/// How long a worker told to exit gets before it is killed.
const EXIT: Duration = Duration::from_secs(5);

/// A worker's reply to one frame, or why there is none.
pub type Reply = Result<Frame, String>;

/// Frames whose buffers total at least this many bytes travel through shared memory.
const SHARED_BYTES: usize = 1 << 20;

/// A new directory that only this user can open, removed with everything in it when dropped.
#[derive(Debug)]
pub struct Private(PathBuf);

impl Private {
    /// Makes one under `parent`, named for this process so leftovers can be found.
    ///
    /// # Errors
    ///
    /// If the directory cannot be made.
    pub fn new(parent: &std::path::Path) -> std::io::Result<Self> {
        use std::os::unix::fs::DirBuilderExt;
        let suffix = crate::setup::random_hex(8).map_err(std::io::Error::other)?;
        let path = parent.join(format!("gf-{}-{suffix}", std::process::id()));
        std::fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }

    /// Removes directories left under `parent` by servers that were killed
    /// before they could, those whose process is no longer running.
    pub fn remove_stale(parent: &std::path::Path) {
        let Ok(entries) = std::fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(rest) = name.to_str().and_then(|name| name.strip_prefix("gf-")) else {
                continue;
            };
            let Some(pid) = rest
                .split('-')
                .next()
                .and_then(|pid| pid.parse::<libc::pid_t>().ok())
                .filter(|&pid| pid > 0)
            else {
                continue;
            };
            let stale = entry.file_type().is_ok_and(|kind| kind.is_dir())
                && u32::try_from(pid).is_ok_and(|pid| pid != std::process::id())
                && !process_running(pid);
            if stale {
                tracing::info!(path = %entry.path().display(), "removing a stale worker directory");
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Private {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Where shared frames go: memory-backed `/dev/shm` where there is one.
#[must_use]
pub fn shared_parent() -> PathBuf {
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        shm
    } else {
        std::env::temp_dir()
    }
}

/// Who waits on a frame's reply: its sender, or each of a `Controls` frame's
/// requests in order.
enum Answer {
    One(oneshot::Sender<Reply>),
    Controls(Vec<oneshot::Sender<Reply>>),
}

impl Answer {
    /// Answers every request the frame carried with the same reply.
    fn send(self, reply: &Reply) {
        match self {
            Self::One(sender) => {
                let _ = sender.send(reply.clone());
            }
            Self::Controls(senders) => {
                for sender in senders {
                    let _ = sender.send(reply.clone());
                }
            }
        }
    }
}

/// A request waiting for its reply, and the shared frame file to remove then.
type Waiting = (Answer, Option<PathBuf>);

/// Requests waiting for replies, in order, and why the pipeline stopped, if it has.
#[derive(Default)]
struct Queue {
    waiting: VecDeque<Waiting>,
    stopped: Option<String>,
}

/// Stops a pipeline: from its reader, its writer, or a task watching the worker exit.
#[derive(Clone)]
struct Stopper {
    queue: Arc<Mutex<Queue>>,
    signal: Arc<watch::Sender<bool>>,
}

impl Stopper {
    /// Fails every waiting request and refuses later ones, keeping the first reason.
    fn stop(&self, reason: &str) {
        let waiting = {
            let mut queue = self.queue.lock().unwrap_or_else(PoisonError::into_inner);
            queue.stopped.get_or_insert_with(|| reason.to_string());
            std::mem::take(&mut queue.waiting)
        };
        for (answer, shared) in waiting {
            if let Some(path) = shared {
                let _ = std::fs::remove_file(path);
            }
            answer.send(&Err(reason.to_string()));
        }
        self.signal.send_replace(true);
    }
}

/// Frames to a worker and its replies, matched in order.
pub struct Pipeline {
    /// Each frame's bytes, for the writing task, in the order sent.
    frames_out: mpsc::UnboundedSender<Vec<u8>>,
    stopper: Stopper,
    /// When the worker last reported progress on a step.
    progressed: Arc<Mutex<Option<Instant>>>,
    /// Keys of programs compiled beside steps, not yet taken, and a wake
    /// for each report of them.
    compiled: Arc<(Mutex<Vec<String>>, Notify)>,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
    /// Where shared frames go, made at the first one.
    shared: Option<Private>,
    /// Where shared frames go instead once `/dev/shm` is full, such as a
    /// container's 64 MiB default; a disk directory, so slower but unbounded.
    fallback: PathBuf,
    frames: u64,
    /// Controls not sent yet, which go in one frame before the next frame
    /// or at `flush`.
    outbox: Vec<(WorkerRequest, oneshot::Sender<Reply>)>,
}

impl Pipeline {
    /// A pipeline over a connected stream, reading replies on its own task.
    /// Shared frames go to `fallback` once shared memory runs out, and the
    /// worker's warm-up progress goes to `warming`.
    #[must_use]
    pub fn new(stream: UnixStream, fallback: PathBuf, warming: Compiling) -> Self {
        let (mut reader, mut writer) = stream.into_split();
        let stopper = Stopper {
            queue: Arc::default(),
            signal: Arc::new(watch::channel(false).0),
        };
        let (frames_out, mut frames) = mpsc::unbounded_channel::<Vec<u8>>();
        let writing = stopper.clone();
        let writer = tokio::spawn(async move {
            while let Some(bytes) = frames.recv().await {
                let written = async {
                    writer
                        .write_all(&(bytes.len() as u64).to_le_bytes())
                        .await?;
                    writer.write_all(&bytes).await?;
                    writer.flush().await
                };
                if let Err(error) = written.await {
                    writing.stop(&format!("worker connection: {error}"));
                    break;
                }
            }
        });
        let reading = stopper.clone();
        let progressed = Arc::new(Mutex::new(None));
        let progressing = Arc::clone(&progressed);
        let compiled: Arc<(Mutex<Vec<String>>, Notify)> = Arc::default();
        let compiling = Arc::clone(&compiled);
        let reader = tokio::spawn(async move {
            // The worker's last warm-up report, so each counts once.
            let mut reported = (0, 0);
            loop {
                let frame = match Frame::read_from(&mut reader).await {
                    Ok(frame) => frame,
                    Err(error) => {
                        reading.stop(&format!("worker connection: {}", error.0));
                        break;
                    }
                };
                if let Ok(WorkerReply::Progress { done, total }) = WorkerReply::of(&frame) {
                    *progressing.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(Instant::now());
                    if let (Some(done), Some(total)) = (done, total) {
                        warming.expect(total.saturating_sub(reported.1));
                        warming.advance(done.saturating_sub(reported.0));
                        reported = (done.max(reported.0), total.max(reported.1));
                    }
                    continue;
                }
                if let Ok(WorkerReply::Compiled { keys }) = WorkerReply::of(&frame) {
                    let (keys_out, wake) = &*compiling;
                    keys_out
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .extend(keys);
                    wake.notify_one();
                    continue;
                }
                let next = reading
                    .queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .waiting
                    .pop_front();
                let Some((next, shared)) = next else {
                    reading.stop("the worker sent a frame nothing asked for");
                    break;
                };
                // The worker removes a shared frame once it maps it; this covers a worker that failed first.
                if let Some(path) = shared {
                    let _ = std::fs::remove_file(path);
                }
                match (WorkerReply::of(&frame), next) {
                    (Ok(WorkerReply::Controlled { failures }), Answer::Controls(senders)) => {
                        let mut failed: HashMap<usize, String> = failures
                            .into_iter()
                            .map(|failure| (failure.index, failure.message))
                            .collect();
                        let done = Frame::header(WorkerReply::Ok.frame().header);
                        for (index, sender) in senders.into_iter().enumerate() {
                            let _ = sender
                                .send(failed.remove(&index).map_or_else(|| Ok(done.clone()), Err));
                        }
                    }
                    (reply, next) => next.send(&match reply {
                        Ok(WorkerReply::Error { message }) => Err(message),
                        Ok(_) => Ok(frame),
                        Err(error) => Err(error.0),
                    }),
                }
            }
        });
        Self {
            frames_out,
            stopper,
            progressed,
            compiled,
            reader,
            writer,
            shared: None,
            fallback,
            frames: 0,
            outbox: Vec::new(),
        }
    }

    /// Queues a control, a table added or removed or tiers' copies and
    /// drops, for the next frame of controls; its reply arrives on the
    /// returned receiver once the worker carried it out.
    pub fn control(&mut self, request: WorkerRequest) -> oneshot::Receiver<Reply> {
        let (sender, receiver) = oneshot::channel();
        self.outbox.push((request, sender));
        receiver
    }

    /// Sends the controls queued since the last frame, in one frame.
    ///
    /// # Errors
    ///
    /// If the pipeline has stopped, which fails the controls' replies.
    pub fn flush(&mut self) -> Result<(), String> {
        if self.outbox.is_empty() {
            return Ok(());
        }
        let (requests, senders): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.outbox).into_iter().unzip();
        let frame = WorkerRequest::Controls { requests }.frame();
        self.write(&frame, Answer::Controls(senders), None)
    }

    /// Queues a frame's reply and hands its bytes to the writing task.
    fn write(&self, frame: &Frame, answer: Answer, shared: Option<PathBuf>) -> Result<(), String> {
        let bytes = frame.to_bytes();
        // Queued before writing, so the reply always finds its request.
        let mut queue = self
            .stopper
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(reason) = &queue.stopped {
            if let Some(path) = shared {
                let _ = std::fs::remove_file(path);
            }
            let reason = reason.clone();
            drop(queue);
            answer.send(&Err(reason.clone()));
            return Err(reason);
        }
        queue.waiting.push_back((answer, shared));
        // The writing task ends only once the pipeline has stopped, which
        // fails this frame's reply.
        let _ = self.frames_out.send(bytes);
        Ok(())
    }

    /// A fresh path for a shared frame, in a directory only this user can open.
    fn shared_path(&mut self) -> std::io::Result<PathBuf> {
        if self.shared.is_none() {
            self.shared = Some(Private::new(&shared_parent())?);
        }
        self.frames += 1;
        let directory = self.shared.as_ref().expect("made above").path();
        Ok(directory.join(format!("{}.frame", self.frames)))
    }

    /// Writes a frame's buffers to a shared file, falling back from shared
    /// memory to the fallback directory for good once it runs out of room.
    fn share(&mut self, frame: &Frame) -> Result<(Frame, PathBuf), String> {
        let path = self
            .shared_path()
            .map_err(|error| format!("shared memory: {error}"))?;
        match frame.shared(&path) {
            Ok(shared) => Ok((shared, path)),
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                let in_memory = self
                    .shared
                    .as_ref()
                    .is_some_and(|shared| shared.path().starts_with(shared_parent()))
                    && shared_parent() != self.fallback;
                if !in_memory {
                    return Err(format!("shared memory: {}", error.0));
                }
                tracing::warn!(
                    fallback = %self.fallback.display(),
                    "shared memory is full ({}); large frames go through disk from now on. \
                     In a container, raise --shm-size",
                    error.0
                );
                std::fs::create_dir_all(&self.fallback)
                    .map_err(|error| format!("shared frames: {error}"))?;
                self.shared =
                    Some(Private::new(&self.fallback).map_err(|error| error.to_string())?);
                let path = self
                    .shared_path()
                    .map_err(|error| format!("shared frames: {error}"))?;
                let shared = frame
                    .shared(&path)
                    .map_err(|error| format!("shared frames: {}", error.0))?;
                Ok((shared, path))
            }
        }
    }

    /// When the worker last reported progress on a step, if it has.
    #[must_use]
    /// Waits for the worker to report programs compiled beside its steps.
    pub async fn compiled(&self) {
        self.compiled.1.notified().await;
    }

    /// The keys of programs the worker reported compiled since last asked.
    pub fn take_compiled(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .compiled
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    pub fn progressed(&self) -> Option<Instant> {
        *self
            .progressed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Resolves once the pipeline has stopped, and the worker answers nothing more.
    pub fn stopped(&self) -> impl Future<Output = ()> + Send + 'static {
        let mut stopped = self.stopper.signal.subscribe();
        async move {
            let _ = stopped.wait_for(|stopped| *stopped).await;
        }
    }

    /// Sends a frame without waiting for it to be written; its reply arrives
    /// on the returned receiver.
    ///
    /// # Errors
    ///
    /// If the pipeline has stopped, or a shared frame cannot be written.
    pub async fn send(&mut self, frame: &Frame) -> Result<oneshot::Receiver<Reply>, String> {
        // Controls queued before a frame are carried out before it.
        self.flush()?;
        let (sender, receiver) = oneshot::channel();
        let size: usize = frame.buffers.iter().map(Vec::len).sum();
        let (frame, shared) = if size >= SHARED_BYTES {
            let (shared, path) = self.share(frame)?;
            (std::borrow::Cow::Owned(shared), Some(path))
        } else {
            (std::borrow::Cow::Borrowed(frame), None)
        };
        self.write(&frame, Answer::One(sender), shared)?;
        Ok(receiver)
    }

    /// Sends a frame and waits for its reply, after the replies to frames sent before it.
    ///
    /// # Errors
    ///
    /// If the connection fails, or the worker replies with an error.
    pub async fn request(&mut self, frame: &Frame) -> Reply {
        self.send(frame)
            .await?
            .await
            .map_err(|_| "worker connection: closed".to_string())?
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.reader.abort();
        self.writer.abort();
    }
}

/// How to start a worker, kept to start it again if it stops.
#[derive(Clone, Debug)]
pub struct Launch {
    /// The worker's place among the server's, which names its socket.
    pub index: usize,
    /// Workers the server starts, which share the machine's memory.
    pub workers: usize,
    pub config: WorkerConfig,
    pub bucket_minimum: usize,
    pub slot_bytes: u64,
    /// The one GPU the worker sees, if it has its own.
    pub gpu: Option<String>,
    /// The cores the worker is pinned to, if it has its own.
    pub cpus: Option<Vec<usize>>,
    /// A file for the worker's disk tier, if the node has one.
    pub disk: Option<PathBuf>,
    /// Environment variables the worker never sees, such as the log key.
    pub secrets: Vec<String>,
    /// Where large frames go once shared memory is full.
    pub shared_fallback: PathBuf,
    /// Times each part of a step.
    pub timers: bool,
    /// Where the worker's warm-up progress is counted.
    pub compiling: Compiling,
}

/// How a worker is made to exit.
enum Stop {
    /// SIGTERM, then SIGKILL if it is still running after `EXIT`.
    Terminate,
    Kill,
}

/// A running worker and the models it loaded.
pub struct WorkerConnection {
    pub pipeline: Pipeline,
    pub launch: Launch,
    /// Waits for the worker to exit, then stops the pipeline; aborting it kills the worker.
    exited: JoinHandle<()>,
    /// Asks that task to stop the worker and wait for it to exit.
    stop: Option<oneshot::Sender<Stop>>,
    /// Holds the socket; dropped after the pipeline.
    _directory: Private,
    /// Model names, in the order the worker numbers them.
    pub models: Vec<String>,
    /// Memory of the worker's device, as the worker reports it.
    pub device_memory: u64,
    /// The tiers the worker's memory budget gives, for those the
    /// configuration leaves unset.
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub row_memory_bytes: u64,
}

/// The cores this process may run on, as its affinity names them in
/// `/proc/self/status`; None where it cannot be read.
#[must_use]
pub fn allowed_cpus() -> Option<Vec<usize>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    let mut cpus = Vec::new();
    for range in list.trim().split(',').filter(|range| !range.is_empty()) {
        let (first, last) = range.split_once('-').unwrap_or((range, range));
        cpus.extend(first.trim().parse::<usize>().ok()?..=last.trim().parse::<usize>().ok()?);
    }
    (!cpus.is_empty()).then_some(cpus)
}

/// The cores a CPU worker has to itself when its configuration gives no
/// `cpu_threads`. On 8 cores, workers of four served new 256-row tabicl
/// contexts faster than workers of two at 1 and 8 callers, and kept repeated
/// contexts within 6 to 15% of before, where workers of two lost 23 to 38%.
pub const CPU_WORKER_CORES: usize = 4;

/// Each worker's cores: the allowed cores in runs of `threads`, one run per
/// worker, at least one worker; the last run takes any cores left over.
#[must_use]
pub fn cpu_groups(cpus: &[usize], threads: usize) -> Vec<Vec<usize>> {
    let threads = threads.max(1);
    let workers = (cpus.len() / threads).max(1);
    (0..workers)
        .map(|worker| {
            let end = if worker + 1 == workers {
                cpus.len()
            } else {
                (worker + 1) * threads
            };
            cpus[worker * threads..end].to_vec()
        })
        .collect()
}

/// The NVIDIA GPUs this process may use: `CUDA_VISIBLE_DEVICES` if set, else
/// every one the driver lists, else every one `nvidia-smi` lists, as in a
/// sandbox without `/proc/driver/nvidia`.
#[must_use]
pub fn visible_gpus() -> Vec<String> {
    if let Ok(devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
        return devices
            .split(',')
            .map(str::trim)
            .filter(|device| !device.is_empty() && *device != "-1")
            .map(str::to_string)
            .collect();
    }
    if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        return (0..entries.count())
            .map(|index| index.to_string())
            .collect();
    }
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

impl WorkerConnection {
    /// Starts a worker and waits for it to load its models.
    ///
    /// # Errors
    ///
    /// If the process cannot start or does not connect in time.
    pub async fn spawn(launch: Launch) -> Result<Self, String> {
        let Launch {
            index,
            workers,
            config,
            bucket_minimum,
            slot_bytes,
            gpu,
            cpus,
            disk,
            secrets,
            shared_fallback,
            timers,
            compiling,
        } = &launch;
        let (program, arguments) = config
            .command
            .split_first()
            .ok_or("a worker command is empty")?;
        // Unix socket paths are short (about 100 bytes), so they go in the temp directory.
        let directory = Private::new(&std::env::temp_dir()).map_err(|error| error.to_string())?;
        let socket = directory.path().join(format!("worker-{index}.sock"));
        let listener = UnixListener::bind(&socket)
            .map_err(|error| format!("{}: {error}", socket.display()))?;
        let mut command = Command::new(program);
        command
            .args(arguments)
            .arg("--socket")
            .arg(&socket)
            .arg("--bucket-minimum")
            .arg(bucket_minimum.to_string())
            .arg("--slot-bytes")
            .arg(slot_bytes.to_string())
            .arg("--workers")
            .arg(workers.to_string())
            .stdin(Stdio::null())
            .kill_on_drop(true);
        // A terminal's Ctrl-C reaches the worker along with the server, which
        // stops it once drained; ignored before exec, the signal cannot
        // interrupt Python's start-up either, before the worker's own handler.
        // SAFETY: `signal` only sets a disposition, which is async-signal-safe.
        unsafe {
            command.pre_exec(|| {
                libc::signal(libc::SIGINT, libc::SIG_IGN);
                Ok(())
            });
        }
        for (name, spec) in &config.models {
            command.arg("--model").arg(format!("{name}={spec}"));
        }
        for secret in secrets {
            command.env_remove(secret);
        }
        if let Some(gpu) = gpu {
            command.env("CUDA_VISIBLE_DEVICES", gpu);
        }
        if let Some(cpus) = cpus {
            let cpus: Vec<String> = cpus.iter().map(ToString::to_string).collect();
            command.arg("--cpus").arg(cpus.join(","));
        }
        if let Some(disk) = disk {
            command.arg("--disk-path").arg(disk);
        }
        if *timers {
            command.arg("--timers");
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("cannot start worker {program}: {error}"))?;
        let (mut stream, _) = tokio::time::timeout(STARTUP, listener.accept())
            .await
            .map_err(|_| "the worker did not connect in time".to_string())?
            .map_err(|error| error.to_string())?;
        let peer = stream
            .peer_cred()
            .map_err(|error| format!("the worker's credentials: {error}"))?;
        let started = child.id().and_then(|id| i32::try_from(id).ok());
        if peer.pid().is_some_and(|pid| Some(pid) != started) {
            return Err(format!(
                "process {:?} connected instead of the worker the server started",
                peer.pid()
            ));
        }
        let ready = tokio::time::timeout(STARTUP, Frame::read_from(&mut stream))
            .await
            .map_err(|_| "the worker did not report ready in time".to_string())?
            .map_err(|error| error.0)?;
        let WorkerReply::Ready {
            protocol,
            models,
            device_memory,
            device_bytes,
            host_bytes,
            row_memory_bytes,
        } = WorkerReply::of(&ready).map_err(|error| error.0)?
        else {
            return Err(format!("the worker sent {} instead of ready", ready.header));
        };
        if protocol != PROTOCOL_VERSION {
            return Err(format!(
                "the worker speaks protocol {protocol} and the server {PROTOCOL_VERSION}; \
                 run the worker from the generalforecast the server came with"
            ));
        }
        let pipeline = Pipeline::new(stream, shared_fallback.clone(), compiling.clone());
        let stopper = pipeline.stopper.clone();
        let (stop, stopping) = oneshot::channel::<Stop>();
        let exited = tokio::spawn(async move {
            let status = tokio::select! {
                status = child.wait() => status,
                stop = stopping => match stop {
                    Ok(Stop::Terminate) if terminate(&child) => {
                        match tokio::time::timeout(EXIT, child.wait()).await {
                            Ok(status) => status,
                            Err(_) => {
                                let _ = child.start_kill();
                                child.wait().await
                            }
                        }
                    }
                    _ => {
                        let _ = child.start_kill();
                        child.wait().await
                    }
                }
            };
            let status = status.map_or_else(|error| error.to_string(), |status| status.to_string());
            stopper.stop(&format!("the worker exited ({status})"));
        });
        Ok(Self {
            pipeline,
            launch,
            exited,
            stop: Some(stop),
            _directory: directory,
            models,
            device_memory,
            device_bytes,
            host_bytes,
            row_memory_bytes,
        })
    }

    /// A connection to a worker on the other end of `stream`, for tests that
    /// stand in for the worker process. Killing it closes the pipeline as an
    /// exit would; starting it again fails, since there is no command.
    ///
    /// # Errors
    ///
    /// If a directory for the connection cannot be made.
    pub fn fake(stream: UnixStream, models: Vec<String>) -> Result<Self, String> {
        let directory = Private::new(&std::env::temp_dir()).map_err(|error| error.to_string())?;
        let pipeline = Pipeline::new(stream, std::env::temp_dir(), Compiling::default());
        let stopper = pipeline.stopper.clone();
        let (stop, stopping) = oneshot::channel::<Stop>();
        let exited = tokio::spawn(async move {
            let _ = stopping.await;
            stopper.stop("the worker exited (killed)");
        });
        Ok(Self {
            pipeline,
            launch: Launch {
                index: 0,
                workers: 1,
                config: WorkerConfig {
                    command: Vec::new(),
                    per_gpu: false,
                    cpu_threads: None,
                    models: std::collections::BTreeMap::new(),
                },
                bucket_minimum: 8,
                slot_bytes: 1 << 20,
                gpu: None,
                cpus: None,
                disk: None,
                secrets: Vec::new(),
                shared_fallback: std::env::temp_dir(),
                timers: false,
                compiling: Compiling::default(),
            },
            exited,
            stop: Some(stop),
            _directory: directory,
            models,
            device_memory: 1 << 30,
            device_bytes: 1 << 29,
            host_bytes: 1 << 28,
            row_memory_bytes: 1 << 28,
        })
    }

    /// Kills the worker, if it is still running, and waits for it to exit,
    /// so its device memory and disk file are free for the next one.
    pub async fn kill(&mut self) {
        self.stop(Stop::Kill).await;
    }

    /// Tells the worker to exit, as the server does once it has drained,
    /// and waits for it; one still running after `EXIT` is killed.
    pub async fn terminate(&mut self) {
        self.stop(Stop::Terminate).await;
    }

    async fn stop(&mut self, how: Stop) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(how);
        }
        if !self.exited.is_finished() {
            let _ = (&mut self.exited).await;
        }
    }

    /// Sends a frame and waits at most `timeout` for its reply. A worker that
    /// takes longer is killed, which stops its pipeline, so the shard starts
    /// it again.
    ///
    /// # Errors
    ///
    /// If the connection fails, the worker replies with an error, or it
    /// takes too long.
    pub async fn request_within(&mut self, frame: &Frame, timeout: Duration) -> Reply {
        if let Ok(reply) = tokio::time::timeout(timeout, self.pipeline.request(frame)).await {
            return reply;
        }
        self.kill().await;
        Err(format!(
            "the worker took over {} seconds to answer a {:?} message",
            timeout.as_secs_f64(),
            frame.kind()
        ))
    }

    /// Sends a frame and waits for its reply while the worker reports
    /// progress at least every `timeout`, as it does for each program it
    /// compiles. A worker silent for longer is killed, as `request_within`
    /// kills it.
    ///
    /// # Errors
    ///
    /// If the connection fails, the worker replies with an error, or it
    /// goes `timeout` without progress.
    pub async fn request_progressing(&mut self, frame: &Frame, timeout: Duration) -> Reply {
        let progressed = Arc::clone(&self.pipeline.progressed);
        let sent = Instant::now();
        let last = move || {
            progressed
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .map_or(sent, |at| at.max(sent))
        };
        let outcome = {
            let request = self.pipeline.request(frame);
            tokio::pin!(request);
            loop {
                match tokio::time::timeout_at(last() + timeout, &mut request).await {
                    Ok(reply) => break Some(reply),
                    Err(_) if last() + timeout > Instant::now() => {}
                    Err(_) => break None,
                }
            }
        };
        if let Some(reply) = outcome {
            return reply;
        }
        self.kill().await;
        Err(format!(
            "the worker went over {} seconds without progress on a {:?} message",
            timeout.as_secs_f64(),
            frame.kind()
        ))
    }

    /// Sends a frame and waits for its reply; see `Pipeline::request`.
    ///
    /// # Errors
    ///
    /// If the connection fails, or the worker replies with an error.
    pub async fn request(&mut self, frame: &Frame) -> Reply {
        self.pipeline.request(frame).await
    }
}

/// Whether process `pid` exists, as a signal to it that sends nothing
/// finds: one another user runs counts, since it refuses the signal rather
/// than lacking a process.
fn process_running(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only checks the process exists, and touches no memory.
    let found = unsafe { libc::kill(pid, 0) } == 0;
    found || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Sends SIGTERM to `child`; false if it has no process id, as after it exited.
fn terminate(child: &tokio::process::Child) -> bool {
    let Some(pid) = child.id().and_then(|id| i32::try_from(id).ok()) else {
        return false;
    };
    // SAFETY: kill(2) takes a process id and a signal, and touches no memory.
    unsafe { libc::kill(pid, libc::SIGTERM) == 0 }
}

impl Drop for WorkerConnection {
    fn drop(&mut self) {
        self.exited.abort();
    }
}
