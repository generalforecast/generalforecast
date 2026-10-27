// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A server started with `--parent-fd` drains and exits once its parent's
//! end of the pipe closes, and refuses a descriptor it cannot watch.

use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use gf_server::protocol::{Frame, PROTOCOL_VERSION, WorkerReply};

/// The argument after `flag`, as the server passes a worker its options.
fn option(arguments: &[String], flag: &str) -> Option<String> {
    let at = arguments.iter().position(|argument| argument == flag)?;
    arguments.get(at + 1).cloned()
}

/// Not a test of its own: the server under test starts this test binary as
/// its worker, filtered down to this function, which answers the server's
/// socket as a worker that loads and answers every request, until it is
/// stopped or its socket closes.
#[test]
fn stand_in_worker_that_answers_every_request() {
    let arguments: Vec<String> = std::env::args().collect();
    let Some(socket) = option(&arguments, "--socket") else {
        return;
    };
    let mut stream = std::os::unix::net::UnixStream::connect(socket).unwrap();
    let send = |stream: &mut std::os::unix::net::UnixStream, reply: Frame| {
        let bytes = reply.to_bytes();
        stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
        stream.write_all(&bytes)
    };
    let ready = WorkerReply::Ready {
        protocol: PROTOCOL_VERSION,
        models: vec!["tiny".to_string()],
        device_memory: 1 << 30,
        device_bytes: 1 << 29,
        host_bytes: 1 << 28,
        row_memory_bytes: 1 << 28,
    };
    send(&mut stream, ready.frame()).unwrap();
    let mut length = [0u8; 8];
    while stream.read_exact(&mut length).is_ok() {
        let mut frame = vec![0u8; usize::try_from(u64::from_le_bytes(length)).unwrap()];
        if stream.read_exact(&mut frame).is_err()
            || send(&mut stream, WorkerReply::Ok.frame()).is_err()
        {
            break;
        }
    }
}

/// A directory of its own under the system's temporary directory, removed
/// when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("gf-parent-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A configuration whose one worker is this test binary standing in.
fn configured(scratch: &Path) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let config = scratch.join("server.toml");
    std::fs::write(
        &config,
        format!(
            "listen = \"127.0.0.1:0\"\ndata = {data:?}\ntokens = [\"{token}\"]\n\
             encryption_key_env = \"GF_LOG_KEY\"\ndrain_seconds = 30\n\n[[workers]]\n\
             command = [{exe:?}, \"--exact\", \"stand_in_worker_that_answers_every_request\", \
             \"--nocapture\", \"--\"]\n\n[workers.models]\ntiny = \"tiny\"\n",
            data = scratch.join("data"),
            token = "f".repeat(64),
            exe = exe.to_string_lossy(),
        ),
    )
    .unwrap();
    config
}

/// The server, started on `config` with its temporary files under `tmp`.
fn server(config: &Path, tmp: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gf-server"));
    command
        .args(["run", "--config"])
        .arg(config)
        .env("GF_LOG_KEY", "ab".repeat(32))
        .env("TMPDIR", tmp)
        .env_remove(gf_server::parent::ENV)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

/// Whether the server has started serving, by what it logged.
fn serving(log: &Path) -> bool {
    std::fs::read_to_string(log).is_ok_and(|text| text.contains("serving"))
}

#[test]
fn a_server_drains_and_exits_once_its_parent_is_gone() {
    let scratch = Scratch::new("drain");
    // The server's temporary files, under a name short enough that its
    // worker's socket path fits.
    let short = Scratch(std::env::temp_dir().join(format!("gfp{}", std::process::id())));
    std::fs::create_dir_all(&short.0).unwrap();
    let tmp = short.0.clone();
    let config = configured(&scratch.0);
    let (read, write) = std::io::pipe().unwrap();
    // The server inherits the read end; this test alone holds the write end.
    // SAFETY: fcntl clears the descriptor's close-on-exec flag alone.
    unsafe { libc::fcntl(read.as_raw_fd(), libc::F_SETFD, 0) };
    let log = scratch.0.join("server.log");
    let mut child = server(&config, &tmp)
        .arg("--parent-fd")
        .arg(read.as_raw_fd().to_string())
        .stderr(std::fs::File::create(&log).unwrap())
        .spawn()
        .unwrap();
    drop(read);
    let started = Instant::now();
    while !serving(&log) {
        assert!(
            child.try_wait().unwrap().is_none(),
            "{}",
            std::fs::read_to_string(&log).unwrap_or_default()
        );
        assert!(started.elapsed() < Duration::from_secs(60), "never served");
        std::thread::sleep(Duration::from_millis(50));
    }
    let pid = child.id();
    let worker_directories = || {
        std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .filter(|name| name.starts_with(&format!("gf-{pid}-")))
            .count()
    };
    assert!(worker_directories() > 0, "the worker's socket directory");

    // The parent goes: its end of the pipe closes.
    drop(write);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(started.elapsed() < Duration::from_secs(90), "never exited");
        std::thread::sleep(Duration::from_millis(50));
    };
    let logged = std::fs::read_to_string(&log).unwrap();
    assert!(status.success(), "{status}: {logged}");
    assert!(logged.contains("draining"), "{logged}");
    assert!(logged.contains("drained"), "{logged}");
    assert_eq!(worker_directories(), 0, "the worker's directory is removed");
}

#[test]
fn a_descriptor_that_is_not_an_open_pipe_is_a_usage_error() {
    let scratch = Scratch::new("usage");
    let config = configured(&scratch.0);
    let file = std::fs::File::open(&config).unwrap();
    // SAFETY: fcntl clears the descriptor's close-on-exec flag alone.
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) };
    let not_a_pipe = file.as_raw_fd().to_string();
    for (flag, env) in [
        // Not open, and open but a file.
        (Some("987"), None),
        (Some(not_a_pipe.as_str()), None),
        (None, Some("not-a-number")),
        (None, Some("987")),
    ] {
        let mut command = server(&config, &scratch.0);
        if let Some(flag) = flag {
            command.args(["--parent-fd", flag]);
        }
        if let Some(env) = env {
            command.env(gf_server::parent::ENV, env);
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{flag:?} {env:?}: {stderr}");
        assert!(
            stderr.contains("parent fd") || stderr.contains("GF_PARENT_FD"),
            "{stderr}"
        );
    }
}
