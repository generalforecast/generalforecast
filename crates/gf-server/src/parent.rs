// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exiting with the parent: a server started by another process, such as
//! `gf serve` or a test fixture, stops when that process goes, however it
//! went.
//!
//! The parent passes `--parent-fd <N>`, or sets `GF_PARENT_FD=<N>` (the flag
//! wins if both are given): `<N>` is a file descriptor the server inherits,
//! the read end of a pipe whose write end the parent alone holds. The server
//! never writes to it. Once a read on it returns end of file, because the
//! parent closed it or exited, the server drains and exits as it does on
//! SIGTERM, within the same forced-exit deadline. A descriptor that is not
//! open, or not a pipe, refuses the start as a usage error.
//!
//! Linux and macOS only, as the server is.

use std::fs::File;
use std::io::Read;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

use tokio::sync::oneshot;

/// The environment variable that names the descriptor when no flag does.
pub const ENV: &str = "GF_PARENT_FD";

/// The descriptor to watch: `flag`'s, else the environment's, if either
/// names one.
///
/// # Errors
///
/// Why the descriptor cannot be watched: the environment's value is not a
/// descriptor number, or the descriptor is not open, or not a pipe.
pub fn named(flag: Option<RawFd>) -> Result<Option<OwnedFd>, String> {
    let fd = match flag {
        Some(fd) => fd,
        None => match std::env::var(ENV) {
            Ok(text) => text
                .trim()
                .parse()
                .map_err(|_| format!("{ENV} must be a file descriptor number, not {text:?}"))?,
            Err(_) => return Ok(None),
        },
    };
    checked(fd).map(Some)
}

/// `fd`, taken over, once it is known to be an open pipe.
fn checked(fd: RawFd) -> Result<OwnedFd, String> {
    if fd < 0 {
        return Err(format!("parent fd {fd} is not a file descriptor"));
    }
    // SAFETY: fstat writes the descriptor's status into the zeroed struct
    // it is given, and touches no other memory.
    let mut status: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &raw mut status) } != 0 {
        return Err(format!(
            "parent fd {fd} is not open: {}",
            std::io::Error::last_os_error()
        ));
    }
    if status.st_mode & libc::S_IFMT != libc::S_IFIFO {
        return Err(format!(
            "parent fd {fd} is not a pipe; pass the read end of a pipe the parent keeps the write end of"
        ));
    }
    // The workers the server starts have no use for it.
    // SAFETY: fcntl sets the descriptor's flags and touches no memory.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    // SAFETY: the descriptor is open, and the server takes it over alone.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Resolves once the parent is gone: a read on `fd` returns end of file, or
/// fails. Anything the parent writes is read and ignored.
pub fn gone(fd: OwnedFd) -> oneshot::Receiver<()> {
    let (told, gone) = oneshot::channel();
    std::thread::Builder::new()
        .name("gf-parent".to_string())
        .spawn(move || {
            let mut pipe = File::from(fd);
            let mut buffer = [0u8; 64];
            loop {
                match pipe.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            let _ = told.send(());
        })
        .expect("a thread to watch the parent");
    gone
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::os::fd::{AsRawFd, IntoRawFd};

    use super::{checked, gone};

    #[test]
    fn only_an_open_pipe_is_watched() {
        let file = tempfile_like();
        let refused = checked(file.as_raw_fd()).unwrap_err();
        assert!(refused.contains("not a pipe"), "{refused}");
        let refused = checked(1_000_000).unwrap_err();
        assert!(refused.contains("not open"), "{refused}");
        assert!(checked(-1).is_err());
    }

    #[tokio::test]
    async fn the_parent_is_gone_once_its_end_of_the_pipe_closes() {
        let (read, write) = std::io::pipe().unwrap();
        let watched = checked(read.into_raw_fd()).unwrap();
        let mut gone = gone(watched);
        let mut write = write;
        write.write_all(b"ignored").unwrap();
        assert!(gone.try_recv().is_err(), "a write is not the parent going");
        drop(write);
        tokio::time::timeout(std::time::Duration::from_secs(5), gone)
            .await
            .expect("told in time")
            .unwrap();
    }

    /// A plain file, which is not a pipe.
    fn tempfile_like() -> std::fs::File {
        std::fs::File::open(std::env::current_exe().unwrap()).unwrap()
    }
}
