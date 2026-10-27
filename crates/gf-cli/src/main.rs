// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `gf`, as a release wheel installs it: runs `python -m generalforecast.cli`
//! with the same arguments, under `GF_PYTHON`, or else the interpreter
//! beside this executable, or else `python3`. The command itself lives in
//! Python; this only finds it.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Why `gf` could not run, in the error grammar the command prints with
/// `--json`: the wire code `Internal`, with the reason that says which part
/// of the install is missing.
const PYTHON_UNAVAILABLE: &str = concat!(
    r#"{"error":{"code":"Internal","reason":"python_unavailable","#,
    r#""message":"no Python interpreter runs generalforecast here","#,
    r#""hint":"install generalforecast, or set GF_PYTHON to the interpreter it is installed in"}}"#,
);

/// The interpreter to run: `GF_PYTHON`, the one beside this executable, or `python3`.
fn python() -> OsString {
    if let Some(python) = std::env::var_os("GF_PYTHON") {
        return python;
    }
    #[cfg(windows)]
    let names = ["python.exe", "python3.exe"];
    #[cfg(not(windows))]
    let names = ["python", "python3"];
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            let directory = exe.parent()?.to_path_buf();
            names
                .iter()
                .map(|name| directory.join(name))
                .find(|candidate: &PathBuf| candidate.is_file())
        })
        .map_or_else(|| "python3".into(), PathBuf::into_os_string)
}

fn main() -> ExitCode {
    let mut command = Command::new(python());
    command
        .args(["-m", "generalforecast.cli"])
        .args(std::env::args_os().skip(1));
    // On Unix the interpreter takes this process's place, so signals and the
    // exit status are its own; `exec` returns only if it could not start.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let _error = command.exec();
        eprintln!("{PYTHON_UNAVAILABLE}");
        ExitCode::FAILURE
    }
    #[cfg(not(unix))]
    {
        match command.status() {
            Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
            Err(_) => {
                eprintln!("{PYTHON_UNAVAILABLE}");
                ExitCode::FAILURE
            }
        }
    }
}
