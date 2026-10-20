//! Thin launcher for the wheel's single canonical gf implementation.

use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let python = std::env::var_os("GF_PYTHON").unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|exe| {
                let directory = exe.parent()?;
                #[cfg(windows)]
                let names = ["python.exe", "python3.exe"];
                #[cfg(not(windows))]
                let names = ["python", "python3"];
                names
                    .iter()
                    .map(|name| directory.join(name))
                    .find(|candidate| candidate.is_file())
                    .map(|path| path.into_os_string())
            })
            .unwrap_or_else(|| "python3".into())
    });
    let mut command = Command::new(python);
    command
        .args(["-m", "generalforecast.cli"])
        .args(std::env::args_os().skip(1));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let _error = command.exec();
        eprintln!(
            "{{\"error\":{{\"code\":\"PythonUnavailable\",\"message\":\"install generalforecast and set GF_PYTHON to its interpreter\"}}}}"
        );
        ExitCode::FAILURE
    }
    #[cfg(not(unix))]
    {
        match command.status() {
            Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
            Err(_) => {
                eprintln!(
                    "{{\"error\":{{\"code\":\"PythonUnavailable\",\"message\":\"install generalforecast and set GF_PYTHON to its interpreter\"}}}}"
                );
                ExitCode::FAILURE
            }
        }
    }
}
