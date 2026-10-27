// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `gf-router run --parent-fd` refuses a descriptor it cannot watch.

use std::process::Command;

#[test]
fn a_descriptor_that_is_not_an_open_pipe_is_a_usage_error() {
    for (flag, env) in [(Some("987"), None), (None, Some("987")), (None, Some("x"))] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_gf-router"));
        command.args(["run", "--config", "/nonexistent/router.toml"]);
        if let Some(flag) = flag {
            command.args(["--parent-fd", flag]);
        }
        match env {
            Some(env) => command.env("GF_PARENT_FD", env),
            None => command.env_remove("GF_PARENT_FD"),
        };
        let output = command.output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{flag:?} {env:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
