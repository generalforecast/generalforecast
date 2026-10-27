// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The launcher's own failure: no interpreter to run the command under.

use std::process::Command;

#[test]
fn without_python_the_launcher_answers_in_the_error_grammar() {
    let output = Command::new(env!("CARGO_BIN_EXE_gf"))
        .arg("--version")
        .env("GF_PYTHON", "/nonexistent/gf-test-python")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let answer: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    let error = &answer["error"];
    // A wire code, with the reason saying what the install lacks.
    assert_eq!(error["code"], "Internal");
    assert_eq!(error["reason"], "python_unavailable");
    assert!(error["hint"].as_str().unwrap().contains("GF_PYTHON"));
}
