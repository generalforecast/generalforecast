// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! W3C trace context: read from `traceparent`, and sent on by child spans.

use gf_server::trace::Trace;

#[test]
fn traces_parse_and_pass_on_their_id() {
    let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    let trace = Trace::parse(header).unwrap();
    let child = trace.child();

    assert_eq!(trace.header(), header);
    assert_eq!(child.id(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_ne!(child.header(), header);
    assert_eq!(Trace::parse(&child.header()), Some(child));
    for invalid in [
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
    ] {
        assert_eq!(Trace::parse(invalid), None, "{invalid}");
    }
}

#[test]
fn logs_are_coloured_on_a_terminal_alone_and_never_under_no_color() {
    use std::ffi::OsStr;

    use gf_server::trace::colour;
    assert!(colour(true, None));
    assert!(
        colour(true, Some(OsStr::new(""))),
        "an empty NO_COLOR is unset"
    );
    assert!(!colour(true, Some(OsStr::new("1"))));
    assert!(!colour(false, None), "a pipe gets no escape codes");
}
