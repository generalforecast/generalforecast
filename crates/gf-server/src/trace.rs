// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! W3C trace context and logging, so one request's logs join up across the
//! router, the node and the step its queries ran in.

use std::time::Instant;

use tonic::{Code, Request, Status};
use tracing::Instrument;

/// A request's trace id, and the id of the span that sent it on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    id: String,
    span: String,
}

fn random_hex(bytes: usize) -> String {
    crate::setup::random_hex(bytes).expect("the system has randomness")
}

fn valid(id: &str, length: usize) -> bool {
    id.len() == length
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && id.bytes().any(|byte| byte != b'0')
}

impl Trace {
    /// A new trace, for a request that arrived without one.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: random_hex(16),
            span: random_hex(8),
        }
    }

    /// The trace a `traceparent` header names, if it is a valid one.
    #[must_use]
    pub fn parse(header: &str) -> Option<Self> {
        let mut parts = header.trim().split('-');
        let (version, id, span) = (parts.next()?, parts.next()?, parts.next()?);
        let _flags = parts.next()?;
        (version.len() == 2 && version != "ff" && valid(id, 32) && valid(span, 16)).then(|| Self {
            id: id.to_string(),
            span: span.to_string(),
        })
    }

    /// The trace a request carries in its `traceparent` metadata, or a new one.
    #[must_use]
    pub fn of<T>(request: &Request<T>) -> Self {
        request
            .metadata()
            .get("traceparent")
            .and_then(|value| value.to_str().ok())
            .and_then(Self::parse)
            .unwrap_or_default()
    }

    /// The same trace, sent on by a new span.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            id: self.id.clone(),
            span: random_hex(8),
        }
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The `traceparent` header that sends this trace on.
    #[must_use]
    pub fn header(&self) -> String {
        format!("00-{}-{}-01", self.id, self.span)
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs a call in a span that carries its trace id, and logs how it ended:
/// failures on the server's side as warnings, the rest at debug.
///
/// # Errors
///
/// The call's.
pub async fn traced<T>(
    kind: &str,
    trace: &Trace,
    call: impl Future<Output = Result<T, Status>>,
) -> Result<T, Status> {
    // At warn, so a terminal's `warn` filter keeps the span and its trace id
    // on the warnings logged inside it; nothing prints the span itself.
    let span = tracing::warn_span!("call", kind, trace = trace.id());
    async move {
        let started = Instant::now();
        let result = call.await;
        let milliseconds = started.elapsed().as_secs_f64() * 1e3;
        match &result {
            Ok(_) => tracing::debug!(milliseconds, "answered"),
            Err(status)
                if [Code::Internal, Code::Unavailable, Code::Unknown].contains(&status.code()) =>
            {
                tracing::warn!(milliseconds, code = ?status.code(), "{}", status.message());
            }
            Err(status) => {
                tracing::debug!(milliseconds, code = ?status.code(), "{}", status.message())
            }
        }
        result
    }
    .instrument(span)
    .await
}

/// Logs to stderr: as JSON lines with `GF_LOG_FORMAT=json`, at the levels
/// `GF_LOG` sets in `RUST_LOG` syntax. Without `GF_LOG`, a terminal gets
/// `warn`, so the status lines stand alone; a pipe or a supervisor gets
/// `info`. `debug` adds a line per request and per step.
pub fn init_logging() {
    use std::io::IsTerminal;

    let filter = tracing_subscriber::EnvFilter::try_from_env("GF_LOG").unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(default_level(std::io::stderr().is_terminal()))
    });
    let logs = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(colour(
            std::io::stderr().is_terminal(),
            std::env::var_os("NO_COLOR").as_deref(),
        ))
        .with_writer(std::io::stderr);
    if std::env::var("GF_LOG_FORMAT").is_ok_and(|format| format == "json") {
        logs.json().init();
    } else {
        logs.init();
    }
}

/// Whether logs are coloured: on a terminal alone, and not where `NO_COLOR`
/// holds anything, as no-color.org asks.
#[must_use]
pub fn colour(terminal: bool, no_color: Option<&std::ffi::OsStr>) -> bool {
    terminal && no_color.is_none_or(std::ffi::OsStr::is_empty)
}

/// The level logged without `GF_LOG`: `warn` on a terminal, `info` elsewhere.
pub fn default_level(terminal: bool) -> &'static str {
    if terminal { "warn" } else { "info" }
}

/// Makes any panic log and end the process, rather than end one thread and
/// leave the rest serving without it, so a supervisor starts it again whole.
pub fn abort_on_panic() {
    let report = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        tracing::error!("{panic}");
        report(panic);
        std::process::abort();
    }));
}

#[cfg(test)]
mod tests {
    use super::default_level;

    #[test]
    fn a_terminal_logs_warnings_only_and_a_pipe_logs_info() {
        assert_eq!(default_level(true), "warn");
        assert_eq!(default_level(false), "info");
    }
}
