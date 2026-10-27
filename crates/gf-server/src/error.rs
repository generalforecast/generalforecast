// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Why a shard refused a request, and what that means to a client of either
//! API: one table gives each error its Flight status and its Models API
//! status and code, so the same error means the same thing on both.

use std::fmt;

use gf_core::node::{ErrorKind, NodeError};
use tonic::Status;

use crate::wal::LogError;

/// Why a request failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerError {
    NotFound(String),
    Exists(String),
    Invalid(String),
    /// A replicated entry skips ahead of the log.
    Gap(String),
    /// A write arrived without this node's lease, or under another term.
    Fenced(String),
    /// The shard holds as many queries as it may; retry shortly.
    Busy(String),
    /// The worker is starting again; retry shortly.
    Unavailable(String),
    Internal(String),
}

/// What an error means to a client: its Flight status code, its Models API
/// status and code, and the message the Models API answers with, None where
/// the error's own message is safe to show.
struct Meaning {
    code: tonic::Code,
    status: u16,
    name: &'static str,
    message: Option<&'static str>,
}

impl ServerError {
    fn meaning(&self) -> Meaning {
        let meaning = |code, status, name, message| Meaning {
            code,
            status,
            name,
            message,
        };
        match self {
            Self::NotFound(_) => meaning(
                tonic::Code::NotFound,
                404,
                "NotFound",
                Some("the resource is not served here"),
            ),
            Self::Exists(_) => meaning(
                tonic::Code::AlreadyExists,
                409,
                "Conflict",
                Some("the resource exists already"),
            ),
            Self::Invalid(_) => meaning(tonic::Code::InvalidArgument, 422, "InvalidContext", None),
            // Apart, so a router restarts a shipment that skipped ahead, and
            // retries a fenced write only once it learns the table's term.
            Self::Gap(_) => meaning(
                tonic::Code::OutOfRange,
                409,
                "Conflict",
                Some("the request skips ahead of the table's log"),
            ),
            Self::Fenced(_) => meaning(
                tonic::Code::FailedPrecondition,
                409,
                "Conflict",
                Some("this server does not take the table's writes now"),
            ),
            Self::Busy(_) => meaning(
                tonic::Code::ResourceExhausted,
                503,
                "CapacityExceeded",
                None,
            ),
            Self::Unavailable(_) => meaning(
                tonic::Code::Unavailable,
                503,
                "ModelUnavailable",
                Some("the model is restarting; retry shortly"),
            ),
            Self::Internal(_) => meaning(
                tonic::Code::Internal,
                503,
                "InferenceFailed",
                Some("native inference failed"),
            ),
        }
    }

    /// The Flight status code the error answers a call with.
    #[must_use]
    pub fn code(&self) -> tonic::Code {
        self.meaning().code
    }

    /// The Flight status the error answers a call with.
    #[must_use]
    pub fn status(self) -> Status {
        let code = self.code();
        let message = self.to_string();
        if code == tonic::Code::AlreadyExists {
            exists(message)
        } else {
            Status::new(code, message)
        }
    }

    /// The Models API's status, code and message for the error: its own
    /// message where that is safe to show a caller, else the kind's.
    #[must_use]
    pub fn http(&self) -> (u16, &'static str, String) {
        let Meaning {
            status,
            name,
            message,
            ..
        } = self.meaning();
        let message = message.map_or_else(|| self.to_string(), str::to_string);
        (status, name, message)
    }
}

/// `ALREADY_EXISTS`, which Arrow clients raise as invalid input: pyarrow has
/// no error of its own for it, so the Arrow status code rides along in the
/// trailer Arrow Flight reads it from.
pub fn exists(message: impl Into<String>) -> Status {
    let mut metadata = tonic::metadata::MetadataMap::new();
    metadata.insert(
        "x-arrow-status",
        tonic::metadata::MetadataValue::from_static("4"),
    );
    Status::with_metadata(tonic::Code::AlreadyExists, message, metadata)
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(message)
            | Self::Exists(message)
            | Self::Invalid(message)
            | Self::Gap(message)
            | Self::Fenced(message)
            | Self::Busy(message)
            | Self::Unavailable(message)
            | Self::Internal(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ServerError {}

impl From<ServerError> for Status {
    fn from(error: ServerError) -> Self {
        error.status()
    }
}

impl From<NodeError> for ServerError {
    fn from(error: NodeError) -> Self {
        let message = error.to_string();
        match error.kind() {
            ErrorKind::NotFound => Self::NotFound(message),
            ErrorKind::Invalid => Self::Invalid(message),
            ErrorKind::Internal => Self::Internal(message),
        }
    }
}

impl From<LogError> for ServerError {
    fn from(error: LogError) -> Self {
        match error {
            LogError::Beyond { .. } => Self::Invalid(error.to_string()),
            LogError::Io(io) if io.kind() == std::io::ErrorKind::AlreadyExists => Self::Exists(
                format!("a log for the table is on disk already, which no table serves now: {io}"),
            ),
            error => Self::Internal(format!("write-ahead log: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServerError;

    #[test]
    fn each_error_means_one_thing_to_flight_and_to_the_models_api() {
        let message = || "table \"ctx-01\" is gone".to_string();
        for (error, code, http) in [
            (
                ServerError::NotFound(message()),
                tonic::Code::NotFound,
                (404, "NotFound"),
            ),
            (
                ServerError::Exists(message()),
                tonic::Code::AlreadyExists,
                (409, "Conflict"),
            ),
            (
                ServerError::Gap(message()),
                tonic::Code::OutOfRange,
                (409, "Conflict"),
            ),
            (
                ServerError::Fenced(message()),
                tonic::Code::FailedPrecondition,
                (409, "Conflict"),
            ),
            (
                ServerError::Invalid(message()),
                tonic::Code::InvalidArgument,
                (422, "InvalidContext"),
            ),
            (
                ServerError::Busy(message()),
                tonic::Code::ResourceExhausted,
                (503, "CapacityExceeded"),
            ),
            (
                ServerError::Unavailable(message()),
                tonic::Code::Unavailable,
                (503, "ModelUnavailable"),
            ),
            (
                ServerError::Internal(message()),
                tonic::Code::Internal,
                (503, "InferenceFailed"),
            ),
        ] {
            assert_eq!(error.code(), code, "{error:?}");
            let (status, name, shown) = error.http();
            assert_eq!((status, name), http, "{error:?}");
            // Only a refusal of the caller's own request says what it was.
            let own = matches!(error, ServerError::Invalid(_) | ServerError::Busy(_));
            assert_eq!(shown == message(), own, "{error:?}: {shown}");
            assert_eq!(error.status().code(), code);
        }
    }
}
