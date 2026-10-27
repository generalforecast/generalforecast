// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Why the node refused a change or a request.

use super::*;

/// Why a change or request was refused. The node is left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeError {
    UnknownTable(TableId),
    /// A table was given no feature columns.
    NoFeatures,
    /// A retrieval vector was requested before its standardizer was fitted.
    Unscaled,
    Strategy(StrategyError),
    /// Queries were sent to a table with no rows to learn from.
    EmptyTable(TableId),
    /// A query's time cutoff precedes every row of the table.
    EmptyContext(TableId),
    /// The table's strategy keeps to a span of time, and no row has a time.
    Untimed(TableId),
    UnknownKey(RowKey),
    DuplicateKey(RowKey),
    /// A batch has a different number of `what` (such as targets) than keys.
    Count {
        what: Counted,
        expected: usize,
        got: usize,
    },
    /// A refit policy limit that is not positive and finite.
    RefitPolicy,
    /// Rows with a different number of columns than the table.
    Width {
        expected: usize,
        got: usize,
    },
    /// A batch of queries with no rows.
    NoQueries,
    /// A class label outside the table's classes.
    Label {
        label: i32,
        classes: u32,
    },
    /// Targets of the wrong kind for the table's task.
    Task,
    /// A compacted log's rows or layout that do not rebuild a table.
    Restore(&'static str),
    /// A given context that is empty or names rows the table does not hold.
    Given,
    /// Rows that could not be read or written.
    Rows(RowsError),
    Table(TableError),
    Index(IndexError),
    Plan(PlanError),
    Cache(CacheError),
    Regions(RegionsError),
    Forecast(ForecastError),
    /// A table's categorical features or their encoding do not fit it.
    Categories(&'static str),
}

impl fmt::Display for NodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTable(table) => write!(formatter, "table {} does not exist", table.0),
            Self::NoFeatures => write!(formatter, "a table needs at least one feature column"),
            Self::Unscaled => write!(formatter, "the table has no fitted retrieval space"),
            Self::Strategy(error) => error.fmt(formatter),
            Self::EmptyTable(_) => write!(
                formatter,
                "the table has no rows, or none labeled, to predict from"
            ),
            Self::EmptyContext(_) => write!(
                formatter,
                "a query's time cutoff precedes every row of the table, so it has no context"
            ),
            Self::Untimed(_) => write!(
                formatter,
                "the table's strategy keeps to a span of time, but none of its rows has a \
                 time, so a query has no context"
            ),
            Self::UnknownKey(key) => write!(formatter, "row key {key} does not exist"),
            Self::DuplicateKey(key) => write!(formatter, "row key {key} appears twice"),
            Self::Count {
                what,
                expected,
                got,
            } => write!(formatter, "expected {expected} {what}, got {got}"),
            Self::Width { expected, got } => write!(
                formatter,
                "rows have {got} columns, but the table has {expected} feature columns"
            ),
            Self::NoQueries => write!(formatter, "there are no query rows"),
            Self::RefitPolicy => write!(
                formatter,
                "a refit policy's limits must be positive and finite, and \
                 refit_cutoffs = \"changed\" needs refit_after"
            ),
            Self::Label { label, classes } => {
                write!(
                    formatter,
                    "label {label} is outside 0 .. {}",
                    classes.saturating_sub(1)
                )
            }
            Self::Task => write!(formatter, "targets do not match the table's task"),
            Self::Restore(why) => write!(formatter, "the table cannot be restored: {why}"),
            Self::Given => write!(formatter, "each given context holds rows of the table"),
            Self::Rows(error) => error.fmt(formatter),
            Self::Table(error) => error.fmt(formatter),
            Self::Index(error) => error.fmt(formatter),
            Self::Plan(error) => error.fmt(formatter),
            Self::Cache(error) => error.fmt(formatter),
            Self::Regions(error) => error.fmt(formatter),
            Self::Forecast(error) => error.fmt(formatter),
            Self::Categories(why) => write!(formatter, "{why}"),
        }
    }
}

impl std::error::Error for NodeError {}

/// What kind of failure an error is, which servers and bindings report alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A table or key that does not exist.
    NotFound,
    /// Input the node refuses, such as rows of the wrong width.
    Invalid,
    /// A fault of the node's own, such as its cache refusing an operation.
    Internal,
}

impl NodeError {
    /// What kind of failure the error is.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::UnknownTable(_)
            | Self::UnknownKey(_)
            | Self::Forecast(ForecastError::UnknownMember(_))
            | Self::Table(TableError::UnknownKey(_)) => ErrorKind::NotFound,
            Self::Index(error) | Self::Plan(PlanError::Index(error)) => match error {
                IndexError::Dimensions { .. } | IndexError::NotFinite => ErrorKind::Invalid,
                IndexError::Duplicate(_)
                | IndexError::Unknown(_)
                | IndexError::UnknownPage(_)
                | IndexError::Rows(_) => ErrorKind::Internal,
            },
            Self::Cache(_)
            | Self::Plan(_)
            | Self::Restore(_)
            | Self::Rows(_)
            | Self::Regions(RegionsError::Rows(_)) => ErrorKind::Internal,
            _ => ErrorKind::Invalid,
        }
    }
}

/// What a batch counts one of per key, or per query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counted {
    Targets,
    Times,
    Cutoffs,
    Groups,
    Rows,
    Contexts,
}

impl fmt::Display for Counted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Targets => "targets",
            Self::Times => "times",
            Self::Cutoffs => "cutoffs",
            Self::Groups => "groups",
            Self::Rows => "rows of features",
            Self::Contexts => "contexts",
        })
    }
}

impl From<RowsError> for NodeError {
    fn from(error: RowsError) -> Self {
        Self::Rows(error)
    }
}

impl From<TableError> for NodeError {
    fn from(error: TableError) -> Self {
        Self::Table(error)
    }
}

impl From<IndexError> for NodeError {
    fn from(error: IndexError) -> Self {
        Self::Index(error)
    }
}

impl From<PlanError> for NodeError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<StrategyError> for NodeError {
    fn from(error: StrategyError) -> Self {
        Self::Strategy(error)
    }
}

impl From<ForecastError> for NodeError {
    fn from(error: ForecastError) -> Self {
        Self::Forecast(error)
    }
}

impl From<CacheError> for NodeError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}
