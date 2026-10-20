//! Flight actions and write commands, parsed once from what a client or
//! router sent, and the answers routers read back, for servers and routers
//! alike.
//!
//! Answers are read tolerantly: unknown fields are ignored and new ones have
//! defaults, so a router and a node of different versions still agree.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

use arrow_flight::Action;
use serde::{Deserialize, Serialize};
use tonic::Status;

use crate::auth::Scope;
use crate::name::TableName;
use crate::wal::{CreationId, TableDefinition, Terms};

/// The longest a client's write id may be.
const WRITE_ID_BYTES: usize = 64;

/// A `DoAction` call, parsed from its type and JSON body.
#[derive(Clone, Debug)]
pub enum FlightAction {
    Health,
    ListTables,
    ListModels,
    /// The nodes that should hold a table, in rank order; routers answer it.
    Locate(TableName),
    CreateTable(Box<TableDefinition>),
    Describe {
        table: TableName,
        words: bool,
    },
    Delete(DeleteAction),
    Truncate(TableName),
    Refit(TableName),
    Compact(TableName),
    DropTable(TableName),
    /// Every table's log length, terms and creation.
    LogLengths,
    /// Stops serving a copy, keeping its log aside with `keep`.
    Release {
        table: TableName,
        keep: bool,
    },
    /// Makes a copy the table's primary under a later term.
    Promote {
        table: TableName,
        term: u64,
        primary: String,
    },
    /// Lets a node take writes for `seconds`, for the router `holder`.
    Lease(LeaseRequest),
    /// Each table's shard and each shard's recent load.
    Placements,
    /// Moves a table to another shard of this server.
    MoveTable {
        table: TableName,
        shard: usize,
    },
    /// Runs a round of moving tables by load now.
    Rebalance,
    /// Has every worker sample its own threads for `seconds`, their
    /// stacks each written collapsed to a file under `directory`, and with
    /// `trace` write JAX's trace of the window to a directory of its own
    /// there. `directory` is relative to the server's profiles directory,
    /// and only an unscoped token may ask. The answer names
    /// the files, which appear once the windows end, `seconds` after they
    /// start. Each worker's window starts after the step it has in flight,
    /// so the windows are offset across shards.
    ProfileWorkers {
        seconds: f64,
        directory: String,
        trace: bool,
    },
}

/// The body of a `profile_workers` action.
#[derive(Deserialize)]
struct ProfileWorkers {
    #[serde(default = "default_profile_seconds")]
    seconds: f64,
    directory: String,
    #[serde(default)]
    trace: bool,
}

fn default_profile_seconds() -> f64 {
    10.0
}

/// A router's request for a node's write lease.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseRequest {
    pub seconds: f64,
    pub holder: String,
    /// When the router sent it, in milliseconds since 1970 by its clock: the
    /// lease ends that long after, so a request a stalled node reads late
    /// cannot grant a lease the router no longer counts on.
    #[serde(default)]
    pub sent_unix_ms: Option<u64>,
    /// Only say whether the lease would be granted and who holds it now,
    /// changing nothing.
    #[serde(default)]
    pub probe: bool,
}

/// A delete of integer keys, as `DoAction` takes it; other keys go through `DoPut`.
#[derive(Clone, Debug, Deserialize)]
pub struct DeleteAction {
    pub table: TableName,
    /// Integer keys, signed or unsigned, as their 128 bits.
    pub keys: Vec<i128>,
    #[serde(default)]
    pub missing_ok: bool,
    /// The client's id for the write, so a retry applies once.
    #[serde(default, deserialize_with = "write_id")]
    pub write_id: Option<String>,
}

#[derive(Deserialize)]
struct Table {
    table: TableName,
}

#[derive(Deserialize)]
struct Describe {
    table: TableName,
    /// Whether to send each categorical column's words, which may be many; a
    /// count of rows or a classifier's labels need none.
    #[serde(default = "yes")]
    words: bool,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct Release {
    table: TableName,
    #[serde(default)]
    keep: bool,
}

#[derive(Deserialize)]
struct MoveTable {
    table: TableName,
    shard: usize,
}

#[derive(Deserialize)]
struct Promote {
    table: TableName,
    term: u64,
    primary: String,
}

impl FlightAction {
    /// The action a call makes.
    ///
    /// # Errors
    ///
    /// `UNIMPLEMENTED` for an unknown type, and `INVALID_ARGUMENT` for a body
    /// that is not the type's.
    pub fn parse(action: &Action) -> Result<Self, Status> {
        let kind = action.r#type.as_str();
        Ok(match kind {
            "health" => Self::Health,
            "list_tables" => Self::ListTables,
            "list_models" => Self::ListModels,
            "log_lengths" => Self::LogLengths,
            "locate" => Self::Locate(body::<Table>(action)?.table),
            "create_table" => Self::CreateTable(Box::new(body(action)?)),
            "describe" => {
                let Describe { table, words } = body(action)?;
                Self::Describe { table, words }
            }
            "delete" => Self::Delete(body(action)?),
            "truncate" => Self::Truncate(body::<Table>(action)?.table),
            "refit" => Self::Refit(body::<Table>(action)?.table),
            "compact" => Self::Compact(body::<Table>(action)?.table),
            "drop_table" => Self::DropTable(body::<Table>(action)?.table),
            "placements" => Self::Placements,
            "rebalance" => Self::Rebalance,
            "profile_workers" => {
                let ProfileWorkers {
                    seconds,
                    directory,
                    trace,
                } = body(action)?;
                if !(seconds > 0.0 && seconds.is_finite()) || directory.is_empty() {
                    return Err(Status::invalid_argument(
                        "a profile names its directory and lasts finite seconds",
                    ));
                }
                Self::ProfileWorkers {
                    seconds,
                    directory,
                    trace,
                }
            }
            "move_table" => {
                let MoveTable { table, shard } = body(action)?;
                Self::MoveTable { table, shard }
            }
            "release" => {
                let Release { table, keep } = body(action)?;
                Self::Release { table, keep }
            }
            "promote" => {
                let Promote {
                    table,
                    term,
                    primary,
                } = body(action)?;
                Self::Promote {
                    table,
                    term,
                    primary,
                }
            }
            "lease" => {
                let lease: LeaseRequest = body(action)?;
                if !lease.seconds.is_finite() || lease.holder.is_empty() {
                    return Err(Status::invalid_argument(
                        "a lease names its holder and lasts finite seconds",
                    ));
                }
                Self::Lease(lease)
            }
            other => return Err(Status::unimplemented(format!("unknown action {other:?}"))),
        })
    }

    /// The action's type, as calls name it.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::ListTables => "list_tables",
            Self::ListModels => "list_models",
            Self::Locate(_) => "locate",
            Self::CreateTable(_) => "create_table",
            Self::Describe { .. } => "describe",
            Self::Delete(_) => "delete",
            Self::Truncate(_) => "truncate",
            Self::Refit(_) => "refit",
            Self::Compact(_) => "compact",
            Self::DropTable(_) => "drop_table",
            Self::LogLengths => "log_lengths",
            Self::Release { .. } => "release",
            Self::Promote { .. } => "promote",
            Self::Lease(_) => "lease",
            Self::Placements => "placements",
            Self::MoveTable { .. } => "move_table",
            Self::Rebalance => "rebalance",
            Self::ProfileWorkers { .. } => "profile_workers",
        }
    }

    /// The table the action is about, if one.
    #[must_use]
    pub fn table(&self) -> Option<&TableName> {
        match self {
            Self::CreateTable(definition) => Some(&definition.name),
            Self::Delete(delete) => Some(&delete.table),
            Self::Locate(table)
            | Self::Describe { table, .. }
            | Self::Truncate(table)
            | Self::Refit(table)
            | Self::Compact(table)
            | Self::DropTable(table)
            | Self::Release { table, .. }
            | Self::Promote { table, .. }
            | Self::MoveTable { table, .. } => Some(table),
            Self::Health
            | Self::ListTables
            | Self::ListModels
            | Self::LogLengths
            | Self::Lease(_)
            | Self::Placements
            | Self::Rebalance
            | Self::ProfileWorkers { .. } => None,
        }
    }

    /// Whether only routers may make it, with the replication token.
    #[must_use]
    pub fn needs_replication(&self) -> bool {
        matches!(
            self,
            Self::LogLengths | Self::Release { .. } | Self::Promote { .. } | Self::Lease(_)
        )
    }

    /// Whether it changes a table, so a node that needs leases takes it only under one.
    #[must_use]
    pub fn needs_lease(&self) -> bool {
        matches!(
            self,
            Self::CreateTable(_)
                | Self::Delete(_)
                | Self::Truncate(_)
                | Self::Refit(_)
                | Self::Compact(_)
                | Self::DropTable(_)
        )
    }

    /// Whether it changes rows, so a router confirms a replica holds it.
    #[must_use]
    pub fn changes_rows(&self) -> bool {
        matches!(self, Self::Delete(_) | Self::Truncate(_))
    }

    /// Refuses the action if a token's scope does not reach its table, or
    /// every table for an action on the whole server.
    ///
    /// # Errors
    ///
    /// As `Scope::check` and `Scope::check_create`, and `PERMISSION_DENIED`
    /// for `profile_workers` under any scope but the unscoped one.
    pub fn check_scope(&self, scope: &Scope) -> Result<(), Status> {
        match self {
            Self::CreateTable(definition) => scope.check_create(&definition.name),
            Self::ProfileWorkers { .. } if *scope != Scope::All => Err(Status::permission_denied(
                "profile_workers needs an unscoped token",
            )),
            other => other.table().map_or(Ok(()), |table| scope.check(table)),
        }
    }
}

/// Where a profile goes: `directory` under `root`, which it may not leave.
///
/// # Errors
///
/// `INVALID_ARGUMENT` for an absolute `directory` or one with a `..`.
pub fn profile_directory(root: &Path, directory: &str) -> Result<PathBuf, Status> {
    let inside = Path::new(directory)
        .components()
        .all(|part| matches!(part, Component::Normal(_) | Component::CurDir));
    if !inside {
        return Err(Status::invalid_argument(
            "a profile's directory is relative, under the server's profiles directory",
        ));
    }
    Ok(root.join(directory))
}

fn body<'a, T: Deserialize<'a>>(action: &'a Action) -> Result<T, Status> {
    serde_json::from_slice(&action.body)
        .map_err(|error| Status::invalid_argument(format!("bad {} body: {error}", action.r#type)))
}

/// Checks a client's id for a write: 1 to 64 characters.
///
/// # Errors
///
/// What is wrong with it.
pub fn check_write_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > WRITE_ID_BYTES {
        return Err(format!("a write id is 1 to {WRITE_ID_BYTES} characters"));
    }
    Ok(())
}

fn write_id<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let id = Option::<String>::deserialize(deserializer)?;
    if let Some(id) = &id {
        check_write_id(id).map_err(serde::de::Error::custom)?;
    }
    Ok(id)
}

/// A `DoPut` call's command, the first part of its descriptor path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutCommand {
    Insert,
    Update,
    Upsert,
    /// Rows' targets alone, keeping their features and times.
    Label,
    Delete,
    /// Log entries from another copy; routers only, with the replication token.
    Replicate,
}

impl PutCommand {
    /// The command's name, as descriptor paths give it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Update => "update",
            Self::Upsert => "upsert",
            Self::Label => "label",
            Self::Delete => "delete",
            Self::Replicate => "replicate",
        }
    }

    /// Whether clients send it: every command but `Replicate`.
    #[must_use]
    pub fn is_client_write(self) -> bool {
        self != Self::Replicate
    }
}

impl FromStr for PutCommand {
    type Err = Status;

    fn from_str(command: &str) -> Result<Self, Status> {
        Ok(match command {
            "insert" => Self::Insert,
            "update" => Self::Update,
            "upsert" => Self::Upsert,
            "label" => Self::Label,
            "delete" => Self::Delete,
            "replicate" => Self::Replicate,
            other => return Err(Status::invalid_argument(format!("unknown write {other:?}"))),
        })
    }
}

impl fmt::Display for PutCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What a write, a create, a compaction or a promotion answers: the table's
/// log length after it, and with a write or a create, the table's creation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<usize>,
    #[serde(default)]
    pub log_length: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "CreationId::optional"
    )]
    pub created: Option<CreationId>,
}

impl Answer {
    /// An answer as a node sent it, or an empty one if it is not one.
    #[must_use]
    pub fn of(value: &serde_json::Value) -> Self {
        Self::deserialize(value).unwrap_or_default()
    }
}

/// One copy of a table, as `log_lengths` answers for each table a node holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopyLength {
    pub length: u64,
    pub terms: Terms,
    #[serde(default, deserialize_with = "CreationId::optional")]
    pub created: Option<CreationId>,
}

/// What `log_lengths` answers: each table's copy on the node.
pub type LogLengths = BTreeMap<String, CopyLength>;

/// What `lease` answers: whether the node granted it, or which router holds it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseAnswer {
    pub granted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holder: Option<String>,
    /// Whether the node takes client writes only under a lease; one that does
    /// not still takes them from a primary a router has replaced.
    #[serde(default)]
    pub enforced: bool,
}

/// Milliseconds since 1970 by this machine's clock, as lease requests carry them.
#[must_use]
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        })
}
