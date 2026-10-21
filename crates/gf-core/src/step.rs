//! What a step asks of the executor: actions, placements and request ids.
//!
//! These name a step's parts without the node's planning around them, so the
//! planners of tabular contexts and of series share them.

use crate::cache::SlotId;

/// What the executor does with a unit or a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Compute it (encode a unit, embed a page) and keep the result.
    Encode,
    /// It is already cached.
    Reuse,
    /// Compute it for this step only: there was no room to keep it.
    Transient,
    /// Fit the table's state on the unit's rows, then encode them and decode
    /// the queries that read them in the same call, keeping only the state:
    /// a context one request brought, which may never be read again.
    Fuse,
    /// Grow a cached series encoding by whole patches of new steps, from
    /// the slots that hold the shorter one, and keep the result.
    Extend,
}

impl Action {
    /// The action's name, as workers read it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Reuse => "reuse",
            Self::Transient => "transient",
            Self::Fuse => "fuse",
            Self::Extend => "extend",
        }
    }
}

/// A unit's or page's action, with the slots that hold its result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Placed {
    /// Computed and kept in these slots.
    Encode(Vec<SlotId>),
    /// Read from the slots that already hold it.
    Reuse(Vec<SlotId>),
    /// Computed for this step only, in no slots.
    Transient,
    /// Fit, encoded and decoded in one call for this step only, in no slots.
    Fuse,
    /// Grown from the encoding in `source` by whole patches of new steps,
    /// and kept in `slots`.
    Extend {
        source: Vec<SlotId>,
        slots: Vec<SlotId>,
    },
}

impl Placed {
    /// What the executor does.
    #[must_use]
    pub fn action(&self) -> Action {
        match self {
            Self::Encode(_) => Action::Encode,
            Self::Reuse(_) => Action::Reuse,
            Self::Transient => Action::Transient,
            Self::Fuse => Action::Fuse,
            Self::Extend { .. } => Action::Extend,
        }
    }

    /// The slots that hold the result; none for a transient one.
    #[must_use]
    pub fn slots(&self) -> &[SlotId] {
        match self {
            Self::Encode(slots) | Self::Reuse(slots) | Self::Extend { slots, .. } => slots,
            Self::Transient | Self::Fuse => &[],
        }
    }

    /// The slots an extension reads the shorter encoding from; none otherwise.
    #[must_use]
    pub fn source(&self) -> &[SlotId] {
        match self {
            Self::Extend { source, .. } => source,
            _ => &[],
        }
    }
}

/// A unit's position in its step's units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepUnit(pub usize);

/// A page's position in its step's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepPage(pub usize);

/// A request waiting for a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);
