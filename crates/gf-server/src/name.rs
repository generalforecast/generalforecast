//! Table names, checked once where they enter a server or router.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The longest a table name may be.
const LONGEST: usize = 128;

/// A table's name: 1 to 128 ASCII letters, digits, `_` or `-`, so it is safe
/// as a file name and as the data a log's records are sealed with.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TableName(String);

/// A name that is not a table name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameError(String);

impl fmt::Display for NameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "table names are 1 to {LONGEST} letters, digits, '_' or '-', not {:?}",
            self.0
        )
    }
}

impl std::error::Error for NameError {}

impl TableName {
    /// A table name, if `name` is one.
    ///
    /// # Errors
    ///
    /// If it is empty, too long, or holds another character.
    pub fn new(name: impl Into<String>) -> Result<Self, NameError> {
        let name = name.into();
        let valid = !name.is_empty()
            && name.len() <= LONGEST
            && name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "_-".contains(character));
        if valid {
            Ok(Self(name))
        } else {
            Err(NameError(name))
        }
    }

    /// The name's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TableName {
    type Error = NameError;

    fn try_from(name: String) -> Result<Self, NameError> {
        Self::new(name)
    }
}

impl FromStr for TableName {
    type Err = NameError;

    fn from_str(name: &str) -> Result<Self, NameError> {
        Self::new(name)
    }
}

impl From<TableName> for String {
    fn from(name: TableName) -> Self {
        name.0
    }
}

impl std::ops::Deref for TableName {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<str> for TableName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for TableName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TableName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
