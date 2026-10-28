// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics a person reads on a terminal, written the way cargo writes
//! them: the one shape gf-server's and gf-router's stderr take, as Python's
//! `generalforecast.diagnostics` writes the gf CLI's.
//!
//! One `error:` or `warning:` line, its wire code in brackets when it has
//! one, then optionally `  --> path:line:col`, then `note:` lines for why
//! and `help:` lines for what to do instead:
//!
//! ```text
//! error[ConfigurationError]: no configuration at `node.toml`
//! help: write one with `gf admin node init`
//! ```
//!
//! Messages start lowercase and end without a period.

use std::fmt::Write as _;
use std::io::Write as _;

/// How serious a diagnostic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One diagnostic, built up and then written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    severity: Severity,
    code: Option<String>,
    message: String,
    location: Option<String>,
    notes: Vec<String>,
    help: Vec<String>,
}

impl Diagnostic {
    /// An error saying `message`.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// A warning saying `message`.
    #[must_use]
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            code: None,
            message: message.into(),
            location: None,
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    /// The wire code the diagnostic carries in brackets.
    #[must_use]
    pub fn code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Where the diagnostic points, `path:line:col`.
    #[must_use]
    pub fn location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Why, as a `note:` line.
    #[must_use]
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// What to do instead, as a `help:` line.
    #[must_use]
    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// The diagnostic's text, one line each, ending in a newline.
    #[must_use]
    pub fn render(&self) -> String {
        let label = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        let head = match &self.code {
            Some(code) => format!("{label}[{code}]"),
            None => label.to_string(),
        };
        let mut out = String::new();
        labelled(&mut out, &head, &self.message);
        if let Some(location) = &self.location {
            let _ = writeln!(out, "  --> {location}");
        }
        for note in &self.notes {
            labelled(&mut out, "note", note);
        }
        for help in &self.help {
            labelled(&mut out, "help", help);
        }
        out
    }

    /// Writes the diagnostic to standard error.
    pub fn report(&self) {
        let mut stderr = std::io::stderr().lock();
        let _ = stderr.write_all(self.render().as_bytes());
        let _ = stderr.flush();
    }
}

/// A labelled line, its later lines under its first.
fn labelled(out: &mut String, label: &str, text: &str) {
    let mut lines = text.split('\n');
    let _ = writeln!(out, "{label}: {}", lines.next().unwrap_or_default());
    let pad = " ".repeat(label.len() + 2);
    for line in lines {
        let _ = writeln!(out, "{pad}{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::Diagnostic;

    #[test]
    fn diagnostics_are_cargo_shaped_lines() {
        let error = Diagnostic::error("no configuration at `node.toml`")
            .code("ConfigurationError")
            .location("node.toml:3:1")
            .note("the file was moved")
            .help("write one with `gf admin node init`");
        assert_eq!(
            error.render(),
            "error[ConfigurationError]: no configuration at `node.toml`\n  \
             --> node.toml:3:1\nnote: the file was moved\n\
             help: write one with `gf admin node init`\n"
        );
        assert_eq!(
            Diagnostic::warning("listening without a key\non every address").render(),
            "warning: listening without a key\n         on every address\n"
        );
    }
}
