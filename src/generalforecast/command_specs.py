# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The gf command table: groups, their actions and help text.

Kept free of heavy imports so `gf --help` and argument errors stay fast.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generalforecast.deployments.boundary import ResourceKind

COLLECTIONS: "dict[str, tuple[ResourceKind, str]]" = {
    "dataset": ("dataset", "datasets"),
    "source": ("source", "sources"),
    "model": ("model", "models"),
    "job": ("job", "jobs"),
    "package": ("package", "packages"),
    "deploy": ("deployment", "deployments"),
    "key": ("api-key", "keys"),
    "secret": ("secret", "secrets"),
    "operator": ("operator", "operators"),
    "auth": ("workspace", "auth"),
    "workspace": ("workspace", "workspace"),
    "organization": ("workspace", "organizations"),
    "project": ("project", "projects"),
    "membership": ("workspace", "memberships"),
    "sso": ("workspace", "sso"),
    "audit": ("workspace", "audit"),
}

OPERATIONS: dict[str, set[str]] = {
    "dataset": {"list", "get", "create", "mutate", "snapshot"},
    "source": {"list", "pin"},
    "model": {"list", "get", "infer"},
    "job": {"list", "get", "submit", "pause", "resume", "cancel", "retry", "results"},
    "package": {"list", "get", "inspect", "register", "test"},
    "deploy": {"list", "get", "apply", "wait", "score", "effect-status", "effect-wait"},
    "key": {"list", "create", "revoke", "rotate"},
    "secret": {"list", "get", "create", "rotate", "delete"},
    "operator": {"doctor", "backup", "restore-check"},
    "auth": {"status"},
    "workspace": {"get"},
    "organization": {"list", "create"},
    "project": {"list", "create"},
    "membership": {"list", "add", "remove"},
    "sso": {"get", "configure"},
    "audit": {"list"},
}

IDENTITY_GROUPS = {
    "key": "keys",
    "secret": "secrets",
    "auth": "auth",
    "workspace": "workspace",
    "organization": "organizations",
    "project": "projects",
    "membership": "memberships",
    "sso": "sso",
    "audit": "audit",
}

LEGACY_DEPLOY = frozenset({"apply", "wait", "score", "effect-status", "effect-wait"})

# Sections of `gf --help`, in order. Other commands work but stay unlisted.
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Account", ("login", "logout", "whoami")),
    ("Packages", ("init", "check", "push", "predict", "serve")),
    ("Resources", ("job", "dataset", "model", "key")),
)

COMMAND_HELP = {
    "login": "Log in through your browser",
    "logout": "Remove this machine's credentials",
    "whoami": "Show the account, workspace and key in use",
    "init": "Create a package",
    "check": "Validate a package and write gf.lock (offline)",
    "push": "Deploy a package",
    "predict": "Run a prediction",
    "serve": "Serve a package or catalog model on this machine",
    "job": "Batch prediction jobs",
    "dataset": "Hosted datasets",
    "model": "Catalog models",
    "key": "API keys",
    "status": "Show a deployment environment",
    "logs": "Show an environment's logs",
    "diff": "Compare a local package with what is deployed",
    "delete": "Delete a deployment",
    "promote": "Promote a revision to an environment",
    "rollback": "Roll an environment back to an earlier revision",
    "datasets": "Import or write dataset rows",
    "runtimes": "Provision and inspect runtimes",
    "secret": "Workspace secrets",
    "source": "Pinned data sources",
    "package": "Registered packages",
    "deploy": "Deployments (low-level API)",
    "operator": "Operator diagnostics",
    "auth": "Authentication status",
    "workspace": "The current workspace",
    "organization": "Organizations",
    "project": "Projects",
    "membership": "Workspace members",
    "sso": "Single sign-on settings",
    "audit": "Audit log",
}

_ACTIONS = {
    "list": "List {plural}",
    "get": "Show one {singular}",
    "create": "Create {a} {singular} from --file",
    "delete": "Delete {a} {singular}",
    "rotate": "Rotate {a} {singular}",
    "revoke": "Revoke {a} {singular}",
    "submit": "Submit {a} {singular} from --file",
    "pause": "Pause {a} {singular}",
    "resume": "Resume a paused {singular}",
    "cancel": "Cancel {a} {singular}",
    "retry": "Retry a failed {singular}",
    "results": "Show {a} {singular}'s results",
    "infer": "Run a model on the request in --file",
    "mutate": "Apply the write in --file",
    "snapshot": "Read a point-in-time snapshot",
    "pin": "Pin a source from --file",
    "inspect": "Inspect a package from --file",
    "register": "Register a package from --file",
    "test": "Test a registered package",
    "doctor": "Check operator health",
    "backup": "Back up the full store (needs host authority)",
    "restore-check": "Verify a backup restores (needs host authority)",
    "status": "Show authentication status",
    "add": "Add a member from --file",
    "remove": "Remove a member",
    "configure": "Configure from --file",
    "apply": "Apply a deployment from --file",
    "wait": "Wait for a deployment",
    "score": "Score a deployment",
    "effect-status": "Show a deployment effect",
    "effect-wait": "Wait for a deployment effect",
}
_NOUNS = {
    "dataset": ("dataset", "datasets"),
    "source": ("source", "sources"),
    "model": ("model", "catalog models"),
    "job": ("job", "jobs"),
    "package": ("package", "packages"),
    "deploy": ("deployment", "deployments"),
    "key": ("API key", "API keys"),
    "secret": ("secret", "secrets"),
    "organization": ("organization", "organizations"),
    "project": ("project", "projects"),
    "membership": ("member", "members"),
    "audit": ("audit event", "audit events"),
}


def action_help(group: str, action: str) -> str:
    """One line describing `gf <group> <action>`."""
    singular, plural = _NOUNS.get(group, (group, group + "s"))
    a = "an" if singular[0].lower() in "aeiou" else "a"
    return _ACTIONS.get(action, action).format(a=a, singular=singular, plural=plural)
