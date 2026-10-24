# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The gf command table: groups, commands, admin nouns and their help text.

Kept free of heavy imports so `gf --help` and argument errors stay fast.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generalforecast.deployments.boundary import ResourceKind

TAGLINE = "General Forecast: Inference for structured data foundation models"
CLOSING = "See 'gf help <command>' for more information"

# `gf --help`: three groups, one line per command, in this order.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Predictors", ("init", "check", "eval", "serve", "predict")),
    ("Hosted", ("login", "logout", "whoami", "push", "status", "promote", "rollback")),
    ("Resources", ("model", "context", "admin")),
)

COMMAND_HELP = {
    "init": "Write a predictor manifest here",
    "check": "Resolve the schema and write gf.lock",
    "eval": "Score the held-out rows",
    "serve": "Serve a predictor on this machine",
    "predict": "Score rows with a predictor",
    "login": "Log in through your browser",
    "logout": "Remove this machine's credentials",
    "whoami": "Show the account and project in use",
    "push": "Deploy a predictor to an environment",
    "status": "Show a predictor's environments",
    "promote": "Point an environment at another's revision",
    "rollback": "Point an environment at an earlier revision",
    "model": "Catalog models: list, show, pull",
    "context": "Live contexts: create, append, label, show",
    "admin": "Keys, secrets, projects, nodes and routers",
}

COMMANDS = tuple(name for _, names in GROUPS for name in names)

MODEL_ACTIONS: tuple[tuple[str, str], ...] = (
    ("list", "List catalog models, newest first"),
    ("show", "Show one model's task, limits and files"),
    ("pull", "Download a model's weights to the cache"),
)

CONTEXT_ACTIONS: tuple[tuple[str, str], ...] = (
    ("create", "Create a live context from a config file"),
    ("append", "Append rows from a file"),
    ("label", "Attach target values from a file"),
    ("show", "Show a live context"),
)

# `gf admin <noun> <action>`: everything an operator needs and a user rarely does.
ADMIN_HELP = {
    "key": "API keys",
    "secret": "Project secrets",
    "workspace": "The current workspace",
    "project": "Projects",
    "organization": "Organizations",
    "membership": "Workspace members",
    "sso": "Single sign-on settings",
    "audit": "Audit log",
    "operator": "Operator diagnostics",
    "source": "Pinned data sources",
    "node": "Serve a node of a hosted cluster",
    "router": "Serve a router in front of nodes",
    "logs": "Show an environment's logs",
    "diff": "Compare a predictor with a revision",
    "delete": "Delete a hosted predictor",
}

ADMIN_NOUNS = tuple(ADMIN_HELP)

# API-backed admin nouns and their actions, each with one help line.
ADMIN_ACTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "key": (
        ("list", "List API keys"),
        ("create", "Create an API key from --file"),
        ("revoke", "Revoke an API key"),
        ("rotate", "Rotate an API key"),
    ),
    "secret": (
        ("list", "List secrets"),
        ("get", "Show one secret's metadata"),
        ("create", "Create a secret from --file"),
        ("rotate", "Rotate a secret"),
        ("delete", "Delete a secret"),
    ),
    "workspace": (("get", "Show the workspace"),),
    "project": (
        ("list", "List projects"),
        ("create", "Create a project from --file"),
    ),
    "organization": (
        ("list", "List organizations"),
        ("create", "Create an organization from --file"),
    ),
    "membership": (
        ("list", "List members"),
        ("add", "Add a member from --file"),
        ("remove", "Remove a member"),
    ),
    "sso": (
        ("get", "Show the single sign-on settings"),
        ("configure", "Configure single sign-on from --file"),
    ),
    "audit": (("list", "List audit events"),),
    "operator": (("doctor", "Check operator health"),),
    "source": (
        ("list", "List pinned sources"),
        ("pin", "Pin a source from --file"),
    ),
}

# Admin nouns that act on hosted predictors rather than API collections.
ADMIN_PREDICTOR_NOUNS = ("logs", "diff", "delete")
# Admin nouns that start a process on this machine.
ADMIN_SERVE_NOUNS = ("node", "router")

# Actions that take no resource name.
UNNAMED_ACTIONS = {
    "list",
    "create",
    "submit",
    "add",
    "pin",
    "doctor",
    "backup",
    "restore",
}

# -- SDK routing tables, shared with the deployments test-suite -------------

COLLECTIONS: "dict[str, tuple[ResourceKind, str]]" = {
    "context": ("context", "contexts"),
    "predictor": ("predictor", "predictors"),
    "source": ("source", "sources"),
    "model": ("model", "models"),
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
    "context": {"list", "get", "create", "mutate", "snapshot"},
    "predictor": {"list", "get"},
    "source": {"list", "pin"},
    "model": {"list", "get"},
    "key": {"list", "create", "revoke", "rotate"},
    "secret": {"list", "get", "create", "rotate", "delete"},
    "operator": {"doctor"},
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


def action_help(group: str, action: str) -> str:
    """One line describing `gf admin <group> <action>`."""
    for name, text in ADMIN_ACTIONS.get(group, ()):
        if name == action:
            return text
    return action
