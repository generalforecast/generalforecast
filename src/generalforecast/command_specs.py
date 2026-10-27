# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The gf command table: groups, commands, admin nouns, their actions and defaults.

The parser, the dispatcher and the renderer all read these tables, so a noun
or an action is named in exactly one place. Kept free of heavy imports so
`gf --help` and argument errors stay fast.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generalforecast.service.boundary import ResourceKind

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
    ("show", "Show one model's task and limits"),
    ("pull", "Download a model's weights to the cache"),
)

CONTEXT_ACTIONS: tuple[tuple[str, str], ...] = (
    ("create", "Create a live context from a config file"),
    ("append", "Append rows from a file"),
    ("label", "Attach target values from a file"),
    ("show", "Show a live context"),
)

# Where `gf serve` listens: the Models API over HTTP.
SERVE_HOST = "127.0.0.1"
SERVE_PORT = 8000
# Where a node or a router listens: tables over Arrow Flight.
TABLE_PORT = 8815
# The hosted environment a command uses when none is named.
DEFAULT_ENV = "production"
# Seconds `push`, `promote` and `rollback` wait for an environment to be ready.
READY_SECONDS = 600.0
# The files `gf admin node init` and `gf admin router init` write by default.
NODE_CONFIG = "generalforecast.toml"
NODE_DATA = "gf-data"
ROUTER_CONFIG = "router.toml"


@dataclass(frozen=True)
class AdminAction:
    """One `gf admin <noun> <action>`.

    `target` names what the NAME argument is, or is None for an action that
    takes none. An action with a `body` reads its request from `--file`; one
    without takes no file. A `destructive` action asks before it runs unless
    `--yes`.
    """

    name: str
    help: str
    target: str | None = None
    body: bool = False
    destructive: bool = False


# `gf admin <noun> <action>` over the API: everything an operator needs and a
# user rarely does.
ADMIN_ACTIONS: dict[str, tuple[AdminAction, ...]] = {
    "key": (
        AdminAction("list", "List API keys"),
        AdminAction("create", "Create an API key from --file", body=True),
        AdminAction("revoke", "Revoke an API key", target="key id", destructive=True),
        AdminAction("rotate", "Rotate an API key", target="key id", destructive=True),
    ),
    "secret": (
        AdminAction("list", "List secrets"),
        AdminAction("get", "Show one secret's metadata", target="secret"),
        AdminAction("create", "Create a secret from --file", body=True),
        AdminAction(
            "rotate", "Rotate a secret from --file", target="secret", body=True
        ),
        AdminAction("delete", "Delete a secret", target="secret", destructive=True),
    ),
    "workspace": (AdminAction("get", "Show the workspace"),),
    "project": (
        AdminAction("list", "List projects"),
        AdminAction("create", "Create a project from --file", body=True),
    ),
    "organization": (
        AdminAction("list", "List organizations"),
        AdminAction("create", "Create an organization from --file", body=True),
    ),
    "sso": (
        AdminAction("get", "Show the single sign-on settings"),
        AdminAction("configure", "Configure single sign-on from --file", body=True),
    ),
    "audit": (AdminAction("list", "List audit events"),),
    "operator": (AdminAction("doctor", "Check operator health"),),
    "source": (
        AdminAction("list", "List pinned sources"),
        AdminAction("pin", "Pin a source from --file", body=True),
    ),
}

# The request field an action's NAME fills, per noun.
NAME_FIELDS = {"key": "id", "secret": "name"}

# Admin nouns that start a server on this machine, and their actions.
ADMIN_SERVE_ACTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "node": (
        ("init", "Write a node's configuration file"),
        ("check", "Check a node's configuration without serving"),
        ("serve", "Serve a node from its configuration"),
    ),
    "router": (
        ("init", "Write a router's configuration file"),
        ("check", "Check a router's configuration and reach its nodes"),
        ("serve", "Route clients to nodes from its configuration"),
    ),
}

# Admin nouns that act on hosted predictors rather than API collections.
ADMIN_PREDICTOR_NOUNS = ("logs", "diff", "delete")

ADMIN_HELP = {
    "key": "API keys",
    "secret": "Project secrets",
    "workspace": "The current workspace",
    "project": "Projects",
    "organization": "Organizations",
    "sso": "Single sign-on settings",
    "audit": "Audit log",
    "operator": "Operator diagnostics",
    "source": "Pinned data sources",
    "node": "Serve a node of a cluster",
    "router": "Route clients to the nodes of a cluster",
    "logs": "Show an environment's logs",
    "diff": "Compare a predictor with a revision",
    "delete": "Delete a hosted predictor",
}

ADMIN_NOUNS = tuple(ADMIN_HELP)


def admin_action(noun: str, action: str) -> AdminAction:
    """The table entry for `gf admin <noun> <action>`."""
    return next(spec for spec in ADMIN_ACTIONS[noun] if spec.name == action)


# The SDK routing tables, which the hosted test suite shares.

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
    "sso": "sso",
    "audit": "audit",
}
