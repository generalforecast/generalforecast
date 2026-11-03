# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Only the worker and the trainer execute models, and only gf-server serves inference over HTTP.

The worker serves: `generalforecast.worker`, the process that hosts it
(`worker_process`, which gf-server's shards start), the runtime under
`generalforecast.runtime` it runs steps with, and the model definitions under
`generalforecast.models` it runs. The trainer fine-tunes:
`generalforecast.training`, which hands a plane's trainer process a catalog
model's calls and parameters, as `hosting.training`. Nothing else executes
models. Every other module, the engine port included, reaches models through
a worker, so none loads weights for execution, calls a model's contract
functions or imports the worker, and nothing in gf runs the trainer's surface:
only a plane's own trainer process calls it, and the serving side never
imports it. The one Python HTTP server is the control
plane, which forwards Models API calls and runs none.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"
WORKER = (
    "generalforecast.worker",
    "generalforecast.worker_process",
    "generalforecast.runtime",
    "generalforecast.models",
    # A plane's trainer loads the base it fine-tunes, as the worker loads it
    # to run it.
    "generalforecast.training",
)
# What executes models: loading them for execution, the worker and runtime.
EXECUTION = (
    "generalforecast.models.registry",
    "generalforecast.models.inference",
    "generalforecast.worker",
    "generalforecast.runtime",
)
# A model's contract calls, as `functions.fit(...)` or `entry.functions.fit(...)`.
CONTRACT = {"fit", "predict", "forecast", "forecast_given", "embed", "transform"}
# The registry calls that load a model's weights to run them.
LOADING = {"load", "load_ref"}
# The trainer's surface: it loads a model's parameters for a plane's trainer
# process to fine-tune, which needs them in that process.
TRAINER = ("generalforecast.training",)
EXECUTORS = (*WORKER, *TRAINER)
CONTROL_PLANE = "generalforecast.server"
HOSTED = "generalforecast.hosted"


def _within(module: str, packages: tuple[str, ...]) -> bool:
    return any(
        module == package or module.startswith(package + ".") for package in packages
    )


def _modules() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted((SOURCE / "generalforecast").rglob("*.py")):
        parts = path.relative_to(SOURCE).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        yield name, ast.parse(path.read_text(), str(path))


def _imports(module: str, tree: ast.Module) -> Iterator[tuple[int, str]]:
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(
                    [
                        *package.split(".")[: len(package.split(".")) - node.level + 1],
                        base,
                    ]
                ).rstrip(".")
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}"


def test_only_the_worker_imports_model_execution():
    found = [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if not _within(module, EXECUTORS)
        for line, name in _imports(module, tree)
        if _within(name, EXECUTION)
    ]
    assert found == []


def _named(node: ast.expr, name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == name) or (
        isinstance(node, ast.Attribute) and node.attr == name
    )


def test_only_the_worker_calls_model_contract_functions():
    found = [
        f"{module}:{node.lineno} calls functions.{node.func.attr}"
        for module, tree in _modules()
        if not _within(module, WORKER)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in CONTRACT
        and _named(node.func.value, "functions")
    ]
    assert found == []


def test_only_the_worker_loads_models_to_run_them():
    found = [
        f"{module}:{node.lineno} calls registry.{node.func.attr}"
        for module, tree in _modules()
        if not _within(module, EXECUTORS)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOADING
        and _named(node.func.value, "registry")
    ]
    assert found == []


def test_nothing_in_gf_runs_the_trainer_surface():
    """Only a plane's trainer process loads through `training`; gf only re-exports it."""
    found = [
        f"{module}:{node.lineno} calls training.{node.func.attr}"
        for module, tree in _modules()
        if not _within(module, TRAINER)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOADING
        and _named(node.func.value, "training")
    ]
    found += [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if not _within(module, TRAINER)
        for line, name in _imports(module, tree)
        if name in {f"{trainer}.{call}" for trainer in TRAINER for call in LOADING}
    ]
    assert found == []


# What serves predictions in gf: the worker, the serve path, the shared service
# contract and the engine port.
SERVING = (
    *WORKER,
    "generalforecast.serve",
    "generalforecast.service",
    "generalforecast.native",
)


def test_the_serving_side_never_imports_the_trainer():
    """What serves predictions reaches no trainer code."""
    serving = (CONTROL_PLANE, *SERVING)
    found = [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if _within(module, serving)
        for line, name in _imports(module, tree)
        if _within(name, TRAINER) or name == "generalforecast.hosting.training"
    ]
    assert found == []


SERVICE = "generalforecast.service"


def test_the_public_path_imports_nothing_hosted():
    """The public wheel ships the SDK, CLI and serve path without `hosted`."""
    found = [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if not _within(module, (CONTROL_PLANE, HOSTED))
        for line, name in _imports(module, tree)
        if name == HOSTED or name.startswith(HOSTED + ".")
    ]
    assert found == []


def test_the_shared_service_contract_imports_nothing_hosted():
    found = [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if _within(module, (SERVICE,))
        for line, name in _imports(module, tree)
        if name == HOSTED or name.startswith(HOSTED + ".")
    ]
    assert found == []


def test_only_the_control_plane_serves_http_in_python():
    found = [
        f"{module}:{line} imports {name}"
        for module, tree in _modules()
        if module != CONTROL_PLANE
        for line, name in _imports(module, tree)
        if name.startswith("http.server.")
    ]
    assert found == []


def test_the_control_plane_runs_no_inference():
    """Its Models API port forwards to the engine's servers, never runs a model here."""
    found = [
        f"{module}:{node.lineno} builds CatalogService"
        for module, tree in _modules()
        if _within(module, (CONTROL_PLANE, "generalforecast.hosted.runtime"))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CatalogService"
    ]
    assert found == []
