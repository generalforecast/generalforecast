# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A predictor's context selection: how each query's context rows are chosen.

`gf check` resolves the manifest's `selection` against the context's size and
the model's row cap, once, and `gf.lock` records the result. Every path binds
the context with the engine strategy `SelectionInfo.strategy` spells from it:
`gf.Predictor`, `gf serve` and hosted serving read the same rows per query.

A selection is one of three strategies:

- `exact`: every query reads the whole context. A snapshot within the
  model's row cap with no declared selection is exact.
- `retrieved`: a `core` of rows every query shares, drawn `kind`-wise with
  `seed`, plus each query's `neighbors` nearest rows, leaning to the newest
  `recent` rows when the context has times. A declared selection, or a
  snapshot past the cap, is retrieved.
- `adaptive`: exact up to the cap, retrieved past it, resolved by the engine
  for each request's rows. A request or live context with no declared
  selection is adaptive, since its size is only known per call.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from generalforecast.errors import InvalidInput

Strategy = Literal["exact", "adaptive", "retrieved"]
CoreKind = Literal["random", "stratified", "balanced", "diverse"]
STRATEGIES = ("exact", "adaptive", "retrieved")
KINDS = ("stratified", "random", "balanced", "diverse")
# How many of the nearest rows the neighbors are drawn from, per neighbor,
# when the context has times; the engine's adaptive rule uses the same.
RECENT_FACTOR = 4


@dataclass(frozen=True)
class SelectionInfo:
    """A resolved selection, in rows.

    Attributes:
        strategy: `exact`, `adaptive` or `retrieved`.
        core: Rows every query sees; None unless retrieved.
        neighbors: Nearest rows per query; 0 unless retrieved.
        kind: How the core is drawn.
        recent: How many newest rows the nearest-row search leans on.
        seed: The core draw's seed.
    """

    strategy: Strategy = "exact"
    core: int | None = None
    neighbors: int = 0
    kind: str = "stratified"
    recent: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks the counts fit the strategy."""
        if self.strategy not in STRATEGIES:
            raise InvalidInput(
                f"selection strategy `{self.strategy}` is not one of "
                + ", ".join(STRATEGIES),
                param="selection",
            )
        if self.kind not in KINDS:
            raise InvalidInput(
                f"selection kind `{self.kind}` is not one of {', '.join(KINDS)}",
                param="selection.kind",
            )
        for key in ("core", "neighbors", "recent", "seed"):
            value = getattr(self, key)
            if value is not None and (
                type(value) is not int or value < 0 or value > 2**32 - 1
            ):
                raise InvalidInput(
                    f"selection {key} must be a whole number of rows",
                    param=f"selection.{key}",
                )
        if self.strategy != "retrieved" and (self.core or self.neighbors):
            raise InvalidInput(
                f"an {self.strategy} selection has no core or neighbors",
                param="selection",
            )

    def to_dict(self) -> dict[str, Any]:
        """The lock's `selection` mapping."""
        return {
            "strategy": self.strategy,
            "core": self.core,
            "neighbors": self.neighbors,
            "kind": self.kind,
            "recent": self.recent,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SelectionInfo":
        """Reads a lock's `selection` mapping.

        Raises:
            InvalidInput: For unknown keys or counts outside the strategy.
        """
        if type(value) is not dict or set(value) != {
            "strategy",
            "core",
            "neighbors",
            "kind",
            "recent",
            "seed",
        }:
            raise InvalidInput(
                "gf.lock `selection` takes strategy, core, neighbors, kind, recent and seed",
                param="lock",
            )
        return cls(**value)

    def strategy_spec(self) -> dict[str, Any]:
        """The engine strategy, as the JSON a table definition takes.

        `gf.Predictor` and hosted serving bind with it in process, and
        `gf serve` passes it to gf-server's table definition.
        """
        from generalforecast._core import Strategy as EngineStrategy

        if self.strategy == "exact":
            engine = EngineStrategy.exact()
        elif self.strategy == "adaptive":
            engine = EngineStrategy.adaptive()
        elif self.neighbors == 0 and self.recent is None:
            engine = (
                EngineStrategy.subsample(
                    self.core, self.seed, cast(CoreKind, self.kind)
                )
                if self.core
                else EngineStrategy.exact()
            )
        else:
            engine = EngineStrategy.retrieved(
                self.neighbors,
                self.core or 0,
                self.seed,
                cast(CoreKind, self.kind),
                among=self.recent,
            )
        return json.loads(engine.spec())

    def describe(self, rows: int | None) -> str:
        """One line as `gf check` prints it."""
        if self.strategy == "exact":
            shown = "" if rows is None else f"  {rows:,} rows"
            return f"whole table{shown}, no neighbors"
        if self.strategy == "adaptive":
            return "adaptive  the whole context up to the row cap, retrieved past it"
        core = f"core {self.core or 0:,} {self.kind}"
        return f"retrieved  {core}, {self.neighbors:,} nearest per query"


def resolve(
    declared: Any,
    *,
    rows: int | None,
    cap: int,
    timed: bool,
    model: str,
) -> tuple[SelectionInfo, str | None]:
    """The selection a lock records, with the reason `gf check` prints for it.

    Args:
        declared: The manifest's `selection`, or None.
        rows: The context's labeled rows, or None when only calls know them.
        cap: The model's context row cap.
        timed: Whether the context has a time column.
        model: The model, for messages.

    Raises:
        InvalidInput: For counts past the model's row cap.
    """
    seed = declared.seed if declared is not None else 0
    kind = (declared.kind if declared is not None else None) or "stratified"
    given = declared is not None and (
        declared.core is not None or declared.neighbors is not None
    )
    if not given:
        if rows is None:
            return SelectionInfo("adaptive", kind=kind, seed=seed), None
        if rows <= cap:
            return (
                SelectionInfo("exact", kind=kind, seed=seed),
                (
                    f"{rows:,} rows fit under the {cap:,} row cap of {model}, "
                    "so every query sees the whole context"
                ),
            )
        from generalforecast._core import Strategy as EngineStrategy

        spec = json.loads(EngineStrategy.adaptive().resolve(rows, cap, timed).spec())
        core = spec.get("core") or {}
        chosen = SelectionInfo(
            "retrieved",
            core=int(core.get("rows", 0)),
            neighbors=int(spec.get("neighbors", 0)),
            kind=core.get("kind", kind),
            recent=spec.get("among"),
            seed=seed,
        )
        return chosen, (
            f"{rows:,} rows exceed the {cap:,} row cap of {model}, so each query "
            f"sees a {chosen.kind} core of {chosen.core:,} rows and its "
            f"{chosen.neighbors:,} nearest rows"
        )
    assert declared is not None
    core = _count(declared.core, cap)
    neighbors = _count(declared.neighbors, cap)
    if neighbors is None:
        neighbors = max(cap - (core or 0), 0)
    total = (core or 0) + neighbors
    if total > cap:
        raise InvalidInput(
            message=f"selection reads {total:,} rows per query; {model} reads at most {cap:,}",
            param="selection",
            hint="lower `selection.core` or `selection.neighbors`",
            limit=cap,
            actual=total,
        )
    recent = declared.recent
    if recent is None and timed and neighbors:
        recent = neighbors * RECENT_FACTOR
    return (
        SelectionInfo(
            "retrieved",
            core=core or 0,
            neighbors=neighbors,
            kind=kind,
            recent=recent,
            seed=seed,
        ),
        None,
    )


def _count(value: float | None, cap: int) -> int | None:
    """A declared count in rows: a float is a fraction of the cap."""
    if value is None:
        return None
    if type(value) is float:
        return round(value * cap)
    return int(value)


__all__ = ["RECENT_FACTOR", "SelectionInfo", "resolve"]
