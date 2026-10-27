# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The public engine tables, on the process's engine with the tiny catalog models."""

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from generalforecast import engine
from generalforecast.engine import LocalEngine
from generalforecast.errors import Conflict, InvalidInput, NotFound

CATALOG = {
    "general-forecast/tabicl-v2-classifier": "tests/tabicl-classifier",
    "general-forecast/chronos-2": "tests/chronos2-catalog-forecaster",
}


@pytest.fixture(scope="module")
def catalog() -> Iterator[None]:
    tiny = LocalEngine(models=CATALOG)

    def local(surface: Any = "local", allow_downloads: bool = False) -> LocalEngine:
        return tiny.gated(surface, allow_downloads)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("generalforecast.engine.local", local)
        patch.setattr("generalforecast.native.estimators._shared", {})
        yield
    tiny.close()


def test_tables_take_unlabeled_classifier_rows_until_labeled(catalog) -> None:
    rows = pd.DataFrame(
        {
            "k": np.arange(32),
            **{f"f{i}": np.random.default_rng(i).normal(size=32) for i in range(3)},
            "y": np.arange(32) % 2,
        }
    )
    pending = rows.tail(4).assign(y=None)
    table = engine.Table(
        "pending",
        model="general-forecast/tabicl-v2-classifier",
        schema=engine.Schema(target="y", id="k"),
        download=False,
    )
    try:
        table.insert(rows.head(28))
        # Rows whose label is not known yet take None (or NaN) for a classifier,
        # as they take NaN for a regressor; they are predicted but not fit on.
        table.insert(pending)
        table.insert(pending.assign(y=np.nan, k=np.arange(100, 104)))
        assert len(table) == 36
        scored = table.predict(pending.drop(columns=["y"]))
        assert scored["k"].tolist() == [28, 29, 30, 31]
        assert set(scored["label"]) <= {0, 1}
        table.label([28, 29, 30, 31], [0, 1, 1, 0])
        with pytest.raises(InvalidInput) as caught:
            table.label([100], [7])
        assert (
            caught.value.message
            == "target value `7` is not one of the classes from context"
        )
        table.label([100, 101, 102, 103], [None, 1, np.nan, 0])
        assert len(table) == 36
    finally:
        table.drop()


def test_series_tables_forecast_every_series_by_default(catalog) -> None:
    def observations(item: str, start: int, seed: int) -> pd.DataFrame:
        times = list(range(start, 20))
        return pd.DataFrame(
            {
                "item": [item] * len(times),
                "t": times,
                "units": np.random.default_rng(seed).normal(size=len(times)),
            }
        )

    table = engine.SeriesTable(
        "every-series",
        model="general-forecast/chronos-2",
        schema=engine.SeriesSchema(
            target="units", time="t", series="item", frequency=1
        ),
        horizon=3,
        download=False,
    )
    try:
        table.insert(observations("shoes", 0, 1))
        table.insert(observations("socks", 4, 2))
        table.insert(observations("shoes", 19, 3))
        everything = table.forecast()
        assert everything["item"].tolist() == ["shoes"] * 3 + ["socks"] * 3
        assert everything["step"].tolist() == [1, 2, 3] * 2
        chosen = table.forecast(["socks"])
        np.testing.assert_allclose(
            chosen["value"].to_numpy(), everything["value"].to_numpy()[3:]
        )
    finally:
        table.drop()


def _rows(count: int = 32, seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": np.arange(count),
            "f0": np.random.default_rng(seed).normal(size=count),
            "f1": (np.arange(count) + seed) % 4,
            "f2": np.random.default_rng(seed + 1).normal(size=count),
            "y": np.where(np.arange(count) % 3 == 0, "no", "yes"),
        }
    )


def test_open_reproduces_an_existing_table_and_predicts(catalog) -> None:
    rows = _rows()
    table = engine.Table(
        "reopened",
        model="general-forecast/tabicl-v2-classifier",
        schema=engine.Schema(target="y", id="k", types={"f1": "categorical"}),
        download=False,
    )
    try:
        table.insert(rows)
        again = engine.Table.open("reopened", download=False)
        assert again.model == table.model
        assert again.schema.target == "y"
        assert again.schema.id == "k"
        assert again.schema.time is None
        assert again.inputs == ("f0", "f1", "f2")
        assert again.schema.types == {
            "f0": "numeric",
            "f1": "categorical",
            "f2": "numeric",
        }
        assert again.task == "classification"
        assert again.classes == ["no", "yes"]
        assert again.selection == table.selection
        assert len(again) == 32
        scored = again.predict(rows.drop(columns=["y"]).head(4))
        assert scored["k"].tolist() == [0, 1, 2, 3]
        assert set(scored["label"]) <= {"no", "yes"}
        assert set(scored.columns) == {"k", "label", "p_no", "p_yes"}
        again.insert(_rows(4, seed=9).assign(k=np.arange(100, 104)))
        again.label([100, 101], ["yes", "no"])
        again.delete([102])
        assert len(again) == 35
        assert len(engine.Table.open("reopened", download=False)) == 35
        # An opened table keeps following its size: no frozen selection, and a
        # table it makes anew is adaptive too.
        assert again._selection is None
        again.drop()
        again.insert(rows)
        assert again._selection is None
        assert json.loads(again._table.describe().strategy.spec())["kind"] == "adaptive"
        assert (
            again.selection is not None and again.selection.describe() == "whole table"
        )
    finally:
        table.drop()


def test_open_raises_not_found_and_names_the_other_kind(catalog) -> None:
    with pytest.raises(NotFound) as missing:
        engine.Table.open("never-made", download=False)
    assert missing.value.message == "table 'never-made' does not exist"
    assert missing.value.hint
    with pytest.raises(NotFound):
        engine.SeriesTable.open("never-made", download=False)

    table = engine.Table(
        "rows-not-series",
        model="general-forecast/tabicl-v2-classifier",
        target="y",
        download=False,
    )
    try:
        table.insert(_rows())
        with pytest.raises(InvalidInput) as wrong:
            engine.SeriesTable.open("rows-not-series", download=False)
        assert wrong.value.message == (
            "table 'rows-not-series' holds rows, not what gf.engine.SeriesTable opens"
        )
        assert (
            wrong.value.hint == "open it with gf.engine.Table.open('rows-not-series')"
        )
    finally:
        table.drop()


def test_a_second_table_of_the_name_raises_conflict_with_a_hint(catalog) -> None:
    first = engine.Table(
        "taken",
        model="general-forecast/tabicl-v2-classifier",
        target="y",
        download=False,
    )
    second = engine.Table(
        "taken",
        model="general-forecast/tabicl-v2-classifier",
        target="y",
        download=False,
    )
    try:
        first.insert(_rows())
        with pytest.raises(Conflict) as caught:
            second.insert(_rows())
        assert caught.value.code == "Conflict"
        assert caught.value.message == "table 'taken' exists"
        assert caught.value.hint == "pass exists=\"open\" or Table.open('taken')"
        assert second._table is None
    finally:
        first.drop()


def test_exists_open_attaches_at_start_up_and_refuses_a_different_table(
    catalog,
) -> None:
    def start(**changes: Any) -> engine.Table:
        settings: dict[str, Any] = {
            "model": "general-forecast/tabicl-v2-classifier",
            "schema": engine.Schema(target="y", id="k"),
            "download": False,
            "exists": "open",
        }
        settings.update(changes)
        return engine.Table("idempotent", **settings)

    first = start()
    try:
        first.insert(_rows())
        second = start()
        second.insert(_rows(8, seed=5).assign(k=np.arange(32, 40)))
        assert len(second) == 40
        assert second.classes == ["no", "yes"]
        assert len(first) == 40

        with pytest.raises(InvalidInput) as target:
            start(schema=engine.Schema(target="k", id="y")).insert(_rows())
        assert (
            target.value.message == "table 'idempotent' exists with target 'y', not 'k'"
        )
        assert target.value.param == "target"
        with pytest.raises(InvalidInput) as role:
            start(schema=engine.Schema(target="y")).insert(_rows())
        assert role.value.message == "table 'idempotent' exists with id 'k', not None"
        with pytest.raises(InvalidInput) as inputs:
            start(schema=engine.Schema(target="y", id="k", exclude=["f2"])).insert(
                _rows()
            )
        assert inputs.value.message == (
            "table 'idempotent' exists with inputs ['f0', 'f1', 'f2'], not ['f0', 'f1']"
        )
        with pytest.raises(InvalidInput) as kind:
            start(
                schema=engine.Schema(target="y", id="k", types={"f1": "categorical"})
            ).insert(_rows())
        assert kind.value.message == (
            "table 'idempotent' exists with kind of f1 'numeric', not 'categorical'"
        )
    finally:
        first.drop()

    with pytest.raises(InvalidInput, match='exists must be "error" or "open"'):
        start(exists="replace")


def _observations(item: str, start: int, seed: int) -> pd.DataFrame:
    times = list(range(start, 20))
    return pd.DataFrame(
        {
            "item": [item] * len(times),
            "t": times,
            "units": np.random.default_rng(seed).normal(size=len(times)),
            "price": np.linspace(1.0, 2.0, len(times)),
        }
    )


def test_series_tables_open_and_attach_with_exists_open(catalog) -> None:
    schema = engine.SeriesSchema(
        target="units", time="t", series="item", frequency=1, known=["price"]
    )
    table = engine.SeriesTable(
        "reopened-series",
        model="general-forecast/chronos-2",
        schema=schema,
        horizon=3,
        download=False,
    )
    try:
        table.insert(_observations("shoes", 0, 1))
        again = engine.SeriesTable.open("reopened-series", download=False)
        assert again.model == table.model
        assert again.schema == engine.SeriesSchema(
            target="units",
            time="t",
            series="item",
            frequency=1.0,
            known=["price"],
            types={"price": "numeric"},
        )
        assert again.horizon == 3
        assert again.grouping == engine.Grouping(alone=True)
        assert len(again) == 20
        again.insert(_observations("socks", 4, 2))
        forecast = again.forecast(["shoes", "socks"])
        assert forecast["item"].tolist() == ["shoes"] * 3 + ["socks"] * 3
        with pytest.raises(InvalidInput) as unnamed:
            again.forecast()
        assert unnamed.value.message == (
            "table reopened-series was opened, so name the series to forecast"
        )
        with pytest.raises(InvalidInput) as wrong:
            engine.Table.open("reopened-series", download=False)
        assert wrong.value.message == (
            "table 'reopened-series' holds series, not what gf.engine.Table opens"
        )

        same = engine.SeriesTable(
            "reopened-series",
            model="general-forecast/chronos-2",
            schema=schema,
            horizon=3,
            download=False,
            exists="open",
        )
        same.insert(_observations("boots", 10, 3))
        assert len(same) == 46
        with pytest.raises(Conflict) as taken:
            engine.SeriesTable(
                "reopened-series",
                model="general-forecast/chronos-2",
                schema=schema,
                horizon=3,
                download=False,
            ).insert(_observations("boots", 10, 3))
        assert taken.value.hint == (
            "pass exists=\"open\" or SeriesTable.open('reopened-series')"
        )
        with pytest.raises(InvalidInput) as horizon:
            engine.SeriesTable(
                "reopened-series",
                model="general-forecast/chronos-2",
                schema=schema,
                horizon=5,
                download=False,
                exists="open",
            ).insert(_observations("boots", 10, 3))
        assert horizon.value.message == (
            "table 'reopened-series' exists with horizon 3, not 5"
        )
        dated = _observations("boots", 10, 3)
        dated["t"] = pd.to_datetime(dated["t"], unit="D")
        with pytest.raises(InvalidInput) as kind:
            engine.SeriesTable(
                "reopened-series",
                model="general-forecast/chronos-2",
                schema=schema,
                horizon=3,
                download=False,
                exists="open",
            ).insert(dated)
        assert kind.value.message == (
            "table 'reopened-series' exists with time kind 'integer', not 'datetime'"
        )
        again.drop()
        again.insert(_observations("shoes", 0, 1))
        assert again.forecast()["item"].tolist() == ["shoes"] * 3
    finally:
        table.drop()


def test_descriptions_carry_a_routed_table_s_partitions() -> None:
    from generalforecast.native.handles import Described

    reply = {
        "name": "routed",
        "model": "tiny",
        "columns": [{"name": "amount", "kind": "numeric"}],
        "classes": 0,
        "outputs": 1,
        "rows": 0,
        "partitions": 4,
        "route_by": {"column": "amount", "kind": "ordered", "bounds": [1, 2, 3]},
    }
    described = Described.of(reply)
    assert described.partitions == 4
    assert described.route_by == reply["route_by"]
    plain = Described.of({**reply, "partitions": None, "route_by": None})
    assert (plain.partitions, plain.route_by) == (1, None)


def test_flight_already_exists_converts_to_conflict() -> None:
    from generalforecast.errors import convert

    mapped = convert(
        ValueError(
            'Flight returned already exists error, with message: table "x" exists'
        )
    )
    assert isinstance(mapped, Conflict)
    assert mapped.message == 'table "x" exists'
    assert isinstance(
        convert(KeyError('Flight returned not found error: table "x" does not exist')),
        NotFound,
    )


def test_open_keeps_the_retrieval_settings_of_an_adaptive_table(catalog) -> None:
    rows = _rows()
    table = engine.Table(
        "kept",
        model="general-forecast/tabicl-v2-classifier",
        schema=engine.Schema(target="y", id="k", types={"f1": "categorical"}),
        selection=engine.Retrieved(core=None, neighbors=0, categoricals="one_hot"),
        download=False,
    )
    try:
        table.insert(rows)
        table.set_selection(None)
        again = engine.Table.open("kept", download=False)
        assert again._selection is None
        assert again._retrieval.categoricals == "one_hot"
        again.drop()
        again.insert(rows)
        described = again._table.describe()
        assert json.loads(described.strategy.spec())["kind"] == "adaptive"
        assert described.retrieval_categoricals == "one_hot"
    finally:
        table.drop()


def _served(*models: str) -> tuple[str, str]:
    """The process engine's Flight endpoint, its workers serving these models."""
    from generalforecast.models.catalog import parse

    local = engine.local()
    for model in models:
        local.load(parse(model))
    return local.serve()


def test_a_client_sees_a_table_another_made_again_as_it_is_now(catalog) -> None:
    from generalforecast.native.remote import RemoteClient

    location, token = _served(
        "general-forecast/tabicl-v2-classifier", "general-forecast/chronos-2"
    )
    first, second = RemoteClient(location, token), RemoteClient(location, token)
    try:
        made = first._create_table(
            "remade",
            model="general-forecast/tabicl-v2-classifier",
            features=2,
            classes=2,
        )
        assert first.table("remade") is made
        second.drop_table("remade")
        second._create_table(
            "remade",
            model="general-forecast/tabicl-v2-classifier",
            features=3,
            classes=2,
        )

        remade = first.table("remade")

        assert remade is not made
        assert len(remade.columns) == 3
        assert first.table("remade") is remade
        series = second._create_series_table(
            "remade-series",
            model="general-forecast/chronos-2",
            series="item",
            time="t",
            target="units",
            step=1,
        )
        assert first.series_table("remade-series").definition == series.definition
        first.drop_table("remade-series")
        first._create_series_table(
            "remade-series",
            model="general-forecast/chronos-2",
            series="item",
            time="t",
            target="units",
            step=2,
        )
        assert second.series_table("remade-series").step == 2
    finally:
        for name in ("remade", "remade-series"):
            first.drop_table(name)
        first.close()
        second.close()


def _calls(kind: type) -> set[str]:
    """A table class's public methods, its properties left out."""
    import inspect

    return {
        name
        for name, value in inspect.getmembers(kind)
        if not name.startswith("_") and inspect.isfunction(value)
    }


def test_async_tables_wrap_every_call_of_their_twins() -> None:
    from generalforecast.native.remote import (
        AsyncRemoteSeriesTable,
        AsyncRemoteTable,
        RemoteSeriesTable,
        RemoteTable,
    )

    assert _calls(RemoteTable) <= set(dir(AsyncRemoteTable))
    assert _calls(RemoteSeriesTable) <= set(dir(AsyncRemoteSeriesTable))
    assert AsyncRemoteTable.predict.__doc__ == RemoteTable.predict.__doc__


def test_async_tables_run_their_twins_calls(catalog) -> None:
    import asyncio

    from generalforecast.native.remote import AsyncRemoteClient, RemoteClient

    location, token = _served("general-forecast/tabicl-v2-classifier")
    features = np.random.default_rng(0).normal(size=(16, 2)).astype(np.float32)
    targets = (np.arange(16) % 2).astype(np.int32)

    async def run() -> tuple[np.ndarray, np.ndarray, int]:
        async with AsyncRemoteClient(RemoteClient(location, token)) as client:
            table = await client._create_table(
                "awaited",
                model="general-forecast/tabicl-v2-classifier",
                features=2,
                classes=2,
            )
            try:
                await table.insert(np.arange(16), features, targets)
                await table.delete([3])
                return (
                    await table.predict(features[:2]),
                    await table.predict(features[:2], quantiles=None),
                    await table.count(),
                )
            finally:
                await client.drop_table("awaited")

    predicted, again, count = asyncio.run(run())

    assert predicted.shape == (2, 2) and count == 15
    np.testing.assert_array_equal(predicted, again)
