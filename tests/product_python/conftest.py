# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from pathlib import Path

import pytest

from generalforecast.python_packages.api import Field, Table, TableSchema
from generalforecast.python_packages.compiler import PythonSpec, compile_python

INPUT = TableSchema((Field("x", "float64"),))
OUTPUT = TableSchema((Field("value", "float64"),))
MODEL = """
from generalforecast.python_packages.api import Table, TableSchema, Field
from small_math import shift
import json
class Model:
    def load(self, artifacts):
        self.bias = json.loads((artifacts / "bias.json").read_text())["bias"]
        self.calls = 0
    def predict(self, table):
        self.calls += 1
        return Table(TableSchema((Field("value", "float64"),)),
                     tuple((shift(row[0], self.bias),) for row in table.rows))
"""


def wheel(bias: float = 0.0) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "small_math.py", f"def shift(x, bias): return x + bias + {bias!r}\n"
        )
        archive.writestr(
            "small_math-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "small_math-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: small_math\nVersion: 1.0\n",
        )
    return stream.getvalue()


@pytest.fixture
def package_root(tmp_path: Path) -> Path:
    (tmp_path / "predictor.py").write_text(MODEL)
    (tmp_path / "small_math-1.0-py3-none-any.whl").write_bytes(wheel())
    (tmp_path / "bias.json").write_text('{"bias":2.0}')
    return tmp_path


@pytest.fixture
def spec() -> PythonSpec:
    return PythonSpec(
        INPUT,
        OUTPUT,
        dependencies=("small_math-1.0-py3-none-any.whl",),
        artifacts=("bias.json",),
    )


@pytest.fixture
def revision(package_root, spec):
    return compile_python(package_root, spec)


@pytest.fixture
def query() -> Table:
    return Table(INPUT, ((1.0,), (4.5,)))
