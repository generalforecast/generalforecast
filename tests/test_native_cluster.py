# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The SkyPilot cluster recipe runs a digest-pinned image's own node and router."""

import re
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1] / "deploy" / "native-cluster.yaml"


def test_the_cluster_recipe_requires_a_pinned_image_with_both_servers():
    text = RECIPE.read_text()
    assert ":latest" not in text
    assert re.search(r"^  GF_IMAGE: \"\"$", text, re.MULTILINE)
    assert "*@sha256:" + "?" * 64 + ")" in text
    assert 'l.binary(\\"gf-server\\"); l.binary(\\"gf-router\\")' in text


def test_the_cluster_runs_gfs_node_and_router_commands():
    text = RECIPE.read_text()
    assert '"$GF_IMAGE" \\\n      admin node serve --config /etc/gf/gf.toml' in text
    assert (
        '"$GF_IMAGE" \\\n        admin router serve --config /etc/gf/router.toml'
        in text
    )
