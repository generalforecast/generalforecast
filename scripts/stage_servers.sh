#!/bin/sh
# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the binaries a release wheel carries and stages them where maturin
# packs them: the versioned gf-native client beside Python, and gf-server and
# gf-router, which `gf serve` runs, inside the package, off PATH. Run from the
# repository root before `maturin build`; gf's release and the hosted plane's
# image build both stage wheels with it.
set -eu
cargo build --release --locked -p gf-cli --bin gf -p gf-server --bin gf-server \
    -p gf-router --bin gf-router
mkdir -p wheel-data/scripts src/generalforecast/bin
cp target/release/gf wheel-data/scripts/gf-native
cp target/release/gf-server target/release/gf-router src/generalforecast/bin/
# One expression for GNU and BSD sed alike.
sed 's/^# data = "wheel-data"/data = "wheel-data"/' pyproject.toml >pyproject.staged
mv pyproject.staged pyproject.toml
