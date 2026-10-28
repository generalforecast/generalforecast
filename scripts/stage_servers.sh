#!/bin/sh
# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Builds gf-server and gf-router, which only gf runs, and stages them inside
# the package, in generalforecast/bin, off PATH, where maturin packs them.
# Run from the repository root before `maturin build`; gf's release and the
# hosted plane's image build both stage wheels with it.
set -eu
cargo build --release --locked -p gf-server --bin gf-server -p gf-router --bin gf-router
mkdir -p src/generalforecast/bin
cp target/release/gf-server target/release/gf-router src/generalforecast/bin/
