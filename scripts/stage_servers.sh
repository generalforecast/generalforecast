#!/bin/sh
# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Builds gf-server and gf-router, which only gf runs, and stages them inside
# the package, in generalforecast/bin, off PATH, where maturin packs them.
# Run from the repository root before `maturin build`; gf's release and the
# hosted plane's image build both stage wheels with it. A build for an
# x86-64 level above baseline, named by `-C target-cpu=x86-64-vN` in
# RUSTFLAGS, records it as generalforecast/bin/cpu-level, which importing
# generalforecast checks the CPU against (src/generalforecast/_cpu.py).
set -eu
cargo build --release --locked -p gf-server --bin gf-server -p gf-router --bin gf-router
mkdir -p src/generalforecast/bin
cp target/release/gf-server target/release/gf-router src/generalforecast/bin/
level=$(printf '%s\n' "${RUSTFLAGS:-}" | sed -n 's/.*target-cpu=\(x86-64-v[0-9]\).*/\1/p')
if [ -n "$level" ]; then
    printf '%s\n' "$level" > src/generalforecast/bin/cpu-level
else
    rm -f src/generalforecast/bin/cpu-level
fi
