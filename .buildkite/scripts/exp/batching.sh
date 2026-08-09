#!/usr/bin/env bash
# bench_batching: batch-width sweep + zpd/workspace sweep.
# BENCH_ELTYPES is pinned to f64.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :package: batching (batch sweep + zpd sweep)"
export BENCH_ELTYPES="f64"
julia --project=bench bench/bench_batching.jl cuda || echo "batching failed (soft_fail)"
publish bench_batching.csv bench_zpd.csv
