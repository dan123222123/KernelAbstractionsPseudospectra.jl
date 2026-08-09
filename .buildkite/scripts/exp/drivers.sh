#!/usr/bin/env bash
# bench_drivers: fixed vs adaptive, timing.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :vs: drivers (fixed vs adaptive)"
export BENCH_ELTYPES="f32,f64"
julia --project=bench bench/bench_drivers.jl cuda || echo "drivers failed (soft_fail)"
publish "bench_drivers*.csv"
