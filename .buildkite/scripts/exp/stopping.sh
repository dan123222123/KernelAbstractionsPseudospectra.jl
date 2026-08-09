#!/usr/bin/env bash
# bench_stopping: stopping-criteria ablation — certified Ritz-residual vs Cauchy retirement at
# nconfirm ∈ {1,2}: replay leg (same-trajectory accuracy/depth vs a dense-SVD oracle) + live leg
# (wall-clock via the unexported criterion kwarg) + ramp torture rows. Pinned to f32/f64 for the
# same reason bench_drivers pins.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :stopwatch: stopping (certified vs cauchy)"
export BENCH_ELTYPES="f32,f64"
julia --project=bench bench/bench_stopping.jl cuda || echo "stopping failed (soft_fail)"
publish "bench_stopping*.csv"
