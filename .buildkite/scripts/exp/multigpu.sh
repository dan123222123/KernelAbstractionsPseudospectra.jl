#!/usr/bin/env bash
# bench_multigpu: strong scaling + size sweep (needs multiple GPUs).
# Cap sizes at build time via BENCH_MG_NS / BENCH_SS_N / BENCH_ELTYPES.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :chart_with_upwards_trend: multigpu"
julia --project=bench bench/bench_multigpu.jl cuda || echo "multigpu failed (soft_fail)"
publish strong_scaling.csv size_sweep.csv bench_multigpu_log.txt
