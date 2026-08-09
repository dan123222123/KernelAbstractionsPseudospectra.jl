#!/usr/bin/env bash
# bench_drivers --trace: kernel budget + retirement (CUDA.@profile). A fatal CUPTI crash can
# take the whole agent down, so the caller schedules this LAST.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :bar_chart: drivers --trace (kernel budget + retirement)"
julia --project=bench bench/bench_drivers.jl cuda --trace || echo "trace failed (soft_fail)"
publish "bench_drivers_trace*"
