#!/usr/bin/env bash
# bench_golub: XL pseudospectra portrait of the golub matrix (adaptive, every device).
# Size/grid/window via BENCH_GOLUB_M / BENCH_GOLUB_GRIDN / BENCH_GOLUB_HW.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :framed_picture: golub portrait"
julia --project=bench bench/bench_golub.jl cuda || echo "golub failed (soft_fail)"
publish "golub_*" bench_golub_log.txt
