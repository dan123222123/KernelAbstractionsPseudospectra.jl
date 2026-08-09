#!/usr/bin/env bash
# bench_kernels: strategy × eltype ladder, isolated + end-to-end.
#   $1  VARIANT suffix — for a second run of this experiment on the SAME box (_tuned: the
#       post-probe re-race). The box tag is separate — publish appends it.
#   $2  label shown in the Buildkite section header.
# The caller pins BENCH_ELTYPES when it wants a specific rung set; the default below is the
# full ladder, which is only fair on a full-rate-FP64 card.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

VARIANT="${1:-}"
LABEL="${2:-kernels}"

echo "--- :abacus: $LABEL"
export BENCH_ELTYPES="${BENCH_ELTYPES:-f32,f64,f32x2,f32x4,f64x2,f64x4}"
julia --project=bench bench/bench_kernels.jl cuda || echo "kernels$VARIANT failed (soft_fail)"
if [ -n "$VARIANT" ]; then
  mv -f bench/results/bench_kernels.csv "bench/results/bench_kernels$VARIANT.csv" 2>/dev/null || true
fi
publish "bench_kernels$VARIANT.csv"
