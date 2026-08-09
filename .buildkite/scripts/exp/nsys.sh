#!/usr/bin/env bash
# nsys TIMELINE of the multigpu scaling leg, pre-Volta
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :stopwatch: 1080Ti nsys multi-device timeline"
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found on this agent — skipping the multi-device timeline."
  exit 0
fi

export TMPDIR="${TMPDIR:-$PWD/bench/results/nsys_tmp}"
mkdir -p "$TMPDIR"
JL_LIBDIR="$(julia -e 'print(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))' 2>/dev/null || true)"
[ "$JL_LIBDIR" != "" ] && export LD_LIBRARY_PATH="$JL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BENCH_ELTYPES="${BENCH_NSYS_ELTYPES:-f32}" BENCH_SS_N="${BENCH_NSYS_N:-1024}" \
  BENCH_MG_NS="${BENCH_NSYS_N:-1024}" BENCH_REPS=1 \
  nsys profile --trace=cuda --sample=none --force-overwrite=true --output=bench/results/nsys_multigpu --stats=true julia --project=bench bench/bench_multigpu.jl cuda >bench/results/nsys_multigpu_stats.txt 2>&1 ||
  echo "nsys multigpu capture failed (soft_fail) — see nsys_multigpu_stats.txt"

if grep -q "does not contain CUDA trace data" bench/results/nsys_multigpu_stats.txt 2>/dev/null; then
  echo "^^^ nsys produced an EMPTY capture (see baseline.sh's note): the traced process"
  echo "    died before any kernel ran. Stacktrace at the top of nsys_multigpu_stats.txt."
fi

publish "nsys_multigpu*"
