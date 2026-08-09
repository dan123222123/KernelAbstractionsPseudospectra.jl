#!/usr/bin/env bash
# bench_baseline: naive GPU baseline — cuBLAS trsm_batched against materialized per-shift triangles vs our
# three strategies. CUDA-only by construction (no portable batched TRSM); the memory-traffic
# argument needs a full-rate-FP64 device. The --counters pass re-runs one launch per leg under
# ncu to get dram__bytes as ground truth for the traffic model the CSV reports analytically.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :vs: baseline (cuBLAS trsm_batched vs KATRSM)"
export BENCH_ELTYPES="f64"
julia --project=bench bench/bench_baseline.jl cuda || echo "baseline failed (soft_fail)"
publish bench_baseline.csv

if command -v ncu >/dev/null 2>&1; then
  BASE_METRICS="dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct"
  # --nvtx captures the leg/… ranges the script emits, so rows can be attributed to a LEG
  # instead of guessed from mangled kernel names (ambiguous — the legs share kernels).
  if BENCH_BASE_MS="${BENCH_BASE_NCU_MS:-512}" ncu --csv --nvtx --metrics "$BASE_METRICS" \
      julia --project=bench bench/bench_baseline.jl cuda --counters \
      > bench/results/ncu_baseline_raw.csv 2> bench/results/ncu_baseline_stderr.log; then
    publish ncu_baseline_raw.csv
  else
    echo "ncu baseline capture failed — skipping (stderr below)."; cat bench/results/ncu_baseline_stderr.log || true
  fi
else
  echo "ncu not found — skipping the baseline counter capture (timing rows still uploaded)."
fi

# nsys for the timeline half — ncu replays kernels in isolation and cannot show launch gaps or
# idle time. Bounded config: a full-sweep trace is gigabytes.
if command -v nsys >/dev/null 2>&1; then
  # nsys wants a scratch dir under /tmp that the agent user cannot create; TMPDIR
  # redirects it into the agent's own build dir.
  export TMPDIR="${TMPDIR:-$PWD/bench/results/nsys_tmp}"; mkdir -p "$TMPDIR"
  # ...and nsys's injection pulls in the SYSTEM libcrypto, which shadows the one Julia
  # bundles and is too old for Julia's libssl — Julia then crashes before reaching a kernel
  # and the capture is empty while nsys itself reports success. Prepending Julia's private
  # libdir makes its own libcrypto win. Same class as the MATLAB LD_LIBRARY_PATH fix in
  # cpu_gpu.sh.
  JL_LIBDIR="$(julia -e 'print(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))' 2>/dev/null || true)"
  [ -n "$JL_LIBDIR" ] && export LD_LIBRARY_PATH="$JL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  BENCH_BASE_MS="${BENCH_BASE_NSYS_MS:-512}" BENCH_REPS=1 \
    nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true \
      --output=bench/results/nsys_baseline --stats=true \
      julia --project=bench bench/bench_baseline.jl cuda \
      > bench/results/nsys_baseline_stats.txt 2>&1 ||
    echo "nsys baseline capture failed (soft_fail) — see nsys_baseline_stats.txt"
  # nsys exits 0 even when the traced app died before issuing a kernel, so a non-zero check
  # is not enough. The report's own 'does not contain CUDA trace data' line is the real signal.
  if grep -q "does not contain CUDA trace data" bench/results/nsys_baseline_stats.txt 2>/dev/null; then
    echo "^^^ nsys produced an EMPTY capture: the traced process died before any kernel ran."
    echo "    Check the top of nsys_baseline_stats.txt for the Julia stacktrace."
  fi
  publish "nsys_baseline*"
else
  echo "nsys not found — skipping the baseline timeline capture."
fi
