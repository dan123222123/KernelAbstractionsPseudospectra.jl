#!/usr/bin/env bash
# nvprof hardware counters for bench_kernels' kernels — the profiler for pre-Volta CUDA cards (ncu
# refuses sm < 7.0; nvprof was dropped after Volta). One capture per (eltype, m): the
# profiler groups by kernel family and profiler_summarize.jl emits one summary shard per
# capture, concatenated by downstream analysis. Hard-errors when the tool is missing or
# the GPU is out of nvprof's range.
#
# Box quirks handled below: CUDA.jl is pinned to the local toolkit with the module libs
# restored (nvprof's injected CUPTI must resolve against the SAME CUDA the app uses), and
# --openacc-profiling off (the OpenACC hook injection kills julia).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :microscope: nvprof saturation counters"
command -v nvprof >/dev/null 2>&1 || {
  echo "nvprof not found on this agent — cannot collect saturation counters." >&2
  exit 1
}
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [ -n "$CAP" ] && [ "${CAP%%.*}" -ge 8 ]; then
  echo "compute capability $CAP: nvprof does not support this GPU — use the ncu leg." >&2
  exit 1
fi

echo "=== reconcile: restore module libs + pin CUDA.jl to the local toolkit ==="
export LD_LIBRARY_PATH="${BENCH_ORIG_LD_LIBRARY_PATH:-}"
julia --project=bench -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)' ||
  echo "set_runtime_version! failed — nvprof may not resolve CUPTI against the app toolkit"

NVP_METRICS="sm_efficiency,achieved_occupancy,single_precision_fu_utilization,dram_utilization"
IFS=',' read -ra ELTS <<<"${BENCH_NVPROF_ELTYPES:-f32,f32x2,f32x4}"
IFS=',' read -ra MS <<<"${BENCH_NVPROF_MS:-64,128,256,512,1024}"
for elt in "${ELTS[@]}"; do
  for msz in "${MS[@]}"; do
    echo "--- :microscope: nvprof $elt m$msz"
    # nvprof writes its metric CSV to --log-file (NOT stdout); julia stdout → /dev/null,
    # stderr → the stderr log (where a perf-counter block surfaces).
    if BENCH_ELTYPES="$elt" BENCH_MS="$msz" \
      nvprof --openacc-profiling off --unified-memory-profiling off \
      --csv --metrics "$NVP_METRICS" \
      --log-file "bench/results/nvprof_raw_${elt}_m${msz}.csv" \
      julia --project=bench bench/bench_kernels.jl cuda --counters \
      >/dev/null 2>"bench/results/nvprof_stderr_${elt}_m${msz}.log"; then
      if grep -q "Metric" "bench/results/nvprof_raw_${elt}_m${msz}.csv" 2>/dev/null; then
        julia --project=bench bench/profiler_summarize.jl \
          "bench/results/nvprof_raw_${elt}_m${msz}.csv" "$elt" "$msz" ||
          echo "profiler_summarize $elt m$msz failed (soft_fail)"
      else
        echo "nvprof $elt m$msz: no metric rows (counters blocked?) — stderr below:"
        tail -5 "bench/results/nvprof_stderr_${elt}_m${msz}.log" || true
      fi
    else
      echo "nvprof $elt m$msz: exited non-zero — stderr below:"
      tail -5 "bench/results/nvprof_stderr_${elt}_m${msz}.log" || true
    fi
  done
done
# un-pin so the local_toolkit preference cannot leak into later runs of this checkout
julia --project=bench -e 'using CUDA; try; CUDA.reset_runtime_version!(); catch; end' >/dev/null 2>&1 || true
publish "nvprof_*"
