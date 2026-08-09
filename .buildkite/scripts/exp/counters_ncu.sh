#!/usr/bin/env bash
# ncu hardware counters for bench_kernels' kernels (any CUDA card with sm ≥ 7.0 — pre-Volta gets the
# nvprof leg). One capture per (eltype, m): ncu groups by kernel family and
# profiler_summarize.jl emits one summary shard per capture, concatenated by downstream analysis.
# dram__bytes gives the ground-truth bytes for bench_kernels' analytic model.
# Hard-errors when the tool is missing or the GPU is out of ncu's range.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :microscope: ncu kernel counters"
command -v ncu >/dev/null 2>&1 || {
  echo "ncu (Nsight Compute) not found on this agent — cannot collect counters." >&2
  exit 1
}
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [ -n "$CAP" ] && [ "${CAP%%.*}" -lt 7 ]; then
  echo "compute capability $CAP: ncu refuses pre-Volta GPUs — use the nvprof leg." >&2
  exit 1
fi

NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum"
IFS=',' read -ra ELTS <<<"${BENCH_NCU_ELTYPES:-f32,f32x2,f32x4,f64,f64x2,f64x4}"
IFS=',' read -ra MS <<<"${BENCH_NCU_MS:-64,128,256,512,1024}"
for elt in "${ELTS[@]}"; do
  for msz in "${MS[@]}"; do
    echo "--- :microscope: ncu $elt m$msz"
    if BENCH_ELTYPES="$elt" BENCH_MS="$msz" \
      ncu --csv --metrics "$NCU_METRICS" \
      julia --project=bench bench/bench_kernels.jl cuda --counters \
      >"bench/results/ncu_raw_${elt}_m${msz}.csv" 2>"bench/results/ncu_stderr_${elt}_m${msz}.log"; then
      if [ -s "bench/results/ncu_raw_${elt}_m${msz}.csv" ]; then
        julia --project=bench bench/profiler_summarize.jl \
          "bench/results/ncu_raw_${elt}_m${msz}.csv" "$elt" "$msz" ||
          echo "profiler_summarize $elt m$msz failed (soft_fail)"
      else
        echo "ncu $elt m$msz: empty CSV (ERR_NVGPUCTRPERM — counters blocked?) — skipping."
        cat "bench/results/ncu_stderr_${elt}_m${msz}.log" || true
      fi
    else
      echo "ncu $elt m$msz: exited non-zero — skipping (stderr below)."
      cat "bench/results/ncu_stderr_${elt}_m${msz}.log" || true
    fi
  done
done
publish "ncu_*"
