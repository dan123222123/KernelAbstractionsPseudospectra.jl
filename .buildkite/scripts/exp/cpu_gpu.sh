#!/usr/bin/env bash
# bench_cpu_gpu: our CPU vs our GPU vs EigTool.
# EigTool is probe-and-skip: MATLAB via PATH or environment-modules; EigTool itself is
# shallow-cloned into bench/eigtool if absent. Degrades to NaN columns when either is missing.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :scales: cpu_gpu (+ EigTool)"
if ! command -v matlab >/dev/null 2>&1; then
  command -v module >/dev/null 2>&1 ||
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
      [ -f "$f" ] && . "$f" && break
    done 2>/dev/null || true
  if command -v module >/dev/null 2>&1; then
    MATLAB_MODULE="${MATLAB_MODULE:-matlab-2026a}"   # site-specific; override on the build
    module load "$MATLAB_MODULE" || echo "module load $MATLAB_MODULE failed"
    # The module also prepends MATLAB's BUNDLED CUDA libs (glnxa64) to LD_LIBRARY_PATH,
    # which shadow CUDA.jl's runtime and SEGFAULT julia's GPU init. Keep matlab on PATH
    # but drop its libs from LD_LIBRARY_PATH — the matlab launcher re-establishes its
    # own LD_LIBRARY_PATH for the MATLAB child, so EigTool still runs.
    export LD_LIBRARY_PATH="$(printf %s "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vi matlab | paste -sd: -)"
  else
    echo "no 'module' command available on this agent"
  fi
fi
command -v matlab >/dev/null 2>&1 && echo "matlab resolved: $(command -v matlab)" || echo "matlab NOT found — EigTool leg self-skips"
[ -f bench/eigtool/eigtool.m ] ||
  git clone --depth 1 https://github.com/eigtool/eigtool.git bench/eigtool ||
  echo "EigTool clone failed — the leg will self-skip"
julia --project=bench bench/bench_cpu_gpu.jl cuda || echo "cpu-gpu failed (soft_fail)"
publish bench_cpu_gpu.csv "eigtool_m*" "depthmap_*"   # depthmap_*: standard pencil
