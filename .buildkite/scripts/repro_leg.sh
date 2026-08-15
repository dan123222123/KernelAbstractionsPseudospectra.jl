#!/usr/bin/env bash
# Cross-vendor correctness leg: same deterministic pencil/grid/x₀, fixed driver, F32.
# Usage: repro_leg.sh <amdgpu|oneapi|cuda>
set -euo pipefail
vendor="$1"

unset LD_LIBRARY_PATH

case "$vendor" in
  amdgpu)
    julia --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.add("AMDGPU")'
    export JULIA_NUM_THREADS=1                  # AMDGPU multithread GC bug (see README)
    ;;
  oneapi)
    julia --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.add("oneAPI")'
    # Arc A380 (Xe-HPG) has NO native FP64 — persist pdiv_accurate=false first.
    julia --project=bench -e 'using KernelAbstractionsPseudospectra; set_pdiv_accurate!(false)'
    ;;
  cuda)
    julia --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA")'
    ;;
  *)
    echo "unknown vendor: $vendor" >&2; exit 1
    ;;
esac

julia --project=bench bench/cross_vendor_repro.jl "$vendor"
