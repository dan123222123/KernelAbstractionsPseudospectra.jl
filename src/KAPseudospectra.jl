"""
    KAPseudospectra

Backend-agnostic pseudospectra computation on CPUs and GPUs via KernelAbstractions.jl.

Compute `σ_min(zB − A)` fields over grids of shifts `z` with either dense SVD sweeps
([`ℂsvdpsa`](@ref), [`ℝsvdpsa`](@ref)) or the inverse-Lanczos iteration [`ihlpsa`](@ref)
on Schur-factored pencils ([`MatrixPencil`](@ref)), with CUDA/AMDGPU/Metal/oneAPI
support through package extensions and extended precision through MultiFloats.jl.
Build shift grids with [`qgrid`](@ref) and plot results with [`psaplot`](@ref).
"""
module KAPseudospectra

using Preferences
using PrecompileTools
using TOML

include("preferences.jl")   # trsm strategy + GPU-kernel-cache + vendor opt-in preferences
export set_trsm_strategy!, enable_gpu_kernel_cache!
export set_intel_force_simd32!, set_metal_warp_trsm!
export tune_profile_path, tune_profile, reload_tuning!

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa, ℝsvdpsa

include("backend.jl")   # general per-backend device interface (CPU defaults; GPU exts override)

include("ihlpsa.jl")    # also brings in the KATRSM triangular-solve submodule
export ihlpsa

include("tune.jl")   # tune_trsm!: per-device probes for the tiled (tile_cols/block_warps/warp_gridpts) + column (wgs) knobs
export set_pdiv_accurate!   # Float16/Float32 GPU-solve precision toggle (from KATRSM)

include("grid.jl")
export qgrid, psaplot, psaplot!

include("precompile.jl")

function __init__()
    # Apply the Intel SIMD32 override early (before the oneAPI driver initializes) when
    # opted in. Harmless on non-Intel setups — the env vars are Intel-specific.
    intel_force_simd32() && _apply_intel_simd32!()
end

end
