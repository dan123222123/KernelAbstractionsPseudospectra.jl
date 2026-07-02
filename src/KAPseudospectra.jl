module KAPseudospectra

using Preferences

# Triangular-solve strategy for the GPU ihlpsa inner solve — a local preference
# (LocalPreferences.toml), with KAPSEUDO_TRSM as a no-recompile runtime override.
# `column` is the always-correct default; `tiled`/`tiled-gemm` self-gate back to it
# where unusable. See DESIGN_TRSM.md for the strategy comparison and routing rules.
const _VALID_TRSM = ("column", "tiled", "tiled-gemm")

function trsm_strategy()
    s = get(ENV, "KAPSEUDO_TRSM", @load_preference("trsm_strategy", "column"))
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    return s
end

# Persist the default strategy into LocalPreferences.toml. Changing it triggers a recompile;
# the change takes effect on the next Julia session (use the KAPSEUDO_TRSM env var to switch live).
function set_trsm_strategy!(s::AbstractString)
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    @set_preferences!("trsm_strategy" => s)
    @info "set trsm_strategy local preference to $(repr(s)); restart Julia to take effect (or set ENV[\"KAPSEUDO_TRSM\"] to switch now)"
end
export set_trsm_strategy!

# ─── opt-in GPU-kernel precompilation + on-disk caching ──────────────────────────────
# The tiled GPU triangular-solve kernels have a non-trivial first-call compile cost.
# Exercising them in the precompile workload plus GPUCompiler's disk cache can make that
# compile persist across sessions (needs JuliaLang/julia#60747, milestone 1.13; on older
# Julia the workload still runs but the compiled code isn't retained). Off by default
# since it lengthens precompilation.
const PRECOMPILE_GPU_KERNELS = @load_preference("precompile_gpu_kernels", false)
const _GPUCOMPILER_UUID = Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55")

"""
    enable_gpu_kernel_cache!(state=true)

Opt into precompiling the tiled GPU triangular-solve kernels and caching their compiled code on
disk, so the first-call compile is paid once at package precompile instead of every session.
Sets the `precompile_gpu_kernels` preference and GPUCompiler's `disk_cache`; **restart Julia**
to take effect. Cross-session persistence needs Julia ≥ the release with JuliaLang/julia#60747
(milestone 1.13); on older Julia this is harmless but doesn't cut first-use latency.
"""
function enable_gpu_kernel_cache!(state::Bool=true)
    @set_preferences!("precompile_gpu_kernels" => state)
    Preferences.set_preferences!(_GPUCOMPILER_UUID, "disk_cache" => string(state); force=true)
    @info "GPU kernel cache $(state ? "enabled" : "disabled") (precompile_gpu_kernels + GPUCompiler disk_cache). Restart Julia to take effect; cross-session reuse needs Julia ≥ 1.13 (JuliaLang/julia#60747)."
    return nothing
end
export enable_gpu_kernel_cache!

# Intel/oneAPI: force a fixed SIMD32 subgroup so the tiled trsm's warp shuffles are correct at
# every m (see ext/oneAPIPseudospectra.jl for why). Must be set before `using oneAPI`; `__init__`
# applies it when opted in.
intel_force_simd32() = @load_preference("intel_force_simd32", false)
function _apply_intel_simd32!()
    get!(ENV, "NEOReadDebugKeys", "1")          # enable Intel NEO debug keys
    get!(ENV, "IGC_ForceOCLSIMDWidth", "32")    # force every kernel to SIMD32
    return nothing
end
"""
    set_intel_force_simd32!(flag::Bool)

Persist whether KAPseudospectra forces Intel GPUs to SIMD32, so the `tiled` trsm strategy's
warp-shuffle panel solve is correct at every `m` (otherwise IGC may narrow the SIMD width and
the warp shuffles break past the subgroup boundary). Stored in LocalPreferences.toml and applied
from `__init__`; for full effect start a fresh session with `using KAPseudospectra`
**before** `using oneAPI`. Without it, use `KAPSEUDO_TRSM=column` on Intel.
"""
function set_intel_force_simd32!(flag::Bool)
    @set_preferences!("intel_force_simd32" => flag)
    flag && _apply_intel_simd32!()
    @info "intel_force_simd32 = $flag (LocalPreferences.toml). For full effect, restart Julia and load KAPseudospectra before oneAPI."
    return flag
end
export set_intel_force_simd32!

# Metal: opt-in for the tiled fast path (a policy gate, not a correctness one — Metal's shuffles
# are correct; see ext/MetalPseudospectra.jl). Mirrors the oneAPI opt-in above.
metal_warp_trsm() = @load_preference("metal_warp_trsm", false)
"""
    set_metal_warp_trsm!(flag::Bool)

Persist whether the `tiled` trsm strategy may use the tiled fast path on Metal (default
`false` → `tiled` falls back to the always-correct `column` solve on Metal). Metal's shuffles are
correct, so this is an opt-in for a modest speedup, mirroring oneAPI's `set_intel_force_simd32!`.
Stored in LocalPreferences.toml; restart Julia (or set `KAPSEUDO_TRSM=tiled`) for it to take effect.
"""
function set_metal_warp_trsm!(flag::Bool)
    @set_preferences!("metal_warp_trsm" => flag)
    @info "metal_warp_trsm = $flag (LocalPreferences.toml); restart Julia or set KAPSEUDO_TRSM=tiled to switch now"
    return flag
end
export set_metal_warp_trsm!

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa, ℝsvdpsa

include("backend.jl")   # general per-backend device interface (CPU defaults; GPU exts override)

include("ihlpsa.jl")
export ihlpsa

include("tune.jl")   # tune_trsm_tc!: per-device probe for the tiled trailing-tile width
export set_pdiv_accurate   # Float16/Float32 GPU-solve precision toggle (from KATRSM)

# Build a 2D grid of complex shifts. Returns (gx, gy, zg) where gx and gy are
# real ranges along the real and imaginary axes and zg[i,j] = gx[i] + im*gy[j].
function qgrid(T, tx, ty, gp)
    nx, ny = gp
    gx = range(tx[1], tx[2], length=nx)
    gy = range(ty[1], ty[2], length=ny)
    zg = [T(complex(x, y)) for x in gx, y in gy]
    return collect(gx), collect(gy), zg
end
export qgrid

"""
    psaplot(gx, gy, σ[, eigenvalues]; levels, kwargs...)
    psaplot!(gx, gy, σ[, eigenvalues]; ...)

Plot recipe for a pseudospectra field: contours of `log10.(σ)` over the grid `(gx, gy)` (as
returned by [`qgrid`](@ref) and [`ihlpsa`](@ref)/[`ℂsvdpsa`](@ref)), on a `log₁₀ σ` colorbar and,
if a fourth argument is given, the `eigenvalues` overlaid as diamonds. Pass `levels` and any Plots
attribute (`size`, `line`, `seriestype=:contourf`, …) as keywords.

Requires a plotting stack: the recipe lives in an extension that activates once `RecipesBase` (pulled
in by `Plots`) is loaded. Example: `using Plots; psaplot(gx, gy, srg, eigvals(A); levels=-6:0)`.
"""
function psaplot end
function psaplot! end
export psaplot, psaplot!

# Shared precompile body for the GPU extensions: exercises the fixed and adaptive `ihlpsa`
# paths, for both a B=I and a true B≠I pencil, on one device, so each extension's
# `@compile_workload` traces the backend-specialized method instances.
function _precompile_ihlpsa(backend, dev, Ts)
    for T in Ts
        _, _, zg = qgrid(T, (-4, 4), (-4, 4), (100, 100))
        P = MatrixPencil(schur(randn(T, 32, 32)))   # B = I (single matrix)
        ihlpsa(backend, zg, P, 5; devs=[dev])   # fixed-nit path
        ihlpsa(backend, zg, P; devs=[dev])      # adaptive (per-point hybrid)
        # True matrix pencil (B ≠ I): also trace the generalized-Schur construction
        # path so a first MatrixPencil(A, B) call isn't a cold compile.
        Pg = MatrixPencil(randn(T, 32, 32), randn(T, 32, 32) + T(5) * I)
        ihlpsa(backend, zg, Pg, 5; devs=[dev])
    end
    return nothing
end

# Opt-in (PRECOMPILE_GPU_KERNELS) extension of the GPU precompile workload: exercises the
# `tiled` solve so its CodeInstances get created during precompilation. Guarded: a flaky
# precompile-worker GPU *execution* is tolerated since the kernel still *compiles* (the
# CodeInstance is what we need cached), so a failure degrades gracefully.
function _precompile_gpu_kernels(backend, dev, Ts)
    try
        withenv("KAPSEUDO_TRSM" => "tiled") do
            _precompile_ihlpsa(backend, dev, Ts)
        end
    catch err
        @debug "tiled precompile skipped" exception = err
    end
    return nothing
end

function __init__()
    # Apply the Intel SIMD32 override early (before the oneAPI driver initializes) when
    # opted in. Harmless on non-Intel setups — the env vars are Intel-specific.
    intel_force_simd32() && _apply_intel_simd32!()
end

## precompile gpu code
using PrecompileTools

@setup_workload begin
    using LinearAlgebra
    using KernelAbstractions
    @compile_workload begin
        for T in [ComplexF32, ComplexF64]
            m = 16
            g = 10
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            ℂsvdpsa(zg, P)
            ℝsvdpsa(zg, P)
            ihlpsa(CPU(), zg, P, m)      # fixed-nit path
            ihlpsa(CPU(), zg, P)         # adaptive (per-point hybrid)
        end
    end
end

end
