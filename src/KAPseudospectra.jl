module KAPseudospectra

using Preferences

# Triangular-solve strategy for the GPU ihlpsa inner solve — a LOCAL PREFERENCE
# (LocalPreferences.toml via Preferences.jl), with KAPSEUDO_TRSM as a no-recompile
# runtime override. `column` is the default, always-correct baseline; `tiled` and
# `tiled-gemm` are opt-in fast paths that self-gate back to `column` wherever they
# aren't usable on the current backend+type. See DESIGN_TRSM.md for the full
# strategy comparison and routing rules.
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
# The tiled GPU triangular-solve kernels (warp-shuffle panel solve + shared-memory trailing
# update) have a non-trivial first-call compile cost. On Julia versions that serialize foreign
# (GPUCompiler) CodeInstances into pkgimages (JuliaLang/julia#60747, milestone 1.13), exercising
# them in the GPU precompile workload together with GPUCompiler's on-disk cache makes that
# compile persist ACROSS sessions, eliminating their TTFP. Off by default: it lengthens
# precompilation, and the cross-session payoff needs the #60747 fix (on older Julia the workload
# still runs but the GPU code isn't retained yet).
const PRECOMPILE_GPU_KERNELS = @load_preference("precompile_gpu_kernels", false)
const _GPUCOMPILER_UUID = Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55")

"""
    enable_gpu_kernel_cache!(state=true)

Opt into precompiling the tiled GPU triangular-solve kernels and caching their compiled code on
disk, so their first-call compile is paid once (at package precompile) instead of on the first
solve of every session. Sets the `precompile_gpu_kernels` preference and enables GPUCompiler's
`disk_cache`; **restart Julia** to take effect (re-precompiles, slower once). Full cross-session
persistence needs Julia ≥ the release containing JuliaLang/julia#60747 (milestone 1.13); on
older Julia the workload still runs but the compiled GPU code is not retained across sessions
yet (harmless — it just won't cut TTFP there).
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

# Shared precompile body for the GPU extensions. Builds a small problem for each
# element type in `Ts` and exercises the fixed and adaptive `ihlpsa` paths (both the
# B=I and a true B≠I pencil) on one device. Called from inside each extension's
# `@compile_workload` so the backend-specialized method instances are traced and
# cached. `LinearAlgebra` (for `schur`) is in module scope via `svdpsa.jl`.
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

# Opt-in (PRECOMPILE_GPU_KERNELS) extension of the GPU precompile workload: exercise the `tiled`
# solve (the only fast-path strategy left — the register-warp tier and its per-R recompile were
# removed, see DESIGN_TRSM.md) so its CodeInstances are created during precompilation. With
# JuliaLang/julia#60747 + GPUCompiler's disk cache these persist across sessions. The launch is
# guarded: a flaky precompile-worker GPU *execution* is tolerated — the kernel still *compiles*
# (the CI is what we need cached), so a failure degrades gracefully instead of breaking precompile.
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
