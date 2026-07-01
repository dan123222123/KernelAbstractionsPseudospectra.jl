module KAPseudospectra

using Preferences

# Triangular-solve strategy for the GPU ihlpsa inner solve. This is a LOCAL PREFERENCE
# (stored in LocalPreferences.toml via Preferences.jl) so it can be set per-checkout, with
# the env var KAPSEUDO_TRSM as a runtime override that needs no recompile.
#
# The DEFAULT is "column": correct for every element type and every backend, no warp/shuffle
# assumptions. The "tiled" fast path is OPT-IN (for now): it uses hardware warp shuffles, and although
# it SELF-GATES to the always-correct `column` solve wherever the shuffle / shared memory isn't usable
# (so it's safe to request on any backend), `column` stays the default as the fully-validated,
# bitwise-stable baseline. Opt in with `set_trsm_strategy!("tiled")` (or `KAPSEUDO_TRSM=tiled`) once
# you've confirmed it on your hardware. Values:
#   "column" (default) – column-oriented solve: barrier-based, shuffle-free, no per-warp register
#              semantics. Correct for every element type / backend, and the solve that "tiled" falls
#              back to wherever the tiled path isn't usable.
#   "tiled"  – the shared-memory A,B-reuse tiled solve where it's usable for this backend+type, else
#              an automatic fall back to `column`. "Usable" = `warp_trsm_safe(backend, wide)` (the
#              hardware shuffle works here; false on stock oneAPI, Metal-unless-opted-in, and wide
#              non-IEEE types) AND the trailing-update tiles fit shared memory (a wide non-IEEE B≠I
#              pencil needs two tiles and typically overflows). The performant choice for
#              ComplexF32/F64 on CUDA / AMDGPU.
#   "tiled-gemm" – same diagonal panel solve as `tiled`, but the trailing update is a vendor-BLAS
#              `mul!` (cuBLAS/rocBLAS) with the grid points as the GEMM's wide dimension — compute-
#              bound where `tiled` is bandwidth-bound. Gated to ComplexF32/F64 (`tiled_gemm_safe`,
#              which has the fast complex GEMM); MultiFloats / non-IEEE and backends without a complex
#              GEMM auto-fall-back to the `tiled` trailing kernel. B=I in this first cut (B≠I keeps
#              the `tiled` trailing). Same correctness self-gating as `tiled`.
# (The earlier register-warp `@generated` solve was removed: it only beat tiled at small m while
# paying a per-R recompile growing to ~18 s at R=16 / minutes at R=32. The former gate-bypassing
# `tiled` was also dropped — `tiled` now always self-gates; what it means here is the old `auto`.)
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

# ── Intel/oneAPI: force a fixed SIMD32 subgroup so the tiled trsm works at all m ──────
# The tiled solve assumes a fixed 32-lane warp. Intel's IGC instead
# picks the SIMD width (8/16/32) PER KERNEL by register pressure, so a 32-lane workgroup
# can span several subgroups and the warp shuffles silently return garbage past the first
# subgroup. Setting IGC's SIMD32 override makes every kernel one 32-lane subgroup, so the
# shuffles are correct for all m. These env vars are read when the Level-Zero/IGC driver
# initializes, so they must be set BEFORE `using oneAPI`; `__init__` (which runs at
# `using KAPseudospectra`) does that when the opt-in preference is set. Opt-in because it
# forces SIMD32 process-wide (lowering occupancy for kernels that would prefer SIMD16).
# Also requires the oneAPI warp-shuffle backend in KernelIntrinsics.
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

# ── Metal: opt-in for the tiled fast path ──────────────────────────────────────────────
# Unlike oneAPI (where the stock KernelIntrinsics shuffle is a stub), Metal's warp shuffles are
# correct, so this is a policy gate, not a correctness one: it keeps the always-correct `column`
# solve as what Metal's `tiled` strategy falls back to by default, matching how oneAPI requires an
# explicit opt-in, so the fast path isn't silently on for a modest speedup. The Metal extension's
# `warp_trsm_safe` reads this. (The `metal_warp_trsm` preference name is retained for back-compat.)
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
