module KAPseudospectra

using Preferences

# Triangular-solve strategy for the GPU ihlpsa inner solve. This is a LOCAL PREFERENCE
# (stored in LocalPreferences.toml via Preferences.jl) so it can be set per-checkout, with
# the env var KAPSEUDO_TRSM as a runtime override that needs no recompile.
#
# The DEFAULT is "column": correct for every element type and every backend, no warp/shuffle
# assumptions. The warp/tiled fast paths are OPT-IN — they assume a fixed hardware-shuffled
# 32-lane warp (CUDA/AMDGPU/Metal, or Intel only with the SIMD32 pin) and IEEE element types,
# and an explicit "warp"/"tiled" bypasses the per-backend safety gate, so they ship off-by-
# default to avoid surprising the unaware user. Opt in with `set_trsm_strategy!("auto")` (or
# `KAPSEUDO_TRSM=auto`) once you know your setup is in the supported set. Values:
#   "column" (default) – column-oriented solve: barrier-based, shuffle-free, no per-warp
#              register semantics. Correct for every element type / backend. Non-hardware-float
#              types (MultiFloats / BigFloat / …) are routed here AUTOMATICALLY by `trsmIHL`
#              regardless of the setting, because the warp/tiled shuffle solves miscompile for them.
#   "auto"   – size-based opt-in: register-warp solve for small m, tiled solve for large m
#              (crossover `trsm_crossover()`); the performant choice for ComplexF32/F64 on a
#              supported backend.
#   "warp"   – always the register-warp (per-warp shuffle) solve.
#   "tiled"  – always the tiled (shared-memory A,B reuse) solve.
const _VALID_TRSM = ("column", "auto", "warp", "tiled")

function trsm_strategy()
    s = get(ENV, "KAPSEUDO_TRSM", @load_preference("trsm_strategy", "column"))
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    return s
end

# m at/above which "auto" switches register-warp → tiled (overridable via KAPSEUDO_TRSM_CROSSOVER).
# Default 512: tiled overtakes the warp solve there, and routing m≥512 to tiled also avoids the
# KA+KI register-warp codegen regression at R=16 (m≈512). NOTE: 512 is calibrated on the 6× GTX
# 1080 Ti dev node; the true crossover is device-specific. The env var gives per-checkout
# tuning today; a `tune_trsm_crossover!(backend, dev)` probe that benchmarks warp-vs-tiled and
# writes the winner into LocalPreferences is a planned follow-up (see PR discussion).
trsm_crossover() = parse(Int, get(ENV, "KAPSEUDO_TRSM_CROSSOVER", "512"))

# Persist the default strategy into LocalPreferences.toml. Changing it triggers a recompile;
# the change takes effect on the next Julia session (use the KAPSEUDO_TRSM env var to switch live).
function set_trsm_strategy!(s::AbstractString)
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    @set_preferences!("trsm_strategy" => s)
    @info "set trsm_strategy local preference to $(repr(s)); restart Julia to take effect (or set ENV[\"KAPSEUDO_TRSM\"] to switch now)"
end
export set_trsm_strategy!

# ── Intel/oneAPI: force a fixed SIMD32 subgroup so warp/tiled trsm work at all m ──────
# The register-warp and tiled solves assume a fixed 32-lane warp. Intel's IGC instead
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

Persist whether KAPseudospectra forces Intel GPUs to SIMD32, so the `warp`/`tiled` trsm
strategies are correct at every `m` (otherwise IGC may narrow the SIMD width and the warp
shuffles break past the subgroup boundary). Stored in LocalPreferences.toml and applied
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

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa, ℝsvdpsa

include("ihlpsa.jl")
export ihlpsa
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
