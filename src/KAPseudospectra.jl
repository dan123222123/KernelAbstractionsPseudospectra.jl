module KAPseudospectra

using Preferences

# Triangular-solve strategy for the GPU ihlpsa inner solve. This is a LOCAL PREFERENCE
# (stored in LocalPreferences.toml via Preferences.jl) so it can be set per-checkout, with
# the env var KAPSEUDO_TRSM as a runtime override that needs no recompile. Values:
#   "auto"   – size-based: register-warp solve for small m, tiled solve for large m
#              (crossover `trsm_crossover()`); the performant default for ComplexF32/F64.
#   "warp"   – always the register-warp (per-warp shuffle) solve.
#   "tiled"  – always the tiled (shared-memory A,B reuse) solve.
#   "column" – the original column-oriented solve: barrier-based, shuffle-free, no per-warp
#              register semantics. Correct for every element type. Non-hardware-float types
#              (MultiFloats / BigFloat / …) are routed here AUTOMATICALLY by `trsmIHL`,
#              overriding any of the above, because the warp/tiled shuffle solves miscompile
#              for them; this value forces it explicitly for the IEEE-float types too.
const _VALID_TRSM = ("auto", "warp", "tiled", "column")

function trsm_strategy()
    s = get(ENV, "KAPSEUDO_TRSM", @load_preference("trsm_strategy", "auto"))
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    return s
end

# m at/above which "auto" switches register-warp → tiled (overridable via KAPSEUDO_TRSM_CROSSOVER).
# Default 512: tiled overtakes the warp solve there, and routing m≥512 to tiled also avoids the
# KA+KI register-warp codegen regression at R=16 (m≈512).
trsm_crossover() = parse(Int, get(ENV, "KAPSEUDO_TRSM_CROSSOVER", "512"))

# Persist the default strategy into LocalPreferences.toml. Changing it triggers a recompile;
# the change takes effect on the next Julia session (use the KAPSEUDO_TRSM env var to switch live).
function set_trsm_strategy!(s::AbstractString)
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    @set_preferences!("trsm_strategy" => s)
    @info "set trsm_strategy local preference to $(repr(s)); restart Julia to take effect (or set ENV[\"KAPSEUDO_TRSM\"] to switch now)"
end
export set_trsm_strategy!

# ─── opt-in GPU-kernel precompilation + on-disk caching ──────────────────────────────
# The warp/tiled GPU triangular-solve kernels are @generated / warp-shuffle kernels with a
# substantial first-call compile cost (seconds, growing with R = ⌈m/32⌉). On Julia versions
# that serialize foreign (GPUCompiler) CodeInstances into pkgimages (JuliaLang/julia#60747,
# milestone 1.13), exercising those kernels in the GPU precompile workload together with
# GPUCompiler's on-disk cache makes that compile persist ACROSS sessions, eliminating their
# TTFP. Off by default: it lengthens precompilation, and the cross-session payoff needs the
# #60747 fix (on older Julia the workload still runs but the GPU code isn't retained yet).
const PRECOMPILE_GPU_KERNELS = @load_preference("precompile_gpu_kernels", false)
const _GPUCOMPILER_UUID = Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55")

"""
    enable_gpu_kernel_cache!(state=true)

Opt into precompiling the warp/tiled GPU triangular-solve kernels and caching their compiled
code on disk, so their first-call compile is paid once (at package precompile) instead of on
the first solve of every session. Sets the `precompile_gpu_kernels` preference and enables
GPUCompiler's `disk_cache`; **restart Julia** to take effect (re-precompiles, slower once).
Full cross-session persistence needs Julia ≥ the release containing JuliaLang/julia#60747
(milestone 1.13); on older Julia the workload still runs but the compiled GPU code is not
retained across sessions yet (harmless — it just won't cut TTFP there).
"""
function enable_gpu_kernel_cache!(state::Bool=true)
    @set_preferences!("precompile_gpu_kernels" => state)
    Preferences.set_preferences!(_GPUCOMPILER_UUID, "disk_cache" => string(state); force=true)
    @info "GPU kernel cache $(state ? "enabled" : "disabled") (precompile_gpu_kernels + GPUCompiler disk_cache). Restart Julia to take effect; cross-session reuse needs Julia ≥ 1.13 (JuliaLang/julia#60747)."
    return nothing
end
export enable_gpu_kernel_cache!

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

# Opt-in (PRECOMPILE_GPU_KERNELS) extension of the GPU precompile workload: compile the warp
# solve for every R up to the crossover (R = ⌈m/32⌉ — i.e. every m the auto/warp paths use it
# for) and the tiled solve once, so their CodeInstances are created during precompilation. With
# JuliaLang/julia#60747 + GPUCompiler's disk cache these persist across sessions. Each launch is
# guarded: a flaky precompile-worker GPU *execution* is tolerated — the kernel still *compiles*
# (the CI is what we need cached), so a failure degrades gracefully instead of breaking precompile.
function _precompile_gpu_kernels(backend, dev, Ts)
    cross = trsm_crossover()
    for T in Ts
        for m in 32:32:cross               # R = 1 … ⌈cross/32⌉, covering every m below the crossover
            try
                _, _, zg = qgrid(T, (-4, 4), (-4, 4), (4, 4))
                P = MatrixPencil(schur(randn(T, m, m)))
                withenv("KAPSEUDO_TRSM" => "warp") do
                    ihlpsa(backend, zg, P, 3; devs=[dev])
                end
            catch err
                @debug "warp precompile skipped" T m exception = err
            end
        end
        try                                # tiled: one size at/above the crossover
            _, _, zg = qgrid(T, (-4, 4), (-4, 4), (4, 4))
            P = MatrixPencil(schur(randn(T, cross, cross)))
            withenv("KAPSEUDO_TRSM" => "tiled") do
                ihlpsa(backend, zg, P, 3; devs=[dev])
            end
        catch err
            @debug "tiled precompile skipped" T exception = err
        end
    end
    return nothing
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
