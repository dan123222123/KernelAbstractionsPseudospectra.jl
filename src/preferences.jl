# Preference-backed configuration: trsm strategy, opt-in GPU-kernel precompilation,
# and the per-vendor tiled-solve opt-ins. All persist via LocalPreferences.toml.

# Triangular-solve strategy for the GPU ihlpsa inner solve — a local preference
# (LocalPreferences.toml), with KAPSEUDO_TRSM as a no-recompile runtime override.
# `column` is the always-correct default; `tiled`/`tiled-gemm` self-gate back to it
# where unusable. See DESIGN_TRSM.md for the strategy comparison and routing rules.
const _VALID_TRSM = ("column", "tiled", "tiled-gemm")

# Lazy cache of the persisted preference: trsm_strategy() runs once per Lanczos
# iteration, and an uncached @load_preference re-parses LocalPreferences.toml on
# every call. ENV is still consulted live so KAPSEUDO_TRSM keeps working mid-session.
const _TRSM_PREF = Ref{Union{Nothing, String}}(nothing)

function trsm_strategy()
    s = get(ENV, "KAPSEUDO_TRSM") do
        pref = _TRSM_PREF[]
        pref === nothing ? (_TRSM_PREF[] = @load_preference("trsm_strategy", "column")) :
        pref
    end
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    return s
end

"""
    set_trsm_strategy!(s)

Persist the default GPU triangular-solve strategy (`"column"`, `"tiled"`, or
`"tiled-gemm"`) into LocalPreferences.toml and apply it to the current session.
`"column"` is the always-correct default; `"tiled"`/`"tiled-gemm"` self-gate back to
it on devices where they are unusable (see DESIGN_TRSM.md). The `KAPSEUDO_TRSM`
environment variable overrides the persisted value at any time.
"""
function set_trsm_strategy!(s::AbstractString)
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    @set_preferences!("trsm_strategy" => s)
    _TRSM_PREF[] = String(s)
    @info "set trsm_strategy local preference to $(repr(s)); effective immediately (KAPSEUDO_TRSM still overrides)"
end

# Opt-in GPU-kernel precompilation + on-disk caching. The tiled GPU triangular-solve
# kernels have a non-trivial first-call compile cost. Exercising them in the precompile
# workload plus GPUCompiler's disk cache can make that compile persist across sessions
# (needs JuliaLang/julia#60747, milestone 1.13; on older Julia the workload still runs
# but the compiled code isn't retained). Off by default since it lengthens precompilation.
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
function enable_gpu_kernel_cache!(state::Bool = true)
    @set_preferences!("precompile_gpu_kernels" => state)
    Preferences.set_preferences!(_GPUCOMPILER_UUID, "disk_cache" => string(state); force = true)
    @info "GPU kernel cache $(state ? "enabled" : "disabled") (precompile_gpu_kernels + GPUCompiler disk_cache). Restart Julia to take effect; cross-session reuse needs Julia ≥ 1.13 (JuliaLang/julia#60747)."
    return nothing
end

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
