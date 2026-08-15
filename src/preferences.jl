# Preference-backed configuration: trsm strategy, GPU-kernel precompilation opt-in, per-vendor
# tiled-solve opt-ins, and the tuning-profile layer. Persists via LocalPreferences.toml or a
# named profile.

# ---------------------------------------------------------------------------------------------
# Tuning profiles: a tracked file of knobs (four × {ComplexF32,ComplexF64} × {eye,generic}, `wgs`
# a schedule over `m`), selected via:
#
#     KAPSEUDO_TUNE_PROFILE=bench/tuning/a100.toml julia --project=bench bench/bench_kernels.jl cuda
#
# Per-knob resolution order: KAPSEUDO_TRSM_* env scalar > profile > LocalPreferences.toml >
# heuristic. File format is a `[KernelAbstractionsPseudospectra]` table of string-valued knob keys, identical to
# LocalPreferences.toml.
const _PROFILE_LOCK = ReentrantLock()
const _PROFILE_SRC = Ref{String}("")                          # path the cache was parsed from
const _PROFILE = Ref{Union{Nothing, Dict{String, String}}}(nothing)

"""
    tune_profile_path() -> Union{String,Nothing}

Path of the active tuning profile (`KAPSEUDO_TUNE_PROFILE`), or `nothing` when unset.
"""
tune_profile_path() = get(ENV, "KAPSEUDO_TUNE_PROFILE", "") |> p -> isempty(p) ? nothing : p

# Every key a complete probe run writes; distinguishes a full profile from a partial one (e.g.
# tiled-only, leaving the column solve on `_auto_wgs` while still looking tuned).
const _TUNED_TYPES = ("ComplexF32", "ComplexF64")
const _TUNED_KNOBS = ("trsm_tilecols", "trsm_blockwarps", "trsm_warpgridpts", "trsm_wgs")
tuning_keys() = [string(k, "_", T, "_", e)
                 for k in _TUNED_KNOBS for T in _TUNED_TYPES for e in ("eye", "gen")]

# Accepts both a `[KernelAbstractionsPseudospectra]`-wrapped file and a bare key table. Values are stringified
# since the knob resolvers all `tryparse`.
function _parse_profile(path)
    raw = TOML.parsefile(path)
    tbl = get(raw, "KernelAbstractionsPseudospectra", raw)
    return Dict{String, String}(k => string(v) for (k, v) in tbl if !(v isa AbstractDict))
end

"""
    tune_profile() -> Union{Dict{String,String},Nothing}

The active tuning profile's knob table, or `nothing` if `KAPSEUDO_TUNE_PROFILE` is unset. Parsed
once per path and cached; an unreadable or missing file warns once and resolves to `nothing`, so a
bad path degrades to `LocalPreferences.toml` + heuristics rather than failing every solve.
"""
function tune_profile()
    path = tune_profile_path()
    path === nothing && return nothing
    @lock _PROFILE_LOCK begin
        if _PROFILE_SRC[] != path
            _PROFILE_SRC[] = path
            _PROFILE[] = try
                _parse_profile(path)
            catch err
                @warn "KAPSEUDO_TUNE_PROFILE unreadable; falling back to LocalPreferences.toml + heuristics" path exception=err
                nothing
            end
        end
        return _PROFILE[]
    end
end

# Single resolution point for every tuned knob: profile first, then LocalPreferences; returns
# `nothing` (⇒ use the heuristic) if neither has the key. Per-knob env override is handled by the
# caller before this call.
#
# `_RENAMED_KNOBS` maps each knob to its previous key spelling; profiles and LocalPreferences.toml
# written before the rename still parse, read as a fallback with a one-time warning.
const _RENAMED_KNOBS = ("trsm_tilecols" => "trsm_tc", "trsm_blockwarps" => "trsm_w",
    "trsm_warpgridpts" => "trsm_gt")
const _LEGACY_KNOB_WARNED = Set{String}()

function _legacy_knob_key(key::AbstractString)
    for (new, old) in _RENAMED_KNOBS
        startswith(key, new * "_") && return old * key[(length(new) + 1):end]
    end
    return nothing
end

function tuned_knob(key::AbstractString)
    p = tune_profile()
    if p !== nothing
        v = get(p, key, nothing)
        v === nothing || return v
    end
    v = @load_preference(key, nothing)
    v === nothing || return v
    legacy = _legacy_knob_key(key)
    legacy === nothing && return nothing
    lv = p !== nothing ? get(p, legacy, nothing) : nothing
    lv === nothing && (lv = @load_preference(legacy, nothing))
    if lv !== nothing && !(legacy in _LEGACY_KNOB_WARNED)
        push!(_LEGACY_KNOB_WARNED, legacy)
        @warn "tuning key renamed: reading the previous name. Re-run the probe (or rename " *
              "the key) so the profile matches what the resolvers ask for." old=legacy new=key
    end
    return lv
end

"""
    reload_tuning!()

Re-read the tuning profile and drop the resolved-knob caches, so a profile written or switched
mid-session takes effect without restarting Julia. The probes call this after persisting.
"""
function reload_tuning!()
    @lock _PROFILE_LOCK (_PROFILE_SRC[] = ""; _PROFILE[] = nothing)
    _clear_tilecols_cache!()
    _clear_blockwarps_cache!()
    _clear_warpgridpts_cache!()
    _clear_wgs_cache!()
    return nothing
end


# Triangular-solve strategy for the GPU ihlpsa inner solve — a local preference
# (LocalPreferences.toml), with KAPSEUDO_TRSM as a no-recompile runtime override.
# `column` is the always-correct default; `tiled`/`tiled-gemm` self-gate back to it
# where unusable.
const _VALID_TRSM = ("column", "tiled", "tiled-gemm")

# Preference is cached (an uncached @load_preference would re-parse LocalPreferences.toml every
# call). ENV is still consulted live so KAPSEUDO_TRSM keeps working mid-session.
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
it on devices where they are unusable. The `KAPSEUDO_TRSM`
environment variable overrides the persisted value at any time.
"""
function set_trsm_strategy!(s::AbstractString)
    s in _VALID_TRSM || error("invalid trsm strategy $(repr(s)); valid: $(_VALID_TRSM)")
    @set_preferences!("trsm_strategy" => s)
    _TRSM_PREF[] = String(s)
    @info "set trsm_strategy local preference to $(repr(s)); effective immediately (KAPSEUDO_TRSM still overrides)"
end

# Opt-in GPU-kernel precompilation + on-disk caching (needs JuliaLang/julia#60747, milestone
# 1.13, for cross-session persistence; older Julia runs the workload but doesn't retain it).
# Off by default since it lengthens precompilation.
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

Persist whether KernelAbstractionsPseudospectra forces Intel GPUs to SIMD32, so the `tiled` trsm strategy's
warp-shuffle panel solve is correct at every `m` (otherwise IGC may narrow the SIMD width and
the warp shuffles break past the subgroup boundary). Stored in LocalPreferences.toml and applied
from `__init__`; for full effect start a fresh session with `using KernelAbstractionsPseudospectra`
**before** `using oneAPI`. Without it, use `KAPSEUDO_TRSM=column` on Intel.
"""
function set_intel_force_simd32!(flag::Bool)
    @set_preferences!("intel_force_simd32" => flag)
    flag && _apply_intel_simd32!()
    @info "intel_force_simd32 = $flag (LocalPreferences.toml). For full effect, restart Julia and load KernelAbstractionsPseudospectra before oneAPI."
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
