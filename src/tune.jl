# Per-device probes for the solve knobs. `tune_trsm_tiled!` picks one occupancy triple (TC,W,gt)
# for the tiled trailing kernel; `tune_trsm_wgs!` picks the column solve's workgroup size as a
# schedule over `m` (its optimum moves; the tiled triple's doesn't). `tune_trsm!` runs both. Run
# once per target device:
#
#     using KernelAbstractionsPseudospectra, CUDA
#     KernelAbstractionsPseudospectra.tune_trsm!(CUDABackend(), collect(CUDA.devices())[1])

# Margin a candidate must win by to displace the incumbent, so sub-percent noise doesn't flip
# the persisted choice. Ties go to whichever candidate is listed first.
const _TUNE_TIE_TOL = 0.03

_beats(t, best) = t < best * (1 - _TUNE_TIE_TOL)

# Best-of-`reps` wall-clock seconds for the device closure `f` (one warm-up, then min over reps;
# each timing includes a device synchronize so async GPU work is actually waited on).
function _bestof_solve(f, backend, reps)
    f();
    KernelAbstractions.synchronize(backend)        # warm up (JIT compile + caches)
    best = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed (f(); KernelAbstractions.synchronize(backend))
        best = min(best, t)
    end
    return best
end

# Representative pencil for a probe: standard (B=I) for eye, diagonally dominant generalized
# otherwise. Extended-precision types reduce at ComplexF64 and round to T (GenericSchur converges
# poorly at MultiFloat for a throughput-only probe) — the same promoted reduction the bench
# harness uses for its generalized MultiFloat rungs.
function _probe_pencil(T, m, eye)
    if real(T) <: Base.IEEEFloat
        return eye ? MatrixPencil(schur(randn(T, m, m))) :
               MatrixPencil(randn(T, m, m), randn(T, m, m) + T(5) * I)
    end
    if eye
        F = schur(randn(ComplexF64, m, m))
        S = T.(F.T)
        B = Diagonal(ones(T, m))
        return SchurMatrixPencil{T, true}(S, collect(S'), B, B, Matrix(T.(F.Z)))
    end
    F = schur(randn(ComplexF64, m, m), randn(ComplexF64, m, m) + ComplexF64(5) * I)
    S, B = T.(F.S), T.(F.T)
    return SchurMatrixPencil{T, false}(S, collect(S'), B, collect(B'), Matrix(T.(F.Z)))
end

# Batch width a real sweep would use at this size (via `findmaxbatchihl`, the driver's `zpd`
# source); wgs is sensitive to batch width. Capped so probing many candidates stays fast, floored
# so a near-full device still gets a representative batch.
function _probe_g(backend, P; nit = 40, cap = 8192, min_g = 512)
    clamp(findmaxbatchihl(backend, P, nit), min_g, cap)
end

# Informational only, never read back by the resolvers — records what a tune was run with, so a
# preference carried to different hardware isn't indistinguishable from a good one.
_tune_meta(note) = @set_preferences!("trsm_tune_meta" => note)

# Joint probe of the tiled trailing kernel's three occupancy knobs, at one `m`. Persists a single
# triple per (element type, eye/generic); probe at your working size if the optimum matters there.
function tune_trsm_tiled!(backend, dev = missing;
        types = (ComplexF32, ComplexF64), m = 512, g = missing, reps = 3, persist = true)
    KernelAbstractions.isgpu(backend) ||
        error("tune_trsm_tiled! needs a GPU backend (the tiled solve is GPU-only); got $(backend).")
    ismissing(dev) || device!(backend, dev)
    bg = get_bgarray(backend)
    chosen = Dict{String, String}()
    @info "tuning tiled trailing-tile width TC + warps-per-block W + grid-points-per-warp gt (joint)" backend types m g reps _TILECOLS_CANDIDATES _BLOCKWARPS_CANDIDATES _WARPGRIDPTS_CANDIDATES _TUNE_TIE_TOL
    for T in types, eye in (true, false)
        wide = !(real(T) <: Base.IEEEFloat)
        P = adapt(bg, _probe_pencil(T, m, eye))
        if !(warp_trsm_safe(backend, wide) && tiled_tiles_fit(backend, P))
            @info "  skipping $(T) $(eye ? "eye" : "gen") — tiled not usable on this backend+type"
            continue
        end
        gm = ismissing(g) ? _probe_g(backend, P) : g   # NB: `gt` below is the tile-reuse knob
        zv = adapt(bg, T(2) .+ T(3 // 10) .* randn(T, gm))
        bV = VectorOfSimilarVectors(adapt(bg, reduce(hcat, [randn(T, m) for _ in 1:gm])))
        wgs = column_wgs(backend, P)
        bV0 = copy(flatview(bV))            # restore the RHS before each (TC,W,gt) so all runs see one input
        ntiles = eye ? 1 : 2
        best_tc, best_w, best_gt, best_t = 0, 0, 0, Inf
        for tc in _TILECOLS_CANDIDATES
            ntiles * tile_warp_rows(backend) * tc * sizeof(T) <= device_smem_bytes(backend) ||
                continue   # tile must fit a block
            for w in _BLOCKWARPS_CANDIDATES, gt in _WARPGRIDPTS_CANDIDATES
                flatview(bV) .= bV0
                t = try
                    withenv("KAPSEUDO_TRSM_TILECOLS" => string(tc), "KAPSEUDO_TRSM_BLOCKWARPS" => string(w),
                        "KAPSEUDO_TRSM_WARPGRIDPTS" => string(gt)) do  # force this (TC,W,gt)
                        _bestof_solve(() -> _tiled_trsm!(backend, bV, zv, P, wgs), backend, reps)
                    end
                catch e   # e.g. W too large for the register file → launch failure; skip this combo
                    @info "  $(T) $(eye ? "eye" : "gen") TC=$tc W=$w gt=$gt — skipped ($(sprint(showerror, e)))"
                    Inf
                end
                isfinite(t) && @info "  $(T) $(eye ? "eye" : "gen") TC=$tc W=$w gt=$gt" seconds=t
                _beats(t, best_t) && ((best_t, best_tc, best_w, best_gt) = (t, tc, w, gt))
            end
        end
        best_tc == 0 && continue
        tckey, wkey, gtkey = _tilecols_pref_key(T, eye), _blockwarps_pref_key(T, eye), _warpgridpts_pref_key(T, eye)
        chosen[tckey] = string(best_tc)
        chosen[wkey] = string(best_w)
        chosen[gtkey] = string(best_gt)
        if persist
            @set_preferences!(tckey => chosen[tckey])
            @set_preferences!(wkey => chosen[wkey])
            @set_preferences!(gtkey => chosen[gtkey])
        end
    end
    if persist
        reload_tuning!()   # so the just-persisted values are picked up this session
        @info "persisted tiled TC widths + W (warps/block) + gt (grid points/warp) to LocalPreferences.toml — " *
              "`tile_cols`/`block_warps`/`warp_gridpts` read them now (this session) and in future sessions; " *
              "ENV overrides still win" chosen
    end
    return chosen
end

# Probe of the column solve's workgroup size, swept over `m` (its optimum moves, unlike the
# tiled triple) at the batch width the driver would pick for each `m`. Persists one
# piecewise-constant schedule per (element type, eye/generic); see `_parse_wgs_schedule` for the
# format.
function tune_trsm_wgs!(backend, dev = missing;
        types = (ComplexF32, ComplexF64), ms = (128, 256, 512, 1024), g = missing,
        reps = 3, persist = true)
    KernelAbstractions.isgpu(backend) ||
        error("tune_trsm_wgs! needs a GPU backend (wgs is 1 on CPU); got $(backend).")
    ismissing(dev) || device!(backend, dev)
    bg = get_bgarray(backend)
    chosen = Dict{String, String}()
    @info "tuning column-solve workgroup size wgs (per element type + eye/generic, over m)" backend types ms g reps _WGS_CANDIDATES _TUNE_TIE_TOL
    for T in types, eye in (true, false)
        sched = Tuple{Int, Int}[]
        for m in ms
            P = adapt(bg, _probe_pencil(T, m, eye))
            gm = ismissing(g) ? _probe_g(backend, P) : g
            zv = adapt(bg, T(2) .+ T(3 // 10) .* randn(T, gm))
            bV = VectorOfSimilarVectors(adapt(bg, reduce(hcat, [randn(T, m) for _ in 1:gm])))
            bV0 = copy(flatview(bV))        # restore before each candidate so all see one input
            best_wgs, best_t = 0, Inf
            for wgs in _WGS_CANDIDATES
                wgs <= m || continue        # wider than the triangle only idles lanes
                flatview(bV) .= bV0
                t = try
                    _bestof_solve(() -> _column_trsm!(backend, bV, zv, P, wgs), backend, reps)
                catch e   # e.g. wgs above the device's max workgroup size → launch failure
                    @info "  $(T) $(eye ? "eye" : "gen") m=$m g=$gm wgs=$wgs — skipped ($(sprint(showerror, e)))"
                    Inf
                end
                isfinite(t) && @info "  $(T) $(eye ? "eye" : "gen") m=$m g=$gm wgs=$wgs" seconds=t
                _beats(t, best_t) && ((best_t, best_wgs) = (t, wgs))
            end
            # Record a breakpoint only where the winner changes, so a stable optimum persists as
            # one entry rather than one per probed size.
            best_wgs == 0 && continue
            (isempty(sched) || last(last(sched)) != best_wgs) && push!(sched, (m, best_wgs))
        end
        isempty(sched) && continue
        key = _wgs_pref_key(T, eye)
        chosen[key] = _format_wgs_schedule(sched)
        persist && @set_preferences!(key => chosen[key])
    end
    if persist
        reload_tuning!()   # so the just-persisted schedules are picked up this session
        @info "persisted column-solve wgs schedules to LocalPreferences.toml — `column_wgs` reads " *
              "them now (this session) and in future sessions; ENV overrides still win" chosen
    end
    return chosen
end

# Tune both strategies via separate probes, one entry point (`column` ships as default; `tiled`/
# `tiled-gemm` are raced against it). `profile`, when given, also writes the result as a tuning
# profile (see `write_tune_profile`) so one probe run serves every later run on the box.
function tune_trsm!(backend, dev = missing;
        types = (ComplexF32, ComplexF64), m = 512, wgs_ms = (128, 256, 512, 1024),
        g = missing, reps = 3, persist = true, profile = nothing, device = "",
        extra = Dict{String, String}())
    ismissing(dev) || device!(backend, dev)
    # A profile outranks LocalPreferences, so persisted results would stay shadowed until it's
    # unset or rewritten.
    persist && tune_profile_path() !== nothing &&
        @warn "tuning with KAPSEUDO_TUNE_PROFILE set: the profile outranks the values this probe " *
              "persists, so they will not take effect until it is unset or rewritten" profile=tune_profile_path()
    tiled = tune_trsm_tiled!(backend; types, m, g, reps, persist)
    wgs = tune_trsm_wgs!(backend; types, ms = wgs_ms, g, reps, persist)
    persist && _tune_meta("tiled(m=$m) wgs(ms=$(collect(wgs_ms))) " *
                  "g=$(ismissing(g) ? "auto (findmaxbatchihl)" : g) tie_tol=$_TUNE_TIE_TOL")
    chosen = (; tiled, wgs)
    # `extra` lands in the file but not the return value (per-machine calibration, not a solve
    # knob — e.g. the contention canary's idle baseline). `device` is passed in rather than
    # queried: it's a bench-harness concern (`bench/backends/<vendor>.jl`'s `device_name`) —
    # querying it from the package raised `UndefVarError` on every GPU tested.
    profile === nothing || write_tune_profile(profile, (; tiled, wgs, extra); device)
    return chosen
end

# TOML bare keys allow only [A-Za-z0-9_-]; a MultiFloat knob key doesn't (braces/space/comma) and
# would emit an unparseable profile bare. Quote whatever isn't bare.
_toml_key(k::AbstractString) = occursin(r"^[A-Za-z0-9_-]+$", k) ? String(k) : repr(String(k))

"""
    write_tune_profile(path, chosen; device="")

Write the knobs a probe just chose to `path` as a tuning profile — the tracked, per-machine file a
later run selects with `KAPSEUDO_TUNE_PROFILE`. `chosen` is what [`tune_trsm!`](@ref) returns (or
any dict of knob keys).

Warns when the result is missing keys from [`tuning_keys`](@ref) (e.g. a probe that skipped the
column sweep leaves `wgs` on the `_auto_wgs` heuristic while the file still reads as tuned).
"""
function write_tune_profile(path::AbstractString, chosen; device::AbstractString = "")
    tbl = Dict{String, String}()
    for d in (chosen isa NamedTuple ? values(chosen) : (chosen,)), (k, v) in d
        tbl[String(k)] = string(v)
    end
    absent = setdiff(tuning_keys(), keys(tbl))
    isempty(absent) ||
        @warn "PARTIAL tuning profile: these knobs stay on their heuristic defaults" path absent
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "# KernelAbstractionsPseudospectra tuning profile — generated by tune_trsm!, do not hand-edit")
        println(io, "# lightly. Select with KAPSEUDO_TUNE_PROFILE=", path)
        isempty(device) || println(io, "# device: ", device)
        println(io, "# written: ", Libc.strftime("%Y-%m-%d %H:%M:%S", time()))
        isempty(absent) || println(io, "# PARTIAL — missing: ", join(sort(absent), " "))
        println(io)
        println(io, "[KernelAbstractionsPseudospectra]")
        for k in sort(collect(keys(tbl)))
            println(io, _toml_key(k), " = ", repr(tbl[k]))
        end
    end
    @info "wrote tuning profile" path nkeys=length(tbl) partial=!isempty(absent)
    return path
end
