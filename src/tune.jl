# Per-device joint tuning of the tiled solve's two occupancy knobs: the trailing-tile column width
# `TC` and the warps-per-block count `W` (see `tiled_tc`/`tiled_w` in src/ihlpsa_trsm.jl; occupancy
# rationale in DESIGN_TRSM.md). `tune_trsm_tc!` times the full tiled trsm over the (TC, W) grid per
# (element type, eye/generic) and persists the fastest pair via Preferences. Run once per target device:
#
#     using KAPseudospectra, CUDA
#     KAPseudospectra.tune_trsm_tc!(CUDABackend(), collect(CUDA.devices())[1])

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

function tune_trsm_tc!(backend, dev = missing;
        types = (ComplexF32, ComplexF64), m = 512, g = 2048, reps = 3, persist = true)
    KernelAbstractions.isgpu(backend) ||
        error("tune_trsm_tc! needs a GPU backend (the tiled solve is GPU-only); got $(backend).")
    ismissing(dev) || device!(backend, dev)
    bg = get_bgarray(backend)
    chosen = Dict{String, Int}()
    @info "tuning tiled trailing-tile width TC + warps-per-block W (joint)" backend types m g reps _TC_CANDIDATES _W_CANDIDATES
    for T in types, eye in (true, false)

        wide = !(real(T) <: Base.IEEEFloat)
        # representative pencil: standard (B=I) for eye, a diagonally-dominant generalized one otherwise
        Phost = eye ? MatrixPencil(schur(randn(T, m, m))) :
                MatrixPencil(randn(T, m, m), randn(T, m, m) + T(5) * I)
        P = adapt(bg, Phost)
        if !(warp_trsm_safe(backend, wide) && tiled_tiles_fit(backend, P))
            @info "  skipping $(T) $(eye ? "eye" : "gen") — tiled not usable on this backend+type"
            continue
        end
        zv = adapt(bg, T(2) .+ T(3 // 10) .* randn(T, g))
        bV = VectorOfSimilarVectors(adapt(bg, reduce(hcat, [randn(T, m) for _ in 1:g])))
        wgs = default_wgs(backend, m)
        bV0 = copy(flatview(bV))            # restore the RHS before each (TC,W) so all runs see one input
        ntiles = eye ? 1 : 2
        best_tc, best_w, best_t = 0, 0, Inf
        for tc in _TC_CANDIDATES
            ntiles * 32 * tc * sizeof(T) <= device_smem_bytes(backend) || continue   # tile must fit a block
            for w in _W_CANDIDATES
                flatview(bV) .= bV0
                t = try
                    withenv("KAPSEUDO_TRSM_TC" => string(tc), "KAPSEUDO_TRSM_W" =>
                        string(w)) do  # force this (TC,W)
                        _bestof_solve(() -> _tiled_trsm!(backend, bV, zv, P, wgs), backend, reps)
                    end
                catch e   # e.g. W too large for the register file → launch failure; skip this combo
                    @info "  $(T) $(eye ? "eye" : "gen") TC=$tc W=$w — skipped ($(sprint(showerror, e)))"
                    Inf
                end
                isfinite(t) && @info "  $(T) $(eye ? "eye" : "gen") TC=$tc W=$w" seconds=t
                t < best_t && ((best_t, best_tc, best_w) = (t, tc, w))
            end
        end
        best_tc == 0 && continue
        tckey, wkey = _tc_pref_key(T, eye), _w_pref_key(T, eye)
        chosen[tckey] = best_tc
        chosen[wkey] = best_w
        if persist
            @set_preferences!(tckey => string(best_tc))
            @set_preferences!(wkey => string(best_w))
        end
    end
    if persist
        _clear_tc_cache!();
        _clear_w_cache!()   # so the just-persisted values are picked up this session
        @info "persisted tiled TC widths + W (warps/block) to LocalPreferences.toml — `tiled_tc`/`tiled_w` " *
              "read them now (this session) and in future sessions; ENV overrides still win" chosen
    end
    return chosen
end
