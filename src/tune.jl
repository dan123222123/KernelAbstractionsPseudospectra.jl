# Per-device tuning of the two tiled-solve occupancy knobs, JOINTLY: the trailing-tile column width
# `TC` (the `Val{TC}` of the trailing kernels) and the warps-per-block count `W` (the `Val{W}`) — see
# `tiled_tc` / `tiled_w` in src/ihlpsa_trsm.jl and the occupancy note in
# src/KATRSM.jl/trsm_tiled_kernels.jl.
#
# Both control occupancy and they INTERACT: TC sets the 32×TC @localmem tile size (→ shared-mem-limited
# resident blocks/SM), W sets warps/block (→ register-limited blocks, and W× the warps per resident
# block). The end-to-end sweet spot depends on the element type's arithmetic intensity AND the device's
# compute:bandwidth + register/shared-mem budgets, so it is best MEASURED. `tune_trsm_tc!` times the
# full tiled trsm (`_tiled_trsm!`, which dominates ihlpsa) over the (TC, W) grid for each (element type,
# eye/generic), persists the fastest pair via Preferences (keyed so `tiled_tc`/`tiled_w` pick them up),
# and skips any (TC, W) whose kernel launch fails (e.g. W too large for the register file). Run once on
# the target hardware:
#
#     using KAPseudospectra, CUDA
#     KAPseudospectra.tune_trsm_tc!(CUDABackend(), collect(CUDA.devices())[1])
#
# A timed probe needs no occupancy MODEL — it captures the tradeoff directly — so it is the accurate
# per-device/per-type path on ANY shuffle-capable backend (CUDA/AMDGPU, and oneAPI/Metal with the
# opt-in), regardless of whether `device_smem_per_sm` is overridden there.

# Best-of-`reps` wall-clock seconds for the device closure `f` (one warm-up, then min over reps;
# each timing includes a device synchronize so async GPU work is actually waited on).
function _bestof_solve(f, backend, reps)
    f(); KernelAbstractions.synchronize(backend)        # warm up (JIT compile + caches)
    best = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed (f(); KernelAbstractions.synchronize(backend))
        best = min(best, t)
    end
    return best
end

function tune_trsm_tc!(backend, dev=missing;
        types=(ComplexF32, ComplexF64), m=512, g=2048, reps=3, persist=true)
    KernelAbstractions.isgpu(backend) ||
        error("tune_trsm_tc! needs a GPU backend (the tiled solve is GPU-only); got $(backend).")
    ismissing(dev) || device!(backend, dev)
    bg = get_bgarray(backend)
    chosen = Dict{String,Int}()
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
                    withenv("KAPSEUDO_TRSM_TC" => string(tc), "KAPSEUDO_TRSM_W" => string(w)) do  # force this (TC,W)
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
        _clear_tc_cache!(); _clear_w_cache!()   # so the just-persisted values are picked up this session
        @info "persisted tiled TC widths + W (warps/block) to LocalPreferences.toml — `tiled_tc`/`tiled_w` " *
              "read them now (this session) and in future sessions; ENV overrides still win" chosen
    end
    return chosen
end
