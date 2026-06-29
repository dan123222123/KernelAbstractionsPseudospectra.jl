# Per-device tuning of the tiled solve's trailing-tile column width `TC` (the `Val{TC}` of the
# trailing kernels — see `tiled_tc` in src/ihlpsa_trsm.jl and the occupancy note in
# src/KATRSM.jl/trsm_tiled_kernels.jl).
#
# `tiled_tc` defaults to a fixed-target analytic estimate, but the true occupancy sweet spot depends
# on the element type's arithmetic intensity AND the device's compute:bandwidth ratio, so it is best
# MEASURED. `tune_trsm_tc!` times the full tiled trsm (`_tiled_trsm!`, which is what dominates ihlpsa)
# at each candidate `TC` for each (element type, eye/generic) and persists the fastest via
# Preferences, keyed so `tiled_tc` picks it up. Run once on the target hardware:
#
#     using KAPseudospectra, CUDA
#     KAPseudospectra.tune_trsm_tc!(CUDABackend(), collect(CUDA.devices())[1])
#
# A timed probe needs no occupancy MODEL — it captures the compute+bandwidth tradeoff directly — so
# it is the accurate per-device/per-type path on ANY shuffle-capable backend (CUDA/AMDGPU, and
# oneAPI/Metal with the opt-in), regardless of whether `device_smem_per_sm` is overridden there.

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
    @info "tuning tiled trailing-tile width TC" backend types m g reps
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
        bV0 = copy(flatview(bV))            # restore the RHS before each TC so all runs see one input
        ntiles = eye ? 1 : 2
        best_tc, best_t = 0, Inf
        for tc in _TC_CANDIDATES
            ntiles * 32 * tc * sizeof(T) <= device_smem_bytes(backend) || continue   # tile must fit a block
            flatview(bV) .= bV0
            t = withenv("KAPSEUDO_TRSM_TC" => string(tc)) do                          # force this TC
                _bestof_solve(() -> _tiled_trsm!(backend, bV, zv, P, wgs), backend, reps)
            end
            @info "  $(T) $(eye ? "eye" : "gen") TC=$tc" seconds=t
            t < best_t && ((best_t, best_tc) = (t, tc))
        end
        best_tc == 0 && continue
        key = _tc_pref_key(T, eye)
        chosen[key] = best_tc
        persist && @set_preferences!(key => string(best_tc))
    end
    persist && _clear_tc_cache!()   # so the just-persisted values are picked up this session
    persist && @info "persisted tiled TC widths to LocalPreferences.toml — `tiled_tc` reads them now " *
                     "(this session) and in future sessions; ENV[\"KAPSEUDO_TRSM_TC\"] still overrides" chosen
    return chosen
end
