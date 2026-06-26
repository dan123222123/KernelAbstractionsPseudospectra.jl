# Per-device tuning of the warp↔tiled crossover for the `auto` trsm strategy.
#
# The default crossover (512) is calibrated on one GPU family; the true value is device-specific.
# `tune_trsm_crossover!` times the warp vs tiled solve for increasing m on the given backend/device,
# finds the smallest m where tiled first wins, and persists it via Preferences so `auto` uses a
# device-appropriate crossover. Run once on the target hardware:
#
#     using KAPseudospectra, CUDA
#     KAPseudospectra.tune_trsm_crossover!(CUDABackend(), collect(CUDA.devices())[1])
#
# Requires a backend where the warp/tiled shuffle solves actually run (`warp_trsm_safe == true`):
# on stock oneAPI (the KI shuffle is a TODO stub) or Metal without `set_metal_warp_trsm!`, it errors
# with guidance rather than benchmarking a broken kernel.

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

function tune_trsm_crossover!(backend, dev=missing;
        ms=(64, 128, 256, 512, 1024), g=2048, T=ComplexF32, reps=3, persist=true)
    KernelAbstractions.isgpu(backend) ||
        error("tune_trsm_crossover! needs a GPU backend (warp/tiled are GPU-only); got $(backend).")
    wide = !(real(T) <: Base.IEEEFloat)   # MultiFloats/BigFloat need the per-limb shuffle backend
    warp_trsm_safe(backend, wide) ||
        error("warp/tiled solves are not usable on $(backend) for element type $(T) " *
              "(warp_trsm_safe == false), so there is nothing to tune. Enable them first: on oneAPI via " *
              "set_intel_force_simd32!(true) + a KernelIntrinsics oneAPI shuffle backend; on Metal via " *
              "set_metal_warp_trsm!(true); wide (non-IEEE) types also need the per-limb shuffle extension.")
    ismissing(dev) || device!(backend, dev)
    bg = get_bgarray(backend)
    # Default: tiled never overtook warp in the probed range → keep warp throughout (crossover past it).
    crossover = last(ms) + 1
    @info "tuning trsm warp↔tiled crossover" backend ms g T
    for m in ms
        P = adapt(bg, MatrixPencil(schur(randn(T, m, m))))
        zv = adapt(bg, T(2) .+ T(3 // 10) .* randn(T, g))
        bV = VectorOfSimilarVectors(adapt(bg, reduce(hcat, [randn(T, m) for _ in 1:g])))
        wgs = default_wgs(backend, m)
        # Each solve mutates bV in place. Snapshot the RHS and restore it before the tiled timing so
        # the two runs measure on the same input (timing is data-independent for a triangular solve,
        # but this keeps the measurements independently valid and robust to future data-dependent work).
        bV0 = copy(flatview(bV))
        tw = _bestof_solve(() -> _warp_trsm!(backend, bV, zv, P, wgs), backend, reps)
        flatview(bV) .= bV0
        tt = _bestof_solve(() -> _tiled_trsm!(backend, bV, zv, P, wgs), backend, reps)
        @info "  m=$m" warp_s=tw tiled_s=tt ratio=tw / tt
        if tt < tw
            crossover = m
            break
        end
    end
    if persist
        @set_preferences!("trsm_crossover" => string(crossover))
        @info "persisted trsm_crossover = $crossover to LocalPreferences.toml " *
              "(effective next session, or override now with ENV[\"KAPSEUDO_TRSM_CROSSOVER\"])"
    end
    return crossover
end
