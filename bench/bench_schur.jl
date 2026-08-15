# Host factorization vs the GPU sweep it enables, at LARGE m. ihlpsa needs a one-time O(m³)
# host reduction (Schur `gees` for a standard pencil, QZ `gges` for (A, B)) before any grid
# point solves. This times that against the converged adaptive sweep and reports the
# grid-free BREAK-EVEN grid size g* = factor_s / per_point_s — past g* points the
# factorization is amortized below the sweep cost. Single device (multi-GPU would confound
# the per-point rate); generalized rows use B = kms (deterministic SPD Toeplitz). Sizes are
# LARGE (BENCH_MS_SCHUR, default 1024,2048,4096; QZ at 8192 alone is tens of minutes).
#   JULIA_NUM_THREADS=auto julia --project=bench bench/bench_schur.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))

backend = select_backend(ARGS)
gpu = KernelAbstractions.isgpu(backend)
devs = gpu ? [first(KernelAbstractionsPseudospectra.devices(backend))] : missing

# ComplexF64 is the reported eltype; f32 is the GPU-performance story only.
const T = get(ENV, "BENCH_SCHUR_ELTYPE", "f64") == "f32" ? ComplexF32 : ComplexF64
const MS = env_ints("BENCH_MS_SCHUR", (1024, 2048, 4096))
const REPS = env_int("BENCH_REPS", 3)
const RESULTS = results_dir()

foreach(println, repro_stamp(backend))
csv = open_csv(joinpath(RESULTS, "bench_schur.csv"),
    "backend,eltype,m,pencil,gridn,grid,max_depth,matgen_s,factor_s,first_call_s," *
    "sweep_s,iters,zpd,per_point_s,gstar,cond_est")
@printf("%s over %s, backend=%s, eltype=%s\n", BENCH_MATRIX, bench_box(), backend, T)
@printf("%-6s %-4s %-6s %-9s %-9s %-9s %-11s %-10s\n",
    "m", "pcl", "grid", "factor", "gpu-1st", "sweep", "pt(µs)", "g*")
for m in MS
    nitmax = converged_nit_max(m)                # adaptive cap; sizes the α/β workspace
    gridn = bench_gridn(backend)
    g = gridn * gridn
    tgen = @elapsed A = bench_matrix(T, m)
    # cond(A) is the standard pencil's conditioning only, not a generalized-eigenproblem
    # condition number. Capped so m = 4096 skips the extra O(m³) SVD.
    condA = m <= 2048 ? cond(A) : NaN
    zg = bench_grid(T, gridn)

    # (pencil-label, matgen_s, factor_s, factored pencil); B's generation time counts as matgen.
    std = ("std", tgen, @elapsed(F = schur(A)), MatrixPencil(F))
    tgenB = @elapsed B = bench_matrix_b(T, m)
    gen = ("gen", tgen + tgenB, @elapsed(Fg = schur(A, B)), MatrixPencil(Fg))

    for (pcl, tmat, tfac, P) in (std, gen)
        zpd = pinned_zpd(backend, T, m, nitmax; ngrid = g, headroom = zpd_headroom())
        # Cold first call doubles as warm-up and yields the per-point retirement grid for the
        # FoM (max_depth, Σ iters).
        local nitg
        t1 = @elapsed ((_, nitg) = KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P;
            rtol = converged_rtol(T), nit_max = nitmax, devs, zpd))
        tf = bestof(() -> converged_solve(backend, zg, P; devs, zpd), backend; reps = REPS)
        pp = tf / g
        gstar = tfac / pp                        # grid points that amortize the factorization
        @printf("%-6d %-4s %-6d %-9.2f %-9.2f %-9.2f %-11.2f %-10.0f\n",
            m, pcl, gridn, tfac, t1, tf, pp * 1e6, gstar)
        println(csv,
            @sprintf("%s,%s,%d,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%s,%.6e,%.0f,%.3e",
                backend_tag(backend), T, m, pcl, gridn, g, maximum(nitg), tmat, tfac, t1, tf,
                total_iters(nitg), string(zpd), pp, gstar, condA))
        flush(csv)
    end
end
close_csv(csv, joinpath(RESULTS, "bench_schur.csv"))
