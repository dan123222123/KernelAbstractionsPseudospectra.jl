# End-to-end ihlpsa timing across trsm strategies (KAPSEUDO_TRSM): column (baseline) vs
# warp (register, KA/KI) vs tiled (KA/KI). Confirms correctness (σ vs column) and the
# size-dependent crossover (warp wins small m, tiled wins large m).
#   unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto julia --project=bench bench/warp_trsm_bench.jl
using KAPseudospectra, LinearAlgebra, Random, Printf, CUDA

backend = CUDABackend()
T = ComplexF32
nit = 20
reps = 3
gridn = 300
ms = (128, 256, 512, 1024)

bestof(f) = minimum(begin
    for d in CUDA.devices(); CUDA.device!(d); CUDA.reclaim(); end; GC.gc()
    t0 = time_ns(); f(); CUDA.synchronize(); (time_ns() - t0) / 1e9
end for _ in 1:reps)

@printf("%-6s %-10s %-10s %-10s %-9s %-9s %-9s\n",
        "m", "warp(s)", "tiled(s)", "column(s)", "warp/col", "tiled/col", "maxdiff")
for m in ms
    rng = Random.seed!(0xBEEF)
    P = MatrixPencil(schur(randn(rng, T, m, m)))
    _, _, zg = KAPseudospectra.qgrid(T, (-3, 3), (-3, 3), (gridn, gridn))
    run(flag) = withenv("KAPSEUDO_TRSM" => flag) do
        ihlpsa(backend, zg, P, nit; devs=collect(CUDA.devices()))
    end
    σw = run("warp"); σt = run("tiled"); σc = run("column")   # warm + correctness
    maxdiff = max(maximum(abs.(σw .- σc)), maximum(abs.(σt .- σc)))

    tw = bestof(() -> run("warp"))
    tt = bestof(() -> run("tiled"))
    tc = bestof(() -> run("column"))
    @printf("%-6d %-10.4f %-10.4f %-10.4f %-9.3f %-9.3f %.2e\n",
            m, tw, tt, tc, tc / tw, tc / tt, maxdiff)
end
