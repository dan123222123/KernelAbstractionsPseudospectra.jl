# Larger test battery for KAPseudospectra on AMDGPU.
#
# Three experiments:
#   1. Grid scaling at fixed m=512: 16×16, 32×32, 64×64, 100×100.
#      Forces real GPU batching as gtotal exceeds zpd.
#   2. nit scaling at fixed m=256, grid=40×40: nit ∈ {8, 16, 32, 64, 128}.
#      Tests how compute scales with iteration count (linear-ish expected).
#   3. Memory-bandwidth scaling: m ∈ {128, 256, 512, 1024, 2048} with grid
#      sized to keep gtotal ~constant (tests how kernel saturation evolves).
#
# All experiments use F32 (the production target), default nit unless overridden,
# CPU vs AMDGPU agreement check (with seeded x₀ for bit-comparable starts).
#
# Usage: julia --project=test --threads=auto bench/bench_battery.jl

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using AMDGPU
using MatrixDepot
using Statistics
using Printf
using Random

@assert AMDGPU.functional()
println("AMDGPU device: ", AMDGPU.device())
println("AMDGPU free / total: ", round(AMDGPU.free()/1024^3, digits=2), " / ",
        round(AMDGPU.total()/1024^3, digits=2), " GB")
println("Threads: ", Threads.nthreads())
flush(stdout)

const T = ComplexF32

function _x₀(::Type{T}, m, seed) where {T}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m); x ./ norm(x)
end

function bench_one(m, grid_n; nit_override=nothing, runs=3, want_cpu=true)
    nit = nit_override === nothing ? max(1, ceil(Int, log2(m))) : nit_override
    A = Matrix{T}(MatrixDepot.matrixdepot("grcar", m))

    @printf("\n--- m=%d, grid=%d×%d, nit=%d, T=%s ---\n",
            m, grid_n, grid_n, nit, T)
    flush(stdout)

    t_schur = @elapsed F = schur(A)
    P = MatrixPencil(F)
    gx, gy, zg = qgrid(T, (-2.0, 2.0), (-2.0, 2.0), (grid_n, grid_n))
    gtotal = grid_n^2
    x₀ = _x₀(T, m, 0xACED)
    @printf("  schur:        %7.3f s\n", t_schur)

    # auto-zpd probe: tells us how many batches the GPU will do
    GC.gc(); AMDGPU.HIP.reclaim()
    auto_zpd = KAPseudospectra.findmaxbatchihl(ROCBackend(), T, m, nit)
    n_batches = ceil(Int, gtotal / auto_zpd)
    @printf("  zpd auto:     %d  (gtotal=%d → %d batch(es))\n",
            auto_zpd, gtotal, n_batches)
    flush(stdout)

    # CPU
    if want_cpu
        t_cpu = @elapsed s_cpu = ihlpsa(CPU(), zg, P, nit; x₀=x₀)
        @assert all(isfinite, s_cpu)
        @printf("  ihlpsa(CPU):  %7.3f s\n", t_cpu)
        flush(stdout)
    else
        s_cpu = nothing
        t_cpu = NaN
    end

    # warmup + timed GPU runs
    GC.gc(); AMDGPU.HIP.reclaim()
    print("  warmup... "); flush(stdout)
    t_warm = @elapsed s_gpu = ihlpsa(ROCBackend(), zg, P, nit; x₀=x₀)
    @printf("(%.2f s)\n", t_warm)
    @assert all(isfinite, s_gpu)
    @assert size(s_gpu) == (grid_n, grid_n)

    times = Float64[]
    for _ in 1:runs
        GC.gc(); AMDGPU.HIP.reclaim()
        t = @elapsed ihlpsa(ROCBackend(), zg, P, nit; x₀=x₀)
        push!(times, t)
    end
    t_gpu = minimum(times)
    @printf("  ihlpsa(ROC):  %7.3f s (min of %d runs; med %.3f)\n",
            t_gpu, runs, median(times))

    if want_cpu
        d_abs = maximum(abs.(s_cpu .- s_gpu))
        d_rel = maximum(abs.(s_cpu .- s_gpu) ./ abs.(s_cpu))
        tol = 1e-3  # F32 cross-backend tolerance per test_consistency.jl
        status = d_rel <= tol ? "PASS" : "FAIL"
        @printf("  CPU vs GPU:   abs %.2e  rel %.2e  [%s]\n",
                d_abs, d_rel, status)
        @printf("  speedup:      %6.2fx  (%.3fs CPU vs %.3fs GPU)\n",
                t_cpu / t_gpu, t_cpu, t_gpu)
    end

    # Memory-traffic estimate, GB/s seen by the kernel
    bytes = gtotal * nit * m^2 * 2 * 4 * sizeof(T)  # 2 trsm passes, 4 matrices each
    @printf("  apparent BW:  %6.1f GB/s  (kernel memory model)\n",
            bytes / 1e9 / t_gpu)
    @printf("  throughput:   %d grid-pts/s\n", round(Int, gtotal / t_gpu))
    flush(stdout)

    return (m, grid_n, nit, t_cpu, t_gpu, gtotal)
end

# ---- Experiment 1: grid scaling at fixed m=512, default nit ----
println("\n", "="^70)
println("EXPERIMENT 1: grid scaling, m=512, default nit")
println("="^70)
for grid_n in (16, 32, 64, 100)
    bench_one(512, grid_n; want_cpu = grid_n <= 32)  # CPU too slow at 64+
end

# ---- Experiment 2: nit scaling at fixed m=256, grid=40×40 ----
println("\n", "="^70)
println("EXPERIMENT 2: nit scaling, m=256, grid=40×40")
println("="^70)
for nit in (8, 16, 32, 64, 128)
    bench_one(256, 40; nit_override=nit, want_cpu = nit <= 32)
end

# ---- Experiment 3: memory-bandwidth scaling, gtotal ~ constant ----
println("\n", "="^70)
println("EXPERIMENT 3: memory-BW scaling, m sweep at ~constant gtotal")
println("="^70)
# Pick grid sizes so gtotal ≈ 600–1000 across m
for (m, grid_n) in [(128, 32), (256, 28), (512, 24), (1024, 24), (2048, 20)]
    bench_one(m, grid_n; want_cpu = m <= 256)
end

println("\n", "="^70)
println("Done.")
println("="^70)
