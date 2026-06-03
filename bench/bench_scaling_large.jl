# Large-m scaling test on AMDGPU.
#
# Sweeps m ∈ {256, 512, 1024, 2048, 4096} at fixed grid (50×50 for the smaller
# sizes, 32×32 for m=4096 to keep wall-clock manageable on this iGPU).
# Reports schur time, ihlpsa time, throughput, and per-grid-point cost.
# Verifies output is finite (no full svdpsa baseline at this m — would take
# hours on CPU).
#
# Usage: julia --project=test --threads=auto bench/bench_scaling_large.jl 2>&1 | tee bench/bench_scaling_large.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using AMDGPU
using MatrixDepot
using Statistics
using Printf

@assert AMDGPU.functional()
println("AMDGPU device: ", AMDGPU.device())
println("AMDGPU free / total: ", round(AMDGPU.free()/1024^3, digits=2), " / ",
        round(AMDGPU.total()/1024^3, digits=2), " GB")
println("Threads: ", Threads.nthreads())
flush(stdout)

# Use ComplexF32 throughout — half the memory traffic of F64, and the algorithm
# is memory-bound. Pseudospectra contour levels rarely care about precision
# below ε ~ 1e-5 anyway.
const T = ComplexF32

function bench_one(m, grid_n; runs=3)
    nit = ceil(Int, log2(m))

    # Grcar matrix: well-known non-normal pseudospectra benchmark.
    A = Matrix{T}(MatrixDepot.matrixdepot("grcar", m))

    @printf("\n=== m=%d, grid=%d×%d, nit=%d, T=%s ===\n", m, grid_n, grid_n, nit, T)
    flush(stdout)

    # Schur (host)
    t_schur = @elapsed F = schur(A)
    @printf("schur(A) [host]:        %8.3f s\n", t_schur)
    flush(stdout)

    P = MatrixPencil(F)
    gtotal = grid_n^2

    # Probe the auto-picked zpd
    GC.gc()
    AMDGPU.HIP.reclaim()
    auto_zpd = KAPseudospectra.findmaxbatchihl(ROCBackend(), T, m, nit)
    n_batches = ceil(Int, gtotal / auto_zpd)
    @printf("zpd auto: %d (gtotal=%d, batches=%d)\n", auto_zpd, gtotal, n_batches)
    flush(stdout)

    gx, gy, zg = qgrid(T, (-2.0, 2.0), (-2.0, 2.0), (grid_n, grid_n))

    # Warmup (forces kernel compile + caches)
    print("warmup... "); flush(stdout)
    t_warm = @elapsed s = ihlpsa(ROCBackend(), zg, P, nit)
    @printf("(%.2f s)\n", t_warm)
    flush(stdout)

    # Sanity: output is finite, has expected shape
    @assert all(isfinite, s)
    @assert size(s) == (grid_n, grid_n)
    @assert minimum(s) > 0

    # Timed runs
    times = Float64[]
    for r in 1:runs
        GC.gc()
        AMDGPU.HIP.reclaim()
        t = @elapsed ihlpsa(ROCBackend(), zg, P, nit)
        push!(times, t)
        @printf("  run %d: %.3f s\n", r, t)
        flush(stdout)
    end
    t_min = minimum(times)
    t_med = median(times)

    # Memory traffic estimate: per inner-loop iter ≈ 4 × sizeof(T) bytes
    # (A[I,j], B[I,j], bv[I] r/w). Total iters: gtotal * nit * m² * 2 (fwd+bwd).
    bytes_total = gtotal * nit * m^2 * 2 * 4 * sizeof(T)
    bw_GBs = bytes_total / 1e9 / t_min

    @printf("ihlpsa(ROC) min/med:    %8.3f / %8.3f s\n", t_min, t_med)
    @printf("throughput:             %8.0f grid-points/sec\n", gtotal / t_min)
    @printf("apparent bandwidth:     %8.1f GB/s (memory-traffic model)\n", bw_GBs)
    flush(stdout)
end

# Scaling sweep
for (m, grid_n) in [(256, 50), (512, 50), (1024, 50), (2048, 50)]
    bench_one(m, grid_n)
end

# Stretch: m=4096 with a smaller grid to keep wall-time manageable.
# Wrap in try/catch in case it OOMs or hits some other limit.
println()
println(repeat("=", 60))
println("Stretch test: m=4096, grid=32×32")
println(repeat("=", 60))
try
    bench_one(4096, 32; runs=2)
catch e
    println("m=4096 failed: ", typeof(e), ": ", first(split(string(e), "\n"), 1))
end
