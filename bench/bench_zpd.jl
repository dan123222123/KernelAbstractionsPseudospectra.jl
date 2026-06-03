# Sweep zpd (batch size) on ROCBackend at fixed problem size.
# Identifies the throughput sweet spot vs. the auto-picked findmaxbatchihl value.
# Usage: julia --project=test --threads=auto bench/bench_zpd.jl 2>&1 | tee bench/bench_zpd.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Statistics
using Printf

using AMDGPU
@assert AMDGPU.functional()

println("Threads: $(Threads.nthreads())")
flush(stdout)

function bench(callable, runs=3)
    times = Float64[]
    for _ in 1:runs
        GC.gc()
        push!(times, @elapsed callable())
    end
    return minimum(times)
end

function sweep(m, grid_n)
    nit = ceil(Int, log2(m))
    A = randn(ComplexF64, m, m)
    F = schur(A)
    P = MatrixPencil(F)
    gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (grid_n, grid_n))
    gtotal = length(zg)

    println()
    println("m=$m,  grid=$(grid_n)×$(grid_n) (gtotal=$gtotal),  nit=$nit")
    println(repeat("-", 60))
    @printf("%8s  %12s  %12s\n", "zpd", "time(s)", "Mpts/s")
    flush(stdout)

    backend = ROCBackend()
    auto_zpd = KAPseudospectra.findmaxbatchihl(backend, ComplexF64, m, nit)
    println("(auto findmaxbatchihl = $auto_zpd)")

    for zpd in [256, 512, 1024, 2048, 4096, min(8192, gtotal), gtotal]
        zpd > gtotal && continue
        # warmup
        ihlpsa(backend, zg, P, nit; zpd=zpd)
        t = bench(() -> ihlpsa(backend, zg, P, nit; zpd=zpd), 3)
        mpps = (gtotal / t) / 1e6
        @printf("%8d  %12.4f  %12.3f\n", zpd, t, mpps)
        flush(stdout)
    end
end

for (m, gn) in [(128, 100), (256, 100), (512, 100), (256, 200)]
    sweep(m, gn)
end
