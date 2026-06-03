# Sweep wgs (workgroup size) of the column-oriented trsm kernel at fixed m.
# Hypothesis: at small m (e.g. m=64) the default wgs=256 underutilizes the
# wavefront — only m of 256 threads do useful work in the column update phase.
# Usage: julia --project=test --threads=auto bench/bench_wgs.jl 2>&1 | tee bench/bench_wgs.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Adapt
using Statistics
using Printf

using AMDGPU
@assert AMDGPU.functional()

using KAPseudospectra: IHLworkspace, lockstep_ihl!

println("Threads: $(Threads.nthreads())")
flush(stdout)

function bench_lockstep(P, m, grid_n, wgs; runs=5)
    nit = ceil(Int, log2(m))
    backend = ROCBackend()
    bgarray = AMDGPU.ROCArray
    gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (grid_n, grid_n))
    g = length(zg)

    Pd = adapt(bgarray, P)
    zv_h = collect(Iterators.flatten(zg))
    dzv = adapt(bgarray, zv_h)
    α = adapt(bgarray, zeros(ComplexF64, nit, g))
    β = adapt(bgarray, zeros(ComplexF64, nit + 1, g))
    ihl = adapt(bgarray, IHLworkspace(Pd, g))
    view(ihl.zv, 1:g) .= view(dzv, 1:g)
    KernelAbstractions.synchronize(backend)

    # warmup
    lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g; wgs=wgs)
    KernelAbstractions.synchronize(backend)

    times = Float64[]
    for _ in 1:runs
        AMDGPU.HIP.reclaim()
        t = @elapsed begin
            lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g; wgs=wgs)
            KernelAbstractions.synchronize(backend)
        end
        push!(times, t)
    end
    return minimum(times)
end

function sweep(m, grid_n)
    A = randn(ComplexF64, m, m)
    F = schur(A)
    P = MatrixPencil(F)

    println()
    println("m=$m,  grid=$(grid_n)×$(grid_n),  nit=$(ceil(Int, log2(m)))")
    println(repeat("-", 60))
    @printf("%6s  %12s  %12s\n", "wgs", "lockstep(s)", "vs wgs=256")
    flush(stdout)

    baseline = NaN
    # try wgs ∈ {32, 64, 128, 256, 512} (typical AMDGPU wavefront 32 or 64)
    for wgs in [32, 64, 128, 256, 512]
        wgs > 1024 && continue  # AMDGPU max workgroup
        wgs < m && wgs ÷ 2 == 0 && continue  # avoid trivially-bad sizes
        t = bench_lockstep(P, m, grid_n, wgs)
        if wgs == 256
            baseline = t
        end
        ratio = isnan(baseline) ? NaN : t / baseline
        @printf("%6d  %12.4f  %11.2fx\n", wgs, t, ratio)
        flush(stdout)
    end
end

for (m, gn) in [(64, 100), (128, 100), (256, 100), (512, 100)]
    sweep(m, gn)
end
