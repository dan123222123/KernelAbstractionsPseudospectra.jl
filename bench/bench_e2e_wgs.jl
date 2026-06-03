# Validate that the wgs=32 win on lockstep_ihl translates to ihlpsa end-to-end.
# Also sample m=512 at wgs ∈ {32, 256} as a small cap on the previous sweep.
# Usage: julia --project=test --threads=auto bench/bench_e2e_wgs.jl 2>&1 | tee bench/bench_e2e_wgs.log

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

function bench(callable, runs=3)
    times = Float64[]
    for _ in 1:runs
        GC.gc()
        push!(times, @elapsed callable())
    end
    return minimum(times)
end

function bench_lockstep(P, m, grid_n, wgs; runs=3)
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

println()
@printf("%-7s %-7s %-9s %-9s %-9s\n", "m", "kind", "wgs=32", "wgs=256", "speedup")
println(repeat("-", 60))

for m in [64, 128, 256, 512]
    nit = ceil(Int, log2(m))
    A = randn(ComplexF64, m, m)
    F = schur(A)
    P = MatrixPencil(F)
    grid = 100
    gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (grid, grid))

    # End-to-end ihlpsa
    ihlpsa(ROCBackend(), zg, P, nit; wgs=32)
    t_e2e_32 = bench(() -> ihlpsa(ROCBackend(), zg, P, nit; wgs=32), 3)
    ihlpsa(ROCBackend(), zg, P, nit; wgs=256)
    t_e2e_256 = bench(() -> ihlpsa(ROCBackend(), zg, P, nit; wgs=256), 3)
    @printf("%-7d %-7s %-9.4f %-9.4f %5.2fx\n",
            m, "e2e", t_e2e_32, t_e2e_256, t_e2e_256/t_e2e_32)
    flush(stdout)
end

println()
println("Validation: m=512 lockstep with wgs=32 (last point of wgs sweep that was killed)")
m = 512
nit = ceil(Int, log2(m))
A = randn(ComplexF64, m, m)
F = schur(A)
P = MatrixPencil(F)
t32 = bench_lockstep(P, m, 100, 32; runs=3)
t256 = bench_lockstep(P, m, 100, 256; runs=3)
@printf("m=512  wgs=32: %.3fs   wgs=256: %.3fs   speedup: %.2fx\n",
        t32, t256, t256/t32)
