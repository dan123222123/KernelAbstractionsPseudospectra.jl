# Test whether lazy Adjoint{T,ROCArray{T,2}} adjoints hurt GPU coalescing
# vs. materialized Matrix{T} adjoints in the forward triangular solve.
# Usage: julia --project=test --threads=auto bench/bench_lazy_adjoint.jl 2>&1 | tee bench/bench_lazy_adjoint.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Adapt
using Statistics
using Printf

using AMDGPU
@assert AMDGPU.functional()

println("Threads: $(Threads.nthreads())")
flush(stdout)

# Reach into KAPseudospectra to construct both flavors of pencil.
using KAPseudospectra: SchurMatrixPencil, IHLworkspace, lockstep_ihl!

function pencil_lazy(F::Schur)
    A = Matrix{eltype(F)}(F.T)
    Iₘ = Matrix{eltype(F)}(I, size(F.T))
    SchurMatrixPencil{eltype(F)}(A, A', Iₘ, Iₘ)
end

function pencil_eager(F::Schur)
    A = Matrix{eltype(F)}(F.T)
    Ac = Matrix{eltype(F)}(F.T')
    Iₘ = Matrix{eltype(F)}(I, size(F.T))
    Iₘc = Matrix{eltype(F)}(I, size(F.T))
    SchurMatrixPencil{eltype(F)}(A, Ac, Iₘ, Iₘc)
end

function bench_lockstep(P::SchurMatrixPencil, m, grid_n; runs=5)
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
    lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g)
    KernelAbstractions.synchronize(backend)

    times = Float64[]
    for _ in 1:runs
        AMDGPU.HIP.reclaim()
        t = @elapsed begin
            lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g)
            KernelAbstractions.synchronize(backend)
        end
        push!(times, t)
    end
    return minimum(times)
end

function compare(m, grid_n)
    A = randn(ComplexF64, m, m)
    F = schur(A)
    Pl = pencil_lazy(F)
    Pe = pencil_eager(F)

    @assert typeof(Pl.Ac) <: Adjoint
    @assert typeof(Pe.Ac) <: Matrix

    tlazy = bench_lockstep(Pl, m, grid_n)
    teager = bench_lockstep(Pe, m, grid_n)

    @printf("m=%-4d grid=%-4d  lazy=%8.3f ms  eager=%8.3f ms  lazy/eager=%.2fx\n",
            m, grid_n, tlazy*1000, teager*1000, tlazy/teager)
    flush(stdout)
end

println()
@printf("%-46s %-9s %-9s %-9s\n", "config", "lazy", "eager", "ratio")
println(repeat("-", 80))
for (m, gn) in [(64, 100), (128, 100), (256, 100), (512, 100),
                 (256, 200), (512, 200)]
    compare(m, gn)
end
