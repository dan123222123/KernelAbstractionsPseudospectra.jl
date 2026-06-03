# Phase breakdown of ihlpsa for one fixed problem size on ROCBackend.
# Measures: schur, MatrixPencil construction, host->device transfer,
# workspace alloc, lockstep_ihl! kernel time, device->host transfer,
# and ihlsrg! post-processing.
#
# Usage: julia --project=test bench/bench_phases.jl 2>&1 | tee bench/bench_phases.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Adapt
using ArraysOfArrays
using Statistics
using Printf

# Reach into KAPseudospectra internals to time individual phases.
using KAPseudospectra: IHLworkspace, lockstep_ihl!, ihlsrg!, findmaxbatchihl

const HAVE_AMDGPU = try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

@assert HAVE_AMDGPU "Requires functional AMDGPU"

println("Threads: $(Threads.nthreads()),  AMDGPU: $(HAVE_AMDGPU)")
flush(stdout)

function timeit(label, f; runs=5)
    times = Float64[]
    for r in 1:runs
        GC.gc()
        push!(times, @elapsed f())
    end
    tmin, tmed = minimum(times), median(times)
    @printf("  %-32s min=%9.4f ms  median=%9.4f ms\n",
            label, tmin*1000, tmed*1000)
    flush(stdout)
    return tmin
end

function profile_one(m, grid)
    nit = ceil(Int, log2(m))
    println()
    println(repeat("=", 80))
    println("Problem: m=$m,  grid=$(grid)×$(grid),  nit=$nit  (T=ComplexF64, ROCBackend)")
    println(repeat("=", 80))

    A = randn(ComplexF64, m, m)
    backend = ROCBackend()
    bgarray = AMDGPU.ROCArray
    dev = AMDGPU.device()
    AMDGPU.device!(dev)

    # Phase 1: Schur (CPU)
    tschur = timeit("schur(A) [host]", () -> schur(A))

    F = schur(A)

    # Phase 2: MatrixPencil construction (now lazy adjoints)
    tmp = timeit("MatrixPencil(F) [host]", () -> MatrixPencil(F))

    P = MatrixPencil(F)

    # Phase 3: qgrid (CPU)
    tqg = timeit("qgrid [host]",
                  () -> qgrid(ComplexF64, (-2, 2), (-2, 2), (grid, grid)))

    gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (grid, grid))

    # Phase 4: adapt(ROCArray, P) -- materializes the lazy adjoint into device memory
    tadapt = timeit("adapt(ROC, P) [host->dev]", () -> begin
        Pd = adapt(bgarray, P)
        KernelAbstractions.synchronize(backend)
        Pd
    end)

    # Phase 5: allocate device workspaces (zv, α, β, ihl)
    gtotal = length(zg)
    zpd = min(findmaxbatchihl(backend, ComplexF64, m, nit), gtotal)
    @printf("  (zpd = %d,  gtotal = %d,  num batches = %d)\n",
            zpd, gtotal, ceil(Int, gtotal / zpd))
    flush(stdout)
    Pd = adapt(bgarray, P)

    talloc = timeit("alloc workspaces [dev]", () -> begin
        zv_h = collect(Iterators.flatten(zg))
        dzv = adapt(bgarray, zv_h)
        α = adapt(bgarray, zeros(ComplexF64, nit, gtotal))
        β = adapt(bgarray, zeros(ComplexF64, nit + 1, gtotal))
        ihl = adapt(bgarray, IHLworkspace(P, zpd))
        KernelAbstractions.synchronize(backend)
        (dzv, α, β, ihl)
    end)

    # Phase 6: just the lockstep_ihl! kernel time (one full batch)
    zv_h = collect(Iterators.flatten(zg))
    dzv = adapt(bgarray, zv_h)
    α = adapt(bgarray, zeros(ComplexF64, nit, gtotal))
    β = adapt(bgarray, zeros(ComplexF64, nit + 1, gtotal))
    ihl = adapt(bgarray, IHLworkspace(Pd, zpd))

    g = min(zpd, gtotal)
    view(ihl.zv, 1:g) .= view(dzv, 1:g)
    KernelAbstractions.synchronize(backend)

    # warmup
    lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g)
    KernelAbstractions.synchronize(backend)

    tlockstep = timeit("lockstep_ihl! [dev kernels]", () -> begin
        lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit, g)
        KernelAbstractions.synchronize(backend)
    end)

    # Phase 7: device -> host of α, β (for ihlsrg post-processing)
    ttransfer = timeit("α,β dev->host", () -> begin
        αh = adapt(Array, α[:, 1:g])
        βh = adapt(Array, β[:, 1:g])
        (αh, βh)
    end)

    # Phase 8: ihlsrg! (CPU post-processing, threaded)
    αh = adapt(Array, α[:, 1:g])
    βh = adapt(Array, β[:, 1:g])
    sr = zeros(Float64, g)
    γ, δ = 1.0, 0.0
    zv_view = view(zv_h, 1:g)
    tihlsrg = timeit("ihlsrg! [host threaded]",
                      () -> ihlsrg!(sr, zv_view, γ, δ, αh, βh))

    # Phase 9: full ihlpsa for reference
    ihlpsa(backend, zg, P, nit)
    tfull = timeit("ihlpsa(ROC) [end-to-end]",
                   () -> ihlpsa(backend, zg, P, nit))

    println()
    println("Phase share of one batch (full grid fits in $(ceil(Int, gtotal/zpd)) batches):")
    accounted = tlockstep + ttransfer + tihlsrg
    @printf("  kernels  : %5.1f%%  (%.4f ms)\n", 100*tlockstep/accounted, tlockstep*1000)
    @printf("  transfer : %5.1f%%  (%.4f ms)\n", 100*ttransfer/accounted, ttransfer*1000)
    @printf("  ihlsrg!  : %5.1f%%  (%.4f ms)\n", 100*tihlsrg/accounted, tihlsrg*1000)
    println()
    @printf("Setup overhead per ihlpsa call:\n")
    @printf("  schur+MP+qgrid+adapt+alloc = %.4f ms\n",
            (tschur+tmp+tqg+tadapt+talloc)*1000)
    @printf("Full ihlpsa : %.4f ms\n", tfull*1000)
    flush(stdout)
end

# Sweep a few sizes
for (m, grid) in [(64, 100), (256, 100), (512, 100), (256, 200)]
    profile_one(m, grid)
end
