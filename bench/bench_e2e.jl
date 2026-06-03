# End-to-end timing sweep: CPU svdpsa baseline vs CPU/ROC ihlpsa.
# Usage: julia --project=test bench/bench_e2e.jl 2>&1 | tee bench/bench_e2e.log

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Statistics
using Printf

const HAVE_AMDGPU = try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

println("Threads: $(Threads.nthreads()),  AMDGPU: $(HAVE_AMDGPU)")
flush(stdout)

function bench(callable, runs=3)
    times = Float64[]
    for _ in 1:runs
        GC.gc()
        push!(times, @elapsed callable())
    end
    return minimum(times), median(times)
end

function run_sweep()
    grid = 100
    println("Grid: $(grid)×$(grid).  Reporting min(s) over 3 runs after 1 warmup.")
    println()
    @printf("%6s %5s %10s %12s %12s %12s %10s\n",
            "m", "nit", "schur", "svdpsa-cpu", "ihlpsa-cpu", "ihlpsa-roc", "ihl-spdup")
    println(repeat("-", 80))
    flush(stdout)

    for m in [64, 128, 256, 512]
        nit = ceil(Int, log2(m))
        A = randn(ComplexF64, m, m)
        gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (grid, grid))

        ts_min, _ = bench(() -> schur(A), 3)

        F = schur(A)
        P = MatrixPencil(F)

        # ℂsvdpsa scales O(m³ × g²); cap at m=128 to keep total runtime sane
        tsvd_min = if m <= 128
            ℂsvdpsa(zg, P)
            tsvd, _ = bench(() -> ℂsvdpsa(zg, P), 2)
            tsvd
        else
            NaN
        end

        ihlpsa(CPU(), zg, P, nit)
        ticc_min, _ = bench(() -> ihlpsa(CPU(), zg, P, nit), 3)

        if HAVE_AMDGPU
            ihlpsa(ROCBackend(), zg, P, nit)
            tigr_min, _ = bench(() -> ihlpsa(ROCBackend(), zg, P, nit), 3)
        else
            tigr_min = NaN
        end

        speedup = isnan(tigr_min) ? NaN : ticc_min / tigr_min
        @printf("%6d %5d %10.4f %12.4f %12.4f %12.4f %9.2fx\n",
                m, nit, ts_min, tsvd_min, ticc_min, tigr_min, speedup)
        flush(stdout)
    end
end

run_sweep()
