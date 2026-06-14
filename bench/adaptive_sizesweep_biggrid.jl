# (a) Clean best-of-N adaptive SIZE SWEEP at the standard 300x300 grid.
# (b) Fixed-vs-adaptive at a LARGER 600x600 grid (4x the points) to test whether
#     adaptive's win widens as per-chunk host overhead amortizes over more points.
#
# Both use best-of-REPS with full per-device CUDA.reclaim() + GC between configs
# (see strong_scaling_adaptive.jl for why that discipline matters).
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/adaptive_sizesweep_biggrid.jl

using KAPseudospectra
using CUDA
using LinearAlgebra, Printf, Dates

const RESULTS = joinpath(@__DIR__, "results_adaptive")
mkpath(RESULTS)
const ALLDEVS = collect(CUDA.devices())
const N_GPU   = length(ALLDEVS)
const T       = ComplexF32
const REGION  = ((-1, 3), (-3, 3))
const GSTD    = 300                 # standard grid (90k pts)
const GBIG    = 600                 # 4x larger grid (360k pts); 600/6 = 100
const REPS    = 2

logio = open(joinpath(RESULTS, "clean_biggrid_log.txt"), "w")
clock() = Dates.format(Dates.now(), "HH:MM:SS")
logln(args...) = (s = string(args...); println(s); println(logio, s); flush(logio))

function grcar(::Type{S}, m, k=3) where {S}
    A = zeros(S, m, m)
    for i in 1:m
        A[i, i] = one(S)
        for j in 1:k; i+j <= m && (A[i, i+j] = one(S)); end
        i+1 <= m && (A[i+1, i] = -one(S))
    end
    A
end
nit_fixed(m) = 4 * ceil(Int, log2(m))
function reclaim_all()
    for d in ALLDEVS; CUDA.device!(d); CUDA.reclaim(); end
    GC.gc(); GC.gc()
end
build(m, g) = (zg = qgrid(T, REGION[1], REGION[2], (g, g))[3]; (; zg, P = MatrixPencil(grcar(T, m))))

function best_adaptive(zg, P, nmax)
    best = Inf; nu = 0
    for _ in 1:REPS
        reclaim_all(); local n = 0
        t = @elapsed ((_, n) = ihlpsa(CUDABackend(), zg, P; nit_max=nmax))
        t < best && (best = t; nu = n)
    end
    (best, nu)
end
function best_fixed(zg, P, nit)
    best = Inf
    for _ in 1:REPS
        reclaim_all()
        best = min(best, @elapsed ihlpsa(CUDABackend(), zg, P, nit))
    end
    best
end

logln("="^78)
logln("ADAPTIVE clean size sweep + big-grid study   ", Dates.now())
logln("GPUs: ", N_GPU, " x ", CUDA.name(ALLDEVS[1]), "   threads: ", Threads.nthreads(),
      "   best-of-", REPS, "  Grcar  T=", T)
logln("="^78)

# warmup compiles both paths
logln("[", clock(), "] warmup...")
let w = build(256, GSTD)
    @elapsed ihlpsa(CUDABackend(), w.zg, w.P, 4)
    tw = @elapsed ihlpsa(CUDABackend(), w.zg, w.P; nit_max=nit_fixed(256))
    logln("[", clock(), "] warmup done (", @sprintf("%.1f s", tw), ")")
end

#############################################
# (a) CLEAN ADAPTIVE SIZE SWEEP  (300x300)
#############################################
logln("\n## (a) CLEAN ADAPTIVE SIZE SWEEP  grid=", GSTD, "x", GSTD, " (", GSTD*GSTD, " pts), all ", N_GPU, " GPUs")
a_csv = open(joinpath(RESULTS, "size_sweep_adaptive_clean.csv"), "w")
println(a_csv, "m,adaptive_s,nit_used,nit_max,gridpts_per_s")
logln(@sprintf("%-7s %-12s %-8s %-8s %-12s", "m", "adapt(s)", "nit", "nit_max", "gridpts/s"))
for m in (512, 1024, 2048)
    nmax = nit_fixed(m)
    b = build(m, GSTD)
    ta, nu = best_adaptive(b.zg, b.P, nmax)
    logln(@sprintf("[%s] %-7d %-12.4f %-8d %-8d %-12.1f", clock(), m, ta, nu, nmax, (GSTD*GSTD)/ta))
    println(a_csv, @sprintf("%d,%.6f,%d,%d,%.3f", m, ta, nu, nmax, (GSTD*GSTD)/ta)); flush(a_csv)
end
close(a_csv)

#############################################
# (b) BIG-GRID FIXED vs ADAPTIVE  (600x600)
#############################################
logln("\n## (b) BIG-GRID FIXED vs ADAPTIVE  grid=", GBIG, "x", GBIG, " (", GBIG*GBIG, " pts), all ", N_GPU, " GPUs")
logln("    (compare speedup column to the 300x300 baseline: m256=1.90x m512=1.90x m1024=2.90x)")
b_csv = open(joinpath(RESULTS, "biggrid_fixed_vs_adaptive.csv"), "w")
println(b_csv, "m,grid,nit_fixed,t_fixed_s,t_adaptive_s,nit_used,speedup,adapt_gridpts_per_s")
logln(@sprintf("%-7s %-9s %-12s %-13s %-8s %-9s %-12s", "m", "nit_fix", "fixed(s)", "adapt(s)", "nit", "speedup", "adapt gp/s"))
for m in (256, 512, 1024)
    nf = nit_fixed(m)
    b = build(m, GBIG)
    tf = best_fixed(b.zg, b.P, nf)
    ta, nu = best_adaptive(b.zg, b.P, nf)
    logln(@sprintf("[%s] %-7d %-9d %-12.4f %-13.4f %-8d %-7.2f× %-12.1f", clock(), m, nf, tf, ta, nu, tf/ta, (GBIG*GBIG)/ta))
    println(b_csv, @sprintf("%d,%d,%d,%.6f,%.6f,%d,%.4f,%.3f", m, GBIG, nf, tf, ta, nu, tf/ta, (GBIG*GBIG)/ta)); flush(b_csv)
end
close(b_csv)

logln("\n", "="^78); logln("DONE. results in ", RESULTS); close(logio)
