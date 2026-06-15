# Multi-GPU strong scaling of the ADAPTIVE ihlpsa: fixed problem, sweep ndev=1..N.
#
# Methodology that keeps the numbers honest: fully reclaim every device's memory
# and GC between configs, and take best-of-REPS. Without the per-device reclaim a
# sweep that reuses one problem across sequential ndev=1..N calls lets the always-
# used device accumulate allocator pressure, inflating the high-ndev points 2-3x.
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/strong_scaling_adaptive.jl

using KAPseudospectra
using CUDA
using LinearAlgebra, Printf, Dates

const RESULTS = joinpath(@__DIR__, "results_adaptive")
mkpath(RESULTS)
const ALLDEVS = collect(CUDA.devices())
const N_GPU   = length(ALLDEVS)
const T       = ComplexF32
const G       = 300
const REGION  = ((-1, 3), (-3, 3))
const M       = 1024
const NMAX    = 4 * ceil(Int, log2(M))     # 40
const REPS    = 2

logio = open(joinpath(RESULTS, "strong_scaling_clean_log.txt"), "w")
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

# reclaim every device so no config inherits another's allocator state
function reclaim_all()
    for d in ALLDEVS
        CUDA.device!(d); CUDA.reclaim()
    end
    GC.gc(); GC.gc()
end

_, _, zg = qgrid(T, REGION[1], REGION[2], (G, G))
P = MatrixPencil(grcar(T, M))

logln("="^74)
logln("ADAPTIVE strong scaling (CLEAN: reclaim-all + best-of-", REPS, ")   ", Dates.now())
logln("Grcar m=", M, "  nit_max=", NMAX, "  grid=", G, "x", G, " (", G*G, " pts)  T=", T)
logln("="^74)

# warmup
logln("[", clock(), "] warmup...")
let
    tw = @elapsed ihlpsa(CUDABackend(), zg, P; nit_max=NMAX)
    logln("[", clock(), "] warmup done (", @sprintf("%.1f s", tw), ")")
end

csv = open(joinpath(RESULTS, "strong_scaling_adaptive_clean.csv"), "w")
println(csv, "ndev,time_s,gridpts_per_s,speedup,efficiency,nit_used")
logln(@sprintf("%-6s %-11s %-12s %-9s %-9s %-7s", "ndev", "time(s)", "gridpts/s", "speedup", "eff", "nit"))
t1 = NaN
for ndev in 1:N_GPU
    devs = ALLDEVS[1:ndev]
    best = Inf; nused = 0
    for _ in 1:REPS
        reclaim_all()
        local nu = 0
        # internal driver returns (σ, nit_used); public ihlpsa returns only σ
        t = @elapsed ((_, nu) = KAPseudospectra._ihlpsa_adaptive(CUDABackend(), zg, P; nit_max=NMAX, devs=devs))
        if t < best; best = t; nused = nu; end
    end
    global t1; ndev == 1 && (t1 = best)
    gpps, spd = (G*G)/best, t1/best
    logln(@sprintf("[%s] %-3d  %-11.4f %-12.1f %-9.2f %-6.1f%% %-7d", clock(), ndev, best, gpps, spd, 100*spd/ndev, nused))
    println(csv, @sprintf("%d,%.6f,%.3f,%.4f,%.4f,%d", ndev, best, gpps, spd, spd/ndev, nused)); flush(csv)
end
close(csv)
logln("\n", "="^74); logln("DONE."); close(logio)
