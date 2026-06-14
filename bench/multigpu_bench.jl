# Multi-GPU benchmark for fixed-`nit` ihlpsa -- runs on the full node (all CUDA devices).
#
#   1. Strong scaling: fixed problem, sweep ndev = 1..N_GPU, report speedup/efficiency
#   2. Size sweep:     all GPUs, sweep matrix size n
#   3. Correctness:    1-GPU vs all-GPU result must match (grid is column-partitioned)
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/multigpu_bench.jl
#
# ihlpsa(...; devs=...) returns a host Matrix, so the call is fully synchronized on
# return -> plain wall-clock @elapsed is valid for the multi-device case.

using KAPseudospectra
using CUDA
using LinearAlgebra, MatrixDepot, Printf, Dates

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)
const ALLDEVS = collect(CUDA.devices())
const N_GPU   = length(ALLDEVS)
const T       = ComplexF32
const NIT     = 8
const G       = 300                 # 300x300 = 90_000 grid points; divisible by 1..6
const REPS    = 2                   # measured reps (best-of); global warmup done once up front

logio = open(joinpath(RESULTS, "bench_log.txt"), "w")
clock() = Dates.format(Dates.now(), "HH:MM:SS")
function logln(args...)
    s = string(args...); println(s); println(logio, s); flush(logio)
end

logln("="^72)
logln("KAPseudospectra.jl multi-GPU benchmark   ", Dates.now())
logln("GPUs: ", N_GPU, " x ", CUDA.name(ALLDEVS[1]),
      @sprintf("  (%.1f GiB each)", CUDA.totalmem(ALLDEVS[1])/2^30))
logln("Julia threads: ", Threads.nthreads(),
      "   T=", T, "  grid=", G, "x", G, " (", G*G, " pts)  nit=", NIT)
logln("="^72)

build(n; box=(-2,5,-4.5,4.5)) = begin
    _, _, zg = qgrid(T, (box[1],box[2]), (box[3],box[4]), (G, G))
    (; zg, P = MatrixPencil(schur(MatrixDepot.golub(T, n))))
end

# best-of-REPS wall-clock seconds (assumes kernels already compiled via warmup)
function bestof(f; reps=REPS)
    best = Inf
    for _ in 1:reps
        GC.gc(); t = @elapsed f(); best = min(best, t)
    end
    best
end

# ---- one-time global warmup: triggers all GPU-kernel compilation ----
logln("[", clock(), "] warmup (compiling GPU kernels)...")
let w = build(256)
    twarm = @elapsed ihlpsa(CUDABackend(), w.zg, w.P, NIT)        # all devices
    logln("[", clock(), "] warmup done in ", @sprintf("%.1f s", twarm))
end

#############################################
# 1. STRONG SCALING
#############################################
const SS_N = 1024
logln("\n## 1. STRONG SCALING  golub n=", SS_N)
ss = build(SS_N)
ss_csv = open(joinpath(RESULTS, "strong_scaling.csv"), "w")
println(ss_csv, "ndev,time_s,gridpts_per_s,speedup,efficiency")
logln(@sprintf("%-6s %-11s %-13s %-9s %-10s", "ndev", "time(s)", "Mgridpts/s", "speedup", "eff"))
t1 = NaN
for ndev in 1:N_GPU
    devs = ALLDEVS[1:ndev]
    t = bestof(() -> ihlpsa(CUDABackend(), ss.zg, ss.P, NIT; devs=devs))
    global t1; ndev == 1 && (t1 = t)
    gpps, spd = (G*G)/t, t1/t
    logln(@sprintf("[%s] %-3d  %-11.4f %-13.3f %-9.2f %-6.1f%%", clock(), ndev, t, gpps/1e6, spd, 100*spd/ndev))
    println(ss_csv, @sprintf("%d,%.6f,%.3f,%.4f,%.4f", ndev, t, gpps, spd, spd/ndev)); flush(ss_csv)
end
close(ss_csv)

#############################################
# 2. SIZE SWEEP  (all GPUs)
#############################################
logln("\n## 2. SIZE SWEEP  (all ", N_GPU, " GPUs)")
sz_csv = open(joinpath(RESULTS, "size_sweep.csv"), "w")
println(sz_csv, "n,ihlpsa_s,schur_s,gridpts_per_s")
logln(@sprintf("%-7s %-11s %-11s %-13s", "n", "ihlpsa(s)", "schur(s)", "Mgridpts/s"))
for n in (512, 1024, 2048, 4096)
    ts = @elapsed P = MatrixPencil(schur(MatrixDepot.golub(T, n)))
    _, _, zg = qgrid(T, (-2,5), (-4.5,4.5), (G, G))
    t = bestof(() -> ihlpsa(CUDABackend(), zg, P, NIT))
    logln(@sprintf("[%s] %-7d %-11.4f %-11.4f %-13.3f", clock(), n, t, ts, (G*G)/t/1e6))
    println(sz_csv, @sprintf("%d,%.6f,%.6f,%.3f", n, t, ts, (G*G)/t)); flush(sz_csv)
end
close(sz_csv)

#############################################
# 3. CORRECTNESS  1-GPU vs ALL-GPU
#############################################
logln("\n## 3. CORRECTNESS  (1 GPU vs ", N_GPU, " GPUs)")
let cp = build(512)
    r1 = ihlpsa(CUDABackend(), cp.zg, cp.P, NIT; devs=ALLDEVS[1:1])
    rN = ihlpsa(CUDABackend(), cp.zg, cp.P, NIT; devs=ALLDEVS)
    rel = maximum(abs.(r1 .- rN)) / maximum(abs.(r1))
    logln(@sprintf("  size=%s  relerr=%.3e  -> %s", string(size(r1)), rel, rel < 1e-5 ? "PASS" : "CHECK"))
end

logln("\n", "="^72)
logln("DONE. results in ", RESULTS)
close(logio)
