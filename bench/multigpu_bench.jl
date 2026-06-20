# Multi-GPU benchmark for ihlpsa -- runs on the full node (all CUDA devices), covering
# BOTH the fixed-`nit` and the adaptive (per-point hybrid) drivers.
#
#   1. Strong scaling: fixed problem, sweep ndev = 1..N_GPU, report speedup/efficiency
#   2. Size sweep:     all GPUs, sweep matrix size n
#
# (1-GPU vs all-GPU correctness is covered by the test suite, not here.)
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/multigpu_bench.jl
#
# ihlpsa(...; devs=...) returns a host Matrix, so the call is fully synchronized on
# return -> plain wall-clock @elapsed is valid for the multi-device case.

using CUDA
include(joinpath(@__DIR__, "bench_common.jl"))   # deps + shared helpers (logger, bestof, …)

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)
const ALLDEVS = collect(CUDA.devices())
const N_GPU   = length(ALLDEVS)
const T       = ComplexF32
const NIT     = 8
const G       = 300                 # 300x300 = 90_000 grid points; divisible by 1..6
const REPS    = 2                   # measured reps (best-of); global warmup done once up front

# The two drivers, as (label, runner) pairs. Fixed runs NIT iterations everywhere;
# adaptive omits nit and retires each point at its own converged depth. Both return a
# host Matrix, so @elapsed is valid.
run_fixed(zg, P, devs)    = ihlpsa(CUDABackend(), zg, P, NIT; devs=devs)
run_adaptive(zg, P, devs) = ihlpsa(CUDABackend(), zg, P; devs=devs)
const MODES = (("fixed", run_fixed), ("adaptive", run_adaptive))

logln, logio = bench_logger(joinpath(RESULTS, "bench_log.txt"))

logln("="^72)
logln("KAPseudospectra.jl multi-GPU benchmark   ", Dates.now())
logln("GPUs: ", N_GPU, " x ", CUDA.name(ALLDEVS[1]),
      @sprintf("  (%.1f GiB each)", CUDA.totalmem(ALLDEVS[1])/2^30))
logln("Julia threads: ", Threads.nthreads(),
      "   T=", T, "  grid=", G, "x", G, " (", G*G, " pts)  fixed nit=", NIT)
logln("="^72)

build(n; box=(-2,5,-4.5,4.5)) = begin
    _, _, zg = qgrid(T, (box[1],box[2]), (box[3],box[4]), (G, G))
    (; zg, P = golub_pencil(T, n))
end

# ---- one-time global warmup: triggers all GPU-kernel compilation (both drivers) ----
logln("[", clock(), "] warmup (compiling GPU kernels)...")
let w = build(256)
    twarm = @elapsed begin
        run_fixed(w.zg, w.P, ALLDEVS)
        run_adaptive(w.zg, w.P, ALLDEVS)
    end
    logln("[", clock(), "] warmup done in ", @sprintf("%.1f s", twarm))
end

#############################################
# 1. STRONG SCALING
#############################################
function strong_scaling()
    SS_N = 1024
    logln("\n## 1. STRONG SCALING  golub n=", SS_N)
    ss = build(SS_N)
    ss_csv = open(joinpath(RESULTS, "strong_scaling.csv"), "w")
    println(ss_csv, "mode,ndev,time_s,gridpts_per_s,speedup,efficiency")
    for (name, runner) in MODES
        logln(@sprintf("  [%s]", name))
        logln(@sprintf("  %-6s %-11s %-13s %-9s %-10s", "ndev", "time(s)", "Mgridpts/s", "speedup", "eff"))
        t1 = NaN
        for ndev in 1:N_GPU
            devs = ALLDEVS[1:ndev]
            t = bestof(() -> runner(ss.zg, ss.P, devs); reps=REPS)
            ndev == 1 && (t1 = t)
            gpps, spd = (G*G)/t, t1/t
            logln(@sprintf("  [%s] %-3d  %-11.4f %-13.3f %-9.2f %-6.1f%%", clock(), ndev, t, gpps/1e6, spd, 100*spd/ndev))
            println(ss_csv, @sprintf("%s,%d,%.6f,%.3f,%.4f,%.4f", name, ndev, t, gpps, spd, spd/ndev)); flush(ss_csv)
        end
    end
    close(ss_csv)
end

#############################################
# 2. SIZE SWEEP  (all GPUs)
#############################################
function size_sweep()
    logln("\n## 2. SIZE SWEEP  (all ", N_GPU, " GPUs)")
    sz_csv = open(joinpath(RESULTS, "size_sweep.csv"), "w")
    println(sz_csv, "n,mode,ihlpsa_s,pencil_s,gridpts_per_s")
    logln(@sprintf("  %-7s %-9s %-11s %-11s %-13s", "n", "mode", "ihlpsa(s)", "pencil(s)", "Mgridpts/s"))
    for n in (512, 1024, 2048, 4096)
        ts = @elapsed P = golub_pencil(T, n)
        _, _, zg = qgrid(T, (-2,5), (-4.5,4.5), (G, G))
        for (name, runner) in MODES
            t = bestof(() -> runner(zg, P, ALLDEVS); reps=REPS)
            logln(@sprintf("  [%s] %-7d %-9s %-11.4f %-11.4f %-13.3f", clock(), n, name, t, ts, (G*G)/t/1e6))
            println(sz_csv, @sprintf("%d,%s,%.6f,%.6f,%.3f", n, name, t, ts, (G*G)/t)); flush(sz_csv)
        end
    end
    close(sz_csv)
end

strong_scaling()
size_sweep()

logln("\n", "="^72)
logln("DONE. results in ", RESULTS)
close(logio)
