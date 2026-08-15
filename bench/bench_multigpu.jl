# Multi-device scaling of ihlpsa, f32 and f64, on every device of the chosen backend (needs
# ≥ 2 devices of one backend; see below).
#
#   1. Strong scaling: fixed problem (BENCH_SS_N), sweep ndev = 1..N, speedup/efficiency
#   2. Size sweep:     all devices, sweep matrix size n (BENCH_MG_NS)
#
# BOTH drivers run here (the sanctioned exception to the fixed-only-outside-bench_drivers
# policy) to check whether the adaptive win survives multi-device load imbalance. The legs
# are NOT the same deliverable — fixed runs `default_nit`, adaptive gets `converged_nit_max`
# headroom — so compare per-driver efficiencies, never absolute times across legs.
#
# The package fan-out partitions grid columns into balanced blocks (sizes differ by ≤ 1), so
# the tiered grids need no divisibility relation to the device count. ihlpsa returns a
# synchronized host Matrix, so plain wall-clock @elapsed is valid multi-device.
#
#   JULIA_NUM_THREADS=auto julia --project=bench bench/bench_multigpu.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))

const backend = select_backend(ARGS; default = "cuda")
KernelAbstractions.isgpu(backend) ||
    error("bench_multigpu needs a GPU backend (got $(backend)); pass cuda|amdgpu|oneapi.")
const RESULTS = results_dir()
const ALLDEVS = collect(KernelAbstractionsPseudospectra.devices(backend))
const NAVAIL = length(ALLDEVS)
# Optional 2nd positional arg caps the device count (strong scaling needs ≥ 2 devices of ONE
# backend; self-skips below when only one is in play).
#   julia --project=bench bench/bench_multigpu.jl cuda 2      # first 2 CUDA devices
const NREQ = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : NAVAIL
NREQ > NAVAIL && @warn "requested $NREQ devices but only $NAVAIL available; using $NAVAIL"
const DEVS = ALLDEVS[1:min(NREQ, NAVAIL)]
const N_GPU = length(DEVS)

const TOKS = runnable_eltypes(backend, eltype_tokens("f32,f64"))
const SS_N = env_int("BENCH_SS_N", 1024)
const NS = env_ints("BENCH_MG_NS", (512, 1024, 2048, 4096))
const REPS = env_int("BENCH_REPS", 3)

run_fixed(zg, P, nit, devs, zpd) = ihlpsa(backend, zg, P, nit; devs, zpd)
# Adaptive gets converged_nit_max headroom (default_nit would cap every point — see bench_drivers).
run_adaptive(zg, P, nit, devs, zpd) =
    ihlpsa(backend, zg, P; nit_max = converged_nit_max(size(P, 1)), devs, zpd)
const MODES = (("fixed", run_fixed), ("adaptive", run_adaptive))

# Total inner iterations for the FoM: fixed = g·nit; adaptive = Σ nit_grid via one untimed
# internal-driver call (total work is ndev-independent, so compute it once per pencil).
mode_iters(name, zg, P, nit, zpd, g) =
    name == "fixed" ? total_iters(g, nit) :
    total_iters(last(KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P;
        nit_max = converged_nit_max(size(P, 1)), devs = DEVS, zpd)))

logln, logio = bench_logger(joinpath(RESULTS, "bench_multigpu_log.txt"))
logln("="^72)
logln("KernelAbstractionsPseudospectra.jl multi-device benchmark   ", Dates.now())
logln("backend=", backend, "  devices: ", N_GPU, " of ", NAVAIL, " x ", device_name(backend),
    @sprintf("  (%.1f GiB free)", KernelAbstractionsPseudospectra.device_bytes_available(backend)/2^30))
logln("Julia threads: ", Threads.nthreads(), "   eltypes=", join(TOKS, ","),
    "  strong-scaling n=", SS_N, "  size sweep n=", NS)
foreach(logln, repro_stamp(backend))
logln("="^72)

function strong_scaling(tok, T)
    if N_GPU < 2
        logln("\n## STRONG SCALING [", tok, "] — skipped (needs ≥ 2 devices of one backend; ",
            "have ", N_GPU, "). Size sweep below still runs on the single device.")
        return
    end
    ss = bench_setup(backend, T, SS_N)
    g = ss.gridn^2
    eye = KernelAbstractionsPseudospectra.b_is_identity(ss.P)
    logln("\n## STRONG SCALING [", tok, "]  n=", SS_N, "  grid=", ss.gridn, "²  nit=", ss.nit)
    for (name, runner) in MODES
        iters = mode_iters(name, ss.zg, ss.P, ss.nit, ss.zpd, g)
        logln(@sprintf("  [%s]  iters=%d", name, iters))
        logln(@sprintf("  %-6s %-11s %-13s %-11s %-9s %-10s",
            "ndev", "time(s)", "Mgridpts/s", "GFLOP/s", "speedup", "eff"))
        t1 = NaN
        for ndev in 1:N_GPU
            devsn = DEVS[1:ndev]
            t = bestof(() -> runner(ss.zg, ss.P, ss.nit, devsn, ss.zpd), backend; reps = REPS)
            ndev == 1 && (t1 = t)
            spd, gf = t1 / t, gflops(t, SS_N, iters; eye)
            logln(@sprintf("  [%s] %-3d  %-11.4f %-13.3f %-11.3f %-9.2f %-6.1f%%",
                clock(), ndev, t, g/t/1e6, gf, spd, 100*spd/ndev))
            println(SS_CSV, @sprintf("%s,%s,%d,%d,%d,%.6f,%.3f,%.6e,%.4f,%.4f,%.4f,%s",
                tok, name, ndev, SS_N, ss.gridn, t, g/t, per_shift_iter(t, iters), gf,
                spd, spd/ndev, pencil_label()))
            flush(SS_CSV)
        end
    end
end

function size_sweep(tok, T)
    logln("\n## SIZE SWEEP [", tok, "]  (", N_GPU, " device", N_GPU == 1 ? "" : "s", ")")
    logln(@sprintf("  %-7s %-9s %-6s %-11s %-13s %-11s", "n", "mode", "grid", "time(s)",
        "Mgridpts/s", "GFLOP/s"))
    for n in NS
        w = bench_setup(backend, T, n)
        g = w.gridn^2
        eye = KernelAbstractionsPseudospectra.b_is_identity(w.P)
        for (name, runner) in MODES
            iters = mode_iters(name, w.zg, w.P, w.nit, w.zpd, g)
            t = bestof(() -> runner(w.zg, w.P, w.nit, DEVS, w.zpd), backend; reps = REPS)
            logln(@sprintf("  [%s] %-7d %-9s %-6d %-11.4f %-13.3f %-11.3f",
                clock(), n, name, w.gridn, t, g/t/1e6, gflops(t, n, iters; eye)))
            println(SZ_CSV, @sprintf("%s,%d,%s,%d,%d,%.6f,%.3f,%.6e,%.4f,%d,%s",
                tok, n, name, w.gridn, w.nit, t, g/t, per_shift_iter(t, iters),
                gflops(t, n, iters; eye), iters, pencil_label()))
            flush(SZ_CSV)
        end
    end
end

const SS_CSV = open_csv(joinpath(RESULTS, "strong_scaling.csv"),
    "eltype,mode,ndev,n,gridn,time_s,gridpts_per_s,per_shift_iter_s,gflops," *
    "speedup,efficiency,pencil")
const SZ_CSV = open_csv(joinpath(RESULTS, "size_sweep.csv"),
    "eltype,n,mode,gridn,nit,time_s,gridpts_per_s,per_shift_iter_s,gflops,iters,pencil")

for tok in TOKS
    T = ELTYPES[tok]
    logln("\n[", clock(), "] warmup [", tok, "] (compiling GPU kernels)...")
    let w = bench_setup(backend, T, 256)
        twarm = @elapsed begin
            run_fixed(w.zg, w.P, w.nit, DEVS, w.zpd)
            run_adaptive(w.zg, w.P, w.nit, DEVS, w.zpd)
        end
        logln("[", clock(), "] warmup done in ", @sprintf("%.1f s", twarm))
    end
    strong_scaling(tok, T)
    size_sweep(tok, T)
end
close(SS_CSV)
close(SZ_CSV)

logln("\n", "="^72)
logln("DONE. results in ", RESULTS)
close(logio)
