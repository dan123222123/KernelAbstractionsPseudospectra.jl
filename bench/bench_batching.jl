# The two batching knobs.
#
#   batch sweep — isolated column solve at fixed m: one batched launch versus a host loop of
#                 batch-of-1 launches, the kernel-launch-bound baseline.
#   zpd sweep   — end-to-end fixed-`nit` ihlpsa across grid batch sizes, against the workspace
#                 model `findmaxbatchihl` budgets against.
#
# FIXED driver only: the adaptive driver's row-gather scratch adds a second, uncontrolled term
# to the footprint the zpd sweep is trying to measure.
#   julia --project=bench bench/bench_batching.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))
using Adapt, ArraysOfArrays

const backend = select_backend(ARGS; default = "cuda")
KernelAbstractions.isgpu(backend) ||
    error("bench_batching needs a GPU backend (got $(backend)); pass cuda|amdgpu|oneapi.")
const devs = KernelAbstractionsPseudospectra.devices(backend)
const RESULTS = results_dir()
const REPS = env_int("BENCH_REPS", 3)
const TOKS = runnable_eltypes(backend, eltype_tokens("f64"))

const BATCH_M = env_int("BENCH_BATCH_M", 128)
const BATCHES = env_ints("BENCH_BATCHES", (1, 4, 16, 64, 256, 1024, 4096, 16384))
# Past UNBATCHED_MAX per-launch latency dominates and the line is flat; only batched runs beyond it.
const UNBATCHED_MAX = env_int("BENCH_UNBATCHED_MAX", 1024)
const ZPD_M = env_int("BENCH_ZPD_M", 512)

# One triangular solve is half an inverse-Lanczos iteration's 8m².
solve_flops(m, batch; eye) = batch * (eye ? 4 : 8) * m^2

# The workspace `findmaxbatchihl` budgets against; mirrors the model in src/ihlpsa.jl.
ihl_workspace_bytes(T, m, nit, zpd) =
    sizeof(T) * ((4 * m^2 + 1) + (4 * m + 2 * nit + 2) * zpd)

# Batched and unbatched closures over one allocation; the solve is in-place, so reps overwrite
# the previous result (values drift, but FLOPs/traffic per rep don't).
function make_solves(T, m, batch)
    bg = KernelAbstractionsPseudospectra.get_bgarray(backend)
    rng = Random.Xoshiro(0xBA7C + m)
    P = adapt(bg, bench_pencil(T, m).P)
    zv = adapt(bg, T.(ComplexF64(2) .+ ComplexF64(3 // 10) .* randn(rng, ComplexF64, batch)))
    bM = adapt(bg, T.(reduce(hcat, [randn(rng, ComplexF64, m) for _ in 1:batch])))
    # `column_wgs`, not the raw heuristic — races batched against unbatched COLUMN solves; a
    # `wgs` the shipped path would never choose mis-states both sides.
    wgs = KernelAbstractionsPseudospectra.column_wgs(backend, P)
    batched = () -> KernelAbstractionsPseudospectra._column_trsm!(
        backend, VectorOfSimilarVectors(bM), zv, P, wgs)
    unbatched = function ()
        for i in 1:batch
            KernelAbstractionsPseudospectra._column_trsm!(backend,
                VectorOfSimilarVectors(view(bM, :, i:i)), view(zv, i:i), P, wgs)
        end
    end
    (; batched, unbatched, eye = KernelAbstractionsPseudospectra.b_is_identity(P))
end

logln, logio = bench_logger(joinpath(RESULTS, "bench_batching_log.txt"))
logln("="^72)
logln("KernelAbstractionsPseudospectra.jl batching benchmark   ", Dates.now())
logln("backend=", backend, "  device: ", device_name(backend),
    @sprintf("  (%.1f GiB free)", KernelAbstractionsPseudospectra.device_bytes_available(backend) / 2^30))
logln("eltypes=", join(TOKS, ","), "  batch m=", BATCH_M, "  zpd m=", ZPD_M)
foreach(logln, repro_stamp(backend))
logln("="^72)

function batch_sweep(tok, T)
    logln("\n## BATCH SWEEP [", tok, "]  m=", BATCH_M)
    logln(@sprintf("  %-8s %-10s %-12s %-13s %-11s", "batch", "mode", "time(s)",
        "per-solve(s)", "GFLOP/s"))
    for batch in BATCHES
        s = make_solves(T, BATCH_M, batch)
        modes = batch <= UNBATCHED_MAX ?
                (("batched", s.batched), ("unbatched", s.unbatched)) :
                (("batched", s.batched),)
        for (name, f) in modes
            t = bestof(f, backend; reps = REPS, warmup = true)
            gf = solve_flops(BATCH_M, batch; eye = s.eye) / t / 1e9
            logln(@sprintf("  [%s] %-8d %-10s %-12.6f %-13.3e %-11.3f",
                clock(), batch, name, t, t / batch, gf))
            println(BATCH_CSV, @sprintf("%s,%d,%d,%s,%.6e,%.6e,%.4f,%s",
                tok, BATCH_M, batch, name, t, t / batch, gf, pencil_label()))
            flush(BATCH_CSV)
        end
    end
end

function zpd_sweep(tok, T)
    w = bench_setup(backend, T, ZPD_M; headroom = 1)
    pin = w.zpd
    zpds = sort(unique(push!([2^k for k in 6:20 if 2^k <= w.g], min(pin, w.g))))
    logln("\n## ZPD SWEEP [", tok, "]  m=", ZPD_M, "  grid=", w.gridn, "²  nit=", w.nit,
        "  findmaxbatchihl pin=", pin)
    logln(@sprintf("  %-9s %-8s %-12s %-13s %-12s", "zpd", "default", "time(s)",
        "Mgridpts/s", "workspace"))
    for zpd in zpds
        t = bestof(() -> ihlpsa(backend, w.zg, w.P, w.nit; devs, zpd), backend; reps = REPS)
        bytes = ihl_workspace_bytes(T, ZPD_M, w.nit, zpd)
        logln(@sprintf("  [%s] %-9d %-8s %-12.4f %-13.3f %-12.2f MiB", clock(), zpd,
            zpd == pin ? "*" : "", t, w.g / t / 1e6, bytes / 2^20))
        println(ZPD_CSV, @sprintf("%s,%d,%d,%d,%d,%.6f,%.3f,%d,%s",
            tok, ZPD_M, w.gridn, zpd, zpd == pin, t, w.g / t, bytes, pencil_label()))
        flush(ZPD_CSV)
    end
end

const BATCH_CSV = open_csv(joinpath(RESULTS, "bench_batching.csv"),
    "eltype,m,batch,mode,time_s,per_solve_s,gflops,pencil")
const ZPD_CSV = open_csv(joinpath(RESULTS, "bench_zpd.csv"),
    "eltype,m,gridn,zpd,is_default,time_s,gridpts_per_s,workspace_bytes,pencil")

for tok in TOKS
    T = ELTYPES[tok]
    logln("\n[", clock(), "] warmup [", tok, "] (compiling GPU kernels)...")
    let s = make_solves(T, BATCH_M, 64), w = bench_setup(backend, T, 256; headroom = 1)
        twarm = @elapsed begin
            s.batched()
            KernelAbstractions.synchronize(backend)
            ihlpsa(backend, w.zg, w.P, w.nit; devs, zpd = w.zpd)
        end
        logln("[", clock(), "] warmup done in ", @sprintf("%.1f s", twarm))
    end
    batch_sweep(tok, T)
    zpd_sweep(tok, T)
end
close(BATCH_CSV)
close(ZPD_CSV)

logln("\n", "="^72)
logln("DONE. results in ", RESULTS)
close(logio)
