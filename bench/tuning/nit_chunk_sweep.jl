# MANUAL tuning study — not wired into CI: its conclusion drives the shipped `nit_chunk`
# default; re-run by hand only when that default is in question on new hardware.
#
# Sweeps the adaptive `nit_chunk` crossover (retirement granularity vs. per-chunk host
# round-trip). Output per (m, nit_chunk): best-of-REPS wall-clock, throughput, nit_used
# (depth reached), est_chunks = ceil(nit_used/nit_chunk); winning nit_chunk per m is flagged.
#
# Backend-selectable: pass cuda | oneapi | amdgpu | cpu as ARGS[1] (default cuda).
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=bench bench/tuning/nit_chunk_sweep.jl oneapi

include(joinpath(@__DIR__, "..", "bench_common.jl"))   # deps + shared helpers

const BACKEND = select_backend(ARGS; default = "cuda")

const RESULTS = results_dir(joinpath("results", "adaptive"))
const T = ComplexF32
const REGION = ((-1, 3), (-3, 3))
# Discrete-GPU defaults; shrink for a weak iGPU via env, e.g.
#   NSWEEP_G=120 NSWEEP_MS=32,64,128,256 julia ... bench/tuning/nit_chunk_sweep.jl oneapi
# Shared BENCH_* knobs take precedence over the older NSWEEP_* names.
# Fixed default, not the suite grid tiers: this sweeps one G across every m.
const G = haskey(ENV, "NSWEEP_G") && !haskey(ENV, "BENCH_GRIDN") ?
          parse(Int, ENV["NSWEEP_G"]) :
          env_int("BENCH_GRIDN", KernelAbstractions.isgpu(BACKEND) ? 128 : 64)
const MS = env_ints("BENCH_MS",
    Tuple(parse.(Int, split(get(ENV, "NSWEEP_MS", "64,128,256,512,1024"), ","))))
const CHUNKS = [1, 2, 3, 4, 6, 8, 12, 16]                 # nit_chunk values to sweep
const REPS = env_int("BENCH_REPS", parse(Int, get(ENV, "NSWEEP_REPS", "3")))

logln, logio = bench_logger(joinpath(RESULTS, "nit_chunk_sweep_log.txt"))

# Device count for the header; `reclaim_all` (from bench_common) resets the per-device
# allocator state between configs so none inherits another's.
const NDEV = KernelAbstractions.isgpu(BACKEND) ?
             length(collect(KAPseudospectra.devices(BACKEND))) : 1

# best-of-REPS adaptive run at a given nit_chunk. `_ihlpsa_adaptive` returns (σ, nit_grid);
# nu = maximum(nit_grid) is the depth reached. nit_max stays fixed across the sweep.
function best_adaptive(zg, P, nit_chunk, nit_max, zpd)
    best = Inf;
    nu = 0;
    iters = 0
    for _ in 1:REPS
        reclaim_all(BACKEND);
        local ng = nothing
        t = @elapsed ((_, ng) = KAPseudospectra._ihlpsa_adaptive(
            BACKEND, zg, P; nit_chunk = nit_chunk, nit_max = nit_max, zpd = zpd))
        # nu = depth reached; iters = Σ nit_grid = true work (g·nit would over-credit skipped
        # iterations). Captured at the best rep.
        t < best && (best = t; nu = maximum(ng); iters = total_iters(ng))
    end
    (best, nu, iters)
end

_, _, zg = qgrid(T, REGION[1], REGION[2], (G, G))

logln("="^88)
logln("ADAPTIVE nit_chunk crossover sweep (best-of-", REPS, ")   ", Dates.now())
logln("Grcar  grid=", G, "x", G, " (", G*G, " pts)  T=", T, "  backend=", BACKEND,
    "  ndev=", NDEV)
foreach(logln, repro_stamp(BACKEND))
logln("="^88)

csv = open_csv(joinpath(RESULTS, "nit_chunk_sweep.csv"),
    "m,nit_chunk,nit_max,time_s,gridpts_per_s,per_shift_iter_s,gflops,iters," *
    "nit_used,est_chunks,rel_to_chunk2,is_best")

for m in MS
    P = MatrixPencil(MatrixDepot.grcar(T, m))
    eye = KAPseudospectra.b_is_identity(P)
    nit_max = 8 * ceil(Int, log2(m))        # = the ihlpsa default cap
    # Pin the batch size for this m so every nit_chunk sees the same batching (else the
    # chunk comparison is confounded by an allocator-driven zpd — see bench_common).
    zpd = pinned_zpd(BACKEND, T, m, nit_max; ngrid = G * G)
    logln("")
    logln("[", clock(), "] m=", m, "  nit_max=", nit_max, "  zpd=", zpd)
    logln(@sprintf("  %-9s %-11s %-12s %-11s %-8s %-9s %-10s",
        "nit_chunk", "time(s)", "gridpts/s", "GFLOP/s", "nit", "chunks", "rel(c=2)"))

    # warmup this m (compile + allocator) before timing
    best_adaptive(zg, P, 2, nit_max, zpd)

    times = Dict{Int, Float64}();
    nuses = Dict{Int, Int}();
    iterz = Dict{Int, Int}()
    for c in CHUNKS
        t, nu, it = best_adaptive(zg, P, c, nit_max, zpd)
        times[c] = t;
        nuses[c] = nu;
        iterz[c] = it
    end
    base = get(times, 2, minimum(values(times)))   # reference = default nit_chunk=2
    bestc = argmin(times)
    for c in CHUNKS
        t, nu, it = times[c], nuses[c], iterz[c]
        chunks = cld(nu, c)                          # ≈ host round-trips over a point's life
        rel = t / base                               # <1 means faster than the default
        gf = gflops(t, m, it; eye)
        flag = c == bestc ? "  <-- best" : ""
        logln(@sprintf("  %-9d %-11.4f %-12.1f %-11.3f %-8d %-9d %-10.3f%s",
            c, t, (G*G)/t, gf, nu, chunks, rel, flag))
        println(csv,
            @sprintf("%d,%d,%d,%.6f,%.3f,%.6e,%.4f,%d,%d,%d,%.4f,%d",
                m, c, nit_max, t, (G*G)/t, per_shift_iter(t, it), gf, it, nu, chunks, rel,
                c == bestc ? 1 : 0))
        flush(csv)
    end
    logln(@sprintf("  best nit_chunk for m=%d: %d  (%.1f%% vs default c=2)",
        m, bestc, 100*(1 - times[bestc]/base)))
end
close(csv)
logln("")
logln("="^88)
logln("DONE.")
close(logio)
