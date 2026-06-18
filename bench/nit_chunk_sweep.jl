# Find the adaptive `nit_chunk` crossover: chunk size is a two-sided tradeoff.
#
#   - small nit_chunk -> finest retirement (each point stops near its true
#     converged depth, least wasted iteration), but MORE host round-trips: every
#     chunk copies α/β back, runs ihlsrg!/the convergence test on the host, and
#     relaunches the next chunk.
#   - large nit_chunk -> FEWER round-trips (amortizes that fixed host sync), but
#     coarser retirement, so points over-iterate up to ≈ the next chunk boundary.
#
# On GPU the per-chunk host sync can cost more than simply running a couple more
# cheap on-device iterations — most so at SMALL m (the m² solve is cheap, the
# round-trip is not) or once few points survive. So the wall-clock-optimal
# nit_chunk should be > the default (2) for small m and drift toward 2 as m grows
# and the solve dominates. This sweep measures that curve so the default / an
# eventual auto-tune (DESIGN.md: balance nit_chunk·m²·survivors against the fixed
# round-trip) can be set from data instead of guessed.
#
# Output per (m, nit_chunk): best-of-REPS wall-clock, throughput, nit_used (depth
# reached — grows with nit_chunk), and est_chunks = ceil(nit_used/nit_chunk), a
# proxy for the number of host round-trips a point's lifetime pays. The winning
# nit_chunk per m is flagged.
#
# Backend-selectable: pass cuda | oneapi | amdgpu | cpu as ARGS[1] (default cuda).
# Uses the package's backend-agnostic device interface (device_reclaim/devices), so
# the same script runs on any backend — e.g. the Intel iGPU via oneAPI.
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/nit_chunk_sweep.jl oneapi

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra, Printf, Dates

const WHICH = isempty(ARGS) ? "cuda" : lowercase(ARGS[1])
const BACKEND = if WHICH == "cuda"
    @eval using CUDA; CUDA.CUDABackend()
elseif WHICH == "oneapi"
    @eval using oneAPI; oneAPI.oneAPIBackend()
elseif WHICH == "amdgpu"
    @eval using AMDGPU; AMDGPU.ROCBackend()
elseif WHICH == "cpu"
    CPU()
else
    error("unknown backend $(WHICH); use cuda|oneapi|amdgpu|cpu")
end

const RESULTS = joinpath(@__DIR__, "results_adaptive")
mkpath(RESULTS)
const T       = ComplexF32
const REGION  = ((-1, 3), (-3, 3))
# Discrete-GPU defaults; shrink for a weak iGPU via env, e.g.
#   NSWEEP_G=120 NSWEEP_MS=32,64,128,256 julia ... bench/nit_chunk_sweep.jl oneapi
const G       = parse(Int, get(ENV, "NSWEEP_G", "300"))    # G×G grid points
const MS      = parse.(Int, split(get(ENV, "NSWEEP_MS", "64,128,256,512,1024"), ","))
const CHUNKS  = [1, 2, 3, 4, 6, 8, 12, 16]                 # nit_chunk values to sweep
const REPS    = parse(Int, get(ENV, "NSWEEP_REPS", "3"))

logio = open(joinpath(RESULTS, "nit_chunk_sweep_log.txt"), "w")
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

# reclaim memory on EVERY device so no config inherits another's allocator state
# (multi-GPU honesty: the adaptive driver fans out across all devices, so reclaim
# all of them, not just the current one — cf. strong_scaling_adaptive.jl). Uses the
# package's backend-agnostic device interface; CPU has nothing to reclaim per device.
const DEVS = KernelAbstractions.isgpu(BACKEND) ? collect(KAPseudospectra.devices(BACKEND)) : []
function reclaim_all()
    for d in DEVS
        KAPseudospectra.device!(BACKEND, d)
        KAPseudospectra.device_reclaim(BACKEND)
    end
    isempty(DEVS) && KAPseudospectra.device_reclaim(BACKEND)
    GC.gc(); GC.gc()
end

# best-of-REPS adaptive run at a given nit_chunk. The internal _ihlpsa_adaptive
# driver returns (σ, nit_grid) — we report maximum(nit_grid) as the depth reached;
# the public ihlpsa returns only σ. nit_max is held
# fixed across the chunk sweep so only the chunking granularity varies.
function best_adaptive(zg, P, nit_chunk, nit_max)
    best = Inf; nu = 0
    for _ in 1:REPS
        reclaim_all(); local ng = nothing
        t = @elapsed ((_, ng) = KAPseudospectra._ihlpsa_adaptive(
            BACKEND, zg, P; nit_chunk=nit_chunk, nit_max=nit_max))
        t < best && (best = t; nu = maximum(ng))
    end
    (best, nu)
end

_, _, zg = qgrid(T, REGION[1], REGION[2], (G, G))

logln("="^88)
logln("ADAPTIVE nit_chunk crossover sweep (best-of-", REPS, ")   ", Dates.now())
logln("Grcar  grid=", G, "x", G, " (", G*G, " pts)  T=", T, "  backend=", BACKEND,
      "  ndev=", max(1, length(DEVS)))
logln("="^88)

csv = open(joinpath(RESULTS, "nit_chunk_sweep.csv"), "w")
println(csv, "m,nit_chunk,nit_max,time_s,gridpts_per_s,nit_used,est_chunks,rel_to_chunk2,is_best")

for m in MS
    P = MatrixPencil(grcar(T, m))
    nit_max = 8 * ceil(Int, log2(m))        # = the ihlpsa default cap
    logln("")
    logln("[", clock(), "] m=", m, "  nit_max=", nit_max)
    logln(@sprintf("  %-9s %-11s %-12s %-8s %-9s %-10s", "nit_chunk", "time(s)", "gridpts/s", "nit", "chunks", "rel(c=2)"))

    # warmup this m (compile + allocator) before timing
    best_adaptive(zg, P, 2, nit_max)

    times = Dict{Int,Float64}(); nuses = Dict{Int,Int}()
    for c in CHUNKS
        t, nu = best_adaptive(zg, P, c, nit_max)
        times[c] = t; nuses[c] = nu
    end
    base = get(times, 2, minimum(values(times)))   # reference = default nit_chunk=2
    bestc = argmin(times)                            # winning chunk size for this m
    for c in CHUNKS
        t, nu = times[c], nuses[c]
        chunks = cld(nu, c)                          # ≈ host round-trips over a point's life
        rel = t / base                               # <1 means faster than the default
        flag = c == bestc ? "  <-- best" : ""
        logln(@sprintf("  %-9d %-11.4f %-12.1f %-8d %-9d %-10.3f%s", c, t, (G*G)/t, nu, chunks, rel, flag))
        println(csv, @sprintf("%d,%d,%d,%.6f,%.3f,%d,%d,%.4f,%d",
            m, c, nit_max, t, (G*G)/t, nu, chunks, rel, c == bestc ? 1 : 0))
        flush(csv)
    end
    logln(@sprintf("  best nit_chunk for m=%d: %d  (%.1f%% vs default c=2)",
        m, bestc, 100*(1 - times[bestc]/base)))
end
close(csv)
logln("")
logln("="^88)
logln("DONE.  Two regimes (see DESIGN.md): on an integrated GPU (shared RAM, ~free")
logln("round-trip) over-iteration dominates, so the best nit_chunk shrinks toward 1 as")
logln("m grows (measured: 2 at m<=128, 1 at m>=256). On a discrete GPU the per-chunk")
logln("PCIe round-trip is real, so a LARGER nit_chunk may win at small m / few survivors —")
logln("the case this sweep on a multi-GPU box is meant to pin down.")
close(logio)
