# The naive GPU baseline: materialize (zB − A) per shift and call a vendor batched TRSM,
# against our three strategies. A memory-traffic comparison — ours shares one triangular
# factor (two for a pencil) across the whole batch (Θ(m²) DRAM, batch-independent); naive
# needs `batch` distinct triangles since the shift lives in the factor (Θ(batch·m²), no reuse
# possible). Reported per leg: wall time, GFLOP/s, the compulsory-traffic model, the implied
# DRAM bandwidth, and the materialization cost.
#
# CUDA/cuBLAS only (no portable batched TRSM exists), and needs a full-rate-FP64 device to be
# meaningful.
#
#   julia --project=bench bench/bench_baseline.jl cuda
#   julia --project=bench bench/bench_baseline.jl cuda --counters   # one launch per leg, for ncu
#
# `--counters` drops all timing and issues exactly ONE launch per (m, leg) after a warm-up, so a
# wrapping `ncu --metrics dram__bytes.sum` attributes traffic cleanly and gives the ground truth
# for the model below. If the model and the counter disagree, trust the counter.

include("bench_common.jl")

using Adapt, ArraysOfArrays
using CUDA
using NVTX
using CUDA.CUBLAS: trsm_batched!

backend = select_backend(filter(a -> !startswith(a, "--"), ARGS))
backend isa typeof(CUDABackend()) || error(
    "bench_baseline is CUDA-only: it compares against cuBLAS trsm_batched, which has no " *
    "portable equivalent. Got $(backend).")

const MS = env_ints("BENCH_BASE_MS", [128, 256, 512, 1024])
const COHORT = env_int("BENCH_BASE_COHORT", 1024)   # shifts solved per batched call
const NIT = env_int("BENCH_BASE_NIT", 8)            # Lanczos iterations to amortize over
const REPS = env_int("BENCH_REPS", 3)
const TOKS = runnable_eltypes(backend, eltype_tokens("f64"))
const RESULTS = results_dir()
const COUNTERS = "--counters" in ARGS

# Compulsory DRAM traffic per (forward + backward) solve of `batch` shifts, in bytes — a LOWER
# bound (perfect caching within a call; `--counters` measures what actually moves). ours: the
# factor(s) are counted ONCE regardless of batch, plus per-shift RHS read+write. naive: `batch`
# distinct m²/2 triangles, each read in full by both sweeps, plus the same RHS traffic. Both
# count the pencil's second factor when B ≠ I.
ours_bytes(T, m, batch; eye) = sizeof(T) * ((eye ? 1 : 2) * m^2 + 4 * batch * m)
naive_bytes(T, m, batch; eye) = sizeof(T) * (batch * m^2 + 4 * batch * m)   # 2 sweeps × m²/2

# One cohort of shifts, the shared factors, and the RHS block — the state both legs solve with.
function make_state(T, m, batch)
    bg = KAPseudospectra.get_bgarray(backend)
    rng = Random.Xoshiro(0xBA5E + m)
    P = adapt(bg, bench_pencil(T, m).P)
    zv = adapt(bg, T.(ComplexF64(2) .+ ComplexF64(3 // 10) .* randn(rng, ComplexF64, batch)))
    bM = adapt(bg, T.(reduce(hcat, [randn(rng, ComplexF64, m) for _ in 1:batch])))
    # `column_wgs`, not `_auto_wgs` — same trap as bench_kernels: the raw heuristic ignores a
    # schedule persisted by `tune_trsm_wgs!`.
    (; P, zv, bM, eye = KAPseudospectra.b_is_identity(P),
        wgs = KAPseudospectra.column_wgs(backend, P))
end

# The naive leg's working set: `batch` explicit m×m upper-triangular shifted factors — an OOM
# here is a result, not a failure.
function materialize(st, T, m, batch)
    A, eye = st.P.A, st.eye
    # A standard pencil stores B as a `Diagonal`, which neither broadcasts reliably against a
    # CuMatrix nor supports scalar indexing. Materializing ONE dense device identity sidesteps
    # both for a single m² buffer against the batch·m² allocated next anyway.
    Bd = eye ? adapt(KAPseudospectra.get_bgarray(backend), Matrix{T}(I, m, m)) : st.P.B
    z = Array(st.zv)
    Ms = [CuMatrix{T}(undef, m, m) for _ in 1:batch]
    for i in 1:batch
        Ms[i] .= z[i] .* Bd .- A
    end
    CUDA.synchronize()
    Ms
end

# ---- legs -----------------------------------------------------------------------------------
# Each returns a zero-argument closure performing ONE (forward + backward) solve of the cohort,
# so the timing harness below is identical across legs and the FLOP count is by construction.

function leg_ours(st, strategy)
    bV = VectorOfSimilarVectors(st.bM)
    strategy == "column" ?
    (() -> KAPseudospectra._column_trsm!(backend, bV, st.zv, st.P, st.wgs)) :
    (() -> KAPseudospectra._tiled_trsm!(backend, bV, st.zv, st.P, st.wgs;
        gemm = strategy == "tiled-gemm"))
end

# cuBLAS against the materialized triangles, in the SAME operator order as `_column_trsm!`:
# forward sweep with 'C' (conj(z), i.e. M*) then backward with 'N' (z), so the composite is
# x = M⁻¹M⁻*b = (M*M)⁻¹b — the normal-equations apply inverse Hermitian Lanczos wants. `uplo`
# is 'U' throughout (the stored triangle is upper either way). `verify` below checks this
# rather than trusting it — an order slip here would silently benchmark the wrong operator.
function leg_naive(st, Ms, T, m, batch)
    Bs = [view(st.bM, :, i:i) for i in 1:batch]
    one_T = one(T)
    function ()
        trsm_batched!('L', 'U', 'C', 'N', one_T, Ms, Bs)
        trsm_batched!('L', 'U', 'N', 'N', one_T, Ms, Bs)
    end
end

# Do the two legs compute the same thing? Without this a transposed/reordered cuBLAS call
# would look like a fast win. Both solves are in place on `st.bM`, so each leg starts from
# the same saved right-hand side.
function verify(st, Ms, T, m, batch)
    Ms === nothing && return NaN
    b0 = copy(st.bM)
    st.bM .= b0
    leg_ours(st, "column")(); KernelAbstractions.synchronize(backend)
    xo = Array(st.bM)
    st.bM .= b0
    leg_naive(st, Ms, T, m, batch)(); KernelAbstractions.synchronize(backend)
    xn = Array(st.bM)
    st.bM .= b0
    scale = maximum(abs, xo)
    maximum(abs, xo .- xn) / (scale > 0 ? scale : one(real(T)))
end

# ---- harness --------------------------------------------------------------------------------

csv = COUNTERS ? nothing : open_csv(joinpath(RESULTS, "bench_baseline.csv"),
    "backend,eltype,m,batch,nit,leg,pencil," *
    "materialize_s,solve_s,per_solve_s,gflops," *
    "model_bytes,model_GBs,workset_bytes,maxdiff_vs_ours")

foreach(println, repro_stamp(backend))
@printf("cohort=%d  nit=%d  pencil=%s\n", COHORT, NIT, pencil_label())
@printf("%-6s %-12s %-11s %-11s %-10s %-9s\n",
    "m", "leg", "solve(s)", "per-solve", "GFLOP/s", "GB/s")

for tok in TOKS
    T = ELTYPES[tok]
    for m in MS
        batch = COHORT
        st = make_state(T, m, batch)
        # 8m² per shift per (fwd+bwd) pair for a standard pencil, 16m² generalized — the same
        # accounting bench_common uses everywhere else, so GFLOP/s is comparable across scripts.
        fl = batch * (st.eye ? 8 : 16) * m^2 * NIT

        # The naive working set can exceed device memory long before ours does, so record it
        # and carry on to the next size rather than dying.
        wset = sizeof(T) * batch * m^2
        free = KAPseudospectra.device_bytes_available(backend)
        Ms = nothing
        tmat = NaN
        if wset < 0.7 * free
            # Warm up the broadcast kernel on a throwaway 2-triangle build first, or the FIRST
            # size's materialize_s is dominated by compilation.
            materialize(st, T, m, 2)
            tmat = @elapsed Ms = NVTX.@range "materialize/m$m" materialize(st, T, m, batch)
        else
            @printf("%-6d %-12s SKIPPED — needs %.1f GiB of %.1f GiB free\n",
                m, "naive", wset / 2^30, free / 2^30)
        end

        # Correctness gate before any timing: if the legs disagree, the comparison is void.
        vd = verify(st, Ms, T, m, batch)
        isnan(vd) || @printf("%-6d %-12s rel. maxdiff vs column = %.3e%s\n", m, "(verify)", vd,
            vd > 1e-8 ? "   *** LEGS DISAGREE — timings below are not comparable ***" : "")

        legs = Any[("column", leg_ours(st, "column")),
            ("tiled", leg_ours(st, "tiled")),
            ("tiled-gemm", leg_ours(st, "tiled-gemm"))]
        Ms === nothing || push!(legs, ("cublas-naive", leg_naive(st, Ms, T, m, batch)))

        for (name, f) in legs
            f(); KernelAbstractions.synchronize(backend)      # warm up (compile + first touch)
            if COUNTERS
                # NVTX-tagged so a profiler attributes kernels to a LEG, not a mangled kernel
                # name — the four legs share kernels (tiled-gemm's cutlass GEMM, the
                # pointer-array setup).
                NVTX.@range "leg/$name/m$m" begin
                    f()
                    KernelAbstractions.synchronize(backend)
                end
                println("  counters: m=$m leg=$name")
                continue
            end
            t = bestof(backend; reps = REPS, warmup = true) do
                for _ in 1:NIT
                    f()
                end
            end
            naive = name == "cublas-naive"
            bytes = NIT * (naive ? naive_bytes(T, m, batch; eye = st.eye) :
                           ours_bytes(T, m, batch; eye = st.eye))
            gbs = bytes / t / 1e9
            @printf("%-6d %-12s %-11.6f %-11.3e %-10.1f %-9.1f\n",
                m, name, t, t / (batch * NIT), fl / t / 1e9, gbs)
            println(csv, @sprintf("%s,%s,%d,%d,%d,%s,%s,%s,%.6e,%.6e,%.3f,%d,%.2f,%d,%s",
                backend_tag(backend), tok, m, batch, NIT, name, pencil_label(),
                isnan(tmat) ? "NaN" : @sprintf("%.6f", tmat),
                t, t / (batch * NIT), fl / t / 1e9,
                bytes, gbs, naive ? wset : ours_bytes(T, m, batch; eye = st.eye),
                isnan(vd) ? "NaN" : @sprintf("%.3e", vd)))
            flush(csv)
        end
        Ms = nothing
        CUDA.reclaim()
    end
end

COUNTERS || close_csv(csv, joinpath(RESULTS, "bench_baseline.csv"))
