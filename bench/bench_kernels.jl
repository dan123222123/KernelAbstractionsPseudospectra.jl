# The trsm kernel study. Every (eltype rung × size × strategy) cell is measured
# at four granularities in one pass:
#   isolated   — one batched triangular solve (the roofline point), reported both LOGICAL
#                (algorithm FLOPs / analytic bytes) and EFFECTIVE / CGMA (native limb FLOPs =
#                logical × mf_expansion(T)).
#   end-to-end — the full ihlpsa over the grid (IEEE rungs converged to the precision floor,
#                MultiFloat rungs fixed-`nit`): cold first call (TTFP) and hot best-of, plus the
#                per-(shift·iteration) FoM; f32/f64 get a dispersion sample.
#   sentinel   — `maxdiff_svd`: each strategy's σ vs the dense-SVD truth (ℂsvdpsa, F64) on a coarse
#                sub-grid, solved to the arithmetic floor (the tiled-vs-column correctness check).
#   cpu        — the column solve timed on CPU at the same batch; a CPU solve that errors
#                writes NaN, never aborts the sweep.
# Isolated rows use the FIXED driver (exact launch count ⇒ exact analytic AI); the
# fixed-vs-adaptive comparison itself is bench_drivers.jl. `--counters`: one launch per cell
# for clean ncu/nvprof attribution. Eltype ladder + per-device gating: README § Policy.
#   julia --project=bench bench/bench_kernels.jl cuda
#   BENCH_ELTYPES=f64 BENCH_MS=512 julia --project=bench bench/bench_kernels.jl cuda --counters
include(joinpath(@__DIR__, "bench_common.jl"))
using Adapt, ArraysOfArrays

backend = select_backend(filter(a -> !startswith(a, "--"), ARGS))
gpu = KernelAbstractions.isgpu(backend)
devs = gpu ? KernelAbstractionsPseudospectra.devices(backend) : missing

# This experiment races KAPSEUDO_TRSM strategies. Where the shuffle isn't usable (stock oneAPI's
# @shfl stub; Metal without opt-in) `tiled` self-gates to column, making the race meaningless —
# skip and exit 0. On CUDA/AMDGPU the shuffle always exists, so a false there is a regression:
# fail red instead.
if gpu && !KernelAbstractionsPseudospectra.warp_trsm_safe(backend, false)
    msg = "tiled solve not usable on $(backend) (warp_trsm_safe == false)"
    nameof(typeof(backend)) in (:CUDABackend, :ROCBackend) &&
        error(msg * " — this backend always supports the warp shuffle; investigate the regression.")
    @warn msg * " — skipping the strategy race here."
    exit(0)
end

const REPS = env_int("BENCH_REPS", 3)
const REPS_STATS = env_int("BENCH_REPS_STATS", 15)
const MS = env_ints("BENCH_MS", gpu ? (16, 32, 64, 128, 256, 512, 1024) : (64, 128, 256))
const MS_MF = env_ints("BENCH_MS_MF", gpu ? (128, 256, 512) : (64, 128))
const CPU_MS = env_ints("BENCH_CPUGPU_MS", (128, 256, 512))
# Isolated-solve batch: kernel geometry, not a σ-grid, so it does NOT follow the grid tiers —
# fixed to keep the roofline points comparable across m and across runs.
const SOLVE_BATCH = env_int("BENCH_SOLVE_BATCH", 4096)
const TOKS = runnable_eltypes(backend, eltype_tokens("f32"))
const RESULTS = results_dir()

# One batched solve closure for `strategy` on backend `bk` (GPU, or CPU for the cpu row).
# RHS/shifts are drawn at ComplexF64 then converted to T (MultiFloats has no randn).
function make_solve(bk, T, m, strategy)
    bg = KernelAbstractionsPseudospectra.get_bgarray(bk)
    rng = Random.Xoshiro(0xF00D + m)
    P = adapt(bg, bench_pencil(T, m).P)
    zv = adapt(bg, T.(ComplexF64(2) .+ ComplexF64(3 // 10) .* randn(rng, ComplexF64, SOLVE_BATCH)))
    bV = VectorOfSimilarVectors(adapt(bg,
        T.(reduce(hcat, [randn(rng, ComplexF64, m) for _ in 1:SOLVE_BATCH]))))
    # `column_wgs`, not `_auto_wgs`: the harness must see the schedule the shipped path uses,
    # not the raw heuristic.
    wgs = KernelAbstractionsPseudospectra.column_wgs(bk, P)
    solve = strategy == "column" ?
            (() -> KernelAbstractionsPseudospectra._column_trsm!(bk, bV, zv, P, wgs)) :
            (() -> KernelAbstractionsPseudospectra._tiled_trsm!(bk, bV, zv, P, wgs;
        gemm = strategy == "tiled-gemm"))
    (; solve, P)
end

ms_for(T) = iswide(T) ? MS_MF : MS

# --counters: single launch per cell so the wrapping profiler (ncu on ≥ Volta, nvprof on
# pre-Volta — ncu categorically refuses sm < 7.0) attributes counters cleanly; no timing here.
if "--counters" in ARGS
    for tok in TOKS, m in ms_for(ELTYPES[tok]),
        strat in strategies_for(backend, ELTYPES[tok])

        s = make_solve(backend, ELTYPES[tok], m, strat)
        s.solve()
        KernelAbstractions.synchronize(backend)
    end
    exit(0)
end

# Analytic DRAM lower bound (a MODEL) for the isolated solve: triangle read once + RHS read/written
# per solve; `sizeof(T)` is where precision enters. Ground-truth bytes come from the --counters pass.
analytic_bytes(T, m, g; eye) = sizeof(T) * ((eye ? 1 : 2) * (m * (m + 1) ÷ 2) + 2 * g * m)

# Resolved tiled-solve occupancy knobs for the CSV — with the ENV > tuned-preference >
# heuristic resolution chain, "tiled" is not one configuration. GPU only; "NA" on CPU.
function resolved_tcw(P)
    gpu || return ("NA", "NA")
    try
        (string(KernelAbstractionsPseudospectra.tile_cols(backend, P)), string(KernelAbstractionsPseudospectra.block_warps(backend, P)))
    catch
        ("NA", "NA")
    end
end

run_fixed(P, zg, strategy, nit, zpd) = withenv("KAPSEUDO_TRSM" => strategy) do
    ihlpsa(backend, zg, P, nit; devs, zpd)
end

# End-to-end solve to the precision floor (adaptive) for the e2e row and the sentinel; `run_fixed`
# above stays the roofline driver (fixed launch count ⇒ exact analytic AI).
run_converged(P, zg, strategy, zpd) = withenv("KAPSEUDO_TRSM" => strategy) do
    converged_solve(backend, zg, P; devs, zpd)
end

const SVD_GRIDN = env_int("BENCH_SVD_GRIDN", 8)   # sentinel sub-grid side (SVD is O(m³)/point); 0 skips

# Each strategy's σ vs the dense-SVD ground truth (ℂsvdpsa, F64) on a coarse SVD_GRIDN² sub-grid,
# solved to the arithmetic floor (the tiled-vs-column correctness check). Reference and strategy
# grids are each built natively by qgrid over the same box (sidesteps Float64(::Float32x2)).
function svd_sentinel(P, T, m, zpd, strats)
    SVD_GRIDN <= 0 && return Dict(s => NaN for s in strats)
    box = bench_box()
    zref = last(qgrid(ComplexF64, box[1], box[2], (SVD_GRIDN, SVD_GRIDN)))
    zsub = last(qgrid(T, box[1], box[2], (SVD_GRIDN, SVD_GRIDN)))
    σref = ℂsvdpsa(zref, bench_pencil(ComplexF64, m).P)
    zpd_sub = gpu ? max(1, min(zpd, SVD_GRIDN^2)) : missing
    Dict(s => maximum(abs.(to64.(run_converged(P, zsub, s, zpd_sub)) .- σref)) for s in strats)
end

# CPU column-solve time at the same batch (NaN when it errors — MultiFloat CPU rungs can be
# minutes; never abort the GPU sweep for the comparison row).
function cpu_solve_time(T, m)
    try
        sc = make_solve(CPU(), T, m, "column")
        bestof(sc.solve, CPU(); reps = max(2, REPS ÷ 2), warmup = true)
    catch err
        @warn "cpu solve failed for $T m=$m — NaN" err
        NaN
    end
end

foreach(println, repro_stamp(backend))
@printf("device = %s   eltypes = %s   solve batch = %d\n",
    device_name(backend), join(TOKS, ","), SOLVE_BATCH)

# Roofline ceilings are applied downstream: the CSV carries measured GFLOP/s and both arithmetic
# intensities, and published per-device peaks are matched to a row by the box stamp in the filename.
csv = open_csv(joinpath(RESULTS, "bench_kernels.csv"),
    "backend,eltype,m,gridn,grid,strategy,trsm_tilecols,trsm_blockwarps,nit,zpd," *
    "iso_time_s,iso_gflops,iso_ai,iso_bw_GBs," *
    "mf_expansion,eff_gflops,eff_ai," *
    "e2e_first_s,e2e_time_s,per_shift_iter_s,gridpts_per_s,e2e_gflops,maxdiff_svd," *
    "e2e_median_s,e2e_iqr_lo_s,e2e_iqr_hi_s,reps_stats," *
    "cpu_iso_time_s,gpu_cpu_speedup,canary_gflops,cond_est,pencil")

for tok in TOKS
    T = ELTYPES[tok]
    use_stats = tok in (:f32, :f64)     # dispersion for the headline rungs only
    E = mf_expansion(T)                  # native-FLOP expansion for the effective CGMA
    @printf("\nT=%s (%s)   native-FLOP expansion E=%.1f\n", tok, T, E)
    @printf("%-6s %-11s %-10s %-9s %-6s %-10s %-10s %-10s %-9s\n",
        "m", "strategy", "iso(ms)", "iso GF/s", "E", "eff GF/s", "e2e(s)", "e2e GF/s", "svd-err")
    for m in ms_for(T)
        # bench_setup's headroom=2 default applies: the IEEE rungs' e2e/sentinel solves run
        # the ADAPTIVE driver, whose row-gather scratch sits on top of the pinned-workspace
        # model.
        (; gridn, g, zg, nit, P, A, eye, zpd) = bench_setup(backend, T, m)
        # Dense O(m³) SVD: IEEE-only and capped at m ≤ 2048 (MultiFloat has no generic SVD path,
        # and at the largest sizes an extra full SVD adds minutes).
        condA = (real(eltype(A)) <: Base.IEEEFloat && m <= 2048) ? cond(A) : NaN
        tc, w = resolved_tcw(P)
        strats = strategies_for(backend, T)

        # IEEE rungs solve to the precision floor (converged adaptive) — the headline
        # time-to-solution. MultiFloat rungs keep the FIXED driver: converging f64x4 on the
        # full grid takes hours; MF's story is the roofline/sentinel, not e2e time.
        e2e_converged = !iswide(T)
        e2e(s) = e2e_converged ? run_converged(P, zg, s, zpd) : run_fixed(P, zg, s, nit, zpd)

        # First calls are TIMED (cold TTFP at a rung's first m), separate from hot best-of. Order
        # column→tiled(→gemm) so shared kernel compile accrues to the earliest caller.
        cold = Dict{String, Float64}()
        for s in strats
            cold[s] = @elapsed e2e(s)
        end
        maxd = svd_sentinel(P, T, m, zpd, strats)   # each strategy vs dense-SVD truth (converged)

        # e2e FoM iteration count: Σ nit_grid for the converged IEEE rungs (one _ihlpsa_adaptive
        # call — strategy-independent), g·nit for the fixed MF rungs.
        iters = if e2e_converged
            total_iters(last(KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P;
                rtol = converged_rtol(T), nit_max = converged_nit_max(m), devs, zpd)))
        else
            total_iters(g, nit)
        end
        timed(f) = bestof(f, backend; reps = REPS)
        timed_stats(f) = bestof(f, backend; reps = REPS_STATS, stats = true)
        for s in strats
            iso = make_solve(backend, T, m, s)
            # Sampled beside the timing it qualifies: a co-tenant that arrives mid-sweep then
            # shows up on the affected ROWS rather than condemning the whole file.
            cgf = device_canary(backend)
            t_iso = bestof(iso.solve, backend; reps = REPS, warmup = true)
            flops1 = SOLVE_BATCH * flops_per_shift(m, 1; eye)
            bytes = analytic_bytes(T, m, SOLVE_BATCH; eye)
            gf_iso, bw, ai = flops1 / t_iso / 1e9, bytes / t_iso / 1e9, flops1 / bytes
            # Effective / CGMA view: native limb FLOPs = logical × E(T).
            eff_gf, eff_ai = gf_iso * E, ai * E

            st = use_stats ? timed_stats(() -> e2e(s)) : nothing
            tf = use_stats ? st.min : timed(() -> e2e(s))
            med, iqlo, iqhi, nreps = st === nothing ? (NaN, NaN, NaN, 0) :
                                     (st.median, st.iqr_lo, st.iqr_hi, st.n)

            tcpu = (gpu && s == "column" && m in CPU_MS) ? cpu_solve_time(T, m) : NaN
            spd = isnan(tcpu) ? NaN : tcpu / t_iso

            @printf("%-6d %-11s %-10.3f %-9.2f %-6.0f %-10.2f %-10.4f %-10.3f %.2e\n",
                m, s, t_iso * 1e3, gf_iso, E, eff_gf, tf, gflops(tf, m, iters; eye), maxd[s])
            isnan(spd) ||
                @printf("       %-11s cpu %-8.1f ms  →  %.1fx GPU/CPU\n", "(column)", tcpu * 1e3, spd)
            # @sprintf needs a literal format string, so the row is assembled in pieces.
            println(csv,
                @sprintf("%s,%s,%d,%d,%d,%s,%s,%s,%d,%s",
                    backend_tag(backend), tok, m, gridn, g, s, tc, w, nit, string(zpd)),
                @sprintf(",%.6e,%.3f,%.4f,%.3f", t_iso, gf_iso, ai, bw),
                @sprintf(",%.2f,%.3f,%.4f", E, eff_gf, eff_ai),
                @sprintf(",%.6f,%.6f,%.6e,%.3f,%.4f,%.3e", cold[s], tf,
                    per_shift_iter(tf, iters), g / tf, gflops(tf, m, iters; eye), maxd[s]),
                @sprintf(",%.6f,%.6f,%.6f,%d,%s,%s,%.1f,%.3e,%s", med, iqlo, iqhi, nreps,
                    isnan(tcpu) ? "NaN" : @sprintf("%.6e", tcpu),
                    isnan(spd) ? "NaN" : @sprintf("%.4f", spd), cgf, condA,
                    pencil_label()))
            flush(csv)
        end
    end
end
close_csv(csv, joinpath(RESULTS, "bench_kernels.csv"))
