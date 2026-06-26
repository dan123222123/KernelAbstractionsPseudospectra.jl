# End-to-end ihlpsa wall-clock timing across the trsm strategies (KAPSEUDO_TRSM: warp / tiled /
# column) AND both drivers (fixed-nit vs adaptive depth), to show the size-dependent warp↔tiled
# crossover and the fixed-vs-adaptive overhead in one table. Also confirms σ vs the column baseline.
#
# Backend-agnostic via bench_common's `select_backend` — pass the backend as the first CLI arg
# (cuda|amdgpu|oneapi|metal|cpu; defaults to cpu). On CPU the strategy is a no-op (the CPU solve
# ignores it), so the three strategy columns coincide there — the distinction is meaningful on GPU.
#   JULIA_NUM_THREADS=auto julia --project=bench bench/warp_trsm_bench.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))
using Random

backend = select_backend(ARGS)
gpu = KernelAbstractions.isgpu(backend)
devs = gpu ? KAPseudospectra.devices(backend) : missing   # devices(::CPU) is scalar → use missing

const T = ComplexF32
const NIT = 20            # fixed-driver depth; also the adaptive nit_max ceiling (apples-to-apples)
const REPS = 3
const GRIDN = gpu ? 300 : 60          # smaller grid on CPU so the bench stays interactive
const MS = gpu ? (128, 256, 512, 1024) : (64, 128, 256)

# Fixed-nit and adaptive drivers, parameterized by strategy. `withenv` selects the strategy per call.
run_fixed(P, zg, flag) = withenv("KAPSEUDO_TRSM" => flag) do
    ihlpsa(backend, zg, P, NIT; devs)
end
run_adapt(P, zg, flag) = withenv("KAPSEUDO_TRSM" => flag) do
    ihlpsa(backend, zg, P; nit_max=NIT, devs)
end

@printf("%-6s %-9s %-9s %-9s | %-9s %-9s %-9s  %-9s\n",
        "m", "warp", "tiled", "column", "warp-adp", "tiled-adp", "col-adp", "maxdiff")
for m in MS
    rng = Random.seed!(0xBEEF)
    P = MatrixPencil(schur(randn(rng, T, m, m)))
    _, _, zg = KAPseudospectra.qgrid(T, (-3, 3), (-3, 3), (GRIDN, GRIDN))

    # warm-up + correctness: every fixed-driver strategy must match the column baseline
    σc = run_fixed(P, zg, "column")
    σw = run_fixed(P, zg, "warp")
    σt = run_fixed(P, zg, "tiled")
    run_adapt(P, zg, "warp"); run_adapt(P, zg, "tiled"); run_adapt(P, zg, "column")   # warm adaptive
    maxdiff = max(maximum(abs.(σw .- σc)), maximum(abs.(σt .- σc)))

    timed(f) = (reclaim_all(backend); bestof(f; reps=REPS))
    tw  = timed(() -> run_fixed(P, zg, "warp"))
    tt  = timed(() -> run_fixed(P, zg, "tiled"))
    tc  = timed(() -> run_fixed(P, zg, "column"))
    twa = timed(() -> run_adapt(P, zg, "warp"))
    tta = timed(() -> run_adapt(P, zg, "tiled"))
    tca = timed(() -> run_adapt(P, zg, "column"))
    @printf("%-6d %-9.4f %-9.4f %-9.4f | %-9.4f %-9.4f %-9.4f  %.2e\n",
            m, tw, tt, tc, twa, tta, tca, maxdiff)
end
