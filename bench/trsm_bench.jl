# End-to-end ihlpsa wall-clock timing of the GPU trsm strategies (KAPSEUDO_TRSM: tiled vs column)
# across BOTH drivers (fixed-nit vs adaptive depth), in one table. Also confirms σ vs the column
# baseline. (The register-warp strategy was removed — it only beat tiled at small m while paying a
# per-R `@generated` recompile up to minutes; see DESIGN_TRSM.md.)
#
# Backend-agnostic via bench_common's `select_backend` — pass the backend as the first CLI arg
# (cuda|amdgpu|oneapi|metal|cpu; defaults to cpu). On CPU the strategy is a no-op (the CPU solve
# ignores it), so the two strategy columns coincide there — the distinction is meaningful on GPU.
#   JULIA_NUM_THREADS=auto julia --project=bench bench/trsm_bench.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))
using Random

backend = select_backend(ARGS)
gpu = KernelAbstractions.isgpu(backend)
devs = gpu ? KAPseudospectra.devices(backend) : missing   # devices(::CPU) is scalar → use missing

# This bench FORCES KAPSEUDO_TRSM=tiled, which runs a warp-shuffle kernel. On a GPU where the
# shuffle isn't usable (stock oneAPI's KernelIntrinsics @shfl is a TODO stub; Metal without the
# opt-in) that would miscompile/crash, so fail fast with guidance. (On CPU the strategy is a no-op.)
if gpu && !KAPseudospectra.warp_trsm_safe(backend, false)
    error("the tiled solve isn't usable on $(backend) (warp_trsm_safe == false) — use a backend " *
          "where it runs (CUDA/AMDGPU, or oneAPI/Metal with the opt-in enabled), or run on cpu.")
end

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

@printf("%-6s %-9s %-9s | %-9s %-9s  %-9s\n",
        "m", "tiled", "column", "tiled-adp", "col-adp", "maxdiff")
for m in MS
    rng = Random.seed!(0xBEEF)
    P = MatrixPencil(schur(randn(rng, T, m, m)))
    _, _, zg = KAPseudospectra.qgrid(T, (-3, 3), (-3, 3), (GRIDN, GRIDN))

    # warm-up + correctness: the fixed-driver tiled solve must match the column baseline
    σc = run_fixed(P, zg, "column")
    σt = run_fixed(P, zg, "tiled")
    run_adapt(P, zg, "tiled"); run_adapt(P, zg, "column")   # warm adaptive
    maxdiff = maximum(abs.(σt .- σc))

    timed(f) = (reclaim_all(backend); bestof(f; reps=REPS))
    tt  = timed(() -> run_fixed(P, zg, "tiled"))
    tc  = timed(() -> run_fixed(P, zg, "column"))
    tta = timed(() -> run_adapt(P, zg, "tiled"))
    tca = timed(() -> run_adapt(P, zg, "column"))
    @printf("%-6d %-9.4f %-9.4f | %-9.4f %-9.4f  %.2e\n",
            m, tt, tc, tta, tca, maxdiff)
end
