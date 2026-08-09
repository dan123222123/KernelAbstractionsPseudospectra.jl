# A pseudospectra PORTRAIT, not a benchmark: one adaptive solve of a single large matrix,
# fanned out across every device of the backend, dumped as a σ field + depth grid for
# plotting. No reps, no strategy sweep; wall-clock is logged for provenance only.
#
# Matrix: golub (Viswanath & Trefethen, SIMAX 19, 1998) — det = 1 exactly at every size
# while the spectral radius grows ≈ 66·m, so the eigenvalues say nothing about the
# resolvent and the σ field IS the picture. Random; seed is pinned and logged.
#
#   BENCH_GOLUB_M       matrix size                        (default 16384)
#   BENCH_GOLUB_GRIDN   grid points per axis               (default 300)
#   BENCH_GOLUB_HW      window half-width, centered at 0   (default 80·m)
#   BENCH_GOLUB_NITMAX  adaptive depth cap                 (default 8⌈log₂ m⌉)
#   BENCH_GOLUB_SEED    RNG seed for the matrix            (default 20260807)
#   BENCH_GOLUB_COLS    output-row slice "lo:hi" this run solves   (default the whole grid)
#   BENCH_ELTYPES       first rung picks the eltype        (default f32)
#
# `on_batch` deliveries flush the output CSVs as batches retire (≥ 60 s apart), so a run
# killed at a CI timeout keeps every checkpointed batch (unsolved cells read as σ = -1,
# depth = -1). BENCH_GOLUB_COLS slices the solve for job-length ceilings the checkpoints
# can't dodge: each slice fills only its own output rows, so disjoint slices from
# separate runs overlay into one grid.
#
# The pencil ships a Diagonal for both B and Z; with no user x₀ the workspace never
# materializes a dense Z, so the uploaded pencil is just the two dense m×m triangular
# factors (4 GiB at m = 16384 in ComplexF32).
#
# Host reduction uses gees with jobvs = 'N': the portrait never needs the Schur vectors.
#
#   julia --project=bench bench/bench_golub.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))

using DelimitedFiles
using LinearAlgebra: LAPACK

const backend = select_backend(ARGS; default = "cuda")
const RESULTS = results_dir()

const TOK = first(runnable_eltypes(backend, eltype_tokens("f32")))
const T = ELTYPES[TOK]
const M = env_int("BENCH_GOLUB_M", 16384)
const GRIDN = env_int("BENCH_GOLUB_GRIDN", 300)
const HW = parse(Float64, get(ENV, "BENCH_GOLUB_HW", string(80.0 * M)))
const NITMAX = env_int("BENCH_GOLUB_NITMAX", 8 * max(1, ceil(Int, log2(M))))
const SEED = env_int("BENCH_GOLUB_SEED", 20260807)
const COLS = let s = get(ENV, "BENCH_GOLUB_COLS", "1:$(GRIDN)")
    lo, hi = parse.(Int, split(s, ':'))
    1 <= lo <= hi <= GRIDN || error("BENCH_GOLUB_COLS=$s outside 1:$(GRIDN)")
    lo:hi
end
const DEVS = KernelAbstractions.isgpu(backend) ?
             collect(KAPseudospectra.devices(backend)) : missing
const NDEV = DEVS === missing ? 1 : length(DEVS)

logln, logio = bench_logger(joinpath(RESULTS, "bench_golub_log.txt"))
logln("="^72)
logln("KAPseudospectra.jl golub portrait   ", Dates.now())
logln("backend=", backend, "  devices=", NDEV, " x ", device_name(backend))
logln("m=", M, "  T=", T, "  grid=", GRIDN, "²  cols=", COLS, "  window=±", HW,
    "  nit_max=", NITMAX, "  seed=", SEED)
foreach(logln, repro_stamp(backend; matrix = "golub"))
logln("="^72)

# Standard (B = I) Schur pencil for golub(m). Returns the pencil, the computed
# eigenvalues (the gees w vector), and the reduction time.
function golub_pencil(T, m)
    A = MatrixDepot.golub(real(T), m)     # real entries; L·U runs at the real eltype
    S = Matrix{T}(A)
    A = nothing
    tsch = @elapsed ((S, _, w) = LAPACK.gees!('N', S))
    Iₘ = Diagonal(ones(T, m))
    (; P = KAPseudospectra.SchurMatrixPencil{T, true}(S, S', Iₘ, Iₘ, Diagonal(ones(T, m))),
        w, tsch)
end

# Fail fast: compile the adaptive multi-device path before investing in the m-sized reduction.
logln("[", clock(), "] warmup (compiling the adaptive multi-device path)...")
let wp = golub_pencil(T, 256), hw = 80.0 * 256
    zgw = bench_grid(T, 24; box = ((-hw, hw), (-hw, hw)))
    tw = @elapsed KAPseudospectra._ihlpsa_adaptive(backend, zgw, wp.P;
        nit_max = 16, devs = DEVS, on_batch = (idx, σv, nitv) -> nothing)
    logln("[", clock(), "] warmup done in ", @sprintf("%.1f s", tw))
end

Random.seed!(SEED)
logln("[", clock(), "] generating golub(", M, ") and reducing (gees, jobvs='N')...")
tpencil = @elapsed pc = golub_pencil(T, M)
logln("[", clock(), "] pencil ready in ", @sprintf("%.1f s", tpencil),
    "  (gees ", @sprintf("%.1f s", pc.tsch), ")",
    "  spectral radius ≈ ", @sprintf("%.3e", maximum(abs, pc.w)))

zpd = pinned_zpd(backend, T, M, NITMAX;
    ngrid = cld(length(COLS) * GRIDN, NDEV), headroom = zpd_headroom(), P = pc.P)
logln("zpd=", zpd, " per device")

gx, gy, zg = qgrid(T, (-HW, HW), (-HW, HW), (GRIDN, GRIDN))

# Eigenvalues and axes first — they are ready before the solve starts.
writedlm(joinpath(RESULTS, "golub_eigvals_$(TOK)_m$(M).csv"),
    [Float64.(real.(pc.w)) Float64.(imag.(pc.w))], ',')
writedlm(joinpath(RESULTS, "golub_axes_$(TOK)_m$(M).csv"),
    [Float64.(gx) Float64.(gy)], ',')

# σ/nitg come back permutedims'd: row i ↔ gy[i], column j ↔ gx[j]; on_batch delivers in the
# same orientation. Unsolved cells hold -1.
σfull = fill(-1.0, GRIDN, GRIDN)
nitfull = fill(-1, GRIDN, GRIDN)
sigma_path = joinpath(RESULTS, "golub_sigma_$(TOK)_m$(M).csv")
nitgrid_path = joinpath(RESULTS, "golub_nitgrid_$(TOK)_m$(M).csv")

ndone = Ref(0)
last_flush = Ref(time())
npts = length(COLS) * GRIDN
roff = Ref(CartesianIndex(first(COLS) - 1, 0))   # slice-local delivery row → output row
checkpoint = (idx, σv, nitv) -> begin
    σfull[idx .+ roff] .= to64.(abs.(σv))
    nitfull[idx .+ roff] .= nitv
    ndone[] += length(idx)
    if time() - last_flush[] >= 60
        writedlm(sigma_path, σfull, ',')
        writedlm(nitgrid_path, nitfull, ',')
        logln("[", clock(), "] checkpoint: ", ndone[], "/", npts, " points on disk",
            "  iters so far=", sum(nitfull[nitfull .>= 0]))
        last_flush[] = time()
    end
end

logln("[", clock(), "] solving rows ", COLS, " of ", GRIDN, "² (", npts,
    " points), adaptive, on ", NDEV, " device(s)...")
tsolve = @elapsed ((σ, nitg) = KAPseudospectra._ihlpsa_adaptive(backend,
    zg[:, COLS], pc.P; nit_max = NITMAX, devs = DEVS, zpd, on_batch = checkpoint))
σfull[COLS, :] .= to64.(abs.(σ))
nitfull[COLS, :] .= nitg
writedlm(sigma_path, σfull, ',')
writedlm(nitgrid_path, nitfull, ',')
logln("[", clock(), "] solve ", @sprintf("%.1f s", tsolve),
    "  total iters=", sum(nitg),
    "  mean depth=", @sprintf("%.1f", sum(nitg) / length(nitg)),
    "  at-cap points=", count(==(NITMAX), nitg))
logln("wrote golub_{sigma,nitgrid,eigvals,axes}_$(TOK)_m$(M).csv in ", RESULTS)
close(logio)
