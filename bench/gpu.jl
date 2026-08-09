# One-command, device-agnostic bench runner: `julia --project=bench bench/gpu.jl <backend>`
# (cuda | amdgpu | oneapi | metal | cpu). Runs the per-device experiments in ISOLATED subprocesses
# (fresh device state per experiment):
#   bench_drivers.jl — fixed vs adaptive
#   bench_kernels.jl — strategy × eltype ladder, both granularities
# Each self-adapts: eltype rungs gate on FP64/shuffle support, and bench_kernels self-skips
# where the tiled solve isn't usable (e.g. Intel's @shfl stub). Launch-time constraints stay
# with the caller: AMDGPU needs JULIA_NUM_THREADS=1 (a multithread GC bug), and an FP64-less
# Intel iGPU needs `set_pdiv_accurate!(false)` persisted first.
#
# Only the two scripts above are in this loop; every other bench_*.jl is run explicitly (the CI
# pipeline drives them per queue).

const WHICH = isempty(ARGS) ? "cpu" : lowercase(first(ARGS))
const SWEEPS = ("bench_drivers.jl", "bench_kernels.jl")

# Bounded CI smoke by default: small sizes so a casual run is minutes, not hours. Set
# KAPSEUDO_BENCH_FULL=1 for the policy-default sizes; BENCH_* vars still override.
if !haskey(ENV, "KAPSEUDO_BENCH_FULL")
    get!(ENV, "BENCH_MS", "64,128")
    get!(ENV, "BENCH_GRIDN", "64")     # 64×64 = 4096 grid points
    get!(ENV, "BENCH_REPS", "2")
    get!(ENV, "BENCH_ELTYPES", "f32")  # one rung only in the smoke
end

jl = Base.julia_cmd()
failed = String[]
for s in SWEEPS
    script = joinpath(@__DIR__, s)
    println("\n", "="^72, "\n===== ", s, "  [", WHICH, "] =====\n", "="^72)
    try
        run(`$jl --project=$(@__DIR__) $script $WHICH`)
    catch err
        @warn "experiment failed; continuing" sweep=s err
        push!(failed, s)
    end
end

if isempty(failed)
    println("\nAll experiments completed for backend=", WHICH, ".")
else
    println("\nCompleted with failures (", join(failed, ", "), ") for backend=", WHICH, ".")
    exit(1)
end
