# Shared policy + helpers included by every bench/ script. Suite-wide policy (depth / grid /
# matrix / eltype ladder): README § Policy. Only bench_drivers, bench_multigpu's scaling mode,
# and bench_golub drive the adaptive driver directly; the converged experiments reach it through
# `converged_solve`. Everything else runs fixed-`nit` so iteration count stays a controlled
# variable.

using KernelAbstractionsPseudospectra, KernelAbstractions
using LinearAlgebra, MatrixDepot, MultiFloats, Printf, Dates, Random, SHA
import GenericLinearAlgebra
import GenericSchur
using JLD2                   # on-disk cache for the once-per-size BigFloat Schur factor
using TOML                   # read back the persisted tuning preferences for the stamp

# Short HH:MM:SS stamp for progress lines.
clock() = Dates.format(Dates.now(), "HH:MM:SS")

# Open `path` and return `(logln, io)`: `logln(args...)` tees one line to stdout and
# the file. Close `io` when the bench finishes.
function bench_logger(path)
    io = open(path, "w")
    logln = (args...) -> (s = string(args...); println(s); println(io, s); flush(io); s)
    return logln, io
end

# Best-of-`reps` wall-clock seconds for `f`. Reclaims every device before each rep (no rep
# inherits allocator state) and synchronizes inside the timed region. `warmup = true` runs one
# untimed call first, for raw kernel closures whose first call compiles. Reports min;
# `stats = true` also returns median/IQR (needs a large `BENCH_REPS_STATS`).
function bestof(f, backend; reps = 3, warmup = false, stats = false)
    warmup && (f(); KernelAbstractions.synchronize(backend))
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        reclaim_all(backend)
        times[i] = @elapsed (f(); KernelAbstractions.synchronize(backend))
    end
    stats || return minimum(times)
    sorted = sort(times)
    q(p) = sorted[clamp(round(Int, 1 + p * (reps - 1)), 1, reps)]   # nearest-rank quantile
    (min = sorted[1], median = q(0.5), mean = sum(times) / reps,
        iqr_lo = q(0.25), iqr_hi = q(0.75), max = sorted[end], n = reps)
end

# Bench-side hygiene on top of the package primitive `device_reclaim`: reclaim every device +
# double GC.
function reclaim_all(backend)
    if KernelAbstractions.isgpu(backend)
        for d in KernelAbstractionsPseudospectra.devices(backend)
            KernelAbstractionsPseudospectra.device!(backend, d)
            KernelAbstractionsPseudospectra.device_reclaim(backend)
        end
    else
        KernelAbstractionsPseudospectra.device_reclaim(backend)
    end
    GC.gc();
    GC.gc()
end

# Display name for the active device — console output and the tune-profile header only.
# CUDA gets a real query (method added in select_backend); other backends read as the type name.
device_name(backend) = string(nameof(typeof(backend)))
device_name(::CPU) = Sys.CPU_NAME

# Backend from `args[1]` (cuda|amdgpu|oneapi|metal|cpu), loading the matching GPU package on
# demand; defaults to `default`. Loads it via runtime `using`, not a Pkg extension: extensions
# only exist for packages, and bench/ is a script environment.
function select_backend(args = ARGS; default = "cpu")
    which = isempty(args) ? default : lowercase(first(args))
    which == "cpu" && return CPU()
    backend = if which == "cuda"
        @eval Main using CUDA
        @eval Main device_name(::CUDA.CUDABackend) = CUDA.name(CUDA.device())
        Base.invokelatest(() -> Main.CUDA.CUDABackend())
    elseif which == "amdgpu"
        @eval Main using AMDGPU
        Base.invokelatest(() -> Main.AMDGPU.ROCBackend())
    elseif which == "oneapi"
        @eval Main using oneAPI
        Base.invokelatest(() -> Main.oneAPI.oneAPIBackend())
    elseif which == "metal"
        @eval Main using Metal
        Base.invokelatest(() -> Main.Metal.MetalBackend())
    else
        error("unknown backend $(which); use cuda|amdgpu|oneapi|metal|cpu")
    end
    return backend
end

# Comma-separated integers from ENV[key] as a Tuple, else `default`. Lets the pipeline
# shrink a sweep's sizes for CI (a bounded smoke) without editing the scripts.
env_ints(key, default) =
    haskey(ENV, key) ? Tuple(parse.(Int, split(ENV[key], ","))) : default
env_int(key, default) = parse(Int, get(ENV, key, string(default)))

# Fixed inverse-Lanczos depth for the roofline / isolated-solve throughput (bench_kernels) ONLY,
# where a known, identical trsm launch count makes the analytic FLOP/byte/AI exact. NOT a
# converged solve (σ_min under-resolved at m ≳ 256) — end-to-end rows use the adaptive floor
# solve below. README § Policy; BENCH_NIT pins a flat depth.
default_nit(m) = env_int("BENCH_NIT", 4 * ceil(Int, log2(m)))

# Per-base-precision rtol for the adaptive solve to the ARITHMETIC FLOOR (the presented
# time-to-solution rows + the SVD sentinel), reaching ~eps·σ. The driver's defaults
# (1e-6/1e-4) target working accuracy; the floor solve overrides them per precision. Kept as
# an explicit table, not computed from eps(T) (routes through the fragile
# Float64(::Float32x2) conversion).
const CONVERGED_RTOL = Dict{DataType, Float64}(
    Float32 => 1e-5, Float64 => 1e-10,
    Float32x2 => 1e-10, Float32x4 => 1e-12,
    Float64x2 => 1e-13, Float64x4 => 1e-16)
converged_rtol(T) = get(CONVERGED_RTOL, real(T)) do
    error("no CONVERGED_RTOL entry for $(real(T)) — add one (precision-floor rtol)")
end
converged_nit_max(m) = env_int("BENCH_NIT_MAX", 20 * max(1, ceil(Int, log2(m))))

# Adaptive solve to the precision floor on `backend` (the fixed-driver analogue is `run_fixed`).
converged_solve(backend, zg, P; devs = missing, zpd = missing) =
    ihlpsa(backend, zg, P; devs, zpd, rtol = converged_rtol(eltype(zg)),
        nit_max = converged_nit_max(size(P, 1)))

# Grid side length: BENCH_GRIDN overrides (512 is the publication size — README § Grid
# sizing); GPU sweeps default to 128², CPU to 64². The reported metrics all normalize the
# grid out, so it's a pure wall-clock multiplier except on the cpu_gpu field dumps.
bench_gridn(backend) = env_int("BENCH_GRIDN", KernelAbstractions.isgpu(backend) ? 128 : 64)

# grcar, suite-wide: a named, deterministic non-normal test matrix (needs m > 4), matching
# MATLAB `gallery('grcar')` for the EigTool leg. BENCH_MATRIX is a stable stamp/cache key;
# grcar is the only family.
const BENCH_MATRIX = "grcar"

bench_matrix(T, m) = MatrixDepot.grcar(T, m)   # deterministic; no RNG state to guard

# B for the generalized (QZ) rows: kms — deterministic, SPD, Toeplitz — keeps the pencil
# nonsingular and well-scaled with no RNG state. kms is real-parameterized; convert to T.
bench_matrix_b(T, m) = T.(MatrixDepot.kms(real(T), m))

# Shift box matched to grcar's spectrum.
bench_box() = ((-1, 3), (-3, 3))

# The shift grid every experiment sweeps: qgrid over the matrix box, returning just the flat
# shift vector `zg` (drivers rarely need the gx/gy axes). Pass `box` to override — the EigTool
# leg reuses its own box for the MATLAB axis string.
bench_grid(T, gridn; box = bench_box()) = last(qgrid(T, box[1], box[2], (gridn, gridn)))

# Per-experiment output directory under bench/, created on demand.
results_dir(name = "results") = (p = joinpath(@__DIR__, name); mkpath(p); p)

# Results-CSV scaffolding shared by the experiments: open + header row, close + "wrote" trailer.
open_csv(path, header) = (io = open(path, "w"); println(io, header); io)
close_csv(io, path) = (close(io); println("\nwrote ", path))

# Schur pencil at eltype T for the suite matrix. Returns `(; P, A)` where `A` is the underlying
# IEEE matrix (ComplexF64 for MultiFloat rungs), so `cond(A)` is meaningful on every rung.
# MultiFloat rungs reduce with the full-precision cached BigFloat GenericSchur factor by default;
# BENCH_FAST_SCHUR=1 swaps in the fast Float64 factor for the kernel-perf sweeps (reduction
# precision is orthogonal to trsm throughput).
const FAST_SCHUR = get(ENV, "BENCH_FAST_SCHUR", "0") == "1"
const HIPREC_BITS = env_int("BENCH_HIPREC_BITS", 256)   # BigFloat precision (256 bits resolves f64x4)
const SCHUR_CACHE = joinpath(@__DIR__, "schur_cache")

# BigFloat triangular Schur factor at size m, cached on disk (once per size — the reduction depends
# only on the matrix). The reduction is the package primitive `bigfloat_schur_factor`; bench only
# adds the cache. Missing/unreadable cache is regenerated.
function bigfloat_schur(A::AbstractMatrix, m::Integer)
    path = joinpath(SCHUR_CACHE, "$(BENCH_MATRIX)_m$(m)_$(HIPREC_BITS)b.jld2")
    if isfile(path)
        try
            return JLD2.load(path, "S")::Matrix{Complex{BigFloat}}
        catch
            @warn "Schur cache unreadable — regenerating" path
        end
    end
    S = KernelAbstractionsPseudospectra.bigfloat_schur_factor(A; bits = HIPREC_BITS)
    mkpath(SCHUR_CACHE)
    JLD2.jldsave(path; S)
    S
end

# Full-precision triangular factor rounded to T (Z = I suffices for ihlpsa).
hiprec_schur(A::AbstractMatrix, ::Type{T}) where {T} = T.(bigfloat_schur(A, size(A, 1)))

# Pencil mode for the suite (zB − A; B = I is the standard case). `gen` is the default;
# `BENCH_PENCIL=std` restores B = I for legs that can't express a pencil at all
# (bench_cpu_gpu, bench_correctness).
const PENCIL_MODE = let v = lowercase(get(ENV, "BENCH_PENCIL", "gen"))
    v in ("gen", "std") || error("BENCH_PENCIL must be \"gen\" or \"std\"; got $(repr(v))")
    Symbol(v)
end

pencil_label(mode = PENCIL_MODE) = String(mode)

# Whether the MultiFloat generalized reduction had to fall back to a promoted Float64 QZ.
# Latched on first use so the warning fires once per process, not once per (T, m) cell.
const MF_GEN_WARNED = Ref(false)

function bench_pencil(T, m; pencil::Symbol = PENCIL_MODE)
    if real(T) <: Base.IEEEFloat
        A = bench_matrix(T, m)
        pencil === :std && return (; P = MatrixPencil(schur(A)), A)
        return (; P = MatrixPencil(schur(A, bench_matrix_b(T, m))), A)
    end
    A = bench_matrix(ComplexF64, m)
    if pencil === :gen
        # Generalized MultiFloat rungs reduce with a promoted Float64 QZ (a full-precision
        # generic QZ is O(hours) at these sizes), capping end-to-end accuracy at the ~1e-15
        # Schur bound.
        if !MF_GEN_WARNED[]
            @warn "generalized MultiFloat pencil: reducing with a promoted Float64 QZ " *
                  "(a full-precision generic QZ is impractically slow here) — throughput " *
                  "is valid, end-to-end accuracy is Schur-bounded"
            MF_GEN_WARNED[] = true
        end
        Fg = schur(A, bench_matrix_b(ComplexF64, m))
        S, TB = T.(Fg.S), T.(Fg.T)
        return (; P = KernelAbstractionsPseudospectra.SchurMatrixPencil{T, false}(
            S, collect(S'), TB, collect(TB'), Matrix(T.(Fg.Z))), A)
    end
    Iₘ = Diagonal(ones(T, m))   # B = Bc = I as a Diagonal, mirroring MatrixPencil(::Schur)
    S, Z = if FAST_SCHUR
        F = schur(A)
        (T.(F.T), Matrix(T.(F.Z)))                 # kernel-perf shortcut: Float64 factor promoted
    else
        (hiprec_schur(A, T), Matrix{T}(I, m, m))   # default: full-precision cached GenericSchur factor
    end
    (; P = KernelAbstractionsPseudospectra.SchurMatrixPencil{T, true}(S, collect(S'), Iₘ, Iₘ, Z), A)
end

const ELTYPES = (;
    f32 = ComplexF32, f64 = ComplexF64,
    f32x2 = Complex{Float32x2}, f32x4 = Complex{Float32x4},
    f64x2 = Complex{Float64x2}, f64x4 = Complex{Float64x4})

iswide(T) = !(real(T) <: Base.IEEEFloat)

# Native-FLOP expansion (μ_mul, μ_add) per real-limb op, for bench_kernels' effective-CGMA
# arithmetic intensity — counted from LLVM IR by tuning/multifloat_flop_costs.jl (re-run to
# refresh for a new MultiFloats version).
const MF_FLOP_COST = Dict{DataType, Tuple{Int, Int}}(
    Float32x2 => (10, 21), Float32x4 => (130, 124),
    Float64x2 => (10, 20), Float64x4 => (130, 124))

# Per-logical-flop expansion E(T) = (μ_mul + μ_add)/2 (balanced mul/add mix; E = 1 for IEEE).
function mf_expansion(T)
    R = real(T)
    R <: Base.IEEEFloat && return 1.0
    c = get(MF_FLOP_COST, R, nothing)
    c === nothing && (@warn "no native-flop cost for $R — effective AI falls back to logical"; return 1.0)
    (c[1] + c[2]) / 2
end

function eltype_tokens(default)
    toks = Symbol.(split(get(ENV, "BENCH_ELTYPES", default), ","))
    all(t -> haskey(ELTYPES, t), toks) ||
        error("BENCH_ELTYPES tokens must be in $(keys(ELTYPES)) (got $(toks))")
    toks
end

# Rungs runnable on this device: F64-limb rungs need native FP64 (an FP64-less iGPU can
# still run f32x2/f32x4); wide rungs additionally need the per-limb warp shuffle. Each skip
# warns, so a green run says what it did NOT measure.
function runnable_eltypes(backend, toks)
    gpu = KernelAbstractions.isgpu(backend)
    filter(collect(toks)) do tok
        T = ELTYPES[tok]
        if gpu && (real(T) === Float64 || real(T) <: Float64x) &&
           !KernelAbstractionsPseudospectra.supports_fp64(backend)
            @warn "skipping $tok — no native FP64 on $(backend)"
            return false
        end
        if gpu && iswide(T) && !KernelAbstractionsPseudospectra.warp_trsm_safe(backend, true)
            @warn "skipping $tok — wide warp shuffle not usable on $(backend)"
            return false
        end
        true
    end
end

# Strategies that are DISTINCT configurations for T on `backend`: tiled self-gates to the
# column solve where the warp shuffle isn't usable (the whole CPU backend, stock oneAPI's
# @shfl stub); tiled-gemm only differs where the vendor GEMM trailing update engages
# (IEEE eltypes on CUDA/AMDGPU — no vendor GEMM exists for MultiFloat).
function strategies_for(backend, T)
    # CPU first: warp_trsm_safe does NOT gate the CPU backend, and KernelIntrinsics' CPU
    # @shfl recurses infinitely if _tiled_trsm! is invoked directly — the package only
    # avoids it by strategy-gating inside its own run paths.
    KernelAbstractions.isgpu(backend) || return ("column",)
    KernelAbstractionsPseudospectra.warp_trsm_safe(backend, iswide(T)) || return ("column",)
    KernelAbstractionsPseudospectra.tiled_gemm_safe(backend, T) ? ("column", "tiled", "tiled-gemm") :
    ("column", "tiled")
end

# Float64 view of a (possibly MultiFloat) magnitude for tables/CSVs. MultiFloats only defines
# same-base conversions (there is no Float64(::Float32x2)), so sum the limbs instead.
to64(x::Real) = Float64(x)
to64(x::MultiFloat) = sum(Float64, x._limbs)

# Pin the per-device batch size (`zpd`) once, so it's a controlled variable rather than a
# function of live free memory at each rep. Sizes the largest batch that fits; `ngrid` caps it
# (also dodges an iGPU Level-Zero per-allocation overflow), `headroom` divides it for the
# adaptive driver's row-gather scratch. Passing the pencil `P` plans against its true adapted
# footprint (Diagonal B/Z ship as m elements, not m²). CPU ⇒ `missing` (whole grid).
function pinned_zpd(backend, T, m, nit; ngrid = nothing, headroom = 1, P = nothing)
    KernelAbstractions.isgpu(backend) || return missing
    reclaim_all(backend)
    z = P === nothing ? KernelAbstractionsPseudospectra.findmaxbatchihl(backend, T, m, nit) :
        KernelAbstractionsPseudospectra.findmaxbatchihl(backend, P, nit)
    ngrid === nothing || (z = min(z, ngrid))
    haskey(ENV, "KAPSEUDO_MAX_ZPD") && (z = min(z, parse(Int, ENV["KAPSEUDO_MAX_ZPD"])))
    max(z ÷ headroom, 1)
end

# The suite-wide zpd headroom: adaptive callers halve the pin for the driver's row-gather
# scratch; the fixed driver shares it so runs stay comparable.
zpd_headroom() = env_int("BENCH_ZPD_HEADROOM", 2)

# Per-(T, m) sweep setup shared by the experiment drivers: tiered grid, fixed depth, pencil,
# pinned batch. A sweep that never runs the adaptive driver may pass `headroom = 1`.
function bench_setup(backend, T, m; gridn = bench_gridn(backend), nit = default_nit(m),
        headroom = zpd_headroom(), pencil = PENCIL_MODE)
    pc = bench_pencil(T, m; pencil)
    (; gridn, g = gridn * gridn, zg = bench_grid(T, gridn), nit, P = pc.P, A = pc.A,
        eye = KernelAbstractionsPseudospectra.b_is_identity(pc.P),
        zpd = pinned_zpd(backend, T, m, nit; ngrid = gridn * gridn, headroom, P = pc.P))
end

# FLOPs per shift: `nit` iterations × two triangular solves = 8m² per iteration for a standard
# pencil (B = I; detect via KernelAbstractionsPseudospectra.b_is_identity), 16m² generalized (the z·B−A term).
# O(m) recurrence + host eigmax excluded (<1%).
flops_per_shift(m, nit; eye = true) = nit * (eye ? 8 : 16) * m^2

# Total inner iterations (the work unit): g·nit fixed, Σ nit_grid adaptive. Using g·nit for
# adaptive would credit it the work it saved — erasing the feature being measured.
total_iters(gridpoints::Integer, nit::Integer) = gridpoints * nit
total_iters(nit_grid::AbstractArray{<:Integer}) = sum(nit_grid)

# Headline FoM: seconds per (shift · iteration) — normalizes across nit and fixed-vs-adaptive
# (co-report m, T, strategy). The workload is latency-bound.
per_shift_iter(t, iters::Integer) = t / iters
gflops(t, m, iters::Integer; eye = true) = flops_per_shift(m, iters; eye) / t / 1e9

# Comma-free backend token for CSV fields (string(backend) embeds a comma that shifts columns).
backend_tag(backend) = string(nameof(typeof(backend)))

# Resolves the tuning knobs in force, through the package's own order (profile, then
# LocalPreferences.toml) so the stamp reports what the solve will actually read. TUNED
# requires ALL of `tuning_keys()`; anything less is PARTIAL.
function _tuning_table()
    src, tbl = String[], Dict{String, String}()
    profile = get(ENV, "KAPSEUDO_TUNE_PROFILE", "")
    for (label, path) in (("profile:$profile", profile),
        ("local", joinpath(@__DIR__, "LocalPreferences.toml")))
        isempty(path) && continue
        # An absent LocalPreferences.toml is normal and stays silent, but a PROFILE pointed at
        # a missing file must not read as "no profile" — that's a config typo, and the run
        # would otherwise silently measure heuristic defaults.
        if !isfile(path)
            startswith(label, "profile:") && push!(src, "$label(MISSING FILE)")
            continue
        end
        prefs = try
            raw = TOML.parsefile(path)
            get(raw, "KernelAbstractionsPseudospectra", raw)
        catch err
            push!(src, "$label(UNREADABLE: $err)")
            continue
        end
        push!(src, label)
        for k in tuning_keys_expected()          # earlier source wins, matching the package
            haskey(tbl, k) || (haskey(prefs, k) && (tbl[k] = string(prefs[k])))
        end
    end
    return src, tbl
end

# Mirror of KernelAbstractionsPseudospectra.tuning_keys(); defined here too so the stamp works against a package
# version that predates it (the bench env resolves the package from the checkout, not a release).
tuning_keys_expected() = [string(k, "_", T, "_", e)
                          for k in ("trsm_tilecols", "trsm_blockwarps", "trsm_warpgridpts", "trsm_wgs")
                          for T in ("ComplexF32", "ComplexF64") for e in ("eye", "gen")]

function tuning_stamp()
    src, tbl = _tuning_table()
    isempty(tbl) &&
        return "tuning=UNTUNED src=[" * join(isempty(src) ? ["none"] : src, ",") * "]"
    absent = setdiff(tuning_keys_expected(), keys(tbl))
    state = isempty(absent) ? "TUNED" : "PARTIAL"
    body = join(("$k=$(tbl[k])" for k in sort(collect(keys(tbl)))), " ")
    stamp = "tuning=$state src=[" * join(src, ",") * "] [" * body * "]"
    isempty(absent) || (stamp *= " MISSING[" * join(sort(absent), " ") * "]")
    return stamp
end

# True only when EVERY tuning knob is configured — the CSV-column form of `tuning_stamp`, so rows
# from a tuned and an untuned run can be told apart. PARTIAL counts as untuned.
is_tuned() = startswith(tuning_stamp(), "tuning=TUNED")

# ── contention canary ────────────────────────────────────────────────────────────────────────
# A fixed-shape GEMM whose time depends only on the device, so a run can report whether it had
# the GPU to itself. Cheap enough to sample per row. Float32 so it also runs on FP64-less
# devices; values are constant (no RNG dependency).
const CANARY_N = 4096

function device_canary(backend; reps = 2)
    KernelAbstractions.isgpu(backend) || return NaN
    try
        A = KernelAbstractions.allocate(backend, Float32, CANARY_N, CANARY_N)
        B = KernelAbstractions.allocate(backend, Float32, CANARY_N, CANARY_N)
        C = KernelAbstractions.allocate(backend, Float32, CANARY_N, CANARY_N)
        fill!(A, 1.0f0); fill!(B, 0.5f0); fill!(C, 0.0f0)
        mul!(C, A, B); KernelAbstractions.synchronize(backend)          # warm up / JIT
        t = minimum(1:reps) do _
            @elapsed (mul!(C, A, B); KernelAbstractions.synchronize(backend))
        end
        return 2 * Float64(CANARY_N)^3 / t / 1e9
    catch err
        @warn "device canary failed — contention cannot be detected for this run" err
        return NaN
    end
end

# Baseline recorded by the tuning probe on an idle device (profile key `canary_gflops`). Absent
# for a profile written before the canary existed, in which case a run still REPORTS its canary
# and cross-run comparison is possible — it just cannot self-diagnose.
canary_baseline() = something(tryparse(Float64, get(_tuning_table()[2], "canary_gflops", "")),
    NaN)

# Below this fraction of baseline, the device is being shared and the run's timings are not
# comparable to anything.
const CANARY_FLOOR = 0.85

function canary_stamp(backend)
    gf = device_canary(backend)
    isnan(gf) && return "canary=NA"
    base = canary_baseline()
    isnan(base) && return @sprintf("canary=%.0f GF/s (no baseline; re-tune to record one)", gf)
    frac = gf / base
    frac < CANARY_FLOOR &&
        @warn "DEVICE CONTENDED: the canary is well below this machine's idle baseline. " *
              "Timings from this run are not comparable." canary_gflops=gf baseline=base frac
    return @sprintf("canary=%.0f/%.0f GF/s (%.0f%% of idle baseline)%s", gf, base, 100frac,
        frac < CANARY_FLOOR ? "  ** CONTENDED **" : "")
end

function repro_stamp(backend = CPU(); matrix = BENCH_MATRIX)
    gitsha = try
        readchomp(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
    manifest = joinpath(@__DIR__, "Manifest.toml")
    # sha256, not Base.hash (unstable across Julia versions).
    mhash = isfile(manifest) ? bytes2hex(sha256(read(manifest)))[1:16] : "none"
    strat = try
        KernelAbstractionsPseudospectra.trsm_strategy()
    catch
        "?"
    end
    pdiv = try
        KernelAbstractionsPseudospectra.KATRSM.PDIV_ACCURATE
    catch
        "?"
    end
    envs = ["$k=$(ENV[k])" for k in
            ("KAPSEUDO_TRSM", "KAPSEUDO_TRSM_TILECOLS", "KAPSEUDO_TRSM_BLOCKWARPS", "KAPSEUDO_TRSM_WARPGRIDPTS",
             "KAPSEUDO_TRSM_WGS", "KAPSEUDO_TUNE_PROFILE",
             "KAPSEUDO_STRIDED", "BENCH_NIT", "BENCH_GRIDN") if haskey(ENV, k)]
    [
        "git=$gitsha",
        "julia=$(VERSION)",
        "cpu=$(Sys.CPU_NAME) ($(Sys.CPU_THREADS) threads)  machine=$(Sys.MACHINE)",
        "julia_threads=$(Threads.nthreads())",
        "blas=$(BLAS.get_config() |> x -> first(x.loaded_libs).libname)  blas_threads=$(BLAS.get_num_threads())",
        "backend=$(backend)",
        "trsm_strategy=$strat  pdiv_accurate=$pdiv",
        tuning_stamp(),
        canary_stamp(backend),
        "matrix=$matrix",
        "kapseudo_env=[" * join(envs, ", ") * "]",
        "bench_manifest_hash=$mhash",
    ]
end
