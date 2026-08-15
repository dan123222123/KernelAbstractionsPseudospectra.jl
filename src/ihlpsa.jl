using ArraysOfArrays
using ThreadsX
using ProgressBars
using Random

# triangular solves using kernel abstractions
include("KATRSM/KATRSM.jl")
using .KATRSM: set_pdiv_accurate!,
               _batched_forward_solve_pencil, _batched_backward_solve_pencil,
               _batched_column_oriented_forward_solve_pencil,
               _batched_column_oriented_backward_solve_pencil,
               _batched_column_oriented_forward_solve_eye,
               _batched_column_oriented_backward_solve_eye,
               _tiled_panel_forward, _tiled_panel_backward,
               _tiled_panel_forward_eye, _tiled_panel_backward_eye,
               _tiled_trailing, _tiled_trailing_eye

# Split out for readability:
include("ihlpsa_workspace.jl")   # Lanczos KA kernels + IHLworkspace (device-resident state)
include("ihlpsa_trsm.jl")        # trsm knob resolution + trsm_strategy routing + lockstep_ihl!

## HOST FUNCTIONS ##

# Largest eigenvalue of the small Lanczos tridiagonal. Float64 uses LAPACK `eigmax`; every other
# eltype uses Gershgorin-bounded Sturm bisection, since iterative eig solvers can return NaN in
# exotic arithmetic (e.g. GenericLinearAlgebra's QL at Float32-limb MultiFloat precision).
function _eigmax_tridiag(d::AbstractVector{Float64}, e::AbstractVector{Float64})
    eigmax(SymTridiagonal(d, e))
end
function _eigmax_tridiag(d::AbstractVector{R}, e::AbstractVector{R}) where {R <: AbstractFloat}
    n = length(d)
    n == 1 && return d[1]
    rad(i) = (i > 1 ? abs(e[i - 1]) : zero(R)) + (i < n ? abs(e[i]) : zero(R))
    lo = minimum(d[i] - rad(i) for i in 1:n)   # Gershgorin bounds bracket every eigenvalue
    hi = maximum(d[i] + rad(i) for i in 1:n)
    pv = sqrt(floatmin(R))   # pivot clamp: e²/q stays in range even for narrow-range eltypes
    for _ in 1:512           # ≥ enough halvings for any practical precision; breaks earlier
        mid = (lo + hi) / 2
        (lo < mid < hi) || break               # interval at arithmetic resolution
        cnt = 0                                # Sturm/LDLᵀ count of eigenvalues below `mid`
        q = one(R)
        for i in 1:n
            q = i == 1 ? d[1] - mid : d[i] - mid - e[i - 1]^2 / q
            abs(q) < pv && (q = copysign(pv, q))
            q < 0 && (cnt += 1)
        end
        cnt == n ? (hi = mid) : (lo = mid)     # all n below mid ⇒ λmax < mid
        (hi - lo) <= eps(R) * max(abs(hi), one(R)) && break
    end
    hi
end

# Solve (SymTridiagonal(d,e) − μI) x = b by Thomas elimination with a pivot clamp (same clamp as
# `_eigmax_tridiag`) so a near-singular inverse-iteration shift can't divide by ~0. Branch-only —
# no LAPACK, so it can't NaN at exotic precision where a QL/QR factorization does.
function _tridiag_shift_solve(d::AbstractVector{R}, e::AbstractVector{R}, μ::R,
        b::AbstractVector{R}) where {R <: AbstractFloat}
    n = length(d)
    pv = sqrt(floatmin(R))
    cp = Vector{R}(undef, n)
    x = Vector{R}(undef, n)
    piv = d[1] - μ
    abs(piv) < pv && (piv = copysign(pv, piv))
    cp[1] = (n > 1 ? e[1] : zero(R)) / piv
    x[1] = b[1] / piv
    for i in 2:n
        piv = (d[i] - μ) - e[i - 1] * cp[i - 1]
        abs(piv) < pv && (piv = copysign(pv, piv))
        cp[i] = (i < n ? e[i] : zero(R)) / piv
        x[i] = (b[i] - e[i - 1] * x[i - 1]) / piv
    end
    for i in (n - 1):-1:1
        x[i] -= cp[i] * x[i + 1]
    end
    x
end

# |last component| of the λmax eigenvector, by inverse iteration with shift μ just above λmax:
# (T−μI)⁻¹ amplifies that eigencomponent by ~1/(μ−λmax), so three steps converge it. Paired with
# the next Lanczos residual β_{k+1}, Paige's identity ‖Ty − θy‖ = β_{k+1}|s_k| turns this into a
# certified error bound on σ (see `ihlsrg!`). Needs no eigen-decomposition, so it is robust in
# exotic arithmetic like `_eigmax_tridiag`.
function _tridiag_top_lastcomp(d::AbstractVector{R}, e::AbstractVector{R}, λ::R) where {R <:
                                                                                        AbstractFloat}
    n = length(d)
    n == 1 && return one(R)
    μ = λ + max(abs(λ), one(R)) * R(64) * eps(R)
    y = fill(inv(sqrt(R(n))), n)
    for _ in 1:3
        y = _tridiag_shift_solve(d, e, μ, y)
        y ./= norm(y)
    end
    abs(y[end])
end

# σ_min = 1/√(eigmax) of [(zB−A)(zB−A)ᴴ]⁻¹; the (γ,δ)-pseudospectral value is
# σ_min/(γ+δ|z|) (Frayssé et al.). Work type `R = promote_type(Float64, real(eltype(α)))` floors
# the eigmax precision at Float64, since an F32 tridiagonal can have an eigenvalue past F32's
# ~1e38 range near a true eigenvalue. R cannot widen the RANGE of a Float32-limb MultiFloat (it
# out-precisions Float64, so R stays e.g. Float32x4 with Float32's range), so the tridiagonal is
# NORMALIZED before the eig — otherwise the eig's internal squarings overflow large-but-finite
# entries and return NaN from finite input. λmax is scale-equivariant, so scale down, solve, and
# rescale. A non-finite tail (Lanczos breakdown past convergence — see below) truncates to the
# finite leading block, the already-converged estimate; only a point dead from iteration 1 (e.g.
# z exactly at an eigenvalue) or an out-of-range λmax pins to `eps(real(eltype(zv)))`, keeping
# `log10(sr)` well-defined.
function ihlsrg!(sr, zv, γ, δ, α, β; resid = nothing)
    R = promote_type(Float64, real(eltype(α)))
    ε = eps(real(eltype(zv)))
    Threads.@threads for i in eachindex(zv)
        αi = R.(real.(α[:, i]))
        βi = R.(real.(β[2:(end - 1), i]))
        # Longest finite leading block: a point that converges early but keeps iterating under
        # the fixed driver drives β to the eltype's precision floor, and ‖v‖² then UNDERFLOWS on
        # eltypes whose range is narrower than their precision warrants (e.g. Float32-limb
        # MultiFloat) — β = 0, division blows up, and the tail goes Inf/NaN. Truncate to the
        # already-converged finite leading block rather than discard the point.
        ka = findfirst(!isfinite, αi)
        kb = findfirst(!isfinite, βi)
        n = min(ka === nothing ? length(αi) : ka - 1,
            kb === nothing ? length(αi) : kb)   # n diagonal entries need βi[1:n-1] finite
        if n < 1
            sr[i] = ε                            # dead from iteration 1 (e.g. z at an eigenvalue)
            resid === nothing || (resid[i] = zero(eltype(resid)))
            continue
        end
        αv, βv = view(αi, 1:n), view(βi, 1:(n - 1))
        s = max(maximum(abs, αv), maximum(abs, βv; init = zero(R)), one(R))
        ds, es = αv ./ s, βv ./ s
        λs = _eigmax_tridiag(ds, es)
        v = 1 / ((γ + δ * abs(zv[i])) * sqrt(s) * sqrt(λs))
        sr[i] = isfinite(v) ? v : ε
        if resid !== nothing
            # Ritz residual bound on the RELATIVE σ error: ½·β_{n+1}·|s_n|/λ (scale-invariant,
            # computed on the scaled block). A truncated or pinned point has already found an
            # invariant subspace, so its residual is 0. Used by the adaptive driver as a certified
            # stopping test (retire when resid·σ ≤ atol + rtol·σ).
            βnext = n < length(αi) ? zero(R) : R(real(β[n + 1, i])) / s
            resid[i] = (isfinite(v) && isfinite(βnext)) ?
                       oftype(sr[i], R(0.5) * abs(βnext) * _tridiag_top_lastcomp(ds, es, λs) / λs) :
                       zero(eltype(resid))
        end
    end
end

## END HOST FUNCTIONS ##

## WRAPPER FUNCTIONS ##

# Flatten the grid to a point vector and split 1:n into contiguous zpd-sized batches. Shared by
# the fixed (`_sdihlpsa`) and adaptive (`_sdihlpsa_adaptive`) single-device workers.
function _grid_batches(zg, zpd)
    zv = collect(Iterators.flatten(zg))
    return zv, collect(Iterators.partition(1:length(zv), min(length(zv), zpd)))
end

# Single-device batched inverse-Lanczos worker (fixed depth) — the per-device unit of
# work. The multi-device drivers fan grid columns out across devices via
# `_ihlpsa_fanout`, which calls one worker per device; `_sdihlpsa` (fixed) and
# `_sdihlpsa_adaptive` (adaptive) are those workers. Neither is exported — the sole
# public entry point is `ihlpsa`.
function _sdihlpsa(
        backend,
        zg::AbstractArray{T, 2},
        P::AbstractMatrixPencil{T},
        γ,
        δ,
        zpd::Integer,
        nit::Integer = max(1, ceil(Integer, log2(size(P, 1)))),
        x₀::Union{Missing, AbstractVector{T}, AbstractArrayOfSimilarArrays{T}} = missing,
        pchnl::Union{Missing, Channel} = missing,
        wgs = missing
) where {T <: Complex}
    dev = device(backend)
    bgarray = get_bgarray(backend)
    zv, idxbatches = _grid_batches(zg, zpd)
    gtotal = length(zv)
    sr = zeros(real(T), gtotal)
    dzv = adapt(bgarray, zv)
    α = adapt(bgarray, zeros(T, nit, gtotal))
    β = adapt(bgarray, zeros(T, nit + 1, gtotal))
    ihl = adapt(bgarray, IHLworkspace(P, zpd, x₀))
    _foreach = !KernelAbstractions.isgpu(backend) ? ThreadsX.foreach : Base.foreach
    @sync _foreach(idxbatches) do idxb
        view(ihl.zv, 1:length(idxb)) .= view(dzv, idxb)
        lockstep_ihl!(view(α, :, idxb), view(β, :, idxb), ihl, nit, length(idxb); wgs)
        Threads.@spawn begin
            device!(backend, dev)
            if !ismissing(pchnl)
                put!(pchnl, length(idxb) * nit)
            end
            ihlsrg!(view(sr, idxb), view(zv, idxb), γ, δ,
                adapt(Array, α[:, idxb]), adapt(Array, β[:, idxb]))
        end
    end
    return Matrix{real(T)}(reshape(sr, size(zg)))
end

# Largest zpd that fits the target device memory within margin `moe`. Accounts for 4*m*m pencil
# bytes (A, Ac, B, Bc), and per-grid (1 zv + 4*m workspace + (2*nit+1) α/β) bytes.
function findmaxbatchihl(backend, T, m, nit; moe = 0.1)
    device_reclaim(backend)
    floor(Integer,
        (device_bytes_available(backend) * (1 - moe) - (sizeof(T) * (4 * m * m + 1))) /
        (sizeof(T) * (1 + 4 * m + 2 * nit + 1)))
end

# Pencil-aware variant: reserves the bytes `_device_pencil_bytes` actually places on the device
# for `P`, instead of the dense-A/Ac/B/Bc envelope above (a Diagonal-B/Z pencil needs ~2m², a
# dense-Z generalized pencil the full ~5m² the envelope under-counts). Distinct arity from the
# (backend, T, m, nit) method, so dispatch is unambiguous.
function findmaxbatchihl(backend, P::AbstractMatrixPencil{T}, nit; moe = 0.1) where {T}
    device_reclaim(backend)
    m = size(P, 1)
    floor(Integer,
        (device_bytes_available(backend) * (1 - moe) - (_device_pencil_bytes(P) + sizeof(T))) /
        (sizeof(T) * (1 + 4 * m + 2 * nit + 1)))
end

# Partition 1:ncols into min(ndev, ncols) balanced blocks (sizes differ by ≤ 1),
# one per device; empty when ncols == 0.
function _device_column_partition(ncols::Integer, ndev::Integer)
    ndev ≥ 1 || throw(ArgumentError("ndev must be ≥ 1, got $ndev"))
    ncols ≥ 0 || throw(ArgumentError("ncols must be ≥ 0, got $ncols"))
    nblocks = min(Int(ndev), Int(ncols))
    nblocks == 0 && return StepRange{Int, Int}[]
    # Round-robin (strided) assignment: device b takes columns b, b+nblocks, …, so clustered
    # hard/deep regions spread across devices (load-balances the adaptive driver, whose per-point
    # work varies). Results are scattered back to original columns afterward. KAPSEUDO_STRIDED=0
    # selects the contiguous bands.
    if get(ENV, "KAPSEUDO_STRIDED", "1") == "0"
        blocks = Vector{StepRange{Int, Int}}(undef, nblocks)
        q, r = divrem(Int(ncols), nblocks)
        lo = 1
        for b in 1:nblocks
            hi = lo + q + (b ≤ r) - 1
            blocks[b] = lo:1:hi
            lo = hi + 1
        end
        return blocks
    end
    return [b:nblocks:Int(ncols) for b in 1:nblocks]
end

# Multi-device GPU fan-out shared by the fixed and adaptive drivers. Partitions zg's columns
# across `devs`, resolves a per-device zpd, then spawns one `worker(zgb, zpd_dev, zgidx)` per
# device — `zgidx` is the device's global column block, so a worker can report results in grid
# coordinates. zpd is resolved sequentially BEFORE the parallel fan-out: device! is process-global
# on CUDA/AMDGPU and findmaxbatchihl queries the *current* device, so racing it across spawns
# could read the wrong device's budget. `zip` truncates to the shorter side, so surplus devices
# (ncols < ndev) go untouched.
function _ihlpsa_fanout(backend, zg::AbstractArray{T, 2}, P::AbstractMatrixPencil{T},
        budget_nit, zpd, devs, worker; pbar = nothing) where {T <: Complex}
    ismissing(devs) && (devs = devices(backend))
    blocks = _device_column_partition(size(zg, 2), length(devs))
    pbar === nothing ||
        set_description(pbar, "$(length(blocks))/$(length(devs)) device(s), grid points * nit:")
    zpd_devs = map(zip(devs, blocks)) do (dev, zgidx)
        device!(backend, dev)
        zgb_len = length(zgidx) * size(zg, 1)
        ismissing(zpd) ? min(findmaxbatchihl(backend, P, budget_nit), zgb_len) : zpd
    end
    results = Vector{Any}(undef, length(blocks))
    @sync for (did, (dev, zgidx)) in enumerate(zip(devs, blocks))
        Threads.@spawn begin
            device!(backend, dev)
            results[did] = worker(zg[:, zgidx], zpd_devs[did], zgidx)
        end
    end
    # Return `blocks` too: with strided partitioning the per-device results are no
    # longer in column order, so callers scatter `results[d]` back to `blocks[d]`.
    return results, blocks
end

# Scatter each device's columns back to their original grid positions (strided partition ⇒
# results aren't in column order). `extract` pulls the array out of each device's result.
function _scatter_columns!(dest::AbstractMatrix, results, blocks, extract = identity)
    for (r, blk) in zip(results, blocks)
        # @inbounds: blocks is a _device_column_partition of 1:size(dest, 2), so every
        # blk index is within dest's column range by construction.
        @inbounds dest[:, blk] = extract(r)
    end
end

# CPU thread fan-out — the thread-analogue of the multi-device `_ihlpsa_fanout`. Partitions the
# grid's `npts` points across `min(npts, nthreads())` threads, runs one independent
# `worker(point_indices)` per chunk under `Threads.@spawn`, and returns the per-chunk results with
# their index blocks for the caller to scatter back. Point granularity (not column) so a 1-column
# w-point shift-batch still uses every thread; each worker allocates its own workspace, so chunks
# share no mutable state.
#
# `strided` picks the partition to match the workload — the CPU mirror of the GPU
# `_device_column_partition`'s strided/contiguous choice: strided (round-robin) is for the
# ADAPTIVE driver, whose per-point depth clusters spatially, so round-robin hands every thread a
# balanced mix; contiguous (even bands) is for the FIXED driver, whose per-point work is uniform.
function _cpu_fanout(worker, npts::Integer; strided::Bool = true)
    nchunks = min(npts, Threads.nthreads())
    blocks = if strided
        [c:nchunks:npts for c in 1:nchunks]                     # round-robin (StepRanges)
    else
        b = round.(Int, range(0, npts; length = nchunks + 1))
        [(b[c] + 1):b[c + 1] for c in 1:nchunks]                # contiguous even bands
    end
    results = Vector{Any}(undef, nchunks)
    @sync for c in 1:nchunks
        Threads.@spawn (results[c] = worker(blocks[c]))
    end
    return results, blocks
end

# Scatter per-chunk results back to their original flat grid positions (round-robin ⇒ chunk results
# aren't in grid order). Flat-index analogue of `_scatter_columns!`; `extract` pulls the per-chunk
# vector out of each worker's return value.
function _scatter_points!(dest::AbstractVector, results, blocks, extract = identity)
    for (r, blk) in zip(results, blocks)
        # @inbounds: blocks partitions 1:npts, so every index is within dest by construction.
        @inbounds dest[blk] = extract(r)
    end
end

# CPU fixed-nit driver — multithreads across the grid via `_cpu_fanout`. A single `_sdihlpsa` over
# the whole grid runs single-threaded on CPU: its `ThreadsX.foreach` parallelizes across
# *batches*, but the CPU path hands it the entire grid as ONE batch (the shared `IHLworkspace` is
# mutated per batch, so concurrent batches would race). So we fan the points out and run one
# independent `_sdihlpsa` per thread-chunk, each with its own workspace. Uniform per-point work ⇒
# `strided=false` (contiguous bands).
#
# `x₀` is resolved ONCE and shared across chunks so every grid point uses the same start vector
# regardless of the chunking — otherwise the σ field could differ across chunk seams.
function _cpu_ihlpsa_threaded(backend, zg::AbstractArray{T, 2}, P::AbstractMatrixPencil{T},
        γ, δ, nit::Integer, x₀, pchnl, wgs) where {T <: Complex}
    npts = length(zg)
    seed = ismissing(x₀) ? (s = randn(T, size(P, 1)); s ./= norm(s); s) : x₀
    zflat = collect(Iterators.flatten(zg))            # column-major point order (matches _sdihlpsa)
    results, blocks = _cpu_fanout(npts; strided = false) do block   # contiguous: uniform per-point work
        chunk = reshape(zflat[block], length(block), 1)
        vec(_sdihlpsa(backend, chunk, P, γ, δ, length(block), nit, seed, pchnl, wgs))
    end
    out = Vector{real(T)}(undef, npts)
    _scatter_points!(out, results, blocks)
    return reshape(out, size(zg))
end

# Fixed-nit engine: multi-device batched inverse-Lanczos at a caller-given depth.
# The public `ihlpsa(..., nit::Integer, ...)` method forwards here.
function _ihlpsa_fixed(
        backend,
        zg::AbstractArray{T, 2},
        P::AbstractMatrixPencil{T},
        nit::Integer,
        γ = 1,
        δ = 0;
        x₀::Union{Missing, AbstractVector{T}} = missing,
        progress = false,
        zpd = missing,
        devs = missing,
        wgs = missing
) where {T <: Complex}
    _validate_weights(γ, δ)
    # progress bar + consumer task only when caller asks for one — otherwise the consumer blocks
    # forever on an empty channel and the spawn leaks at precompile time.
    pbar = progress ? ProgressBar(total = nit * length(zg), printing_delay = 0.001) :
           nothing
    pchnl = progress ? Channel() : nothing
    if progress
        Threads.@spawn begin
            for delta in pchnl
                ProgressBars.update(pbar, delta)
            end
        end
    end
    if KernelAbstractions.isgpu(backend)
        results, blocks = _ihlpsa_fanout(backend, zg, P, nit, zpd, devs,
            (zgb, zpd_dev, _) -> _sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀,
                progress ? pchnl : missing, wgs); pbar)
        result = Matrix{real(T)}(undef, size(zg))
        _scatter_columns!(result, results, blocks)
    else
        # CPU: split the grid across threads, one independent _sdihlpsa (own workspace) per
        # chunk — see _cpu_ihlpsa_threaded for why a single _sdihlpsa call runs single-threaded
        # here despite its internal ThreadsX.foreach.
        progress && set_description(pbar, "CPU device, grid points * nit:")
        result = _cpu_ihlpsa_threaded(
            backend, zg, P, γ, δ, nit, x₀, progress ? pchnl : missing, wgs)
    end
    progress && close(pchnl)
    return permutedims(result)
end

# Deterministic unit-norm complex start vector for the adaptive driver. The same vector is reused
# across chunks so successive σ are one Lanczos run sampled at increasing depth — a fresh x₀ per
# chunk would compare independent runs and make convergence meaningless.
function _adaptive_x₀(::Type{T}, m, seed) where {T <: Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    return x ./ norm(x)
end

_adaptive_default_rtol(::Type{T}) where {T <: Complex} = real(T) == Float32 ? 1.0f-4 : 1e-6
_adaptive_default_atol(::Type{T}) where {T <: Complex} = eps(real(T))

# Certified convergence test via a point's Ritz residual bound `resid` (= ½ β_{k+1}|s_k|/λ, a
# rigorous bound on the RELATIVE error of σ; see `ihlsrg!`). Retire when `resid·σ ≤ atol + rtol·σ`.
# Unlike a successive-σ-change (Cauchy) test, this bounds the true error rather than just
# measuring that the iterate stopped moving, so it reaches the precision floor uniformly in the
# matrix size. eps short-circuit: `ihlsrg!` pins a point at/near a true eigenvalue to
# `eps(real(T))` (already converged; its residual is 0).
@inline function _adaptive_converged(σ::R, resid::R, rtol::R, atol::R) where {R <: Real}
    σ <= eps(R) && return true
    return resid * σ <= atol + rtol * σ
end

# Ablation-only successive-σ-change (Cauchy) test, selectable via `criterion = :cauchy` for
# bench/bench_stopping.jl. Not a bound: near slow-converging points (Lanczos ratio ρ→1)
# consecutive iterates can agree to rtol while both sit ~rtol/(1−ρ) from the limit.
@inline function _cauchy_converged(σ_new::R, σ_prev::R, rtol::R, atol::R) where {R <: Real}
    σ_new <= eps(R) && return true
    return abs(σ_new - σ_prev) <= atol + rtol * σ_new
end

"""
    ihlpsa(backend, zg, P, nit::Integer; γ=1, δ=0, x₀, progress, zpd, devs, wgs)
    ihlpsa(backend, zg, P; γ=1, δ=0, nit_chunk=2, nit_max=…, rtol, atol, nconfirm,
                           x₀, seed, zpd, devs, wgs, verbose, on_batch)

Batched inverse-Lanczos pseudospectra of the matrix pencil `P` over the complex
grid `zg`, fanned out across all devices of `backend` (pass `devs` to restrict).
Each grid point's value is the (γ,δ)-pseudospectral value `σ_min(zB − A)/(γ + δ|z|)`
(Frayssé et al.; `z ∈ σ_ε^{(γ,δ)}` iff this is `< ε`). With the default `γ=1, δ=0`
this is the standard pseudospectrum value `σ_min(zB − A)`. Both forms return the
grid-shaped `Matrix` of σ values. Perturbation scaling `γ`,`δ` are keyword
arguments.

**Fixed depth** — pass `nit::Integer`: every grid point runs exactly `nit`
Lanczos iterations. `progress=true` shows a progress bar.

**Adaptive depth** — omit `nit`: runs lockstep inverse-Lanczos in chunks of
`nit_chunk` with resident per-point state, retiring each grid point once its σ
has converged and gathering the survivors so kernels only touch live points.
Convergence is a certified Ritz residual bound on the relative error of σ:
`½·β_{k+1}·|s_k|/λ ≤ rtol` (with absolute floor `atol`), confirmed over `nconfirm`
consecutive chunks, where `β_{k+1}` is the next Lanczos residual norm and `|s_k|`
the last component of the extreme Ritz vector. Unlike a successive-change test,
this bounds the true error and so reaches the precision floor uniformly in the
matrix size (a change-based test stalls early near small-gap points). Per-point
work tracks each point's own convergence depth rather than the slowest point's,
capped at `nit_max`. A fixed deterministic start vector (`seed`) is reused across
chunks so the residual compares one Lanczos run at successive depths. The σ match
the fixed layout
(`ihlpsa(...) ≈ ihlpsa(..., nit)` at the converged depth). The convergence depth
reached is a diagnostic, not a routine return value: pass `verbose=true` to log
the deepest depth, or call the un-exported `KernelAbstractionsPseudospectra._ihlpsa_adaptive`
driver, which returns `(σ::Matrix, nit_grid::Matrix{Int})` — `nit_grid[i]` is the
depth at which grid point `i` retired (`maximum(nit_grid)` is the deepest point).

`on_batch(idx, σ, nit)` delivers each retired batch as it finalizes — positions
(`Vector{CartesianIndex{2}}` into the returned matrices) with the matching host
value vectors, every grid point exactly once, values identical to the final
return. Deliveries fire synchronously from worker threads but are serialized, so
the callback need not be thread-safe; a slow callback delays the solve (throttle
writes), and an exception aborts it. Made for checkpointing hours-scale solves.

# Examples
```julia
using KernelAbstractionsPseudospectra, KernelAbstractions
P = MatrixPencil(A)                                   # or MatrixPencil(A, B)
_, _, zg = qgrid(ComplexF64, (-2, 5), (-4.5, 4.5), (300, 300))

srg = ihlpsa(CPU(), zg, P, 16)                        # fixed nit=16
srg = ihlpsa(CPU(), zg, P)                            # adaptive depth
srg = ihlpsa(CUDABackend(), zg, P; rtol=1e-8, verbose=true)  # tighter tol, logs depth, multi-GPU
```

See also [`MatrixPencil`](@ref), [`qgrid`](@ref), [`ℂsvdpsa`](@ref).
"""
function ihlpsa(
        backend,
        zg::AbstractArray{T, 2},
        P::AbstractMatrixPencil{T},
        nit::Integer;
        γ = 1,
        δ = 0,
        x₀::Union{Missing, AbstractVector{T}} = missing,
        progress = false,
        zpd = missing,
        devs = missing,
        wgs = missing
) where {T <: Complex}
    return _ihlpsa_fixed(backend, zg, P, nit, γ, δ; x₀, progress, zpd, devs, wgs)
end

# Adaptive form — omit `nit`. Mirrors the fixed-nit form: returns just `srg::Matrix`; the
# convergence depth is available via `verbose=true` or by calling the un-exported
# `_ihlpsa_adaptive` driver below, which returns `(srg, nit_grid)`.
function ihlpsa(backend, zg::AbstractArray{T, 2}, P::AbstractMatrixPencil{T};
        kwargs...) where {T <: Complex}
    return first(_ihlpsa_adaptive(backend, zg, P; kwargs...))
end

# Wrap a user `on_batch` for one worker: `pos` maps the worker's flat point index to the
# point's position in the RETURNED (permutedims'd) matrices, and the shared lock serializes
# concurrent workers' deliveries so the user callback need not be thread-safe. `nothing`
# stays `nothing` — workers then skip delivery entirely.
_batch_relay(::Nothing, lk, pos) = nothing
function _batch_relay(on_batch, lk, pos)
    (idxb, σv, nitv) -> begin
        idx = [pos(fi) for fi in idxb]
        @lock lk on_batch(idx, σv, nitv)
    end
end

# Multi-device per-point adaptive inverse Lanczos. Returns `(srg::Matrix, nit_grid::Matrix{Int})`;
# the public `ihlpsa(...)` wraps this and drops `nit_grid` — exposed via `verbose=true` logging, or
# by calling this directly (the test suite does). Fans columns out across devices via
# `_ihlpsa_fanout`, but each device's adaptive loop (`_sdihlpsa_adaptive`) stops at its own
# converged depth rather than lockstepping to the global worst point. `nit_grid[i]` is point i's
# retirement depth; `maximum(nit_grid)` is the deepest.
function _ihlpsa_adaptive(
        backend,
        zg::AbstractArray{T, 2},
        P::AbstractMatrixPencil{T};
        γ = 1,
        δ = 0,
        nit_chunk::Integer = 2,
        nit_max::Integer = 8 * max(1, ceil(Integer, log2(size(P, 1)))),
        rtol::Real = _adaptive_default_rtol(T),
        atol::Real = _adaptive_default_atol(T),
        nconfirm::Integer = 2,
        criterion::Symbol = :certified,
        x₀::Union{Missing, AbstractVector{T}} = missing,
        seed::Integer = 0x61646170,
        zpd = missing,
        devs = missing,
        wgs = missing,
        verbose = false,
        on_batch = nothing
) where {T <: Complex}
    _validate_weights(γ, δ)
    # `criterion` is an unexported ablation surface (bench/bench_stopping.jl): :certified is
    # the shipped Ritz-residual test, :cauchy the successive-σ-change test.
    criterion in (:certified, :cauchy) ||
        throw(ArgumentError("criterion must be :certified or :cauchy (got $(criterion))"))
    R = real(T)
    m = size(P, 1)
    rtol = R(rtol)
    atol = R(atol)
    x₀_fixed = ismissing(x₀) ? _adaptive_x₀(T, m, seed) : x₀
    n_x = size(zg, 1)
    cb_lock = ReentrantLock()
    if KernelAbstractions.isgpu(backend)
        # findmaxbatchihl is sized with nit_max since α/β hold the full budget.
        results, blocks = _ihlpsa_fanout(backend, zg, P, nit_max, zpd, devs,
            (zgb, zpd_dev, zgidx) -> _sdihlpsa_adaptive(backend, zgb, P, γ, δ, nit_chunk,
                nit_max, rtol, atol, nconfirm, criterion, x₀_fixed, zpd_dev, wgs;
                on_batch = _batch_relay(on_batch, cb_lock,
                    # Worker-flat fi over its column block: row mod1(fi, n_x); the block
                    # gives the global column; the return is permutedims'd, so swap.
                    fi -> CartesianIndex(zgidx[(fi - 1) ÷ n_x + 1], mod1(fi, n_x)))))
        sr = Matrix{real(T)}(undef, size(zg))
        nit_grid = Matrix{Int}(undef, size(zg))
        _scatter_columns!(sr, results, blocks, r -> r[1])
        _scatter_columns!(nit_grid, results, blocks, r -> r[2])
        unconverged = any(r[3] for r in results)
    else
        # CPU: fan the grid's points out across threads with `_cpu_fanout` (round-robin — the fixed
        # driver uses the contiguous variant), one INDEPENDENT `_sdihlpsa_adaptive` per thread. This
        # is ORTHOGONAL to each worker's own per-chunk survivor-gather: fan-out parallelizes the
        # *space* axis, survivor-gather compacts the *nit-depth* axis. Round-robin matters because
        # deep points cluster spatially — without it they'd all land on one thread, killing
        # parallelism in the (wall-clock-dominant) adaptive tail. A user-supplied zpd still caps
        # each chunk's batch. Per-point σ is independent of chunk-mates, so the result is
        # thread-count invariant.
        npts = length(zg)
        zflat = collect(Iterators.flatten(zg))
        results, blocks = _cpu_fanout(npts) do block
            _sdihlpsa_adaptive(backend, reshape(zflat[block], length(block), 1), P, γ, δ,
                nit_chunk, nit_max, rtol, atol, nconfirm, criterion, x₀_fixed,
                ismissing(zpd) ? length(block) : zpd, wgs;
                on_batch = _batch_relay(on_batch, cb_lock,
                    # The worker sees its block reshaped to one column, so its flat fi maps
                    # through the block to a global flat grid index first.
                    fi -> begin
                        gk = block[fi]
                        CartesianIndex((gk - 1) ÷ n_x + 1, mod1(gk, n_x))
                    end))
        end
        σflat = Vector{R}(undef, npts)
        nitflat = Vector{Int}(undef, npts)
        _scatter_points!(σflat, results, blocks, r -> vec(r[1]))
        _scatter_points!(nitflat, results, blocks, r -> vec(r[2]))
        sr = reshape(σflat, size(zg))
        nit_grid = reshape(nitflat, size(zg))
        unconverged = any(r -> r[3], results)
    end
    unconverged &&
        @warn "ihlpsa adaptive hit nit_max=$nit_max with unconverged point(s) (rtol=$rtol)"
    verbose && @info "ihlpsa adaptive done" nit = maximum(nit_grid)
    return permutedims(sr), permutedims(nit_grid)
end

# Gather rows `keep` (host Int vector) from a (g, m, k) device array into a packed (nkeep, m, k)
# array via the `_qv_gather!` kernel — a direct `A[keep,:,:]` is miscompiled on oneAPI. `keep` is
# moved to the device once for the kernel to index.
function _gather_rows(backend, A, keep)
    g, m, k = size(A)
    dst = similar(A, length(keep), m, k)
    keepd = adapt(get_bgarray(backend), keep)
    _qv_gather!(backend)(dst, A, keepd; ndrange = (length(keep), m, k))
    return dst
end

# Single-device adaptive worker — per-point retirement with resident state. After each chunk the
# converged points retire and survivors' per-point state (workspace rows + α/β columns) is
# gathered into a packed prefix (the 3-D Qv gather goes through `_gather_rows`, see its note) so
# subsequent chunks run kernels over live points only. `idx_glob` maps packed position → original
# flat grid index. Returns (sr_matrix, nit_grid, unconverged::Bool); nit_grid[i] is grid point i's
# retirement depth (or nit_max if it never converged).
function _sdihlpsa_adaptive(backend, zg::AbstractArray{T, 2}, P::AbstractMatrixPencil{T},
        γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, criterion, x₀, zpd,
        wgs; on_batch = nothing) where {T <: Complex}
    R = real(T)
    bgarray = get_bgarray(backend)
    zv_h, idxbatches = _grid_batches(zg, zpd)
    gtotal = length(zv_h)
    σ_out = zeros(R, gtotal)
    σ_prev = zeros(R, gtotal)            # indexed by original flat grid index
    nit_at = zeros(Int, gtotal)          # per-point retirement depth (orig flat index)
    unconverged = false
    # One workspace for the whole call: the pencil uploads once, and `lockstep_ihl!` reseeds
    # per-point state at start=1 for each batch. The survivor-gather below rebinds `ihl` to packed
    # copies built around the same device pencil.
    ihl_full = adapt(bgarray, IHLworkspace(P, length(first(idxbatches)), x₀))
    for (bi, idxb) in enumerate(idxbatches)
        # Release the device pool between batches: survivor-gather copies strand until a
        # host-pressure GC happens to run, and the batch planner leaves no slack to absorb them.
        # GPU-only — CPU's reclaim hook is a full GC, far costlier than the stranded copies.
        bi == 1 || !KernelAbstractions.isgpu(backend) || device_reclaim(backend)
        g = length(idxb)
        ihl = ihl_full
        α = adapt(bgarray, zeros(T, nit_max, g))
        β = adapt(bgarray, zeros(T, nit_max + 1, g))
        view(ihl.zv, 1:g) .= adapt(bgarray, zv_h[idxb])
        idx_glob = collect(idxb)
        streak = zeros(Int, g)        # per packed point, gathered with survivors
        nit_done = 0
        while g > 0 && nit_done < nit_max
            nit_new = min(nit_done + nit_chunk, nit_max)
            lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl,
                nit_new, g; wgs, start = nit_done + 1)
            sr_a = zeros(R, g)
            # :cauchy skips the residual (it only compares successive σ), so its timing
            # doesn't pay for the eigenvector the certified bound needs.
            resid_a = criterion === :certified ? zeros(R, g) : nothing
            ihlsrg!(sr_a, view(zv_h, idx_glob), γ, δ,
                adapt(Array, α[1:nit_new, 1:g]), adapt(Array, β[1:(nit_new + 1), 1:g]);
                resid = resid_a)
            if nit_done == 0
                for k in 1:g
                    σ_prev[idx_glob[k]] = sr_a[k]
                end
            else
                keep = Int[]
                for k in 1:g
                    gi = idx_glob[k]
                    # Retire only after nconfirm consecutive passes: the Ritz residual can dip
                    # spuriously for one chunk (a momentary near-interior-eigenvalue Ritz value, or a
                    # small |s_k|), so a single small bound is not trusted — two consecutive are.
                    if criterion === :certified ?
                       _adaptive_converged(sr_a[k], resid_a[k], rtol, atol) :
                       _cauchy_converged(sr_a[k], σ_prev[gi], rtol, atol)
                        streak[k] += 1
                    else
                        streak[k] = 0
                    end
                    if streak[k] >= nconfirm
                        σ_out[gi] = sr_a[k]
                        nit_at[gi] = nit_new        # depth this point retired at
                    else
                        σ_prev[gi] = sr_a[k]
                        push!(keep, k)
                    end
                end
                if length(keep) < g
                    if isempty(keep)
                        g = 0
                    else
                        # Gather survivors' state to a packed prefix. v/x₀/α/β gather on the last
                        # axis (`[:, keep]`); Qv must gather its first axis, which a fancy index on
                        # a 3-D GPUArray miscompiles on oneAPI — route it through `_gather_rows`.
                        Qv = VectorOfSimilarArrays(_gather_rows(backend, flatview(ihl.Qv), keep))
                        v = VectorOfSimilarVectors(flatview(ihl.v)[:, keep])
                        x₀p = VectorOfSimilarVectors(flatview(ihl.x₀)[:, keep])
                        zvp = ihl.zv[keep]
                        ihl = IHLworkspace{T, get_backend(ihl)}(
                            length(keep), zvp, ihl.P, x₀p, Qv, v)
                        α = α[:, keep]
                        β = β[:, keep]
                        idx_glob = idx_glob[keep]
                        streak = streak[keep]
                        g = length(keep)
                    end
                end
            end
            nit_done = nit_new
        end
        if g > 0
            unconverged = true
            for k in 1:g
                σ_out[idx_glob[k]] = σ_prev[idx_glob[k]]
                nit_at[idx_glob[k]] = nit_done      # never converged → used full budget
            end
        end
        # The batch's σ/nit entries are final in every path above, so deliver them now —
        # a caller can checkpoint hours-scale solves batch by batch.
        on_batch === nothing || on_batch(idxb, σ_out[idxb], nit_at[idxb])
    end
    return Matrix{R}(reshape(σ_out, size(zg))),
    Matrix{Int}(reshape(nit_at, size(zg))), unconverged
end

## END WRAPPER FUNCTIONS ##
