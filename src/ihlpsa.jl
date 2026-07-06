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
               _tiled_trailing_forward, _tiled_trailing_backward,
               _tiled_trailing_forward_eye, _tiled_trailing_backward_eye

# Split out for readability (this file was the largest in src/):
include("ihlpsa_workspace.jl")   # Lanczos KA kernels + IHLworkspace (device-resident state)
include("ihlpsa_trsm.jl")        # device hooks + trsm_strategy routing + lockstep_ihl!

## HOST FUNCTIONS ##

# Largest eigenvalue of the small Lanczos tridiagonal, in the eltype's own precision. Float64 uses
# LAPACK `eigmax`; extended-precision types use GenericLinearAlgebra's `eigen`, which stays reliable
# near a true eigenvalue (σ_min → 0 ⇒ λmax spans a huge range) where `eigmax`'s square-root-free
# algorithm is not.
function _eigmax_tridiag(d::AbstractVector{Float64}, e::AbstractVector{Float64})
    eigmax(SymTridiagonal(d, e))
end
function _eigmax_tridiag(d::AbstractVector{<:AbstractFloat}, e::AbstractVector{<:AbstractFloat})
    maximum(eigen(SymTridiagonal(d, e)).values)
end

# σ_min = 1/√(eigmax) of [(zB−A)(zB−A)ᴴ]⁻¹; the (γ,δ)-pseudospectral value is
# σ_min/(γ+δ|z|) (Frayssé et al.). Work type `R = promote_type(Float64, real(eltype(α)))` floors
# the eigmax precision at Float64, lifting F32 (whose tridiagonal can have an eigenvalue past
# F32's ~1e38 range near a true eigenvalue). isfinite guard: pathological Lanczos can produce
# NaN/Inf entries (e.g. z exactly at an eigenvalue); `eps(real(eltype(zv)))` there keeps
# `log10(sr)` well-defined (resolvent norm → ∞, so the stability radius is ~0).
function ihlsrg!(sr, zv, γ, δ, α, β)
    R = promote_type(Float64, real(eltype(α)))
    Threads.@threads for i in eachindex(zv)
        αi = R.(real.(α[:, i]))
        βi = R.(real.(β[2:(end - 1), i]))
        if all(isfinite, αi) && all(isfinite, βi)
            sr[i] = 1 / ((γ + δ * abs(zv[i])) * sqrt(_eigmax_tridiag(αi, βi)))
        else
            sr[i] = eps(real(eltype(zv)))
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
    # selects the legacy contiguous bands.
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

# Multi-device GPU fan-out shared by the fixed and adaptive drivers. Partitions zg's
# columns across `devs`, resolves a per-device zpd, then spawns one `worker(zgb,
# zpd_dev)` per device. zpd is resolved sequentially BEFORE the parallel fan-out:
# device! is process-global on CUDA/AMDGPU and findmaxbatchihl queries the *current*
# device, so racing it across spawns could read the wrong device's budget. `zip`
# truncates to the shorter side, so surplus devices (ncols < ndev) go untouched.
function _ihlpsa_fanout(backend, zg::AbstractArray{T, 2}, P::AbstractMatrixPencil{T},
        budget_nit, zpd, devs, worker; pbar = nothing) where {T <: Complex}
    m = size(P, 1)
    ismissing(devs) && (devs = devices(backend))
    blocks = _device_column_partition(size(zg, 2), length(devs))
    pbar === nothing ||
        set_description(pbar, "$(length(blocks))/$(length(devs)) device(s), grid points * nit:")
    zpd_devs = map(zip(devs, blocks)) do (dev, zgidx)
        device!(backend, dev)
        zgb_len = length(zgidx) * size(zg, 1)
        ismissing(zpd) ? min(findmaxbatchihl(backend, T, m, budget_nit), zgb_len) : zpd
    end
    results = Vector{Any}(undef, length(blocks))
    @sync for (did, (dev, zgidx)) in enumerate(zip(devs, blocks))
        Threads.@spawn begin
            device!(backend, dev)
            results[did] = worker(zg[:, zgidx], zpd_devs[did])
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
            (zgb, zpd_dev) -> _sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀,
                progress ? pchnl : missing, wgs); pbar)
        result = Matrix{real(T)}(undef, size(zg))
        _scatter_columns!(result, results, blocks)
    else
        # CPU runs the whole grid as one batch: `ThreadsX.foreach` inside `_sdihlpsa` already
        # parallelizes across threads, and the single shared IHLworkspace can't be split without
        # per-batch buffers. (The adaptive driver does batch on CPU — its resident workers run
        # batches sequentially, no shared-state race.)
        progress && set_description(pbar, "CPU device, grid points * nit:")
        result = _sdihlpsa(
            backend, zg, P, γ, δ, length(zg), nit, x₀, progress ? pchnl : missing, wgs)
    end
    progress && close(pchnl)
    return permutedims(result)
end

# Deterministic unit-norm complex start vector for the adaptive driver. The same vector is reused
# across chunks so successive σ are one Lanczos run sampled at increasing depth — a fresh x₀ per
# chunk would compare independent runs and make convergence meaningless. Aliased as `_seeded_x₀`
# in the tests so the test default can't drift from the driver's.
function _adaptive_x₀(::Type{T}, m, seed) where {T <: Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    return x ./ norm(x)
end

_adaptive_default_rtol(::Type{T}) where {T <: Complex} = real(T) == Float32 ? 1.0f-4 : 1e-6
_adaptive_default_atol(::Type{T}) where {T <: Complex} = eps(real(T))

# Combined absolute+relative convergence test on consecutive chunk σ values. eps short-circuit:
# `ihlsrg!` pins a point at/near a true eigenvalue to `eps(real(T))` (already physically converged),
# and a bare |Δ|/|σ| there would divide by ~eps and never retire the point.
@inline function _adaptive_converged(σ_new::R, σ_prev::R, rtol::R, atol::R) where {R <:
                                                                                   Real}
    σ_new <= eps(R) && return true
    return abs(σ_new - σ_prev) <= atol + rtol * abs(σ_new)
end

"""
    ihlpsa(backend, zg, P, nit::Integer; γ=1, δ=0, x₀, progress, zpd, devs, wgs)
    ihlpsa(backend, zg, P; γ=1, δ=0, nit_chunk=2, nit_max=…, rtol, atol, nconfirm,
                           x₀, seed, zpd, devs, wgs, verbose)

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
has converged (relative `rtol` / absolute `atol`, confirmed over `nconfirm`
consecutive chunks) and gathering the survivors so kernels only touch live
points. Per-point work tracks each point's own convergence depth rather than the
slowest point's, capped at `nit_max`. A fixed deterministic start vector (`seed`)
is reused across chunks so the convergence check compares one Lanczos run at
successive depths. The σ match the fixed layout
(`ihlpsa(...) ≈ ihlpsa(..., nit)` at the converged depth). The convergence depth
reached is a diagnostic, not a routine return value: pass `verbose=true` to log
the deepest depth, or call the un-exported `KAPseudospectra._ihlpsa_adaptive`
driver, which returns `(σ::Matrix, nit_grid::Matrix{Int})` — `nit_grid[i]` is the
depth at which grid point `i` retired (`maximum(nit_grid)` is the deepest point).

# Examples
```julia
using KAPseudospectra, KernelAbstractions
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
        x₀::Union{Missing, AbstractVector{T}} = missing,
        seed::Integer = 0x61646170,
        zpd = missing,
        devs = missing,
        wgs = missing,
        verbose = false
) where {T <: Complex}
    _validate_weights(γ, δ)
    R = real(T)
    m = size(P, 1)
    rtol = R(rtol)
    atol = R(atol)
    x₀_fixed = ismissing(x₀) ? _adaptive_x₀(T, m, seed) : x₀
    if KernelAbstractions.isgpu(backend)
        # findmaxbatchihl is sized with nit_max since α/β hold the full budget.
        results, blocks = _ihlpsa_fanout(backend, zg, P, nit_max, zpd, devs,
            (zgb, zpd_dev) -> _sdihlpsa_adaptive(backend, zgb, P, γ, δ, nit_chunk,
                nit_max, rtol, atol, nconfirm, x₀_fixed, zpd_dev, wgs))
        sr = Matrix{real(T)}(undef, size(zg))
        nit_grid = Matrix{Int}(undef, size(zg))
        _scatter_columns!(sr, results, blocks, r -> r[1])
        _scatter_columns!(nit_grid, results, blocks, r -> r[2])
        unconverged = any(r[3] for r in results)
    else
        # Unlike _sdihlpsa (whose batches run under ThreadsX and would race on the
        # shared workspace), the resident workers process batches sequentially,
        # so honoring a user-supplied zpd is safe on CPU too — useful for
        # memory-capping and for exercising the multi-batch path in tests.
        sr, nit_grid, unconverged = _sdihlpsa_adaptive(backend, zg, P,
            γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, x₀_fixed,
            ismissing(zpd) ? length(zg) : zpd, wgs)
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
        γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, x₀, zpd, wgs) where {T <: Complex}
    R = real(T)
    bgarray = get_bgarray(backend)
    zv_h, idxbatches = _grid_batches(zg, zpd)
    gtotal = length(zv_h)
    σ_out = zeros(R, gtotal)
    σ_prev = zeros(R, gtotal)            # indexed by original flat grid index
    nit_at = zeros(Int, gtotal)          # per-point retirement depth (orig flat index)
    unconverged = false
    for idxb in idxbatches
        g = length(idxb)
        ihl = adapt(bgarray, IHLworkspace(P, g, x₀))
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
            ihlsrg!(sr_a, view(zv_h, idx_glob), γ, δ,
                adapt(Array, α[1:nit_new, 1:g]), adapt(Array, β[1:(nit_new + 1), 1:g]))
            if nit_done == 0
                for k in 1:g
                    σ_prev[idx_glob[k]] = sr_a[k]
                end
            else
                keep = Int[]
                for k in 1:g
                    gi = idx_glob[k]
                    # Retire only after nconfirm consecutive passes — one small successive
                    # difference can be a slow-convergence plateau, not real convergence.
                    if _adaptive_converged(sr_a[k], σ_prev[gi], rtol, atol)
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
                        # Gather survivors' state to a packed prefix. Layouts: Qv flat (batch, m, 2);
                        # v/x₀ flat (m, batch); zv (batch). The v/x₀/α/β gathers index the last axis
                        # (`[:, keep]`), correct on every backend. Qv must gather its first axis, and
                        # a first-axis fancy index on a 3-D GPUArray is miscompiled on oneAPI
                        # (silently wrong rows + out-of-bounds) — route it through `_gather_rows`,
                        # which does the row copy with an explicit KA kernel.
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
    end
    return Matrix{R}(reshape(σ_out, size(zg))),
    Matrix{Int}(reshape(nit_at, size(zg))), unconverged
end

## END WRAPPER FUNCTIONS ##
