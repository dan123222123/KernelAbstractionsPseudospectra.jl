using ArraysOfArrays
using ThreadsX
using ProgressBars
using Random

# triangular solves using kernel abstractions
include("KATRSM.jl/KATRSM.jl")
using .KATRSM

## KERNELS ##

# Copy a (g × m) 2D source V into a g-vector-of-m-vectors destination W.
# Note: V here is a 2D SubArray (e.g. view(Qv[2], 1:g, :)), NOT a
# VectorOfSimilarVectors — so we must use size(V) for the dimensions.
# (Earlier versions used length(V) and length(V[1]), which gave g*m and 1
# respectively; the kernel still ran and Lanczos converged anyway, but only
# the first element of each destination vector was being seeded.)
@kernel function _v2v(V, W)
    I = @index(Global, Linear)
    g, m = size(V)
    if I <= g
        for i = 1:m
            W[I][i] = V[I, i]
        end
    end
end

# TODO polish
# each thread will take one grid point and do all of its calculations independently
# v is a length g vector of m-dimensional vectors
# βₙ₊₁v is a g-dimensional vector
# Qₙ₊₁v is a length g vector of m-dimensional vectors
@kernel function _qₙnext(v, βₙ₊₁, Qₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        # norm of v
        vnorm = zero(real(eltype(v[I])))
        for i = 1:m
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        # update qₙ₊₁
        for j = 1:m
            Qₙ₊₁[I, j] = v[I][j] / vnorm
        end
        ## set βₙ₊₁
        βₙ₊₁[I] = vnorm
    end
end

# TODO polish
# kernel for the Lanczos 3-term recurance + qₙnext
# each thread block will handle a grid point
@kernel function _ihl_ttr_qₙnext(βₙ₋₁, Qₙ₋₁, αₙ, Qₙ, v, βₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        # ttr
        for i = 1:m
            v[I][i] -= βₙ₋₁[I] * Qₙ₋₁[I, i]
        end
        αₙ[I] = zero(eltype(v[1]))
        for i = 1:m
            αₙ[I] += conj(Qₙ[I, i]) * v[I][i]
        end
        for i = 1:m
            v[I][i] -= αₙ[I] * Qₙ[I, i]
            Qₙ₋₁[I, i] = Qₙ[I, i] # gvecv
        end
        # qₙnext
        vnorm = zero(real(eltype(v[I])))
        for i = 1:m
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        for j = 1:m
            Qₙ[I, j] = v[I][j] / vnorm
            v[I][j] = Qₙ[I, j] # gvecv no2
        end
        βₙ₊₁[I] = vnorm
    end
end

# Gather rows `keepd` of a (g, m, k) source into a packed (nkeep, m, k) prefix:
# dst[i,:,:] = src[keepd[i],:,:]. Used for the adaptive survivor gather of the Qv
# workspace. We can't use a plain `src[keep,:,:]` fancy index: GPUArrays' first-axis
# fancy indexing of a 3-D array is miscompiled on oneAPI (silently wrong rows + device
# out-of-bounds, even through a 2-D reshape), so we do the row copy with this explicit
# kernel — correct and on-device on every backend, like the other gathers (`[:, keep]`
# on the last axis) which are fine.
@kernel function _qv_gather!(dst, @Const(src), @Const(keepd))
    i, j, l = @index(Global, NTuple)
    @inbounds dst[i, j, l] = src[keepd[i], j, l]
end

## END KERNELS ##

struct IHLworkspace{T,B}
    maxbatch::Int
    zv::AbstractVector{T}
    P::AbstractMatrixPencil{T}
    x₀
    Qv
    v
end

function IHLworkspace(P::AbstractMatrixPencil{T}, maxbatch, x₀=missing) where {T<:Complex}
    m = size(P, 1)
    zv = zeros(T, maxbatch)
    if ismissing(x₀)
        # Random x₀ is rotationally invariant — no basis transform needed.
        x = randn(T, m)
        x₀ = VectorOfSimilarVectors(repeat(x / norm(x), outer=(1, maxbatch)))
    elseif !(x₀ isa VectorOfSimilarVectors)
        # Textbook Lanczos parity: the user provides x₀ in the original-A basis,
        # but ihlpsa applies (zI - T_schur)^{-1}(zI - T_schur)^{-H} in the Schur
        # basis. The transformation v_T = Z' * v_A maps a vector across — so
        # Lanczos on the Schur side starting from Z' * x₀ produces the same Ritz
        # values per iteration as textbook Lanczos on the original side starting
        # from x₀. P.Z is the right Schur transform (= F.Z for both Schur and
        # GeneralizedSchur ctors), or a Diagonal(ones) identity for raw direct
        # construction. Random x₀ skips this branch and the identity case is a
        # no-op multiply on a Diagonal of ones.
        x₀ = P.Z' * x₀
        x₀ = VectorOfSimilarVectors(repeat(x₀ / norm(x₀), outer=(1, maxbatch)))
    end
    Qv = VectorOfSimilarArrays(zeros(T, maxbatch, m, 2))
    # v starts as zeros; lockstep_ihl! reseeds v[1:g] from x₀ at the top of every batch.
    v = VectorOfSimilarVectors(zeros(T, m, maxbatch))
    IHLworkspace{T,get_backend(P)}(maxbatch, zv, P, x₀, Qv, v)
end

function Adapt.adapt_structure(to, ihl::IHLworkspace)
    zv = adapt(to, ihl.zv)
    P = adapt(to, ihl.P)
    x₀ = adapt(to, ihl.x₀)
    Qv = adapt(to, ihl.Qv)
    v = adapt(to, ihl.v)
    IHLworkspace{eltype(zv),get_backend(P)}(ihl.maxbatch, zv, P, x₀, Qv, v)
end

# extend get_backend for IHLworkspace
KernelAbstractions.get_backend(x::IHLworkspace{T,B}) where {T,B} = B

## DEVICE FUNCTIONS ##

# Auto-tune workgroup size for the column-oriented trsm kernels.
# Empirically wgs=32 (one RDNA wavefront / one CUDA warp) wins by 2-3x across
# m∈{64,128,256} vs the prior wgs=256 default. The kernel's per-column
# `@synchronize()` is cheap when the workgroup is one wavefront and gets
# expensive across multiple wavefronts; smaller wgs also lets more workgroups
# co-reside per CU. CDNA users (wavefront=64) may find wgs=64 slightly better;
# override via the wgs kwarg to ihlpsa.
default_wgs(backend, m) = KernelAbstractions.isgpu(backend) ? min(m, 32) : 1

# non-cpu solve step in lockstep_ihl!
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=missing)
    wgs = ismissing(wgs) ? default_wgs(backend, size(P, 1)) : wgs
    g = length(zv)
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc; ndrange=(wgs, g))
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B; ndrange=(wgs, g))
end

# cpu solve step in lockstep_ihl!
# wgs is accepted (and ignored) so this method isn't shadowed by the generic
# trsmIHL when called from lockstep_ihl! with `; wgs`. Without the kwarg,
# Julia's dispatch falls back to the generic (column-oriented) method even
# on CPU, silently bypassing the naive per-workitem CPU kernel.
function trsmIHL(backend::CPU, bV, zv, P::SchurMatrixPencil; wgs=missing)
    g = length(zv)
    _batched_forward_solve_pencil(backend)(bV, conj(zv), P.A', P.B', ndrange=g)
    _batched_backward_solve_pencil(backend)(bV, zv, P.A, P.B, ndrange=g)
end

# Runs Lanczos iterations `start:nit` on the first g grid points of the batch.
# `start == 1` (the default) seeds q₁ from x₀ first; `start > 1` RESUMES a prior
# call: the loop body at iteration n only reads β[n] plus the Qv[1]/Qv[2]/v state
# left in the workspace, so continuing is just running more iterations. Resume
# contract: same `ihl` instance (Qv/v/zv untouched since the previous call), the
# same `α`/`β` arrays (sized for the deepest nit up front), the same g, and the
# previous call ended at iteration start-1.
function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=missing, start::Integer=1)
    backend = get_backend(ihl)
    if start == 1
        # `_qₙnext` writes Qv[2] (= q₁); `_v2v` then copies all m elements of
        # each q₁ into v[1:g] before the trsm loop runs. v[1:g] is fully
        # overwritten there, so no separate seed copy is needed.
        _qₙnext(backend)(view(ihl.x₀, 1:g), view(β, 2, 1:g), view(ihl.Qv[2], 1:g, :), ndrange=g)
        _v2v(backend)(view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), ndrange=g)
    end
    for n = start:nit
        trsmIHL(backend, view(ihl.v, 1:g), view(ihl.zv, 1:g), ihl.P; wgs)
        _ihl_ttr_qₙnext(backend)(view(β, n, 1:g), view(ihl.Qv[1], 1:g, :), view(α, n, 1:g), view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), view(β, n + 1, 1:g), ndrange=g)
    end
    synchronize(backend)
end

# device operations "interface" for kernel abstractions.
#
# Why a package-local interface and not KA's own device functions: KernelAbstractions
# 0.9 does expose `device(::Backend)::Int` / `device!(::Backend, ::Int)` / `ndevices`,
# but those address devices by *ordinal* and KA has no equivalent of `devices`
# (an iterable of concrete device handles), `get_bgarray`, `device_bytes_available`,
# or `device_reclaim` — all of which the multi-GPU fan-out (`_ihlpsa_fanout`) and the
# VRAM-budget batch sizing (`findmaxbatchihl`) need. So we keep a small handle-based
# interface (CPU defaults here; each GPU extension overrides the six methods) rather
# than bolt the missing pieces onto KA's ordinal model. `supports_fp64` below is the
# one method that *does* delegate to KA (see its note).
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

# Whether `backend`'s device can run Float64/ComplexF64 kernels. This IS the KA
# interface — the default just forwards to `KernelAbstractions.supports_float64`
# (`true` for CPU/CUDA/AMDGPU, declared `false` by Metal, which has no `double`
# type). The thin `supports_fp64` wrapper exists for exactly one reason: it gives
# the oneAPI extension a method to override *without* committing type piracy on
# `supports_float64` (oneAPI.jl owns that method for its backend and statically
# declares `false`, even on FP64-capable Arc/Max parts). The oneAPI override does a
# device-accurate Level-Zero query so F64 auto-enables where the hardware really
# supports it. The F64 grid/test/precompile paths consult this and skip unsupported
# devices.
supports_fp64(B) = KernelAbstractions.supports_float64(B)

## END DEVICE FUNCTIONS ##

## HOST FUNCTIONS ##

# separate srg computations
#
# Two robustness measures vs the naive `eigmax(SymTridiagonal(real.(α), real.(β)))`:
#
#  1. Promote α/β to Float64 before forming the SymTridiagonal. F32 Lanczos at
#     a grid point near a true eigenvalue produces a tridiagonal whose largest
#     eigenvalue exceeds F32 dynamic range (~1e38), and LAPACK's `stegr!`
#     errors out with code 11. The α/β arrays are tiny (nit per grid point),
#     so the promotion is essentially free.
#  2. isfinite guard: even in F64, pathological Lanczos can produce NaN/Inf
#     entries (e.g. at z exactly at an eigenvalue). When that happens we set
#     sr[i] = eps(real(eltype(zv))) — the resolvent norm is effectively
#     infinity at this point, so the structured stability radius is zero;
#     using `eps` as a sentinel keeps `log10(sr)` well-defined for plotting.
function ihlsrg!(sr, zv, γ, δ, α, β)
    Threads.@threads for i in eachindex(zv)
        αi = Float64.(real.(α[:, i]))
        βi = Float64.(real.(β[2:end-1, i]))
        if all(isfinite, αi) && all(isfinite, βi)
            sr[i] = (γ + δ * abs(zv[i])) / sqrt(eigmax(SymTridiagonal(αi, βi)))
        else
            sr[i] = eps(real(eltype(zv)))
        end
    end
end

## END HOST FUNCTIONS ##

## WRAPPER FUNCTIONS ##

# Flatten the grid to a point vector and split 1:n into contiguous zpd-sized
# batches. Shared by the fixed (`sdihlpsa`) and adaptive (`_sdihlpsa_adaptive`)
# single-device workers.
function _grid_batches(zg, zpd)
    zv = collect(Iterators.flatten(zg))
    return zv, collect(Iterators.partition(1:length(zv), min(length(zv), zpd)))
end

# single-device batched inverse lanczos pseudospectra
function sdihlpsa(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    γ,
    δ,
    zpd::Integer,
    nit::Integer=max(1, ceil(Integer, log2(size(P, 1)))),
    x₀::Union{Missing,AbstractVector{T},AbstractArrayOfSimilarArrays{T}}=missing,
    pchnl::Union{Missing,Channel}=missing,
    wgs=missing
) where {T<:Complex}
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
            ihlsrg!(view(sr, idxb), view(zv, idxb), γ, δ, adapt(Array, α[:, idxb]), adapt(Array, β[:, idxb]))
        end
    end
    return Matrix{real(T)}(reshape(sr, size(zg)))
end

# computes the largest maxbatch that will fit in the target device memory up to the specified margin of error (moe)
# accounts for: 4*m*m pencil bytes (A, Ac, B, Bc), and per-grid (1 zv + 4*m workspace + (2*nit+1) α/β) bytes
function findmaxbatchihl(backend, T, m, nit; moe=0.1)
    device_reclaim(backend)
    floor(Integer, (device_bytes_available(backend) * (1 - moe) - (sizeof(T) * (4 * m * m + 1))) / (sizeof(T) * (1 + 4 * m + 2 * nit + 1)))
end

# Balanced contiguous partition of 1:ncols into min(ndev, ncols) blocks whose
# sizes differ by at most 1 (the first r blocks get the extra column). Returns
# Vector{UnitRange{Int}} in column order; empty when ncols == 0. Replaces the
# old ceil-based Iterators.partition in the multi-device dispatch, which could
# yield FEWER blocks than devices (e.g. 9 cols / 4 devs → blocks of 3,3,3) and
# BoundsError the device fan-out, which assumed one block per device — besides
# idling devices the balanced split now uses (9/4 → 3,2,2,2 on all four).
function _device_column_partition(ncols::Integer, ndev::Integer)
    ndev ≥ 1 || throw(ArgumentError("ndev must be ≥ 1, got $ndev"))
    ncols ≥ 0 || throw(ArgumentError("ncols must be ≥ 0, got $ncols"))
    nblocks = min(Int(ndev), Int(ncols))
    blocks = Vector{UnitRange{Int}}(undef, nblocks)
    nblocks == 0 && return blocks
    q, r = divrem(Int(ncols), nblocks)
    lo = 1
    for b in 1:nblocks
        hi = lo + q + (b ≤ r) - 1
        blocks[b] = lo:hi
        lo = hi + 1
    end
    return blocks
end

# Multi-device GPU fan-out shared by the fixed and adaptive drivers. Partitions
# zg's columns across `devs`, resolves a per-device zpd (device-memory budget
# sized by `budget_nit`), then spawns `worker(zgb, zpd_dev)` on each device with
# that device active; returns the per-device results in column order. zpd is
# resolved sequentially BEFORE the parallel fan-out because device! is
# process-global on CUDA/AMDGPU and findmaxbatchihl queries the *current* device's
# free memory — racing it across spawns can read the wrong device's budget. `zip`
# pairs blocks with devices and truncates to the shorter side, so when ncols < ndev
# the surplus devices are simply never touched (devs need only be iterable; only
# the blocks are indexed). Pass `pbar` to label a progress bar with the block count.
function _ihlpsa_fanout(backend, zg::AbstractArray{T,2}, P::AbstractMatrixPencil{T},
    budget_nit, zpd, devs, worker; pbar=nothing) where {T<:Complex}
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
    return results
end

# Fixed-nit engine: multi-device batched inverse-Lanczos at a caller-given depth.
# The public `ihlpsa(..., nit::Integer, ...)` method forwards here.
function _ihlpsa_fixed(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    nit::Integer,
    γ=1,
    δ=0;
    x₀::Union{Missing,AbstractVector{T}}=missing,
    progress=false,
    zpd=missing,
    devs=missing,
    wgs=missing
) where {T<:Complex}
    # progress bar + consumer task only when caller asks for one. Otherwise the
    # consumer just blocks forever on an empty channel and the spawn leaks at
    # precompile time.
    pbar = progress ? ProgressBar(total=nit * length(zg), printing_delay=0.001) : nothing
    pchnl = progress ? Channel() : nothing
    if progress
        Threads.@spawn begin
            for delta in pchnl
                ProgressBars.update(pbar, delta)
            end
        end
    end
    if KernelAbstractions.isgpu(backend)
        results = _ihlpsa_fanout(backend, zg, P, nit, zpd, devs,
            (zgb, zpd_dev) -> sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀,
                progress ? pchnl : missing, wgs); pbar)
        result = (hcat(results...))::Matrix{real(T)}
    else
        # CPU runs the whole grid as one batch (zpd = length(zg)). It can't split
        # the grid the way the GPU path does: the single preallocated IHLworkspace
        # (Qv/v/zv) is shared, and `ThreadsX.foreach` inside `sdihlpsa` already
        # parallelizes that one batch across threads — a second level of batching
        # would need a per-batch (or per-thread) workspace to avoid racing on those
        # buffers. That's the GPU's memory-budget concern, not the CPU's: on CPU the
        # workspace is cheap and threading already saturates the cores, so batching
        # would add allocation churn for no throughput. If a grid is too large to
        # hold in host RAM at once, use the GPU path. (The adaptive driver DOES batch
        # on CPU — its resident workers run batches sequentially, no shared-state race.)
        progress && set_description(pbar, "CPU device, grid points * nit:")
        result = sdihlpsa(backend, zg, P, γ, δ, length(zg), nit, x₀, progress ? pchnl : missing, wgs)
    end
    progress && close(pchnl)
    return permutedims(result)
end

# Deterministic unit-norm complex start vector for the adaptive driver. The
# SAME vector is reused for every chunk so that eigmax(T_k) vs eigmax(T_{k+chunk})
# compares one Lanczos run at two depths (a principled convergence monitor). If
# instead x₀ were `missing`, the IHLworkspace constructor would draw a FRESH randn
# per chunk and consecutive chunks would be independent runs, making
# the convergence test meaningless. The test suite reuses this exact routine (as
# `_seeded_x₀`, aliased in test_consistency.jl) so the seeded x₀ the tests pass and
# the driver's own default x₀ can never drift apart.
function _adaptive_x₀(::Type{T}, m, seed) where {T<:Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    return x ./ norm(x)
end

# eltype-branched default tolerances. The F32 relative floor ~1e-4 matches the
# SVD-oracle tolerance in test_consistency.jl; below it, F32 Lanczos roundoff in
# σ_min dominates and the criterion would never trip.
_adaptive_default_rtol(::Type{T}) where {T<:Complex} = real(T) == Float32 ? 1.0f-4 : 1e-6
_adaptive_default_atol(::Type{T}) where {T<:Complex} = eps(real(T))

# Per-grid-point convergence test between two consecutive chunk results. σ are
# the resolvent-derived values from `ihlsrg!` (= (γ + δ|z|)·σ_min). Combined
# absolute + relative tolerance, with an explicit eps-sentinel short-circuit:
# `ihlsrg!` pins a point at/near a true eigenvalue to `eps(real(T))` (σ_min ≈ 0,
# resolvent norm → ∞), which is already physically converged — no further
# iteration moves it — and a bare |Δ|/|σ| there would divide by ~eps and never
# retire the point. The atol floor likewise keeps the test well-defined as σ → 0.
@inline function _adaptive_converged(σ_new::R, σ_prev::R, rtol::R, atol::R) where {R<:Real}
    σ_new <= eps(R) && return true
    return abs(σ_new - σ_prev) <= atol + rtol * abs(σ_new)
end

"""
    ihlpsa(backend, zg, P, nit::Integer; γ=1, δ=0, x₀, progress, zpd, devs, wgs)
    ihlpsa(backend, zg, P; γ=1, δ=0, nit_chunk=2, nit_max=…, rtol, atol, nconfirm,
                           x₀, seed, zpd, devs, wgs, verbose)

Batched inverse-Lanczos pseudospectra of the matrix pencil `P` over the complex
grid `zg`, fanned out across all devices of `backend` (pass `devs` to restrict).
Each grid point's value is `(γ + δ|z|)·σ_min(zB − A)`. Both forms return the
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
it, or call the un-exported `KAPseudospectra._ihlpsa_adaptive` driver, which
returns `(σ::Matrix, nit_used::Integer)`.

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
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    nit::Integer;
    γ=1,
    δ=0,
    x₀::Union{Missing,AbstractVector{T}}=missing,
    progress=false,
    zpd=missing,
    devs=missing,
    wgs=missing
) where {T<:Complex}
    return _ihlpsa_fixed(backend, zg, P, nit, γ, δ; x₀, progress, zpd, devs, wgs)
end

# Adaptive form — omit `nit`. Always returns just `srg::Matrix`, mirroring the
# fixed-nit form: the convergence depth reached is a diagnostic, not a routine
# return value. To read the depth, pass `verbose=true` (it is logged) or call the
# un-exported `_ihlpsa_adaptive` driver below, which returns `(srg, nit_used)`.
function ihlpsa(backend, zg::AbstractArray{T,2}, P::AbstractMatrixPencil{T};
    kwargs...) where {T<:Complex}
    return first(_ihlpsa_adaptive(backend, zg, P; kwargs...))
end

# Internal adaptive driver — multi-device per-point adaptive inverse Lanczos.
# Returns `(srg::Matrix, nit_used::Integer)`; the public `ihlpsa(...; …)` wraps
# this and drops `nit_used`. NOT exported: the depth is exposed to power users via
# `verbose=true` logging, and to the test suite by calling this directly.
#
# Fans grid columns out across devices via `_ihlpsa_fanout` (shared with the fixed
# engine), but each device runs its own per-point adaptive loop
# (`_sdihlpsa_adaptive`) and stops at its OWN converged depth — devices over easy
# regions retire early instead of lockstepping to the global worst point.
# `nit_used` is the deepest depth across devices.
function _ihlpsa_adaptive(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T};
    γ=1,
    δ=0,
    nit_chunk::Integer=2,
    nit_max::Integer=8 * max(1, ceil(Integer, log2(size(P, 1)))),
    rtol::Real=_adaptive_default_rtol(T),
    atol::Real=_adaptive_default_atol(T),
    nconfirm::Integer=2,
    x₀::Union{Missing,AbstractVector{T}}=missing,
    seed::Integer=0x61646170,
    zpd=missing,
    devs=missing,
    wgs=missing,
    verbose=false
) where {T<:Complex}
    R = real(T)
    m = size(P, 1)
    rtol = R(rtol)
    atol = R(atol)
    x₀_fixed = ismissing(x₀) ? _adaptive_x₀(T, m, seed) : x₀
    if KernelAbstractions.isgpu(backend)
        # findmaxbatchihl is sized with nit_max since α/β hold the full budget.
        results = _ihlpsa_fanout(backend, zg, P, nit_max, zpd, devs,
            (zgb, zpd_dev) -> _sdihlpsa_adaptive(backend, zgb, P, γ, δ, nit_chunk,
                nit_max, rtol, atol, nconfirm, x₀_fixed, zpd_dev, wgs))
        sr = (hcat((r[1] for r in results)...))::Matrix{real(T)}
        nit_used = maximum(r[2] for r in results)
        unconverged = any(r[3] for r in results)
    else
        # Unlike sdihlpsa (whose batches run under ThreadsX and would race on the
        # shared workspace), the resident workers process batches sequentially,
        # so honoring a user-supplied zpd is safe on CPU too — useful for
        # memory-capping and for exercising the multi-batch path in tests.
        sr, nit_used, unconverged = _sdihlpsa_adaptive(backend, zg, P,
            γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, x₀_fixed,
            ismissing(zpd) ? length(zg) : zpd, wgs)
    end
    unconverged &&
        @warn "ihlpsa adaptive hit nit_max=$nit_max with unconverged point(s) (rtol=$rtol)"
    verbose && @info "ihlpsa adaptive done" nit = nit_used
    return permutedims(sr), nit_used
end

# Gather rows `keep` (a host Int vector) from a (g, m, k) device array into a fresh
# packed (nkeep, m, k) array, via the `_qv_gather!` kernel (see its note — a direct
# `A[keep,:,:]` is miscompiled on oneAPI). `keep` is moved to the device once for the
# kernel to index.
function _gather_rows(backend, A, keep)
    g, m, k = size(A)
    dst = similar(A, length(keep), m, k)
    keepd = adapt(get_bgarray(backend), keep)
    _qv_gather!(backend)(dst, A, keepd; ndrange=(length(keep), m, k))
    return dst
end

# Single-device adaptive worker — per-point retirement with resident state. Each
# batch starts with its own workspace and full-budget α/β; after each chunk the
# converged points retire and the survivors' per-point state — workspace rows
# (Qv batch-major, v/x₀ batch-minor, zv) plus α/β columns — is GATHERED into a
# packed prefix via array indexing (GPUArrays fancy indexing; no custom
# kernels; the 3-D Qv gather goes through `_gather_rows`, see its note).
# Subsequent chunks therefore run kernels over live points only,
# continuing each point's own Lanczos recurrence uninterrupted: per-point work
# ≈ its converged depth + one confirm chunk, instead of the batch-worst depth.
# The gather traffic is O(m·survivors) per chunk — negligible against the
# O(nit_chunk·m²·survivors) solve work it avoids re-running.
# `idx_glob` maps packed position → original flat grid index throughout.
# Returns (sr_matrix, nit_deepest, unconverged::Bool).
function _sdihlpsa_adaptive(backend, zg::AbstractArray{T,2}, P::AbstractMatrixPencil{T},
    γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, x₀, zpd, wgs) where {T<:Complex}
    R = real(T)
    bgarray = get_bgarray(backend)
    zv_h, idxbatches = _grid_batches(zg, zpd)
    gtotal = length(zv_h)
    σ_out = zeros(R, gtotal)
    σ_prev = zeros(R, gtotal)            # indexed by ORIGINAL flat grid index
    nit_deepest = 0
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
            lockstep_ihl!(view(α, :, 1:g), view(β, :, 1:g), ihl, nit_new, g; wgs, start=nit_done + 1)
            sr_a = zeros(R, g)
            ihlsrg!(sr_a, view(zv_h, idx_glob), γ, δ,
                adapt(Array, α[1:nit_new, 1:g]), adapt(Array, β[1:nit_new+1, 1:g]))
            if nit_done == 0
                for k in 1:g
                    σ_prev[idx_glob[k]] = sr_a[k]
                end
            else
                keep = Int[]
                for k in 1:g
                    gi = idx_glob[k]
                    # Retire only after nconfirm consecutive passing checkpoints
                    # — one small successive difference can be a slow-convergence
                    # plateau rather than convergence.
                    if _adaptive_converged(sr_a[k], σ_prev[gi], rtol, atol)
                        streak[k] += 1
                    else
                        streak[k] = 0
                    end
                    if streak[k] >= nconfirm
                        σ_out[gi] = sr_a[k]
                    else
                        σ_prev[gi] = sr_a[k]
                        push!(keep, k)
                    end
                end
                if length(keep) < g
                    if isempty(keep)
                        g = 0
                    else
                        # Gather survivors' state to a packed prefix. Layouts:
                        # Qv flat (batch, m, 2); v/x₀ flat (m, batch); zv (batch).
                        # The v/x₀/α/β gathers index the LAST axis (`[:, keep]`),
                        # which is correct on every backend. Qv must gather its
                        # FIRST axis — and a first-axis fancy index on a 3-D
                        # GPUArray is miscompiled on oneAPI (silently returns wrong
                        # rows + out-of-bounds), so route it through `_gather_rows`,
                        # which does the row copy with an explicit KA kernel.
                        Qv = VectorOfSimilarArrays(_gather_rows(backend, flatview(ihl.Qv), keep))
                        v = VectorOfSimilarVectors(flatview(ihl.v)[:, keep])
                        x₀p = VectorOfSimilarVectors(flatview(ihl.x₀)[:, keep])
                        zvp = ihl.zv[keep]
                        ihl = IHLworkspace{T,get_backend(ihl)}(length(keep), zvp, ihl.P, x₀p, Qv, v)
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
            end
        end
        nit_deepest = max(nit_deepest, nit_done)
    end
    return Matrix{R}(reshape(σ_out, size(zg))), nit_deepest, unconverged
end

## END WRAPPER FUNCTIONS ##