using ArraysOfArrays
using ThreadsX
using ProgressBars
using Random

# triangular solves using kernel abstractions
include("KATRSM.jl/KATRSM.jl")
using .KATRSM

## KERNELS ##

# Copy a (g × m) 2D source V into a g-vector-of-m-vectors destination W.
# V is a 2D SubArray (e.g. view(Qv[2], 1:g, :)), not a VectorOfSimilarVectors,
# so the dimensions come from size(V).
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
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = src[keepd[i], j, k]
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

# Warp width assumed by the register-warp / tiled solves (one shuffle domain): the
# subgroup/warp size, capped at the matrix size. 32 on CUDA / Metal / AMDGPU (≤ wavefront)
# / Intel with the SIMD32 pin (see `set_intel_force_simd32!`). Overridable per backend.
warp_width(backend) = 32
# Workgroup size for the trsm kernels: the warp width on GPU, 1 on CPU. Override via the
# `wgs` kwarg to ihlpsa.
default_wgs(backend, m) = KernelAbstractions.isgpu(backend) ? min(m, warp_width(backend)) : 1

# Whether the register-warp / tiled solves (which broadcast pivots with warp shuffles) are
# correct under the "auto" strategy on this backend. True for CUDA / AMDGPU / Metal: fixed
# warp/wavefront/SIMD width + a hardware shuffle. On oneAPI it requires BOTH the
# KernelIntrinsics oneAPI shuffle backend AND a pinned SIMD width, so the oneAPI extension
# overrides this; without them, `auto` stays on the shuffle-free `column` solve — correct,
# just not the fast path. (Explicit KAPSEUDO_TRSM=warp/tiled is opt-in and not gated.)
# `wide` is true for non-IEEE element types (MultiFloats / BigFloat); the oneAPI override
# keeps those on `column` regardless (their warp/tiled SPIR-V codegen is not yet functional).
warp_trsm_safe(backend, wide) = true

# non-cpu solve step in lockstep_ihl!
#
# Two GPU solve kernels are available:
#   * warp-register (default): one warp/grid-point with the RHS held in registers and
#     pivots broadcast by warp shuffle — no block barriers, no global round-trips on b.
#     Specialized on R = cld(m, ws) (a Val), so one compile per matrix size m (m is fixed
#     per ihlpsa run; the survivor count g stays the dynamic ndrange).
#   * column-oriented (fallback): the original `@synchronize()`-per-column kernel. Select
#     it with KAPSEUDO_WARP_TRSM=0.
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=missing)
    wgs = ismissing(wgs) ? default_wgs(backend, size(P, 1)) : wgs
    strat = trsm_strategy()
    # Element-type + size routing for the GPU inner solve.
    #  * The warp solve is `@generated` on R = ⌈m/32⌉: excellent at small m, but its compile
    #    time and register pressure blow up with R (≈26 s compile at R=16, register-bound
    #    occupancy collapse for wide elements). So large problems avoid warp.
    #  * The tiled solve's trailing-update `@localmem` tiles must fit GPU shared memory (48 KB).
    #    IEEE floats always fit (≤2·32²·sizeof ≤ 32 KB). Wide (non-IEEE: MultiFloats/BigFloat)
    #    elements only fit when B = I, where the sB-free trailing kernels use a single tile
    #    (e.g. 32 KB for Complex{Float64x2}); a wide generalized pencil (B ≠ I) needs two tiles
    #    (64 KB, overflow) and falls back to the shuffle-free column solve. Wide warp/tiled use
    #    the per-limb `_trsm_shfl` override (KAPseudospectraMultiFloatsExt). Threshold = `trsm_crossover()`.
    wide = !(real(eltype(P.A)) <: Base.IEEEFloat)
    big = size(P, 1) >= trsm_crossover()
    tiled_ok = !wide || P.b_is_identity            # tiled's tiles fit shared memory
    if strat == "column"
        _column_trsm!(backend, bV, zv, P, wgs)
    elseif strat == "warp"
        _warp_trsm!(backend, bV, zv, P, wgs)
    elseif strat == "tiled"
        tiled_ok ? _tiled_trsm!(backend, bV, zv, P, wgs) :
                   (big ? _column_trsm!(backend, bV, zv, P, wgs) : _warp_trsm!(backend, bV, zv, P, wgs))
    elseif !warp_trsm_safe(backend, wide)           # "auto" where shuffle isn't safe (stock oneAPI) / unsupported (oneAPI MultiFloats) → column
        _column_trsm!(backend, bV, zv, P, wgs)
    elseif big && tiled_ok                          # "auto", large m, tiles fit (IEEE, or wide B=I)
        _tiled_trsm!(backend, bV, zv, P, wgs)
    elseif big                                      # "auto", large m, wide B≠I → no tiled
        _column_trsm!(backend, bV, zv, P, wgs)
    else                                            # "auto", small m
        _warp_trsm!(backend, bV, zv, P, wgs)
    end
end

# Warp-register solve. The generic implementation uses the portable KA + KernelIntrinsics
# kernels; the CUDA extension can override `_warp_trsm!` with hand-rolled `@cuda` kernels
# (opt-in via KAPSEUDO_CUDA_NATIVE=1) that avoid a KA+KI codegen regression at R=16.
# `_warp_trsm_ka!` is the portable default and is kept callable for that override to fall back to.
_warp_trsm!(backend, bV, zv, P, wgs) = _warp_trsm_ka!(backend, bV, zv, P, wgs)

# Tiled / blocked solve: right-looking panel sweep with shared-memory A,B-tile reuse across
# the grid-point batch (see src/KATRSM.jl/trsm_tiled_kernels.jl). `z` is conjugated once for
# the forward (lower-tri Ac,Bc) sweep; the backward sweep uses (A,B,z) directly. `gt` (grid
# points per trailing tile = the A,B reuse factor) is tunable via KAPSEUDO_TRSM_GT.
function _tiled_trsm!(backend, bV, zv, P, wgs)
    m = size(P, 1)
    g = length(zv)
    gt = parse(Int, get(ENV, "KAPSEUDO_TRSM_GT", "32"))
    nblk = cld(m, 32)
    zc = conj(zv)
    eye = P.b_is_identity   # B = I ⇒ sB-free trailing kernels (half the shared memory)
    # forward (lower-triangular), panels ascending
    for k in 1:nblk
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        @views _tiled_panel_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen; ndrange=(32, g))
        rbase = koff + plen
        ntrail = m - rbase
        if ntrail > 0
            rtiles = cld(ntrail, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_forward_eye(backend, 32)(bV, P.Ac, koff, plen, rbase, m, gt, rtiles; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen, rbase, m, gt, rtiles; ndrange=32 * rtiles * ggrid)
            end
        end
    end
    # backward (upper-triangular), panels descending
    for k in nblk:-1:1
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        @views _tiled_panel_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen; ndrange=(32, g))
        if koff > 0
            rtiles = cld(koff, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_backward_eye(backend, 32)(bV, P.A, koff, plen, gt, rtiles; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen, gt, rtiles; ndrange=32 * rtiles * ggrid)
            end
        end
    end
end
function _warp_trsm_ka!(backend, bV, zv, P, wgs)
    g = length(zv)
    R = cld(size(P, 1), wgs)
    @views _batched_warp_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc, Val(R); ndrange=(wgs, g))
    @views _batched_warp_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B, Val(R); ndrange=(wgs, g))
end
function _column_trsm!(backend, bV, zv, P, wgs)
    g = length(zv)
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc; ndrange=(wgs, g))
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B; ndrange=(wgs, g))
end

# cpu solve step in lockstep_ihl!. `wgs` is accepted and ignored so this CPU method
# isn't shadowed by the generic (column-oriented) trsmIHL when called with `; wgs`.
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

# Device-operations interface: CPU defaults here; each GPU extension overrides these
# six methods (array type, device handle access, free-memory query, reclaim).
# `supports_fp64` below delegates to KA.
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

# Whether `backend`'s device can run Float64/ComplexF64 kernels. Default defers to
# `KernelAbstractions.supports_float64`; overridable so the oneAPI extension can
# substitute a device-accurate FP64 query (see its note). F64 paths skip unsupported devices.
supports_fp64(B) = KernelAbstractions.supports_float64(B)

## END DEVICE FUNCTIONS ##

## HOST FUNCTIONS ##

# Largest eigenvalue of the small Lanczos tridiagonal `SymTridiagonal(d, e)`, computed in
# the eltype's own precision. Float64 uses the LAPACK `eigmax` (fast, well-tested);
# extended-precision element types (MultiFloats / BigFloat) use GenericLinearAlgebra's
# `eigen` and take the top value. `ihlsrg!` only ever calls this with `d`/`e` of the work
# type `R` it selected, so the result follows the input precision.
#
# `eigen` rather than `eigmax`/`eigvals` for the generic path: near a true eigenvalue
# σ_min → 0, so λmax = 1/σ_min² is large and the tridiagonal spans a wide dynamic range.
# GenericLinearAlgebra's `eigen` (plain QL) resolves the extreme eigenvalue to ~machine-eps
# on such matrices; its square-root-free `eigvals` (which `eigmax` calls) is less reliable
# there, so we go through `eigen`. Using it means extended precision needs
# `GenericLinearAlgebra` loaded — the same generic-linear-algebra stack the dense Schur
# factorization already requires. The tridiagonal is tiny (nit per grid point), so the
# eigenvectors `eigen` also returns cost nothing here.
_eigmax_tridiag(d::AbstractVector{Float64}, e::AbstractVector{Float64}) = eigmax(SymTridiagonal(d, e))
_eigmax_tridiag(d::AbstractVector{<:AbstractFloat}, e::AbstractVector{<:AbstractFloat}) =
    maximum(eigen(SymTridiagonal(d, e)).values)

# separate srg computations
#
# Two notes on the σ extraction:
#
#  1. Work type `R = promote_type(Float64, real(eltype(α)))` — a Float64 *floor* on the
#     eigmax precision. It is a no-op (`R == real(eltype(α))`) for Float64 and for every
#     extended-precision type (MultiFloats/BigFloat), so those follow the input precision
#     and the returned σ is as accurate as the resolvent solves that produced α/β. The
#     floor only lifts the sub-Float64 types: an F32 tridiagonal near a true eigenvalue has
#     a largest eigenvalue past F32's ~1e38 range, so its eigmax is taken in Float64.
#     `_eigmax_tridiag` then dispatches LAPACK `eigmax` for Float64 and `eigen` otherwise;
#     α/β are tiny (nit per grid point) so the cost is negligible either way.
#  2. isfinite guard: even in F64, pathological Lanczos can produce NaN/Inf
#     entries (e.g. at z exactly at an eigenvalue). When that happens we set
#     sr[i] = eps(real(eltype(zv))) — the resolvent norm is effectively
#     infinity at this point, so the structured stability radius is zero;
#     using `eps` as a sentinel keeps `log10(sr)` well-defined for plotting.
function ihlsrg!(sr, zv, γ, δ, α, β)
    R = promote_type(Float64, real(eltype(α)))
    Threads.@threads for i in eachindex(zv)
        αi = R.(real.(α[:, i]))
        βi = R.(real.(β[2:end-1, i]))
        if all(isfinite, αi) && all(isfinite, βi)
            # σ_min = 1/√(eigmax) of [(zB−A)(zB−A)ᴴ]⁻¹; the (γ,δ)-pseudospectral
            # value is σ_min/(γ+δ|z|) (Frayssé et al.) = 1/((γ+δ|z|)·√eigmax).
            # `_eigmax_tridiag` follows the input eltype's precision (LAPACK for
            # Float64, GenericLinearAlgebra `eigen` for extended-precision types).
            sr[i] = 1 / ((γ + δ * abs(zv[i])) * sqrt(_eigmax_tridiag(αi, βi)))
        else
            sr[i] = eps(real(eltype(zv)))
        end
    end
end

## END HOST FUNCTIONS ##

## WRAPPER FUNCTIONS ##

# Flatten the grid to a point vector and split 1:n into contiguous zpd-sized
# batches. Shared by the fixed (`_sdihlpsa`) and adaptive (`_sdihlpsa_adaptive`)
# single-device workers.
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

# Partition 1:ncols into min(ndev, ncols) balanced blocks (sizes differ by ≤ 1),
# one per device; empty when ncols == 0.
function _device_column_partition(ncols::Integer, ndev::Integer)
    ndev ≥ 1 || throw(ArgumentError("ndev must be ≥ 1, got $ndev"))
    ncols ≥ 0 || throw(ArgumentError("ncols must be ≥ 0, got $ncols"))
    nblocks = min(Int(ndev), Int(ncols))
    nblocks == 0 && return StepRange{Int,Int}[]
    # Round-robin (strided) assignment: device b takes columns b, b+nblocks, …, so
    # clustered hard/deep-iteration regions spread across devices (load-balances the
    # adaptive driver, whose per-point work varies). Results are scattered back to
    # original columns afterward, so output order is preserved. KAPSEUDO_STRIDED=0
    # selects the legacy contiguous bands.
    if get(ENV, "KAPSEUDO_STRIDED", "1") == "0"
        blocks = Vector{StepRange{Int,Int}}(undef, nblocks)
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
    # Return `blocks` too: with strided partitioning the per-device results are no
    # longer in column order, so callers scatter `results[d]` back to `blocks[d]`.
    return results, blocks
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
    _validate_weights(γ, δ)
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
        results, blocks = _ihlpsa_fanout(backend, zg, P, nit, zpd, devs,
            (zgb, zpd_dev) -> _sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀,
                progress ? pchnl : missing, wgs); pbar)
        # Scatter each device's columns back to their original grid positions
        # (strided partition ⇒ device results are not in column order).
        result = Matrix{real(T)}(undef, size(zg))
        for (r, blk) in zip(results, blocks)
            @inbounds result[:, blk] = r
        end
    else
        # CPU runs the whole grid as one batch: `ThreadsX.foreach` inside `_sdihlpsa`
        # already parallelizes across threads, and the single shared IHLworkspace
        # can't be split without per-batch buffers. (The adaptive driver does batch on
        # CPU — its resident workers run batches sequentially, no shared-state race.)
        progress && set_description(pbar, "CPU device, grid points * nit:")
        result = _sdihlpsa(backend, zg, P, γ, δ, length(zg), nit, x₀, progress ? pchnl : missing, wgs)
    end
    progress && close(pchnl)
    return permutedims(result)
end

# Deterministic unit-norm complex start vector for the adaptive driver. The SAME
# vector is reused across chunks so successive σ are one Lanczos run sampled at
# increasing depth — a fresh x₀ per chunk would compare independent runs and make
# the convergence test meaningless. Aliased as `_seeded_x₀` in the tests so the
# tests' x₀ and the driver's default can't drift apart.
function _adaptive_x₀(::Type{T}, m, seed) where {T<:Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    return x ./ norm(x)
end

_adaptive_default_rtol(::Type{T}) where {T<:Complex} = real(T) == Float32 ? 1.0f-4 : 1e-6
_adaptive_default_atol(::Type{T}) where {T<:Complex} = eps(real(T))

# Per-grid-point convergence test between two consecutive chunk results. σ are
# the resolvent-derived values from `ihlsrg!` (= σ_min/(γ + δ|z|)). Combined
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
# un-exported `_ihlpsa_adaptive` driver below, which returns `(srg, nit_grid)`.
function ihlpsa(backend, zg::AbstractArray{T,2}, P::AbstractMatrixPencil{T};
    kwargs...) where {T<:Complex}
    return first(_ihlpsa_adaptive(backend, zg, P; kwargs...))
end

# Internal adaptive driver — multi-device per-point adaptive inverse Lanczos.
# Returns `(srg::Matrix, nit_grid::Matrix{Int})`; the public `ihlpsa(...; …)` wraps
# this and drops `nit_grid`. NOT exported: the per-point depth is exposed to power
# users via `verbose=true` logging, and to the test suite by calling this directly.
#
# Fans grid columns out across devices via `_ihlpsa_fanout` (shared with the fixed
# engine), but each device runs its own per-point adaptive loop
# (`_sdihlpsa_adaptive`) and stops at its OWN converged depth — devices over easy
# regions retire early instead of lockstepping to the global worst point.
# `nit_grid[i]` is point i's retirement depth; `maximum(nit_grid)` is the deepest.
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
        # Scatter each device's columns back to their original grid positions
        # (strided partition ⇒ device results are not in column order).
        sr = Matrix{real(T)}(undef, size(zg))
        nit_grid = Matrix{Int}(undef, size(zg))
        for (r, blk) in zip(results, blocks)
            @inbounds sr[:, blk] = r[1]
            @inbounds nit_grid[:, blk] = r[2]
        end
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

# Single-device adaptive worker — per-point retirement with resident state. After
# each chunk the converged points retire and the survivors' per-point state
# (workspace rows + α/β columns) is gathered into a packed prefix via array indexing
# (the 3-D Qv gather goes through `_gather_rows`, see its note), so subsequent chunks
# run kernels over live points only. `idx_glob` maps packed position → original flat
# grid index. Returns (sr_matrix, nit_grid, unconverged::Bool); nit_grid[i] is the
# depth at which grid point i retired (or nit_max if it never did).
function _sdihlpsa_adaptive(backend, zg::AbstractArray{T,2}, P::AbstractMatrixPencil{T},
    γ, δ, nit_chunk, nit_max, rtol, atol, nconfirm, x₀, zpd, wgs) where {T<:Complex}
    R = real(T)
    bgarray = get_bgarray(backend)
    zv_h, idxbatches = _grid_batches(zg, zpd)
    gtotal = length(zv_h)
    σ_out = zeros(R, gtotal)
    σ_prev = zeros(R, gtotal)            # indexed by ORIGINAL flat grid index
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
                nit_at[idx_glob[k]] = nit_done      # never converged → used full budget
            end
        end
    end
    return Matrix{R}(reshape(σ_out, size(zg))),
        Matrix{Int}(reshape(nit_at, size(zg))), unconverged
end

## END WRAPPER FUNCTIONS ##
