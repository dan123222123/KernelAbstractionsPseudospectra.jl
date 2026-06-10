using ArraysOfArrays
using ThreadsX
using ProgressBars

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
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bV, conj(zv), P.Ac, P.Bc)
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs, (wgs, g))(bV, zv, P.A, P.B)
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

function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=missing)
    backend = get_backend(ihl)
    # `_qₙnext` writes Qv[2] (= q₁); `_v2v` then copies all m elements of
    # each q₁ into v[1:g] before the trsm loop runs. v[1:g] is fully
    # overwritten there, so no separate seed copy is needed.
    _qₙnext(backend)(view(ihl.x₀, 1:g), view(β, 2, 1:g), view(ihl.Qv[2], 1:g, :), ndrange=g)
    _v2v(backend)(view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), ndrange=g)
    for n = 1:nit
        trsmIHL(backend, view(ihl.v, 1:g), view(ihl.zv, 1:g), ihl.P; wgs)
        _ihl_ttr_qₙnext(backend)(view(β, n, 1:g), view(ihl.Qv[1], 1:g, :), view(α, n, 1:g), view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), view(β, n + 1, 1:g), ndrange=g)
    end
    synchronize(backend)
end

# device operations "interface" for kernel abstractions
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

# Whether `backend`'s device can run Float64/ComplexF64 kernels. The default
# defers to KernelAbstractions' own capability flag: `supports_float64` is `true`
# for CPU/CUDA/AMDGPU and declared `false` by Metal (Metal Shading Language has no
# `double` type). The oneAPI extension overrides this with a device-accurate query
# (oneAPI's KA flag is a conservative static `false`), so F64 auto-enables on
# FP64-capable Intel GPUs (Arc/Max) and stays off on FP64-less iGPUs. The F64
# grid/test/precompile paths consult this and skip unsupported devices.
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

# single-device batched inverse lanczos pseudospectra
function sdihlpsa(
    backend;
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
    zv = collect(Iterators.flatten(zg))
    gtotal = length(zv)
    sr = zeros(real(T), length(zv))
    idxbatches = Vector(collect(Iterators.partition(1:gtotal, min(gtotal, zpd))))
    batches = idxbatches
    dzv = adapt(bgarray, zv)
    α = adapt(bgarray, zeros(T, nit, gtotal))
    β = adapt(bgarray, zeros(T, nit + 1, gtotal))
    ihl = adapt(bgarray, IHLworkspace(P, zpd, x₀))
    _foreach = !KernelAbstractions.isgpu(backend) ? ThreadsX.foreach : Base.foreach
    @sync _foreach(batches) do idxb
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
    sdargs = (; zg, P, γ, δ, zpd, nit, x₀, pchnl, wgs)
    sdihlpsa(backend; sdargs...)
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
# BoundsError the device loops that index zgidxbatches[1:ndev] — besides idling
# devices the balanced split now uses (9/4 → 3,2,2,2 on all four).
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

# multi-device, general purpose batched inverse lanczos pseudospectra caller
function ihlpsa(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    nit::Integer=max(1, ceil(Integer, log2(size(P, 1)))),
    γ=1,
    δ=0;
    x₀::Union{Missing,AbstractVector{T}}=missing,
    progress=false,
    zpd=missing,
    devs=missing,
    wgs=missing
) where {T<:Complex}
    m = size(P.A, 1)
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
        if ismissing(devs)
            devs = devices(backend)
        end
        zgidxbatches = _device_column_partition(size(zg, 2), length(devs))
        progress && set_description(pbar, "$(length(zgidxbatches))/$(length(devs)) device(s), grid points * nit:")
        # Resolve per-device zpd sequentially BEFORE the parallel fan-out: device!
        # is process-global on AMDGPU/CUDA, and findmaxbatchihl reclaims+queries
        # free memory of "the current device" — racing this across spawns can
        # query the wrong device's memory budget.
        # zip pairs each block with a device and truncates to the shorter side,
        # so when ncols < ndev the surplus devices are simply never touched.
        # devs need only be iterable (CUDA.devices() is a non-indexable
        # DeviceIterator); only zgidxbatches is ever indexed, and zip guarantees
        # did ≤ length(zgidxbatches).
        zpd_devs = map(zip(devs, zgidxbatches)) do (dev, zgidx)
            device!(backend, dev)
            zgb_len = length(zgidx) * size(zg, 1)
            ismissing(zpd) ? min(findmaxbatchihl(backend, T, m, nit), zgb_len) : zpd
        end
        results = Vector{Any}(undef, length(zgidxbatches))
        @sync begin
            for (did, (dev, zgidx)) in enumerate(zip(devs, zgidxbatches))
                Threads.@spawn begin
                    device!(backend, dev)
                    zgb = zg[:, zgidx]
                    zpd_dev = zpd_devs[did]
                    if progress
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀, pchnl, wgs)
                    else
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd_dev, nit, x₀, missing, wgs)
                    end
                end
            end
        end
        result = (hcat(results...))::Matrix{real(T)}
    else
        # note, cpu CANNOT currently batch zg -- there are race conditions present due to pre-allocation of ihl for device codes
        # if you run out of memory here...you should have just used the gpu anyways!
        progress && set_description(pbar, "CPU device, grid points * nit:")
        result = sdihlpsa(backend, zg, P, γ, δ, length(zg), nit, x₀, progress ? pchnl : missing, wgs)
    end
    progress && close(pchnl)
    return permutedims(result)
end

## END WRAPPER FUNCTIONS ##