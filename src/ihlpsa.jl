using ArraysOfArrays
using ThreadsX
using ProgressBars

# triangular solves using kernel abstractions
include("KATRSM.jl/KATRSM.jl")
using .KATRSM

## KERNELS ##

# 1-tensor -> vector of vectors
@kernel function _v2v(V, W)
    I = @index(Global, Linear)
    g = length(V)
    m = length(V[1])
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
        vnorm = zero(eltype(v[1]))
        for i = 1:m
            vnorm += conj(v[I][i]) * v[I][i]
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

"""
    IHLworkspace{T,B}

Workspace container for Inverse Hermitian Lanczos (IHL) pseudospectra computations.

# Type parameters
- `T`: Complex element type (e.g., `ComplexF64`)
- `B`: Backend type from KernelAbstractions (e.g., `CPU`, `CUDABackend`)

# Fields
- `maxbatch`: Maximum batch size for simultaneous grid point processing
- `zv::AbstractVector{T}`: Vector of complex shift values for current batch
- `P::AbstractMatrixPencil{T}`: Matrix pencil (A, B) in Schur form
- `x₀`: Initial vectors for Lanczos iteration (VectorOfSimilarVectors)
- `Qv`: Lanczos basis vectors (VectorOfSimilarArrays with 2 time levels)
- `v`: Working vectors for Lanczos iteration

# Details
This workspace is pre-allocated and reused across batches of grid points to minimize allocation
overhead. The structure supports automatic adaptation to different backends (CPU/GPU) via
`Adapt.adapt_structure`.

See also: [`ihlpsa`](@ref), [`lockstep_ihl!`](@ref)
"""
struct IHLworkspace{T,B}
    maxbatch
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
        x = randn(T, m)
        x₀ = VectorOfSimilarVectors(repeat(x / norm(x), outer=(1, maxbatch)))
    elseif !(x₀ isa VectorOfSimilarVectors)
        x₀ = VectorOfSimilarVectors(repeat(x₀ / norm(x₀), outer=(1, maxbatch)))
    end
    Qv = VectorOfSimilarArrays(zeros(T, maxbatch, m, 2))
    v = VectorOfSimilarVectors(zeros(T, m, maxbatch))
    v .= deepcopy(x₀)
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

# non-cpu solve step in lockstep_ihl!
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=256)
    g = length(zv)
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bV, conj(zv), P.Ac, P.Bc)
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs, (wgs, g))(bV, zv, P.A, P.B)
end

# cpu solve step in lockstep_ihl!
function trsmIHL(backend::CPU, bV, zv, P::SchurMatrixPencil)
    g = length(zv)
    _batched_forward_solve_pencil(backend)(bV, conj(zv), P.A', P.B', ndrange=g)
    _batched_backward_solve_pencil(backend)(bV, zv, P.A, P.B, ndrange=g)
end

"""
    lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=256)

Execute batched Inverse Hermitian Lanczos iteration on device.

This kernel-level function performs the core Lanczos iteration for `g` grid points simultaneously,
storing the resulting tridiagonal matrix coefficients in `α` (diagonal) and `β` (off-diagonal).

# Arguments
- `α`: Output matrix for diagonal Lanczos coefficients (nit × gtotal, views used)
- `β`: Output matrix for off-diagonal Lanczos coefficients ((nit+1) × gtotal, views used)
- `ihl::IHLworkspace`: Pre-allocated workspace with matrix pencil and working arrays
- `nit::Integer`: Number of Lanczos iterations
- `g::Integer`: Number of grid points in current batch
- `wgs::Integer`: Workgroup size for GPU kernels (default: 256)

# Algorithm
For each grid point z[i] in parallel:
1. Initialize: q₁ = x₀/‖x₀‖, β₁ = ‖x₀‖
2. For n = 1 to nit:
   - Solve: v = (zB - A)⁻¹(z̄B - A)⁻ᴴ qₙ  [via batched triangular solves]
   - Three-term recurrence: v ← v - βₙ₋₁qₙ₋₁
   - Compute: αₙ = qₙᴴv
   - Update: v ← v - αₙqₙ
   - Normalize: qₙ₊₁ = v/‖v‖, βₙ₊₁ = ‖v‖

The resulting (α, β) define a Hermitian tridiagonal matrix whose largest eigenvalue
approximates the squared smallest singular value of (zB - A).

# Implementation details
- Uses custom kernels: `_qₙnext` (normalization), `_ihl_ttr_qₙnext` (three-term recurrence)
- Batched triangular solves via `trsmIHL` wrapper to `KATRSM` module
- All operations execute on device; synchronizes before returning
- Lockstep execution ensures coherent state across all grid points in batch

See also: [`ihlpsa`](@ref), [`IHLworkspace`](@ref), [`ihlsrg!`](@ref)
"""
function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=256)
    backend = get_backend(ihl)
    ihl.v .= ihl.x₀
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

## END DEVICE FUNCTIONS ##

## HOST FUNCTIONS ##

"""
    ihlsrg!(sr, zv, γ, δ, α, β)

Compute resolvent norms from Lanczos tridiagonal matrices.

This host-side function extracts smallest singular values from the tridiagonal matrices
produced by the Inverse Hermitian Lanczos iteration and converts them to resolvent norms.

# Arguments
- `sr`: Output vector for resolvent norms (modified in-place)
- `zv`: Vector of complex shift values
- `γ::Real`: Scaling weight for perturbations to A
- `δ::Real`: Scaling weight for perturbations to B
- `α`: Diagonal coefficients from Lanczos iteration (nit × g matrix)
- `β`: Off-diagonal coefficients from Lanczos iteration ((nit+1) × g matrix)

# Details
For each grid point i, constructs the Hermitian tridiagonal Lanczos matrix:
```
T = tridiag(β[2:nit], α[1:nit-1], β[2:nit])
```

The resolvent norm is computed as:
```
sr[i] = (γ + δ|z[i]|) / sqrt(λₘₐₓ(T))
```
where `λₘₐₓ(T)` approximates `σₘᵢₙ²(zB - A)`.

Multi-threaded across grid points using `Threads.@threads`.

# Error handling
If eigenvalue computation fails, sets `sr[i] = eps(real(eltype(zv)))` as a fallback.

See also: [`lockstep_ihl!`](@ref), [`ihlpsa`](@ref)
"""
function ihlsrg!(sr, zv, γ, δ, α, β)
    Threads.@threads for i in eachindex(zv)
        Tv = Hermitian(diagm(0 => real(α[:, i]), -1 => real(β[2:end, i]), 1 => real(β[2:end, i]))[1:end-1, 1:end-1])
        try
            sr[i] = (γ + δ * abs(zv[i])) / sqrt(eigmax(Tv))
        catch
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
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    x₀::Union{Missing,AbstractVector{T},AbstractArrayOfSimilarArrays{T}}=missing,
    pchnl::Union{Missing,Channel}=missing,
    wgs=256
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
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    x₀::Union{Missing,AbstractVector{T},AbstractArrayOfSimilarArrays{T}}=missing,
    pchnl::Union{Missing,Channel}=missing,
    wgs=256
) where {T<:Complex}
    sdargs = (; zg, P, γ, δ, zpd, nit, x₀, pchnl, wgs)
    sdihlpsa(backend; sdargs...)
end

"""
    findmaxbatchihl(backend, T, m; moe=0.1)

Compute maximum batch size that fits in device memory.

Determines the largest number of grid points that can be processed simultaneously in a single
batch, given the available device memory and the workspace requirements for IHL iteration.

# Arguments
- `backend`: KernelAbstractions backend (CPU, CUDA, AMDGPU, Metal)
- `T`: Element type (e.g., `ComplexF64`)
- `m`: Matrix dimension
- `moe`: Margin of error / safety factor (default: 0.1 = 10%)

# Returns
- `Integer`: Maximum batch size (number of grid points)

# Memory calculation
The memory footprint per batch includes:
- Matrix pencil storage: `4m²` elements (A, A', B, B' for SchurMatrixPencil)
- Per-gridpoint workspace: `4m + 1` elements per gridpoint
  - Lanczos vectors: `2m` (Q current and previous)
  - Working vectors: `2m` (v and storage)
  - Shift value: `1`

Total bytes: `sizeof(T) * (4m² + batchsize*(4m + 1))`

The function ensures memory usage stays below `(1 - moe)` of available device memory.

# Notes
Call `device_reclaim(backend)` before this to trigger garbage collection and get accurate
available memory estimate.

See also: [`ihlpsa`](@ref), [`IHLworkspace`](@ref)
"""
function findmaxbatchihl(backend, T, m; moe=0.1)
    device_reclaim(backend)
    floor(Integer, (device_bytes_available(backend) * (1 - moe) - (sizeof(T) * (4 * m * m + 1))) / (sizeof(T) * (1 + 4 * m)))
end

"""
    ihlpsa(backend, zg, P, nit=ceil(Int, log2(m)), γ=1, δ=0;
           x₀=missing, progress=false, zpd=missing, devs=missing, wgs=256)

Compute pseudospectra using GPU-accelerated Inverse Hermitian Lanczos method.

This is the main entry point for IHL-based pseudospectra computation, supporting multi-device
execution and automatic memory management. Unlike SVD-based methods, IHL scales to large matrices
and leverages GPU acceleration through batched triangular solves.

# Arguments
- `backend`: KernelAbstractions backend (CPU(), CUDA.CUDABackend(), AMDGPU.ROCBackend(), etc.)
- `zg::AbstractArray{T,2}`: Grid of complex shift values (nx × ny)
- `P::AbstractMatrixPencil{T}`: Matrix pencil in Schur form (use `MatrixPencil(schur(...))`)
- `nit::Integer`: Number of Lanczos iterations (default: `ceil(log2(m))`)
- `γ::Real`: Scaling weight for perturbations to A (default: 1)
- `δ::Real`: Scaling weight for perturbations to B (default: 0)

# Keyword arguments
- `x₀`: Initial vector(s) for Lanczos iteration (default: random)
- `progress::Bool`: Show progress bar tracking iteration completion (default: false)
- `zpd`: Grid points per device batch, overrides automatic sizing (default: auto)
- `devs`: Device list for multi-GPU execution (default: all available devices)
- `wgs::Integer`: Workgroup size for GPU kernels (default: 256; use 16 for AMDGPU)

# Returns
- `Matrix{real(T)}`: Transposed resolvent norm matrix (ny × nx for plotting)

# Algorithm
The Inverse Hermitian Lanczos method approximates the smallest singular value of `(zB - A)`
without computing the full SVD:

1. For each shift `z`, apply Lanczos iteration to `(zB - A)⁻¹(z̄B - A)⁻ᴴ`
2. Build tridiagonal matrix T from Lanczos coefficients α, β
3. Estimate `σₘᵢₙ(zB - A) ≈ 1/√λₘₐₓ(T)`
4. Scale by normalization factor: `(γ + δ|z|) * σₘᵢₙ`

The implementation uses:
- Batched triangular solves via `KATRSM` submodule
- Multi-device parallelism (distributes grid columns across GPUs)
- Automatic memory-aware batching via `findmaxbatchihl`

# Performance notes
- GPU backends: Work distributed across multiple devices, within-device batching
- CPU backend: Currently processes all grid points in one batch (no sub-batching due to race conditions)
- Typical iteration count: `O(log m)` iterations suffice for accurate pseudospectra
- Workgroup size (`wgs`): Use 256 for CUDA, 16 for AMDGPU for optimal performance

# Examples
```julia
using LinearAlgebra, KernelAbstractions, CUDA

# Create test problem
m = 1000
A = randn(ComplexF64, m, m)
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (200, 200))
P = MatrixPencil(schur(A))

# GPU computation with progress bar
psa = ihlpsa(CUDABackend(), zg, P, progress=true)

# Multi-GPU with 4 devices
psa = ihlpsa(CUDABackend(), zg, P, devs=CUDA.devices()[1:4])

# CPU computation
psa = ihlpsa(CPU(), zg, P)

# AMDGPU with tuned workgroup size
using AMDGPU
psa = ihlpsa(ROCBackend(), zg, P, wgs=16)
```

# References
- Lui, S.H. (1997). "Computation of pseudospectra by continuation." SIAM J. Sci. Comput., 18(2), 565-573.
- Trefethen, L.N. & Embree, M. (2005). "Spectra and Pseudospectra", Princeton University Press.

See also: [`ℂsvdpsa`](@ref), [`MatrixPencil`](@ref), [`findmaxbatchihl`](@ref), [`qgrid`](@ref)
"""
function ihlpsa(
    backend,
    zg::AbstractArray{T,2},
    P::AbstractMatrixPencil{T},
    nit::Integer=ceil(Integer, log2(size(P, 1))),
    γ=1,
    δ=0;
    x₀::Union{Missing,AbstractVector{T},AbstractMatrix{T}}=missing,
    progress=false,
    zpd=missing,
    devs=missing,
    wgs=256
) where {T<:Complex}
    m = size(P.A, 1)
    # this progress bar can probably be generalized to all psa methods via a channel -- TODO
    pbar = ProgressBar(total=nit * length(zg), printing_delay=0.001)
    pchnl = Channel()
    Threads.@spawn begin
        while (isopen(pchnl))
            wait(pchnl)
            ProgressBars.update(pbar, take!(pchnl))
        end
    end
    if KernelAbstractions.isgpu(backend)
        if ismissing(devs)
            devs = devices(backend)
        end
        set_description(pbar, "$(length(devs)) device(s), grid points * nit:")
        zgidxbatches = Vector(collect(Iterators.partition(1:size(zg, 2), ceil(Integer, size(zg, 2) / length(devs)))))
        results = Vector{Any}(undef, length(devs))
        @sync begin
            for (did, dev) in enumerate(devs)
                Threads.@spawn begin
                    device!(backend, dev)
                    zgb = zg[:, zgidxbatches[did]]
                    if ismissing(zpd)
                        zpd = min(findmaxbatchihl(backend, T, m), length(zgb))
                    end
                    if progress
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd, nit, x₀, pchnl, wgs)
                    else
                        results[did] = sdihlpsa(backend, zgb, P, γ, δ, zpd, nit, x₀, missing, wgs)
                    end
                end
            end
        end
        result = (hcat(results...))::Matrix{real(T)}
    else
        # note, cpu CANNOT currently batch zg -- there are race conditions present due to pre-allocation of ihl for device codes
        # if you run out of memory here...you should have just used the gpu anyways!
        set_description(pbar, "CPU device, grid points * nit:")
        result = sdihlpsa(backend, zg, P, γ, δ, length(zg), nit, x₀, missing, wgs)
    end
    close(pchnl)
    return result'
end

## END WRAPPER FUNCTIONS ##