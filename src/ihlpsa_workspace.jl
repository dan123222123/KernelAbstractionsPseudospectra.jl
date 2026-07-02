# Lanczos KA kernels and the device-resident per-batch workspace (IHLworkspace) for ihlpsa.
# Split out of ihlpsa.jl for readability; included from there after `using .KATRSM`.

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
# One thread per grid point. v is a length-g vector of m-dim vectors; βₙ₊₁ is g-dim;
# Qₙ₊₁ is a length-g vector of m-dim vectors.
@kernel function _qₙnext(v, βₙ₊₁, Qₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        vnorm = zero(real(eltype(v[I])))
        for i = 1:m
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        for j = 1:m
            Qₙ₊₁[I, j] = v[I][j] / vnorm
        end
        βₙ₊₁[I] = vnorm
    end
end

# TODO polish
# Fused Lanczos 3-term recurrence + qₙnext, one thread per grid point.
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
            Qₙ₋₁[I, i] = Qₙ[I, i]
        end
        # qₙnext
        vnorm = zero(real(eltype(v[I])))
        for i = 1:m
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        for j = 1:m
            Qₙ[I, j] = v[I][j] / vnorm
            v[I][j] = Qₙ[I, j]
        end
        βₙ₊₁[I] = vnorm
    end
end

# Gather rows `keepd` of a (g, m, k) source into a packed (nkeep, m, k) prefix; used
# for the adaptive survivor gather of the Qv workspace. Do NOT replace with a plain
# `src[keep,:,:]` fancy index: GPUArrays' first-axis fancy indexing of a 3-D array is
# miscompiled on oneAPI (silently wrong rows + device out-of-bounds, even through a
# 2-D reshape). Last-axis gathers (`[:, keep]`) are unaffected and fine as fancy indexing.
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
        # User-supplied x₀ is in the original-A basis, but ihlpsa runs Lanczos in the
        # Schur basis; Z'x₀ maps it across (P.Z = identity for a raw, non-Schur pencil,
        # making this a no-op) so Ritz values match textbook Lanczos on x₀ in A's basis.
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

KernelAbstractions.get_backend(x::IHLworkspace{T,B}) where {T,B} = B
