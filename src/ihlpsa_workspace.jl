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
