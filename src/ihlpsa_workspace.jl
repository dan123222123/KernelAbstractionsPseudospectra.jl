# Lanczos KA kernels and the device-resident per-batch workspace (IHLworkspace) for ihlpsa.
# Split out of ihlpsa.jl for readability; included from there after `using .KATRSM`.

# Copy a (g × m) 2D source V into a g-vector-of-m-vectors destination W.
# V is a 2D SubArray (e.g. view(Qv[2], 1:g, :)), not a VectorOfSimilarVectors,
# so the dimensions come from size(V).
@kernel function _v2v!(W, V)
    I = @index(Global, Linear)
    g, m = size(V)
    if I <= g
        # @inbounds: I ≤ g = size(V, 1) = length(W); i ≤ m = size(V, 2) = length(W[I])
        @inbounds for i in 1:m
            W[I][i] = V[I, i]
        end
    end
end

# Normalize v[I] into row I of Q and return the norm — the shared normalize step of
# the two Lanczos kernels below. `reseed` also overwrites v[I] with the normalized
# vector (the next iteration's solve RHS); it is a literal at both call sites, so the
# branch folds away after inlining.
@inline function _normalize_row!(Q, v, I, reseed)
    vnorm = zero(real(eltype(v[I])))
    # @inbounds: callers guard I ≤ g = length(v); loops run over m = length(v[I]) = size(Q, 2)
    @inbounds begin
        for i in 1:length(v[I])
            vnorm += real(conj(v[I][i]) * v[I][i])
        end
        vnorm = sqrt(vnorm)
        for j in 1:length(v[I])
            q = v[I][j] / vnorm
            Q[I, j] = q
            reseed && (v[I][j] = q)
        end
    end
    return vnorm
end

# Normalize each v[I] into Qₙ₊₁ and record the norms in βₙ₊₁, one thread per grid
# point. v is a length-g vector of m-dim vectors; βₙ₊₁ is g-dim; Qₙ₊₁ is a (g × m)
# matrix view of the Qv workspace.
@kernel function _qₙnext!(Qₙ₊₁, βₙ₊₁, v)
    I = @index(Global, Linear)
    g = length(v)
    if I <= g
        @inbounds βₙ₊₁[I] = _normalize_row!(Qₙ₊₁, v, I, false)
    end
end

# Fused Lanczos three-term recurrence + qₙnext, one thread per grid point.
@kernel function _ihl_ttr_qₙnext!(βₙ₋₁, Qₙ₋₁, αₙ, Qₙ, v, βₙ₊₁)
    I = @index(Global, Linear)
    g = length(v)
    m = length(v[1])
    if I <= g
        # @inbounds: I ≤ g bounds every g-dim array; i ≤ m = length(v[I]) = size(Q*, 2)
        @inbounds begin
            for i in 1:m
                v[I][i] -= βₙ₋₁[I] * Qₙ₋₁[I, i]
            end
            αₙ[I] = zero(eltype(v[1]))
            for i in 1:m
                αₙ[I] += conj(Qₙ[I, i]) * v[I][i]
            end
            for i in 1:m
                v[I][i] -= αₙ[I] * Qₙ[I, i]
                Qₙ₋₁[I, i] = Qₙ[I, i]
            end
            βₙ₊₁[I] = _normalize_row!(Qₙ, v, I, true)
        end
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

# B is the backend instance (a compile-time tag, like SchurMatrixPencil's STD); the
# array-family parameters keep field access concrete on both host and device.
struct IHLworkspace{
    T, B, PT <: AbstractMatrixPencil{T}, VZ <: AbstractVector{T}, VX, VQ, VV}
    maxbatch::Int
    zv::VZ
    P::PT
    x₀::VX
    Qv::VQ
    v::VV
end
function IHLworkspace{T, B}(maxbatch, zv, P, x₀, Qv, v) where {T, B}
    IHLworkspace{T, B, typeof(P), typeof(zv), typeof(x₀), typeof(Qv), typeof(v)}(
        maxbatch, zv, P, x₀, Qv, v)
end

function IHLworkspace(P::AbstractMatrixPencil{T}, maxbatch, x₀ = missing) where {T <:
                                                                                 Complex}
    m = size(P, 1)
    zv = zeros(T, maxbatch)
    if ismissing(x₀)
        # Random x₀ is rotationally invariant — no basis transform needed.
        x = randn(T, m)
        x₀ = VectorOfSimilarVectors(repeat(x / norm(x), outer = (1, maxbatch)))
    elseif !(x₀ isa VectorOfSimilarVectors)
        # User-supplied x₀ is in the original-A basis, but ihlpsa runs Lanczos in the
        # Schur basis; Z'x₀ maps it across (P.Z = identity for a raw, non-Schur pencil,
        # making this a no-op) so Ritz values match textbook Lanczos on x₀ in A's basis.
        x₀ = P.Z' * x₀
        x₀ = VectorOfSimilarVectors(repeat(x₀ / norm(x₀), outer = (1, maxbatch)))
    end
    Qv = VectorOfSimilarArrays(zeros(T, maxbatch, m, 2))
    # v starts as zeros; lockstep_ihl! reseeds v[1:g] from x₀ at the top of every batch.
    v = VectorOfSimilarVectors(zeros(T, m, maxbatch))
    IHLworkspace{T, get_backend(P)}(maxbatch, zv, P, x₀, Qv, v)
end

function Adapt.adapt_structure(to, ihl::IHLworkspace)
    zv = adapt(to, ihl.zv)
    P = adapt(to, ihl.P)
    x₀ = adapt(to, ihl.x₀)
    Qv = adapt(to, ihl.Qv)
    v = adapt(to, ihl.v)
    IHLworkspace{eltype(zv), get_backend(P)}(ihl.maxbatch, zv, P, x₀, Qv, v)
end

KernelAbstractions.get_backend(x::IHLworkspace{T, B}) where {T, B} = B
