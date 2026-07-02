using LinearAlgebra, KernelAbstractions, Adapt, Optim

abstract type AbstractMatrixPencil{T} end

struct MatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
end

# User-facing constructor → a Schur-factored pencil (standard for B = I, generalized for B ≠ I) so
# the GPU trsm path just works. The raw, non-factored struct is still reachable via MatrixPencil{T}(A, B).
function MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}
    if B isa UniformScaling
        return MatrixPencil(schur(A))
    end
    @assert size(A) == size(B)
    return MatrixPencil(schur(A, B))
end

# Schur-factored pencil for the GPU trsm path. The compile-time `STD` tag (true ⇒ B = I standard,
# false ⇒ B ≠ I generalized) drives kernel selection and the adapt below at compile time, not via a
# runtime flag. Z is the right Schur transform (A_orig = Z·A·Z'), used by ihlpsa to bring x₀ into the
# Schur basis. Ac/Bc are lazy conjugate-transpose views on the host, materialized on device below.
struct SchurMatrixPencil{T, STD} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
    Z::AbstractMatrix{T}
end
const StandardSchurMatrixPencil{T}    = SchurMatrixPencil{T, true}     # B = I (standard)
const GeneralizedSchurMatrixPencil{T} = SchurMatrixPencil{T, false}    # B ≠ I (generalized)

# Whether B = I — read straight off the compile-time `STD` tag.
b_is_identity(::SchurMatrixPencil{T, STD}) where {T, STD} = STD

MatrixPencil(F::Schur{T}) where {T<:Complex} =
    (Iₘ = Diagonal(ones(T, size(F.T, 1)));                  # B = Bc = I as a Diagonal (m elts, not m×m)
     SchurMatrixPencil{T, true}(F.T, F.T', Iₘ, Iₘ, F.Z))
MatrixPencil(F::GeneralizedSchur{T}) where {T<:Complex} =
    SchurMatrixPencil{T, false}(F.S, F.S', F.T, F.T', F.Z)

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

Adapt.@adapt_structure MatrixPencil

# On device, materialize the lazy Ac/Bc conjugate-transpose views as dense arrays so the GPU forward
# read is coalesced, and for a standard (B = I) pencil ship a Diagonal identity instead of a dense m×m
# — no kernel reads B after the eye path. (The host keeps the lazy views; the CPU solve reads `P.A'`.)
function Adapt.adapt_structure(to, P::SchurMatrixPencil{T, STD}) where {T, STD}
    A = adapt(to, P.A); Ac = adapt(to, Matrix(P.Ac)); Z = adapt(to, P.Z)
    if STD
        Id = Diagonal(adapt(to, ones(T, size(P.A, 1))))    # device B = Bc = I (m elements; unread after eye)
        SchurMatrixPencil{T, true}(A, Ac, Id, Id, Z)
    else
        SchurMatrixPencil{T, false}(A, Ac, adapt(to, P.B), adapt(to, Matrix(P.Bc)), Z)
    end
end

"""
    validate(zg, A, B, γ=missing, δ=missing)
    validate(zg, P, γ=missing, δ=missing)

Checks the following:
- the grid of shifts (zg) is not empty
- A and B are of the same size (if B!=UniformScaling)
- γ, δ are valid (γ, δ ≥ 0 and not both zero)
"""
function validate(zg, A::AbstractMatrix, B, γ=missing, δ=missing)
    @assert !isempty(zg)
    if !(B isa UniformScaling)
        @assert size(A) == size(B)
    end
    if !(ismissing(γ) && ismissing(δ))
        _validate_weights(γ, δ)
    end
end
function validate(zg, P::AbstractMatrixPencil, γ=missing, δ=missing)
    validate(zg, P.A, P.B, γ, δ)
end

# Validate the (γ,δ) perturbation weights (value `σ_min(zB − A)/(γ + δ|z|)`): γ, δ ≥ 0, not both
# zero. The set only depends on (γ,δ) up to a common scale. Shared by ℂsvdpsa, ihlpsa, and the
# adaptive driver so all paths accept the same inputs.
function _validate_weights(γ, δ)
    @assert γ ≥ 0 && δ ≥ 0 "perturbation weights must be ≥ 0 (got γ=$γ, δ=$δ)"
    @assert γ + δ > 0 "perturbation weights γ, δ must not both be zero"
end
