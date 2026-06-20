using LinearAlgebra, KernelAbstractions, Adapt, Optim

abstract type AbstractMatrixPencil{T} end

struct MatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
end

# User-facing constructor: always returns a SchurMatrixPencil so the GPU trsm
# path (which dispatches on SchurMatrixPencil) just works. Direct access to the
# raw, non-factored MatrixPencil struct is still possible via MatrixPencil{T}(A, B).
function MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}
    if B isa UniformScaling
        return MatrixPencil(schur(A))
    end
    @assert size(A) == size(B)
    return MatrixPencil(schur(A, B))
end

struct SchurMatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
    # Right Schur transform: A_orig = Z * A * Z' (standard) or A_orig = Q * A * Z',
    # B_orig = Q * B * Z' (generalized). Used by ihlpsa to bring a user-supplied
    # x₀ from the original-A basis into the Schur basis so that ihlpsa's Lanczos
    # iterates match textbook Lanczos applied to the original problem with the
    # same x₀. Lazy adjoint by default to avoid duplicating m×m bytes.
    Z::AbstractMatrix{T}
end
# Backward-compat constructor for direct use (no Schur transform known).
# Stores a typed Diagonal of ones (O(m) memory) so x₀-transform via Z'*x is a no-op
# and Adapt-to-GPU is essentially free.
SchurMatrixPencil{T}(A, Ac, B, Bc) where {T<:Complex} =
    SchurMatrixPencil{T}(A, Ac, B, Bc, Diagonal(ones(T, size(A, 1))))

function MatrixPencil(F::Schur{T}) where {T<:Complex}
    Iₘ = Matrix{T}(I, size(F.T))
    SchurMatrixPencil{T}(F.T, F.T', Iₘ, Iₘ, F.Z)
end
function MatrixPencil(F::GeneralizedSchur{T}) where {T<:Complex}
    SchurMatrixPencil{T}(F.S, F.S', F.T, F.T', F.Z)
end

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

Adapt.@adapt_structure MatrixPencil
Adapt.@adapt_structure SchurMatrixPencil

"""
    validate(zg, A, B, γ=missing, δ=missing)
    validate(zg, P, γ=missing, δ=missing)

Checks the following:
- the grid of shifts (zg) is not empty
- A and B are of the same size (if B!=UniformScaling)
- γ, δ are valid (γ, δ ≥ 0 and not both zero) -- see [`_validate_weights`](@ref)
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

"""
    _validate_weights(γ, δ)

Validate the (γ,δ) perturbation weights of the Frayssé et al. pseudospectrum
(value `σ_min(zB − A)/(γ + δ|z|)`). The set is invariant to a common scaling of
`(γ,δ)` (it only rescales `ε`), so any `γ, δ ≥ 0` not both zero is admissible —
e.g. the literature's `(1,0)` (standard) and `(1,1)`. Shared by `ℂsvdpsa`,
`ihlpsa`, and the adaptive driver so all paths accept the same inputs.
"""
function _validate_weights(γ, δ)
    @assert γ ≥ 0 && δ ≥ 0 "perturbation weights must be ≥ 0 (got γ=$γ, δ=$δ)"
    @assert γ + δ > 0 "perturbation weights γ, δ must not both be zero"
end
