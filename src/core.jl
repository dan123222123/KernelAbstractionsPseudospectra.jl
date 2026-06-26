using LinearAlgebra, KernelAbstractions, Adapt, Optim

abstract type AbstractMatrixPencil{T} end

struct MatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
end

# User-facing constructor: always returns a `SchurMatrixPencil` (a `StandardSchurMatrixPencil`
# for B = I, a `GeneralizedSchurMatrixPencil` for B ≠ I) so the GPU trsm path (which dispatches
# on `SchurMatrixPencil`) just works. Direct access to the raw, non-factored MatrixPencil struct
# is still possible via MatrixPencil{T}(A, B).
function MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}
    if B isa UniformScaling
        return MatrixPencil(schur(A))
    end
    @assert size(A) == size(B)
    return MatrixPencil(schur(A, B))
end

# Schur-factored pencils used by the GPU trsm path. `SchurMatrixPencil` is the abstract umbrella
# (so `::SchurMatrixPencil` dispatch and `isa` checks cover both); the concrete subtype encodes
# whether B = I, so the tiled solve's B-tile skip (and the pencil arithmetic) is chosen by
# DISPATCH on the type rather than a runtime `b_is_identity` flag. Both carry the same fields.
#
# Z is the right Schur transform: A_orig = Z * A * Z' (standard) or A_orig = Q * A * Z',
# B_orig = Q * B * Z' (generalized). Used by ihlpsa to bring a user-supplied x₀ from the
# original-A basis into the Schur basis so ihlpsa's Lanczos iterates match textbook Lanczos on
# the original problem with the same x₀. Lazy adjoint by default to avoid duplicating m×m bytes.
abstract type SchurMatrixPencil{T} <: AbstractMatrixPencil{T} end

struct StandardSchurMatrixPencil{T} <: SchurMatrixPencil{T}     # B = I (standard, non-generalized)
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
    Z::AbstractMatrix{T}
end
struct GeneralizedSchurMatrixPencil{T} <: SchurMatrixPencil{T}  # B ≠ I (generalized pencil)
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
    Z::AbstractMatrix{T}
end

# Whether B = I — resolved by type, replacing the former runtime `b_is_identity` field. Lets the
# tiled GPU trailing update skip the B tile (off-diagonal B[i,j]=0 ⇒ the z·B term vanishes),
# halving its shared memory (better occupancy; a wide element type's single tile fits the 48 KB
# limit). The non-`eye` solver paths still read B/Bc (they hold the actual identity matrix).
b_is_identity(::StandardSchurMatrixPencil) = true
b_is_identity(::GeneralizedSchurMatrixPencil) = false

function MatrixPencil(F::Schur{T}) where {T<:Complex}
    Iₘ = Matrix{T}(I, size(F.T))
    StandardSchurMatrixPencil{T}(F.T, F.T', Iₘ, Iₘ, F.Z)   # B = I (standard pencil)
end
function MatrixPencil(F::GeneralizedSchur{T}) where {T<:Complex}
    GeneralizedSchurMatrixPencil{T}(F.S, F.S', F.T, F.T', F.Z)
end

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

Adapt.@adapt_structure MatrixPencil
Adapt.@adapt_structure StandardSchurMatrixPencil
Adapt.@adapt_structure GeneralizedSchurMatrixPencil

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
