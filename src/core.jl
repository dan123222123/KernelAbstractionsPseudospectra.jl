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
    Iₘ = Diagonal(ones(T, size(F.T, 1)))                   # B = Bc = I, stored as a Diagonal (m elts,
    StandardSchurMatrixPencil{T}(F.T, F.T', Iₘ, Iₘ, F.Z)   #   not a dense m×m); no solve path needs dense B
end
function MatrixPencil(F::GeneralizedSchur{T}) where {T<:Complex}
    GeneralizedSchurMatrixPencil{T}(F.S, F.S', F.T, F.T', F.Z)
end

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

Adapt.@adapt_structure MatrixPencil

# The Schur pencils' Ac/Bc are lazy conjugate-transpose views (`F.T'`, …). On the host that shares
# storage; but the GPU forward solve reads them with a TRANSPOSED (column-strided) access pattern,
# which is uncoalesced and ~1.4–1.8× slower than a contiguous read (measured at the kernel level).
# So when adapting a pencil to a device we MATERIALIZE the adjoint fields as dense contiguous arrays
# (`Matrix(P.Ac)`): values are unchanged — only the layout — so results stay bitwise-identical, while
# the forward solve becomes coalesced. It is ~free in device memory: `adapt` already makes a separate
# device copy of each field, so this stores the same bytes laid out for coalescing instead of for a
# transposed view. The host struct keeps its lazy views (the CPU solve path reads `P.A'`, never Ac/Bc).
#
# B/Bc memory: a standard pencil has B = Bc = I, and after the B=I "eye" kernels (column / warp /
# tiled) NO GPU kernel reads B/Bc for a standard pencil — so neither host nor device needs a dense
# m×m identity. The pencil is BUILT with a `Diagonal` identity (see `MatrixPencil(::Schur)`, m
# elements), and `adapt` ships a `Diagonal` to the device too (shared between B and Bc) — cutting the
# standard pencil's footprint from 5·m² to ~3·m² on host and on every GPU. `ℂsvdpsa!` takes
# `AbstractMatrix`, so the dense SVD oracle and the CPU solve handle the `Diagonal` (its `z*B − A`
# just materialises). A generalized pencil's B/Bc are real data and stay dense (Bc materialized like Ac).
function Adapt.adapt_structure(to, P::StandardSchurMatrixPencil{T}) where {T}
    Id = Diagonal(adapt(to, ones(T, size(P.A, 1))))   # device B = Bc = I (m elements; unread after eye)
    StandardSchurMatrixPencil{T}(adapt(to, P.A), adapt(to, Matrix(P.Ac)), Id, Id, adapt(to, P.Z))
end
Adapt.adapt_structure(to, P::GeneralizedSchurMatrixPencil{T}) where {T} =
    GeneralizedSchurMatrixPencil{T}(adapt(to, P.A), adapt(to, Matrix(P.Ac)),
                                    adapt(to, P.B), adapt(to, Matrix(P.Bc)), adapt(to, P.Z))

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
