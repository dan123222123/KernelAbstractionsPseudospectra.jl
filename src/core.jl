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

# Schur-factored pencils used by the GPU trsm path. ONE parametric struct with a compile-time `STD`
# Bool tag (true ⇒ B = I standard; false ⇒ B ≠ I generalized). `b_is_identity`, the eye-vs-generic
# kernel selection, the tiled B-tile skip and the adapt B/Bc handling all dispatch on `STD` (a
# compile-time branch, not a runtime flag). `const` aliases keep the old `Standard…`/`Generalized…`
# names. (`::SchurMatrixPencil` bare still matches any instance for dispatch.)
#
# Z is the right Schur transform: A_orig = Z * A * Z' (standard) or A_orig = Q * A * Z',
# B_orig = Q * B * Z' (generalized). Used by ihlpsa to bring a user-supplied x₀ from the original-A
# basis into the Schur basis so ihlpsa's Lanczos iterates match textbook Lanczos with the same x₀.
# Ac/Bc are lazy conjugate-transpose views on the host (shared storage); see `adapt_structure` below.
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

# Adapting a Schur pencil to a device does two memory/perf things, branched on the compile-time `STD`:
#  * Ac (and Bc for a generalized pencil) are lazy conjugate-transpose views (`F.T'`); the GPU forward
#    solve reads them TRANSPOSED (column-strided) → uncoalesced, ~1.4–1.8× slower. We MATERIALIZE them
#    as dense contiguous arrays (`Matrix(P.Ac)`) so the forward read is coalesced — values unchanged,
#    ~free in device memory (`adapt` already copies each field; this just lays the bytes out for
#    coalescing). The host struct keeps the lazy views (the CPU solve reads `P.A'`, never Ac/Bc).
#  * For a STANDARD pencil B = Bc = I, and after the B=I "eye" kernels NO GPU kernel reads B/Bc, so
#    `adapt` ships a tiny `Diagonal` identity (m elements, shared between B and Bc) instead of a dense
#    m×m — cutting the standard pencil from 5·m² to ~3·m² per device. A generalized pencil's B/Bc are
#    real data → stay dense (Bc materialized like Ac). The host build already uses a Diagonal identity
#    (ctor above); `ℂsvdpsa!` takes `AbstractMatrix`, so the SVD oracle / CPU solve handle the Diagonal.
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
