using LinearAlgebra, KernelAbstractions, Adapt, Optim

abstract type AbstractMatrixPencil{T} end

struct MatrixPencil{T, MA <: AbstractMatrix{T}, MB <: AbstractMatrix{T}} <:
       AbstractMatrixPencil{T}
    A::MA
    B::MB
end

# Raw, non-factored wrapper: keeps exactly the arrays passed, no Schur factoring.
function MatrixPencil{T}(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    MatrixPencil{T, typeof(A), typeof(B)}(A, B)
end

"""
    MatrixPencil(A[, B=I]) -> SchurMatrixPencil

Wrap the matrix pencil `(A, B)` for pseudospectra computation, Schur-factoring it
(standard for `B = I`, generalized for `B ≠ I`) so the GPU triangular-solve path
applies directly. A `Schur`/`GeneralizedSchur` factorization can also be passed in
place of the matrices. A scaled `cI` is materialized and treated as a generalized
pencil. The raw, non-factored wrapper remains reachable via `MatrixPencil{T}(A, B)`
for the dense-SVD paths ([`ℂsvdpsa`](@ref)/[`ℝsvdpsa`](@ref)); [`ihlpsa`](@ref)
requires the Schur form.
"""
function MatrixPencil(A::AbstractMatrix{T},
        B::Union{AbstractMatrix{T}, UniformScaling} = I) where {T <: Complex}
    if B isa UniformScaling
        # Literal I takes the standard path; a scaled cI must keep its scale, so
        # materialize it and go generalized (same treatment as ℂsvdpsa/ℝsvdpsa).
        isone(B.λ) && return MatrixPencil(schur(A))
        B = Matrix{T}(B, size(A))
    end
    size(A) == size(B) ||
        throw(DimensionMismatch("pencil matrices differ in size: A $(size(A)), B $(size(B))"))
    return MatrixPencil(schur(A, B))
end

# Schur-factored pencil for the GPU trsm path. The compile-time `STD` tag (B = I standard vs
# B ≠ I generalized) drives kernel selection and the adapt rule below, not a runtime flag. Z is
# the right Schur transform (A_orig = Z·A·Z'), used by ihlpsa to bring x₀ into the Schur basis.
# Ac/Bc are lazy conjugate-transpose views on the host, materialized on device below.
struct SchurMatrixPencil{T, STD, MA <: AbstractMatrix{T}, MAc <: AbstractMatrix{T},
    MB <: AbstractMatrix{T}, MBc <: AbstractMatrix{T},
    MZ <: AbstractMatrix{T}} <: AbstractMatrixPencil{T}
    A::MA
    Ac::MAc
    B::MB
    Bc::MBc
    Z::MZ
end
function SchurMatrixPencil{T, STD}(A, Ac, B, Bc, Z) where {T, STD}
    SchurMatrixPencil{T, STD, typeof(A), typeof(Ac), typeof(B), typeof(Bc), typeof(Z)}(
        A, Ac, B, Bc, Z)
end
const StandardSchurMatrixPencil{T} = SchurMatrixPencil{T, true}        # B = I (standard)
const GeneralizedSchurMatrixPencil{T} = SchurMatrixPencil{T, false}    # B ≠ I (generalized)

# Whether B = I — read straight off the compile-time `STD` tag.
b_is_identity(::SchurMatrixPencil{T, STD}) where {T, STD} = STD

function MatrixPencil(F::Schur{T}) where {T <: Complex}
    Iₘ = Diagonal(ones(T, size(F.T, 1)))   # B = Bc = I as a Diagonal (m elts, not m×m)
    return SchurMatrixPencil{T, true}(F.T, F.T', Iₘ, Iₘ, F.Z)
end
function MatrixPencil(F::GeneralizedSchur{T}) where {T <: Complex}
    SchurMatrixPencil{T, false}(F.S, F.S', F.T, F.T', F.Z)
end

"""
    bigfloat_schur_factor(A; bits=256) -> Matrix{Complex{BigFloat}}

Upper-triangular Schur factor `T` of `A` (with `A = Z T Z'`), computed at `bits` of
`BigFloat` precision. Lives in the `GenericSchurPseudospectra` extension, available once
`GenericSchur` and `MultiFloats` are both loaded. Needed because LAPACK's `schur` is
IEEE-only; backs `MatrixPencil(A::AbstractMatrix{<:Union{MultiFloat, Complex{<:MultiFloat}}})`.
"""
function bigfloat_schur_factor end

"""
    bigfloat_qz_factor(A, B; bits=256) -> (S, T)

Upper-triangular generalized Schur (QZ) factors of the pencil `(A, B)`
(`A = Q S Z'`, `B = Q T Z'`), computed at `bits` of `BigFloat` precision — the
generalized analogue of [`bigfloat_schur_factor`](@ref). Lives in the
`GenericSchurPseudospectra` extension; backs the `B ≠ I` case of `MatrixPencil` for
MultiFloat element types. O(hours) at `m ≳ 512` — cache the factors if a size is revisited.
"""
function bigfloat_qz_factor end

Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

# Explicit (not @adapt_structure): the generated `MatrixPencil(A, B)` call would route
# through the Schur-factoring user constructor; `MatrixPencil{T}` keeps the pencil raw.
function Adapt.adapt_structure(to, P::MatrixPencil{T}) where {T}
    MatrixPencil{T}(adapt(to, P.A), adapt(to, P.B))
end

# On device, materialize the lazy Ac/Bc conjugate-transpose views as dense arrays so the GPU forward
# read is coalesced, and for a standard (B = I) pencil ship a Diagonal identity instead of a dense m×m
# — no kernel reads B after the eye path. (The host keeps the lazy views; the CPU solve reads `P.A'`.)
function Adapt.adapt_structure(to, P::SchurMatrixPencil{T, STD}) where {T, STD}
    A = adapt(to, P.A)
    Ac = adapt(to, Matrix(P.Ac))
    Z = adapt(to, P.Z)
    if STD
        Id = Diagonal(adapt(to, ones(T, size(P.A, 1))))    # device B = Bc = I (m elements; unread after eye)
        SchurMatrixPencil{T, true}(A, Ac, Id, Id, Z)
    else
        SchurMatrixPencil{T, false}(A, Ac, adapt(to, P.B), adapt(to, Matrix(P.Bc)), Z)
    end
end

# Elements an array contributes once adapted: dense arrays their full length, a Diagonal its
# diagonal (Adapt keeps the wrapper, so it never inflates to m²).
_stored_elems(A::AbstractMatrix) = length(A)
_stored_elems(D::Diagonal) = length(D.diag)

# Bytes the adapt rules above place on device for `P`: Ac/Bc materialize dense, Z ships as
# stored, and a standard pencil's B/Bc share ONE Diagonal identity.
_device_pencil_bytes(P::AbstractMatrixPencil{T}) where {T} = sizeof(T) * 4 * size(P, 1)^2
function _device_pencil_bytes(P::MatrixPencil{T}) where {T}
    sizeof(T) * (_stored_elems(P.A) + _stored_elems(P.B))
end
function _device_pencil_bytes(P::SchurMatrixPencil{T, STD}) where {T, STD}
    m = size(P, 1)
    elems = _stored_elems(P.A) + m^2 + _stored_elems(P.Z)      # A + dense Ac + Z
    elems += STD ? m : (_stored_elems(P.B) + m^2)              # shared Id, or B + dense Bc
    sizeof(T) * elems
end

# Input validation shared by ℂsvdpsa/ℝsvdpsa/ihlpsa: non-empty shift grid, matching
# A/B sizes (unless B is a UniformScaling), and valid (γ, δ) weights.
function validate(zg, A::AbstractMatrix, B, γ = missing, δ = missing)
    isempty(zg) && throw(ArgumentError("the shift grid zg is empty"))
    if !(B isa UniformScaling) && size(A) != size(B)
        throw(DimensionMismatch("pencil matrices differ in size: A $(size(A)), B $(size(B))"))
    end
    if !(ismissing(γ) && ismissing(δ))
        _validate_weights(γ, δ)
    end
end
function validate(zg, P::AbstractMatrixPencil, γ = missing, δ = missing)
    validate(zg, P.A, P.B, γ, δ)
end

# Validate the (γ,δ) perturbation weights (value `σ_min(zB − A)/(γ + δ|z|)`): γ, δ ≥ 0, not both
# zero (the set only depends on (γ,δ) up to a common scale). Shared by ℂsvdpsa, ihlpsa, and the
# adaptive driver so all paths accept the same inputs.
function _validate_weights(γ, δ)
    (γ ≥ 0 && δ ≥ 0) ||
        throw(ArgumentError("perturbation weights must be ≥ 0 (got γ=$γ, δ=$δ)"))
    γ + δ > 0 || throw(ArgumentError("perturbation weights γ, δ must not both be zero"))
end
