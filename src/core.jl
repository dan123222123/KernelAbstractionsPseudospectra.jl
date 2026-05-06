using LinearAlgebra, KernelAbstractions, Adapt, Optim

"""
    AbstractMatrixPencil{T}

Abstract type for matrix pencils representing generalized eigenvalue problems of the form
`(zB - A)x = 0`.
"""
abstract type AbstractMatrixPencil{T} end

"""
    MatrixPencil{T} <: AbstractMatrixPencil{T}

Standard matrix pencil storing matrices A and B.

# Fields
- `A::AbstractMatrix{T}`: The A matrix in the pencil
- `B::AbstractMatrix{T}`: The B matrix in the pencil

The pencil represents the matrix-valued function `z ↦ zB - A`.
"""
struct MatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
end

"""
    MatrixPencil(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}

Construct a matrix pencil from matrices A and B.

Matrices must have the same size. The pencil represents the matrix-valued function `z ↦ zB - A`.

# Arguments
- `A::AbstractMatrix{T}`: First matrix (complex-valued)
- `B::AbstractMatrix{T}`: Second matrix (complex-valued, same size as A)
"""
function MatrixPencil(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    @assert size(A) == size(B)
    MatrixPencil{T}(A, B)
end

"""
    MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}

Construct a matrix pencil, defaulting to B = I for standard eigenvalue problems.

# Arguments
- `A::AbstractMatrix{T}`: First matrix (complex-valued)
- `B::Union{AbstractMatrix{T},UniformScaling}`: Second matrix, defaults to identity

If `B` is a `UniformScaling` (e.g., `I`), it is converted to an identity matrix of the same size as `A`.
"""
function MatrixPencil(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},UniformScaling}=I) where {T<:Complex}
    if B isa UniformScaling
        B = Matrix{T}(I, size(A))
    end
    MatrixPencil(A, B)
end

"""
    SchurMatrixPencil{T} <: AbstractMatrixPencil{T}

Optimized matrix pencil storing both Schur forms and their conjugate transposes.

# Fields
- `A::AbstractMatrix{T}`: Schur form of A (upper triangular)
- `Ac::AbstractMatrix{T}`: Conjugate transpose of A
- `B::AbstractMatrix{T}`: Schur form of B (upper triangular)
- `Bc::AbstractMatrix{T}`: Conjugate transpose of B

This representation enables efficient triangular solves for pseudospectra computations,
particularly in GPU-accelerated inverse Lanczos methods.
"""
struct SchurMatrixPencil{T} <: AbstractMatrixPencil{T}
    A::AbstractMatrix{T}
    Ac::AbstractMatrix{T}
    B::AbstractMatrix{T}
    Bc::AbstractMatrix{T}
end

"""
    MatrixPencil(F::Schur)

Construct a `SchurMatrixPencil` from a standard Schur decomposition.

# Arguments
- `F::Schur`: Schur decomposition from `LinearAlgebra.schur(A)`

Creates a pencil `(A, I)` where A is in Schur form (upper triangular).
Both A and A' are stored for efficient triangular solves.
"""
function MatrixPencil(F::Schur)
    SchurMatrixPencil{eltype(F)}(Matrix{eltype(F)}(F.T), Matrix{eltype(F)}(F.T'), Matrix{eltype(F)}(I, size(F.T)), Matrix{eltype(F)}(I, size(F.T)))
end

"""
    MatrixPencil(F::GeneralizedSchur)

Construct a `SchurMatrixPencil` from a generalized Schur decomposition.

# Arguments
- `F::GeneralizedSchur`: Generalized Schur decomposition from `LinearAlgebra.schur(A, B)`

Creates a pencil `(S, T)` where both are in Schur form (upper triangular).
Both S, S', T, and T' are stored for efficient triangular solves.
"""
function MatrixPencil(F::GeneralizedSchur)
    SchurMatrixPencil{eltype(F)}(Matrix{eltype(F)}(F.S), Matrix{eltype(F)}(F.S'), Matrix{eltype(F)}(F.T), Matrix{eltype(F)}(F.T'))
end

"""
    Base.size(x::AbstractMatrixPencil)
    Base.size(x::AbstractMatrixPencil, i)

Get the size of the matrices in the matrix pencil.

Returns the dimensions of matrix A (which must match B).
"""
Base.size(x::AbstractMatrixPencil) = size(x.A)
Base.size(x::AbstractMatrixPencil, i) = size(x.A, i)

"""
    KernelAbstractions.get_backend(x::AbstractMatrixPencil)

Get the compute backend associated with the matrix pencil.

Returns the KernelAbstractions backend (CPU, CUDA, AMDGPU, etc.) of the underlying arrays.
"""
KernelAbstractions.get_backend(x::AbstractMatrixPencil) = get_backend(x.A)

# Enable automatic adaptation to different array types/backends
Adapt.@adapt_structure MatrixPencil
Adapt.@adapt_structure SchurMatrixPencil

"""
    validate(zg, A, B, γ=missing, δ=missing)
    validate(zg, P::AbstractMatrixPencil, γ=missing, δ=missing)

Validate inputs for pseudospectra computations.

# Arguments
- `zg`: Grid of complex shifts (must be non-empty)
- `A::AbstractMatrix`: First matrix in the pencil
- `B`: Second matrix in the pencil (or `UniformScaling`)
- `P::AbstractMatrixPencil`: Matrix pencil object
- `γ`: Optional scaling parameter for perturbations to A (default: missing)
- `δ`: Optional scaling parameter for perturbations to B (default: missing)

# Validation checks
- Grid `zg` is not empty
- Matrices `A` and `B` have the same dimensions (if `B ≠ I`)
- Normalization constraint `γ + δ = 1` holds (if scaling parameters are provided)

# Throws
- `AssertionError` if any validation check fails
"""
function validate(zg, A::AbstractMatrix, B, γ=missing, δ=missing)
    @assert !isempty(zg)
    if B != I
        @assert size(A) == size(B)
    end
    if !(ismissing(γ) && ismissing(δ))
        @assert γ + δ == 1
    end
end

function validate(zg, P::AbstractMatrixPencil, γ=missing, δ=missing)
    validate(zg, P.A, P.B, γ, δ)
end
