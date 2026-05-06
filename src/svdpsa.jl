# Pseudospectra calculations using the SVD (LAPACK GESDD)
# svdpsa computations are multi-threaded, but run solely on the CPU

"""
    ℂsvdpsa!(srg, zg, A::Matrix, B::Matrix, γ, δ)
    ℂsvdpsa!(srg, zg, P::AbstractMatrixPencil, γ=1, δ=0)

Compute unstructured (complex) pseudospectra using singular value decomposition.

This function computes the smallest singular value of `zB - A` at each point `z` in the grid `zg`,
scaled by the normalization factor `(γ + δ|z|)`. The result is the resolvent norm, which gives
the distance to singularity in the operator norm.

# Arguments
- `srg::Matrix{S}`: Output matrix for resolvent norms (real-valued, modified in-place)
- `zg::Matrix{T}`: Grid of complex shift values where pseudospectra are computed
- `A::Matrix{T}`: First matrix in the pencil (complex-valued)
- `B::Matrix{T}`: Second matrix in the pencil (complex-valued)
- `P::AbstractMatrixPencil`: Matrix pencil object containing A and B
- `γ::Real`: Scaling weight for perturbations to A (default: 1)
- `δ::Real`: Scaling weight for perturbations to B (default: 0)

# Details
At each grid point `z ∈ zg`, computes:
```
srg[z] = (γ + δ|z|) * σₘᵢₙ(zB - A)
```
where `σₘᵢₙ` denotes the smallest singular value.

The computation uses LAPACK's GESDD for SVD and is multi-threaded across grid points.
The constraint `γ + δ = 1` should hold for proper normalization.

# Implementation
This is an in-place function that modifies `srg`. Multi-threading uses `Threads.@threads`
for parallel computation across grid points.
"""
function ℂsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, A::Matrix{T}, B::Matrix{T}, γ::Real, δ::Real) where {T<:Complex{S}} where {S<:Real}
    Threads.@threads for Mind in eachindex(zg)
        srg[Mind] = (γ + δ * abs(zg[Mind])) * svdvals(zg[Mind] * B - A)[end]
    end
    return nothing
end
function ℂsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, P::AbstractMatrixPencil, γ::Real, δ::Real) where {T<:Complex{S}} where {S<:Real}
    ℂsvdpsa!(srg, zg, P.A, P.B, γ, δ)
    return nothing
end

"""
    ℂsvdpsa(zg, A, B=I, γ=1, δ=0)
    ℂsvdpsa(zg, P::AbstractMatrixPencil, γ=1, δ=0)

Compute unstructured (complex) pseudospectra and return result matrix.

This is the allocating version of `ℂsvdpsa!` that validates inputs, allocates the output matrix,
and returns the transposed result for standard plotting conventions.

# Arguments
- `zg::Matrix{T}`: Grid of complex shift values (size: nx × ny)
- `A::AbstractMatrix{T}`: First matrix in the pencil (complex-valued)
- `B`: Second matrix in the pencil (default: `I` for identity)
- `P::AbstractMatrixPencil`: Matrix pencil object
- `γ::Real`: Scaling weight for perturbations to A (default: 1)
- `δ::Real`: Scaling weight for perturbations to B (default: 0)

# Returns
- `Matrix{real(T)}`: Transposed resolvent norm matrix (size: ny × nx for plotting)

# Examples
```julia
using LinearAlgebra
m = 50
A = randn(ComplexF64, m, m)
gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (100, 100))
P = MatrixPencil(schur(A))

# Standard eigenvalue problem pseudospectra
psa = ℂsvdpsa(zg, P)

# Generalized eigenvalue problem
B = randn(ComplexF64, m, m)
psa = ℂsvdpsa(zg, A, B, 0.5, 0.5)
```

See also: [`ℂsvdpsa!`](@ref), [`ℝsvdpsa`](@ref)
"""
function ℂsvdpsa(zg::Matrix{T}, A::AbstractMatrix{T}, B=I, γ=1, δ=0) where {T<:Complex}
    validate(zg, A, B, γ, δ)
    if B isa UniformScaling
        B = Matrix{T}(B, size(A))
    end
    srg = Matrix{real(T)}(undef, size(zg))
    ℂsvdpsa!(srg, zg, A, B, γ, δ)
    return srg'
end
function ℂsvdpsa(zg::Matrix{T}, P::AbstractMatrixPencil, γ=1, δ=0) where {T<:Complex}
    ℂsvdpsa(zg, P.A, P.B, γ, δ)
end

## Structured (real) pseudospectra ##

"""
    distzeigAB(z, A, B) where {T<:Complex}

Compute the 2-norm of the smallest real perturbation making z an eigenvalue.

This function computes the structured real stability radius, which is the norm of the smallest
_real_ matrix perturbation E such that z becomes an eigenvalue of the perturbed pencil (A+E, B).

# Arguments
- `z::T`: Complex shift value
- `A::AbstractMatrix{T}`: First matrix in the pencil (complex-valued)
- `B::AbstractMatrix{T}`: Second matrix in the pencil (complex-valued)

# Returns
- `Real`: Minimum 2-norm of real perturbation matrix E such that `det(zB - (A + E)) = 0`

# Algorithm
Implements the optimization-based algorithm from §50 of Trefethen & Embree (2005), "Spectra and
Pseudospectra: The Behavior of Nonnormal Matrices and Operators" (SAP2005). The method computes:

```
R = (zB - A)⁻¹
min_{γ ∈ (0,1]} σ₂([Re(R)  -γ Im(R); γ⁻¹ Im(R)  Re(R)])⁻¹
```

Uses `Optim.jl` to optimize over the parameter γ.

# References
- Qiu, L., Bernhardsson, B., Rantzer, A., Davison, E.J., Young, P.M., & Doyle, J.C. (1995).
  "A formula for computation of the real stability radius." Automatica, 31(6), 879-890.
- Trefethen, L.N. & Embree, M. (2005). "Spectra and Pseudospectra", Princeton University Press.

See also: [`ℝsvdpsa`](@ref), [`ℂsvdpsa`](@ref)
"""
function distzeigAB(z::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    R = inv(z * B - A)
    f = γ -> svdvals([real(R) -γ*imag(R); inv(γ)*imag(R) real(R)])[2]
    return 1 / optimize(f, eps(real(T)), one(real(T))).minimum
end

"""
    ℝsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex{S}} where {S<:Real}
    ℝsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, P::AbstractMatrixPencil) where {T<:Complex{S}} where {S<:Real}

Compute structured (real) pseudospectra using optimization-based method.

This function computes the minimum norm of real-valued perturbations that make `zB - A` singular
at each point `z` in the grid `zg`. Unlike the unstructured (complex) case, this restricts
perturbations to be real matrices, which is relevant for applications with real-valued systems.

# Arguments
- `srg::Matrix{S}`: Output matrix for real stability radii (real-valued, modified in-place)
- `zg::AbstractMatrix{T}`: Grid of complex shift values
- `A::AbstractMatrix{T}`: First matrix in the pencil (complex-valued)
- `B::AbstractMatrix{T}`: Second matrix in the pencil (complex-valued)
- `P::AbstractMatrixPencil`: Matrix pencil object containing A and B

# Details
At each grid point `z ∈ zg`, computes the smallest real perturbation norm using `distzeigAB`.
The computation is multi-threaded across grid points using `Threads.@threads`.

This structured variant typically produces smaller stability radii than the unstructured version
since real perturbations are more restrictive than complex perturbations.

# Implementation
This is an in-place function that modifies `srg`. Each grid point is processed independently
by calling `distzeigAB(z, A, B)`.

See also: [`distzeigAB`](@ref), [`ℝsvdpsa`](@ref), [`ℂsvdpsa!`](@ref)
"""
function ℝsvdpsa!(srg::Matrix{S}, zg::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex{S}} where {S<:Real}
    Threads.@threads for i in eachindex(zg)
        srg[i] = distzeigAB(zg[i], A, B)
    end
    return nothing
end
function ℝsvdpsa!(srg::Matrix{S}, zg::AbstractMatrix{T}, P::AbstractMatrixPencil) where {T<:Complex{S}} where {S<:Real}
    ℝsvdpsa!(srg, zg, P.A, P.B)
    return nothing
end

"""
    ℝsvdpsa(zg, A, B=I)
    ℝsvdpsa(zg, P::AbstractMatrixPencil)

Compute structured (real) pseudospectra and return result matrix.

This is the allocating version of `ℝsvdpsa!` that validates inputs, allocates the output matrix,
and returns the transposed result for standard plotting conventions.

# Arguments
- `zg::Matrix{T}`: Grid of complex shift values (size: nx × ny)
- `A::Matrix{T}`: First matrix in the pencil (complex-valued)
- `B`: Second matrix in the pencil (default: `I` for identity)
- `P::AbstractMatrixPencil`: Matrix pencil object

# Returns
- `Matrix{real(T)}`: Transposed real stability radius matrix (size: ny × nx for plotting)

# Examples
```julia
using LinearAlgebra
m = 50
A = randn(ComplexF64, m, m)
gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (100, 100))
P = MatrixPencil(schur(A))

# Compute structured pseudospectra
psa_real = ℝsvdpsa(zg, P)

# Compare with unstructured pseudospectra
psa_complex = ℂsvdpsa(zg, P)

# Real perturbations give larger ε-pseudospectra (smaller stability radii)
@assert all(psa_real .>= psa_complex)
```

# Notes
The structured (real) pseudospectra represents physically realistic perturbations for systems
described by real matrices, even when analyzed in the complex plane. It typically shows larger
ε-pseudospectra compared to the unstructured case.

See also: [`ℝsvdpsa!`](@ref), [`ℂsvdpsa`](@ref), [`distzeigAB`](@ref)
"""
function ℝsvdpsa(zg::Matrix{T}, A::Matrix{T}, B=I) where {T<:Complex}
    validate(zg, A, B)
    if B isa UniformScaling
        B = Matrix{T}(B, size(A))
    end
    srg = Matrix{real(T)}(undef, size(zg))
    ℝsvdpsa!(srg, zg, A, B)
    return srg'
end
function ℝsvdpsa(zg::Matrix{T}, P::AbstractMatrixPencil) where {T<:Complex}
    ℝsvdpsa(zg, P.A, P.B)
end