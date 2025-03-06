# pseudospectra calculations using the svd (LAPACK GESDD)
# svdpsa computations are multi-threaded, but run solely on the CPU

"""
    ℂsvdpsa!(srg, zg, A::Matrix, B::Matrix, γ, δ)
    ℂsvdpsa!(srg, zg, P::AbstractMatrixPencil, γ=1, δ=0)

Compute matrix of resolvent norms for `zB-A` using the SVD.

`z` is an element of `zg`.
Scaling for perturbations to and `A` and `B` is given by γ, δ ∈ [0, 1], respectively.
Resulting resolvent grid is stored in `srg`.
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
    ℂsvdpsa(zg, P, γ=1, δ=0)

Validates inputs and calls `ℂsvdpsa!`.
Allocates and returns srg instead of modifying in-place.
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

## structured (real) pseudospectra ##

"""
    distzeigAB(z, A, B) where {T<:Complex}

Computes the 2-norm of the smallest _real_ matrix E that makes z an eigenvalue of (A, B)
This is the real stability radii (see Qiu et al., 1995) computed for s ∈ {z}.
The algorithm is essentially verbatim from §50 of SAP2005.
"""
function distzeigAB(z::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex}
    R = inv(z * B - A)
    f = γ -> svdvals([real(R) -γ*imag(R); inv(γ)*imag(R) real(R)])[2]
    return 1 / optimize(f, eps(real(T)), one(real(T))).minimum
end

"""
    ℝsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Complex{S}} where {S<:Real}
    ℝsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, P::AbstractMatrixPencil) where {T<:Complex{S}} where {S<:Real}

Compute matrix of minimum normed _real_ perturbations which make `zB - A` singular.
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
    ℝsvdpsa(zg, A, B)
    ℝsvdpsa(zg, P)

Validates inputs and calls `ℝsvdpsa!`.
Allocates and returns srg instead of modifying in-place.
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