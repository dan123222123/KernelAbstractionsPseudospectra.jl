# pseudospectra calculations using the svd (LAPACK GESDD)
# svdpsa computations are multi-threaded, but run solely on the CPU

# In-place kernel behind ℂsvdpsa. A, B are `AbstractMatrix` (not `Matrix`) so a standard pencil's
# `Diagonal` identity B — and any lazy/wrapped operand — works: the body only needs `z*B - A`
# and `svdvals`, both generic.
function ℂsvdpsa!(
        srg::Matrix{S}, zg::Matrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
        γ::Real, δ::Real) where {T <: Complex{S}} where {S <: Real}
    Threads.@threads for Mind in eachindex(zg)
        srg[Mind] = svdvals(zg[Mind] * B - A)[end] / (γ + δ * abs(zg[Mind]))
    end
    return nothing
end
function ℂsvdpsa!(srg::Matrix{S}, zg::Matrix{T}, P::AbstractMatrixPencil,
        γ::Real, δ::Real) where {T <: Complex{S}} where {S <: Real}
    ℂsvdpsa!(srg, zg, P.A, P.B, γ, δ)
    return nothing
end

"""
    ℂsvdpsa(zg, A, B=I, γ=1, δ=0) -> Matrix{real(T)}
    ℂsvdpsa(zg, P::AbstractMatrixPencil, γ=1, δ=0)

Compute the (γ,δ)-pseudospectral value of the pencil ``zB - A`` at each `z ∈ zg` by
dense SVD: ``σ_min(zB − A) / (γ + δ|z|)``. Multi-threaded, CPU only.

This is the ε-level function of the Frayssé–Gueury–Nicoud–Toumazou (γ,δ)-
pseudospectrum: `z ∈ σ_ε^{(γ,δ)}` iff this value is `< ε`, where `γ`,`δ ≥ 0`
(not both zero) scale perturbations to `A` and `B` respectively (a pencil
perturbation `zΔ − Γ` has norm `≤ ε(γ + δ|z|)`). Common choices are `(1,0)`
(standard pseudospectrum, `= σ_min(zB − A)`, the default) and `(1,1)`. Returns the
field transposed relative to `zg` (rows follow the imaginary axis), ready for
[`psaplot`](@ref); cross-check for [`ihlpsa`](@ref).
"""
function ℂsvdpsa(
        zg::Matrix{T}, A::AbstractMatrix{T}, B = I, γ = 1, δ = 0) where {T <: Complex}
    validate(zg, A, B, γ, δ)
    if B isa UniformScaling
        B = Matrix{T}(B, size(A))
    end
    srg = Matrix{real(T)}(undef, size(zg))
    ℂsvdpsa!(srg, zg, A, B, γ, δ)
    return permutedims(srg)
end
function ℂsvdpsa(zg::Matrix{T}, P::AbstractMatrixPencil, γ = 1, δ = 0) where {T <: Complex}
    ℂsvdpsa(zg, P.A, P.B, γ, δ)
end

## structured (real) pseudospectra ##

# 2-norm of the smallest *real* E making z an eigenvalue of (A, B) — the real stability radius
# (Qiu et al., 1995) at s = z; verbatim from §50 of Trefethen & Embree, "Spectra and
# Pseudospectra" (2005).
function distzeigAB(z::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Complex}
    R = (z * B - A) \ Matrix{T}(I, size(A))
    Rr, Ri = real(R), imag(R)
    f = γ -> svdvals([Rr -γ*Ri; inv(γ)*Ri Rr])[2]
    return 1 / optimize(f, eps(real(T)), one(real(T))).minimum
end

# In-place kernel behind ℝsvdpsa: distzeigAB at every grid point.
function ℝsvdpsa!(srg::Matrix{S}, zg::AbstractMatrix{T}, A::AbstractMatrix{T},
        B::AbstractMatrix{T}) where {T <: Complex{S}} where {S <: Real}
    Threads.@threads for i in eachindex(zg)
        srg[i] = distzeigAB(zg[i], A, B)
    end
    return nothing
end
function ℝsvdpsa!(srg::Matrix{S}, zg::AbstractMatrix{T},
        P::AbstractMatrixPencil) where {T <: Complex{S}} where {S <: Real}
    ℝsvdpsa!(srg, zg, P.A, P.B)
    return nothing
end

"""
    ℝsvdpsa(zg, A, B=I) -> Matrix{real(T)}
    ℝsvdpsa(zg, P::AbstractMatrixPencil)

Compute the real structured pseudospectral value at each `z ∈ zg`: the 2-norm of the
smallest *real* perturbation `E` making `z` an eigenvalue of the pencil (the real
stability radius of Qiu et al., 1995, evaluated pointwise). Multi-threaded, CPU only.
Returns the field transposed relative to `zg` (rows follow the imaginary axis), ready
for [`psaplot`](@ref).
"""
function ℝsvdpsa(zg::Matrix{T}, A::Matrix{T}, B = I) where {T <: Complex}
    validate(zg, A, B)
    if B isa UniformScaling
        B = Matrix{T}(B, size(A))
    end
    srg = Matrix{real(T)}(undef, size(zg))
    ℝsvdpsa!(srg, zg, A, B)
    return permutedims(srg)
end
function ℝsvdpsa(zg::Matrix{T}, P::AbstractMatrixPencil) where {T <: Complex}
    ℝsvdpsa(zg, P.A, P.B)
end
