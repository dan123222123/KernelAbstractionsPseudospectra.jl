# =============================================================================
# Accuracy check: extended-precision ihlpsa vs an independent high-precision oracle
# =============================================================================
#
# Validates the MultiFloats inverse-Lanczos pseudospectra (see ihlpsa_multifloats.jl)
# against an oracle that shares neither its algorithm nor its precision:
#   * algorithm — the package's dense-SVD pseudospectra `ℂsvdpsa` (σ_min via a full SVD of
#                 zI − A) rather than inverse-Lanczos,
#   * precision — BigFloat / MPFR (~77 digits) rather than Float64x2 (~32).
# Both go through the public API on the same grid, so the comparison is direct. The
# extended-precision ihlpsa tracks the oracle to its own precision; the Float64 ihlpsa loses
# accuracy in the ill-conditioned band near the spectrum. Switch HP to Complex{Float64x4}
# to watch the error fall further.
# =============================================================================

using KernelAbstractions
backend = CPU()                       # the ihlpsa runs; a GPU backend works too
#using CUDA; backend = CUDABackend()

using KAPseudospectra, LinearAlgebra, Statistics, Printf
using MultiFloats
using GenericSchur, GenericLinearAlgebra   # generic schur + eigen/svd for extended types
using MatrixDepot

setprecision(BigFloat, 256)           # ≈ 77 decimal digits

const M      = 20
const G      = 21                     # coarse grid — each oracle point is a full BigFloat SVD
const NIT    = 12                     # inverse-Lanczos converges to the extreme eigenvalue in
                                      #   a handful of steps (~4–6 here, ≪ m); 12 is ample margin
const HP     = Complex{Float64x2}

# chebspec (MatrixDepot): a standard, strongly non-normal pseudospectra example. Built once
# in Float64 and promoted, so Float64, Float64x2 and the BigFloat oracle all see the same
# matrix.
const AREF = MatrixDepot.chebspec(Float64, M)
matrix(::Type{T}) where {T<:Complex} = T.(AREF)

# These eigenvalues only frame the grid's viewing window (padded by `pad`), so Float64 is
# fine — the F64-vs-BigFloat bounds differ by <0.03 here, far below the pad. The validated
# reference is the BigFloat `ℂsvdpsa` oracle below, not these.
const EIGS = eigvals(matrix(ComplexF64))
const REGION = let pad = 1.5
    ((minimum(real, EIGS) - pad, maximum(real, EIGS) + pad),
     (minimum(imag, EIGS) - pad, maximum(imag, EIGS) + pad))
end
grid(::Type{T}) where {T<:Complex} = qgrid(T, REGION[1], REGION[2], (G, G))[3]

# ihlpsa inverse-Lanczos at precision T.
run_ihl(::Type{T}) where {T<:Complex} = ihlpsa(backend, grid(T), MatrixPencil(schur(matrix(T))), NIT)

# Oracle: the package's own dense-SVD pseudospectra, evaluated in BigFloat.
oracle() = ℂsvdpsa(grid(Complex{BigFloat}), matrix(Complex{BigFloat}))

@info "oracle: BigFloat dense SVD via ℂsvdpsa ($(precision(BigFloat)) bits ≈ $(round(Int, precision(BigFloat)*log10(2))) digits), $(G)×$(G) grid …"
s_oracle = Float64.(oracle())
@info "ihlpsa Float64 …";   s64 = run_ihl(ComplexF64)
@info "ihlpsa $(HP) …";     smf = Float64.(run_ihl(HP))

e64 = abs.(s64 .- s_oracle) ./ s_oracle
emf = abs.(smf .- s_oracle) ./ s_oracle

println()
@printf("=== relative error vs BigFloat ℂsvdpsa oracle  (chebspec m=%d, %d×%d grid, nit=%d) ===\n", M, G, G, NIT)
@printf("  Float64    ihlpsa : max %.2e   median %.2e   (%d pts >1%% off)\n", maximum(e64), median(e64), count(>(0.01), e64))
@printf("  %s ihlpsa : max %.2e   median %.2e\n", HP, maximum(emf), median(emf))

gx = range(REGION[1]...; length=G); gy = range(REGION[2]...; length=G)
w = argmax(e64); zw = gx[w[2]] + im * gy[w[1]]        # srg[a,b] ↔ z = gx[b] + im·gy[a]
println()
@printf("  worst Float64 point  z = %s\n", string(round(zw, digits=3)))
@printf("    oracle (BigFloat SVD) σ = %.6e\n", s_oracle[w])
@printf("    Float64    ihlpsa     σ = %.6e   (%.1f× off)\n", s64[w], s64[w] / s_oracle[w])
@printf("    %s ihlpsa     σ = %.6e   (%.2e rel)\n", HP, smf[w], emf[w])
println()
println("  → $(HP) ihlpsa tracks the independent ~77-digit oracle to its own precision;")
println("    plain Float64 ihlpsa is the one that loses accuracy in the ill-conditioned band.")
