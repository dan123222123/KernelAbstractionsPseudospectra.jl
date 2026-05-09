# Sanity checks on ℝsvdpsa (structured/real pseudospectra).
#
# distzeigAB uses Optim per grid point, so we keep the grid tiny (4×4) to
# stay under ~1s. We don't have an external reference for the real radii;
# instead we check structural invariants:
#
#  - Output is finite and real.
#  - Real perturbations are MORE restrictive than complex ones, so the
#    minimum-norm real perturbation distance is ≥ the complex one. Hence
#    psa_real ≥ psa_complex pointwise (modulo a small numerical slack).

using Test, KAPseudospectra, LinearAlgebra
using Random

@testset "ℝsvdpsa basic" begin
    Random.seed!(0xBABE)
    m = 8
    A = randn(ComplexF64, m, m)
    P = MatrixPencil(A)
    gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))

    psa_real = ℝsvdpsa(zg, P)
    psa_complex = ℂsvdpsa(zg, P)

    @test all(isfinite, psa_real)
    @test eltype(psa_real) <: Real
    # Real ≥ complex (pointwise) — real perts more restrictive ⇒ larger radius.
    # Allow 1e-10 absolute slack for Optim convergence noise.
    @test all(psa_real .>= psa_complex .- 1e-10)
end
