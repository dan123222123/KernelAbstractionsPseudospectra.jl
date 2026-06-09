# Tests that exercise the actual algorithmic correctness of ihlpsa, not just
# regression vs an external eigtool reference. Three flavors:
#
#  1. ihlpsa vs ℂsvdpsa — at fixed (small) m the inverse-Lanczos result should
#     match the dense SVD baseline within Lanczos convergence tolerance.
#  2. Generalized pencil — exercises the GeneralizedSchur ctor path that
#     parter16 doesn't touch.
#  3. Cross-backend — same problem, same x₀, must agree to ~1e-12 across
#     CPU and the requested GPU backend.

using Test, KAPseudospectra, KernelAbstractions, LinearAlgebra
using Random

# Use a non-default explicit x₀ so cross-backend comparisons can match
# bit-equivalent (modulo floating-point accumulation order).
function _seeded_x₀(::Type{T}, m, seed) where {T<:Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    x ./ norm(x)
end

@testset "ihlpsa vs ℂsvdpsa" begin
    # Standard pencil, dense random A. nit > log2(m) by a healthy margin to
    # over-converge so we can use a tight tolerance.
    Random.seed!(0xCAFE)
    for T in (ComplexF32, ComplexF64)
        m = 24
        nit = 2 * ceil(Int, log2(m))
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))

        x₀ = _seeded_x₀(T, m, 0xBEEF)
        s_ihl = ihlpsa(CPU(), zg, P, nit; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P)
        # Tolerance: F32 ~ 1e-4 (Lanczos + F32 roundoff in σ_min); F64 ~ 1e-10.
        rtol = T == ComplexF32 ? 1e-4 : 1e-10
        @test isapprox(s_ihl, s_svd; rtol=rtol)
    end
end

@testset "generalized pencil round-trip" begin
    # A and B both random, B made diagonally-dominant so it's well-conditioned.
    Random.seed!(0xC0DE)
    for T in (ComplexF32, ComplexF64)
        m = 16
        nit = 2 * ceil(Int, log2(m))
        A = randn(T, m, m)
        B = randn(T, m, m) + T(5) * I
        P = MatrixPencil(A, B)
        @test P isa KAPseudospectra.SchurMatrixPencil

        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        γ, δ = T <: Complex ? (0.5, 0.5) : (0.5, 0.5)

        x₀ = _seeded_x₀(T, m, 0xFEED)
        s_ihl = ihlpsa(CPU(), zg, P, nit, γ, δ; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-4 : 1e-10
        @test isapprox(s_ihl, s_svd; rtol=rtol)
    end
end

# Cross-backend: same problem, same explicit x₀, results agree element-wise.
# `types` lets FP64-less backends (Intel iGPUs) restrict to ComplexF32.
function test_cross_backend(backend; types=(ComplexF32, ComplexF64))
    @testset "cross-backend: CPU vs $(backend)" begin
        Random.seed!(0xDADA)
        for T in types
            m = 32
            nit = ceil(Int, log2(m))
            A = randn(T, m, m)
            P = MatrixPencil(A)
            gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (16, 16))

            x₀ = _seeded_x₀(T, m, 0xACED)
            s_cpu = ihlpsa(CPU(), zg, P, nit; x₀=x₀)
            s_gpu = ihlpsa(backend, zg, P, nit; x₀=x₀)
            # Different reduction orders across backends can perturb σ_min by
            # ~few×eps(T)·κ. F32: 1e-3 generous; F64: 1e-10.
            rtol = T == ComplexF32 ? 1e-3 : 1e-10
            @test isapprox(s_cpu, s_gpu; rtol=rtol)
        end
    end
end
