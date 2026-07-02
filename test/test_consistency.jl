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

# Explicit x₀ so cross-backend comparisons can match modulo floating-point
# accumulation order. This is the adaptive driver's internal deterministic seeded
# start vector, reused (un-exported) so the test x₀ can't drift from the driver's default.
const _seeded_x₀ = KAPseudospectra._adaptive_x₀

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
        s_ihl = ihlpsa(CPU(), zg, P, nit; γ=γ, δ=δ, x₀=x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-4 : 1e-10
        @test isapprox(s_ihl, s_svd; rtol=rtol)
    end
end

@testset "ihlpsa adaptive vs ℂsvdpsa" begin
    # Adaptive (per-point hybrid) driver: must match the dense SVD oracle and the
    # fixed-nit control (same algorithm, same x₀, run to the cap), and must stop
    # before the iteration cap on an easy grid.
    Random.seed!(0xAD0F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))

        x₀ = _seeded_x₀(T, m, 0xBEEF)
        nit_max = 8 * ceil(Int, log2(m))                  # adaptive cap (= default)
        s_fixed = ihlpsa(CPU(), zg, P, nit_max; x₀=x₀)    # over-converged control
        # Public `ihlpsa(…; …)` returns only σ; the internal driver also returns the
        # per-point convergence depth grid this testset asserts on.
        s_adp, nit_grid = KAPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P)

        # Comparison tol ~10× the adaptive stopping rtol (default 1e-4 F32 /
        # 1e-6 F64) to absorb the successive-change criterion's slack on slow points.
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol=rtol)           # vs dense SVD oracle
        @test isapprox(s_adp, s_fixed; rtol=rtol)         # vs fixed-nit control
        @test size(nit_grid) == size(s_adp)               # per-point depth grid
        @test maximum(nit_grid) < nit_max                  # genuinely stopped early
    end
end

@testset "ihlpsa adaptive generalized pencil (B≠I)" begin
    # B ≠ I path with structured weights γ,δ (validate requires γ+δ≈1).
    Random.seed!(0xAD1F)
    for T in (ComplexF32, ComplexF64)
        m = 16
        A = randn(T, m, m)
        B = randn(T, m, m) + T(5) * I
        P = MatrixPencil(A, B)
        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        γ, δ = 0.5, 0.5

        x₀ = _seeded_x₀(T, m, 0xFEED)
        s_adp, nit_grid = KAPseudospectra._ihlpsa_adaptive(CPU(), zg, P; γ=γ, δ=δ, x₀=x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol=rtol)
        @test maximum(nit_grid) < 8 * ceil(Int, log2(m))
    end
end

@testset "ihlpsa adaptive multi-batch" begin
    # The per-point hybrid gathers surviving points' resident Lanczos state across
    # chunks; a small zpd forces many batches sharing that machinery and must agree
    # with the single-batch default (batching only changes stopping granularity).
    Random.seed!(42)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_one = ihlpsa(CPU(), zg, P; x₀=x₀)
        s_many = ihlpsa(CPU(), zg, P; x₀=x₀, zpd=37)      # 144 = 3·37 + 33 → 4 batches
        @test isapprox(s_one, s_many; rtol=(T == ComplexF32 ? 1e-3 : 1e-5))
    end
end

@testset "ihlpsa adaptive nit_max cap" begin
    # With nconfirm=2 a point needs ≥2 real checkpoints to retire, which nit_max=3
    # (chunks at 2,3) cannot provide — so every point hits the cap. The driver must
    # @warn, set every point's depth to nit_max, and fall back to the deepest-chunk
    # σ (≈ the fixed run at the cap), not error or return garbage.
    Random.seed!(0xCA9F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (8, 8))
        x₀ = _seeded_x₀(T, m, 0xBEEF)
        nit_cap = 3

        s_adp, nit_grid = @test_logs (:warn,) match_mode = :any KAPseudospectra._ihlpsa_adaptive(
            CPU(), zg, P; x₀=x₀, nit_max=nit_cap, rtol=1e-12)
        s_fix = ihlpsa(CPU(), zg, P, nit_cap; x₀=x₀)

        @test all(==(nit_cap), nit_grid)
        @test all(isfinite, s_adp)
        @test isapprox(s_adp, s_fix; rtol=(T == ComplexF32 ? 1e-4 : 1e-10))
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

# Cross-backend adaptive: same problem, same fixed x₀. Values agree to the
# adaptive stopping tolerance; the converged nit agrees within one chunk
# (cross-backend reduction-order differences can flip a borderline point across
# the rtol boundary at a chunk edge, changing where it stops).
function test_adaptive_backend(backend; types=(ComplexF32, ComplexF64))
    @testset "ihlpsa adaptive: CPU vs $(backend)" begin
        Random.seed!(0xADDA)
        for T in types
            m = 32
            A = randn(T, m, m)
            P = MatrixPencil(A)
            # A deliberately larger grid (40x40, not a token 16x16): the on-device
            # survivor gather runs once per retirement round, and a too-small grid
            # retires in a couple of rounds whose index-sets can miss a backend's
            # indexing bugs. oneAPI in particular miscompiles a first-axis fancy
            # index on the 3-D Qv backing — latent at 16x16, exposed here — which
            # `_qv_gather!` fixes.
            gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (40, 40))

            x₀ = _seeded_x₀(T, m, 0xACED)
            rtol = T == ComplexF32 ? 1e-3 : 1e-5

            # Adaptive (per-point hybrid) across the device fan-out: exercises the
            # on-device state gather (custom `_qv_gather!` kernel) and multi-device partition.
            s_cpu, grid_cpu = KAPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
            s_gpu, grid_gpu = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; x₀=x₀)
            @test isapprox(s_cpu, s_gpu; rtol=rtol)
            @test abs(maximum(grid_cpu) - maximum(grid_gpu)) <= ceil(Int, log2(m))   # within one chunk
        end
    end
end
