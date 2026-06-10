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

@testset "ihlpsa_adaptive vs ℂsvdpsa" begin
    # Tier-1 adaptive driver: must match the dense SVD oracle AND the fixed-nit
    # control (same algorithm, same x₀, run to the cap), and must stop before
    # the iteration cap on an easy grid.
    Random.seed!(0xAD0F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))

        x₀ = _seeded_x₀(T, m, 0xBEEF)
        nit_max = 8 * ceil(Int, log2(m))                  # adaptive cap (= default)
        s_fixed = ihlpsa(CPU(), zg, P, nit_max; x₀=x₀)    # over-converged control
        s_adp, nit_used = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P)

        # Comparison tol ~10× the adaptive stopping rtol (default 1e-4 F32 /
        # 1e-6 F64) to absorb the successive-change criterion's slack on slow points.
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol=rtol)           # vs dense SVD oracle
        @test isapprox(s_adp, s_fixed; rtol=rtol)         # vs fixed-nit control
        @test nit_used < nit_max                           # genuinely stopped early
    end
end

@testset "ihlpsa_adaptive generalized pencil (B≠I)" begin
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
        s_adp, nit_used = ihlpsa_adaptive(CPU(), zg, P, γ, δ; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol=rtol)
        @test nit_used < 8 * ceil(Int, log2(m))
    end
end

@testset "ihlpsa_adaptive compaction (Tier 2)" begin
    # compact=true (per-point pruning) must produce the SAME pseudospectrum as
    # compact=false (the pruning is pure work-reduction) and match the dense SVD
    # oracle. This is the regression test for the column-major pack + back-map
    # across the internal (nx,ny) ↔ returned (ny,nx) transpose.
    Random.seed!(0xAD2F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_global, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)                # Tier 1
        s_compact, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, compact=true) # Tier 2
        s_svd = ℂsvdpsa(zg, P)

        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test size(s_compact) == size(s_svd)              # shape/orientation preserved
        @test isapprox(s_compact, s_svd; rtol=rtol)       # vs dense SVD oracle
        @test isapprox(s_compact, s_global; rtol=rtol)    # Tier 2 == Tier 1
    end

    # B ≠ I pencil with structured weights γ,δ.
    Random.seed!(0xAD3F)
    for T in (ComplexF32, ComplexF64)
        m = 16
        A = randn(T, m, m)
        B = randn(T, m, m) + T(5) * I
        P = MatrixPencil(A, B)
        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        γ, δ = 0.5, 0.5
        x₀ = _seeded_x₀(T, m, 0xFEED)

        s_compact, _ = ihlpsa_adaptive(CPU(), zg, P, γ, δ; x₀=x₀, compact=true)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_compact, s_svd; rtol=rtol)
    end
end

@testset "ihlpsa_adaptive resumable (Tier 3)" begin
    # resumable=true continues ONE Lanczos run across chunks instead of
    # restarting, with convergence checkpoints at the same nit values as Tier 1.
    # On CPU (single batch, deterministic kernels) it therefore reproduces
    # Tier 1's exact convergence path: same nit_used, near-identical σ (same
    # floating-point operation sequence — only the bookkeeping differs).
    Random.seed!(0xAD4F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_global, n_global = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
        s_res, n_res = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, resumable=true)
        s_svd = ℂsvdpsa(zg, P)

        @test n_res == n_global                           # identical convergence path
        @test isapprox(s_res, s_global; rtol=(T == ComplexF32 ? 1e-6 : 1e-12))
        @test isapprox(s_res, s_svd; rtol=(T == ComplexF32 ? 1e-3 : 1e-5))
    end

    # B ≠ I pencil with structured weights γ,δ.
    Random.seed!(0xAD5F)
    for T in (ComplexF32, ComplexF64)
        m = 16
        A = randn(T, m, m)
        B = randn(T, m, m) + T(5) * I
        P = MatrixPencil(A, B)
        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        x₀ = _seeded_x₀(T, m, 0xFEED)

        s_res, _ = ihlpsa_adaptive(CPU(), zg, P, 0.5, 0.5; x₀=x₀, resumable=true)
        s_svd = ℂsvdpsa(zg, P, 0.5, 0.5)
        @test isapprox(s_res, s_svd; rtol=(T == ComplexF32 ? 1e-3 : 1e-5))
    end

    # Forced multi-batch: a small zpd splits the grid into many batches sharing
    # one resident workspace (sequential, per-batch chunk loops, stale-state
    # reuse across batches). Must agree with the single-batch default. zpd=37
    # gives 4 batches incl. a partial last one (144 = 3·37 + 33).
    Random.seed!(0xAD6F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_one, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, resumable=true)
        s_many, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, resumable=true, zpd=37)
        # Identical per-point arithmetic; batching only changes stopping
        # granularity (per-batch vs whole grid), so points in early-converging
        # batches may stop at a shallower nit — compare at the convergence tol.
        @test isapprox(s_one, s_many; rtol=(T == ComplexF32 ? 1e-3 : 1e-5))
    end
end

@testset "ihlpsa_adaptive hybrid (compact + resumable)" begin
    # The hybrid retires converged points per chunk AND continues survivors'
    # Lanczos state without restart (state gather). Per-point stopping at a
    # finer granularity (default nit_chunk=2) means each point's σ is taken at
    # its own confirmed-converged depth — equal to the other tiers within the
    # convergence tolerance, and to the SVD oracle.
    Random.seed!(0xAD7F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_hyb, n_hyb = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, compact=true, resumable=true)
        s_glob, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
        s_svd = ℂsvdpsa(zg, P)

        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_hyb, s_svd; rtol=rtol)           # vs dense SVD oracle
        @test isapprox(s_hyb, s_glob; rtol=rtol)          # vs Tier 1
        @test n_hyb <= 8 * ceil(Int, log2(m))             # within budget

        # Forced multi-batch: gather logic must survive batch boundaries.
        s_many, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, compact=true, resumable=true, zpd=37)
        @test isapprox(s_hyb, s_many; rtol=rtol)
    end

    # B ≠ I pencil with structured weights γ,δ.
    Random.seed!(0xAD8F)
    for T in (ComplexF32, ComplexF64)
        m = 16
        A = randn(T, m, m)
        B = randn(T, m, m) + T(5) * I
        P = MatrixPencil(A, B)
        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        x₀ = _seeded_x₀(T, m, 0xFEED)

        s_hyb, _ = ihlpsa_adaptive(CPU(), zg, P, 0.5, 0.5; x₀=x₀, compact=true, resumable=true)
        s_svd = ℂsvdpsa(zg, P, 0.5, 0.5)
        @test isapprox(s_hyb, s_svd; rtol=(T == ComplexF32 ? 1e-3 : 1e-5))
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
    @testset "ihlpsa_adaptive: CPU vs $(backend)" begin
        Random.seed!(0xADDA)
        for T in types
            m = 32
            A = randn(T, m, m)
            P = MatrixPencil(A)
            gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (16, 16))

            x₀ = _seeded_x₀(T, m, 0xACED)
            rtol = T == ComplexF32 ? 1e-3 : 1e-5

            s_cpu, n_cpu = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀)
            s_gpu, n_gpu = ihlpsa_adaptive(backend, zg, P; x₀=x₀)
            @test isapprox(s_cpu, s_gpu; rtol=rtol)
            @test abs(n_cpu - n_gpu) <= ceil(Int, log2(m))   # within one chunk

            # Tier 2 (compact) exercises the multi-device pack/back-map on GPU.
            s_cpu_c, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, compact=true)
            s_gpu_c, _ = ihlpsa_adaptive(backend, zg, P; x₀=x₀, compact=true)
            @test isapprox(s_cpu_c, s_gpu_c; rtol=rtol)

            # Tier 3 (resumable) exercises state-resident chunk continuation on GPU.
            s_cpu_r, n_cpu_r = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, resumable=true)
            s_gpu_r, n_gpu_r = ihlpsa_adaptive(backend, zg, P; x₀=x₀, resumable=true)
            @test isapprox(s_cpu_r, s_gpu_r; rtol=rtol)
            @test abs(n_cpu_r - n_gpu_r) <= ceil(Int, log2(m))   # within one chunk

            # Hybrid exercises the on-device state gather (fancy indexing).
            s_cpu_h, _ = ihlpsa_adaptive(CPU(), zg, P; x₀=x₀, compact=true, resumable=true)
            s_gpu_h, _ = ihlpsa_adaptive(backend, zg, P; x₀=x₀, compact=true, resumable=true)
            @test isapprox(s_cpu_h, s_gpu_h; rtol=rtol)
        end
    end
end
