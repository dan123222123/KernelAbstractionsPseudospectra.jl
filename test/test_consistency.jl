# Tests that exercise the actual algorithmic correctness of ihlpsa, not just
# regression vs an external eigtool reference. Three flavors:
#
#  1. ihlpsa vs ℂsvdpsa — at fixed (small) m the inverse-Lanczos result should
#     match the dense SVD baseline within Lanczos convergence tolerance.
#  2. Generalized pencil — exercises the GeneralizedSchur ctor path that
#     parter16 doesn't touch.
#  3. Cross-backend — same problem, same x₀, must agree to ~1e-12 across
#     CPU and the requested GPU backend.

using Test, KernelAbstractionsPseudospectra, KernelAbstractions, LinearAlgebra
using Random

# Explicit x₀ so cross-backend comparisons can match modulo floating-point
# accumulation order. This is the adaptive driver's internal deterministic seeded
# start vector, reused (un-exported) so the test x₀ can't drift from the driver's default.
const _seeded_x₀ = KernelAbstractionsPseudospectra._adaptive_x₀

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
        s_ihl = ihlpsa(CPU(), zg, P, nit; x₀ = x₀)
        s_svd = ℂsvdpsa(zg, P)
        # Tolerance: F32 ~ 1e-4 (Lanczos + F32 roundoff in σ_min); F64 ~ 1e-10.
        rtol = T == ComplexF32 ? 1e-4 : 1e-10
        @test isapprox(s_ihl, s_svd; rtol = rtol)
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
        @test P isa KernelAbstractionsPseudospectra.SchurMatrixPencil

        gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (8, 8))
        γ, δ = T <: Complex ? (0.5, 0.5) : (0.5, 0.5)

        x₀ = _seeded_x₀(T, m, 0xFEED)
        s_ihl = ihlpsa(CPU(), zg, P, nit; γ = γ, δ = δ, x₀ = x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-4 : 1e-10
        @test isapprox(s_ihl, s_svd; rtol = rtol)
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
        s_fixed = ihlpsa(CPU(), zg, P, nit_max; x₀ = x₀)    # over-converged control
        # Public `ihlpsa(…; …)` returns only σ; the internal driver also returns the
        # per-point convergence depth grid this testset asserts on.
        s_adp, nit_grid = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀)
        s_svd = ℂsvdpsa(zg, P)

        # Comparison tol ~10× the adaptive stopping rtol (default 1e-4 F32 /
        # 1e-6 F64) to absorb the successive-change criterion's slack on slow points.
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol = rtol)           # vs dense SVD oracle
        @test isapprox(s_adp, s_fixed; rtol = rtol)         # vs fixed-nit control
        @test size(nit_grid) == size(s_adp)               # per-point depth grid
        @test maximum(nit_grid) < nit_max                  # genuinely stopped early
    end
end

@testset "ihlpsa adaptive criterion ablation kwarg" begin
    # `criterion = :cauchy` selects the successive-σ-change stopping test, as an alternative to
    # the default certified Ritz bound — an unexported ablation surface for bench/bench_stopping.jl.
    # On an easy grid both criteria must agree with the oracle to the comparison tol; the kwarg
    # must reject unknown symbols.
    Random.seed!(0xAD2F)
    for T in (ComplexF32, ComplexF64)
        m = 24
        A = randn(T, m, m)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (12, 12))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        s_cau, nit_cau = KernelAbstractionsPseudospectra._ihlpsa_adaptive(
            CPU(), zg, P; criterion = :cauchy, x₀ = x₀)
        s_svd = ℂsvdpsa(zg, P)
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_cau, s_svd; rtol = rtol)
        @test maximum(nit_cau) < 8 * ceil(Int, log2(m))
    end
    A64 = randn(ComplexF64, 8, 8)
    P8 = MatrixPencil(A64)
    zg8 = last(qgrid(ComplexF64, (-1, 1), (-1, 1), (4, 4)))
    @test_throws ArgumentError KernelAbstractionsPseudospectra._ihlpsa_adaptive(
        CPU(), zg8, P8; criterion = :bogus)
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
        s_adp, nit_grid = KernelAbstractionsPseudospectra._ihlpsa_adaptive(
            CPU(), zg, P; γ = γ, δ = δ, x₀ = x₀)
        s_svd = ℂsvdpsa(zg, P, γ, δ)
        rtol = T == ComplexF32 ? 1e-3 : 1e-5
        @test isapprox(s_adp, s_svd; rtol = rtol)
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

        s_one, ng_one = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀)
        s_many, ng_many = KernelAbstractionsPseudospectra._ihlpsa_adaptive(
            CPU(), zg, P; x₀ = x₀, zpd = 37)                  # 144 = 3·37 + 33 → 4 batches
        @test isapprox(s_one, s_many; rtol = (T == ComplexF32 ? 1e-3 : 1e-5))
        # A point's Lanczos sequence is independent of its batch-mates, so batching must not
        # move any retirement depth — catches per-batch state bleeding through the reused
        # workspace that a loose σ comparison would absorb.
        @test ng_many == ng_one
    end
end

@testset "ihlpsa adaptive batch reuse across singular shifts" begin
    # zpd = 1 pushes every point through the same workspace rows in sequence, and shifts
    # placed exactly on eigenvalues break the solve down (singular zI − A ⇒ ±Inf/NaN in the
    # resident state). Points solved after a breakdown point must still match the
    # single-batch run bit for bit — the reused workspace may not leak the breakdown.
    for T in (ComplexF32, ComplexF64)
        m = 8
        P = MatrixPencil(Matrix{T}(Diagonal(T.(1:m))))
        x₀ = _seeded_x₀(T, m, 0xB0B0)
        # First half exact eigenvalues, second half generic, so a strided thread share meets
        # a breakdown point before a generic one.
        zg = reshape([T.(1:4); T[0.5 + 0.25im, 1.5 - 0.25im, 2.5 + 0.5im, 3.5 - 0.5im]], 8, 1)

        s_one, ng_one = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀)
        s_many, ng_many = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀, zpd = 1)
        @test all(isfinite, s_many)
        @test s_many == s_one
        @test ng_many == ng_one
    end
end

@testset "ihlpsa adaptive on_batch delivery" begin
    # Contract: every grid point delivered exactly once, positions in returned-matrix
    # coordinates, values identical to the returned matrices — single- and multi-batch.
    # The 13×9 grid is deliberately non-square so a transposed index would be caught.
    Random.seed!(0x0BAC)
    for T in (ComplexF32, ComplexF64), zpd in (missing, 37)
        m = 24
        P = MatrixPencil(randn(T, m, m))
        gx, gy, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (13, 9))
        x₀ = _seeded_x₀(T, m, 0xBEEF)

        deliveries = Tuple{Vector{CartesianIndex{2}}, Vector{real(T)}, Vector{Int}}[]
        s, ng = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀, zpd = zpd,
            on_batch = (idx, σv, nitv) -> push!(deliveries, (idx, σv, nitv)))

        @test sort(reduce(vcat, first.(deliveries))) == vec(collect(CartesianIndices(s)))
        σacc = fill(real(T)(-1), size(s))
        nacc = fill(-1, size(ng))
        for (idx, σv, nitv) in deliveries
            σacc[idx] .= σv
            nacc[idx] .= nitv
        end
        @test σacc == s
        @test nacc == ng
    end

    # Generalized pencil leg: the delivery path is pencil-agnostic, but the γ,δ-weighted σ
    # must still round-trip through the callback untouched.
    T = ComplexF64
    m = 16
    Random.seed!(0x0BAD)
    P = MatrixPencil(randn(T, m, m), randn(T, m, m) + T(5) * I)
    zg = last(qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (7, 5)))
    x₀ = _seeded_x₀(T, m, 0xFEED)
    σacc = fill(-1.0, 5, 7)
    s, _ = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; γ = 0.5, δ = 0.5, x₀ = x₀,
        zpd = 4, on_batch = (idx, σv, nitv) -> (σacc[idx] .= σv))
    @test σacc == s
end

@testset "lockstep workspace reuse after breakdown NaN" begin
    # A reused IHLworkspace can carry a Lanczos-breakdown NaN in Qv[1] from an
    # earlier batch. β[1] ≡ 0 cancels any finite residue there but not a NaN
    # (0·NaN = NaN), so the start=1 seed must present a fresh q₀; a poisoned
    # workspace must reproduce a fresh workspace's run exactly.
    Random.seed!(0x0DDB)
    for T in (ComplexF32, ComplexF64)
        m = 16
        g = 5
        nit = 8
        P = MatrixPencil(randn(T, m, m))
        x₀ = _seeded_x₀(T, m, 0xACE5)
        zpts = T[complex(x, 0.3) for x in range(-1, 1; length = g)]

        runs = map((zero(T), T(NaN))) do q₀fill
            ihl = KernelAbstractionsPseudospectra.IHLworkspace(P, g, x₀)
            fill!(ihl.Qv[1], q₀fill)
            ihl.zv[1:g] .= zpts
            α = zeros(T, nit, g)
            β = zeros(T, nit + 1, g)
            KernelAbstractionsPseudospectra.lockstep_ihl!(α, β, ihl, nit, g)
            (α, β)
        end
        (α_fresh, β_fresh), (α_reused, β_reused) = runs
        @test all(isfinite, α_reused) && all(isfinite, β_reused)
        @test α_reused == α_fresh && β_reused == β_fresh
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

        s_adp,
        nit_grid = @test_logs (:warn,) match_mode = :any KernelAbstractionsPseudospectra._ihlpsa_adaptive(
            CPU(), zg, P; x₀ = x₀, nit_max = nit_cap, rtol = 1e-12)
        s_fix = ihlpsa(CPU(), zg, P, nit_cap; x₀ = x₀)

        @test all(==(nit_cap), nit_grid)
        @test all(isfinite, s_adp)
        @test isapprox(s_adp, s_fix; rtol = (T == ComplexF32 ? 1e-4 : 1e-10))
    end
end

# Cross-backend: same problem, same explicit x₀, results agree element-wise.
# `types` lets FP64-less backends (Intel iGPUs) restrict to ComplexF32.
function test_cross_backend(backend; types = (ComplexF32, ComplexF64))
    @testset "cross-backend: CPU vs $(backend)" begin
        Random.seed!(0xDADA)
        for T in types
            m = 32
            nit = ceil(Int, log2(m))
            A = randn(T, m, m)
            P = MatrixPencil(A)
            gx, gy, zg = qgrid(T, (-1.0, 1.0), (-1.0, 1.0), (16, 16))

            x₀ = _seeded_x₀(T, m, 0xACED)
            s_cpu = ihlpsa(CPU(), zg, P, nit; x₀ = x₀)
            s_gpu = ihlpsa(backend, zg, P, nit; x₀ = x₀)
            # Different reduction orders across backends can perturb σ_min by
            # ~few×eps(T)·κ. F32: 1e-3 generous; F64: 1e-10.
            rtol = T == ComplexF32 ? 1e-3 : 1e-10
            @test isapprox(s_cpu, s_gpu; rtol = rtol)
        end
    end
end

# Cross-backend adaptive: same problem, same fixed x₀. Values agree to the
# adaptive stopping tolerance; the converged nit agrees within one chunk
# (cross-backend reduction-order differences can flip a borderline point across
# the rtol boundary at a chunk edge, changing where it stops).
function test_adaptive_backend(backend; types = (ComplexF32, ComplexF64))
    @testset "ihlpsa adaptive: CPU vs $(backend)" begin
        Random.seed!(0xADDA)
        for T in types
            m = 32
            A = randn(T, m, m)
            P = MatrixPencil(A)
            # A larger grid (40x40, not a token 16x16): the on-device
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
            s_cpu, grid_cpu = KernelAbstractionsPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀)
            s_gpu, grid_gpu = KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P; x₀ = x₀)
            @test isapprox(s_cpu, s_gpu; rtol = rtol)
            @test abs(maximum(grid_cpu) - maximum(grid_gpu)) <= ceil(Int, log2(m))   # within one chunk

            # on_batch across the strided multi-device fan-out, with a zpd small enough to
            # force several outer batches per device: exactly-once coverage in
            # returned-matrix coordinates, values identical to the returned matrices.
            seen = CartesianIndex{2}[]
            σacc = fill(real(T)(-1), size(s_gpu))
            s_cb, _ = KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P; x₀ = x₀, zpd = 217,
                on_batch = (idx, σv, nitv) -> (append!(seen, idx); σacc[idx] .= σv))
            @test sort(seen) == vec(collect(CartesianIndices(s_cb)))
            @test σacc == s_cb
        end
    end
end
