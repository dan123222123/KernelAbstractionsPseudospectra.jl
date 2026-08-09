# Edge cases that catch off-by-ones in the IHL pipeline:
#  - m=1: degenerate Lanczos, trivially-sized SymTridiagonal in ihlsrg
#  - m=2: smallest non-trivial pencil
#  - nit=1: single Lanczos step, β has 2 rows but β[2:end-1] is empty
#  - long-aspect grid: nx ≠ ny, exercises the transpose/permutedims at return
#  - findmaxbatchihl returns sane positive values on each backend

using Test, KAPseudospectra, KernelAbstractions, LinearAlgebra, Random
using KAPseudospectra: findmaxbatchihl, _device_column_partition

@testset "edge cases" begin
    @testset "m=1" begin
        A = reshape([2.0 + 0.0im], 1, 1)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        # Fixed nit=1: degenerate single-step Lanczos on a 1×1 pencil.
        s = ihlpsa(CPU(), zg, P, 1)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "m=2" begin
        A = ComplexF64[1.0 0.5; 0.0 2.0]
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        s = ihlpsa(CPU(), zg, P, 2)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "nit=1" begin
        A = randn(ComplexF64, 16, 16)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        s = ihlpsa(CPU(), zg, P, 1)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "long-aspect grid" begin
        A = randn(ComplexF64, 8, 8)
        P = MatrixPencil(A)
        # nx=4, ny=32; ihlpsa permutedims the result, so output shape is (ny, nx) = (32, 4).
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 32))
        s = ihlpsa(CPU(), zg, P, 3)
        @test size(s) == (32, 4)
        @test all(isfinite, s)
    end
end

@testset "error paths" begin
    A = randn(ComplexF64, 4, 4)
    gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (3, 3))
    # (γ, δ) weights scale the perturbation norm ε(γ + δ|z|): each must be ≥ 0, not both 0.
    @test_throws ArgumentError ℂsvdpsa(zg, A, I, -1, 0)
    @test_throws ArgumentError ℂsvdpsa(zg, A, I, 0, 0)
    @test_throws ArgumentError ℂsvdpsa(Matrix{ComplexF64}(undef, 0, 0), A)
    @test_throws DimensionMismatch ℂsvdpsa(zg, A, randn(ComplexF64, 3, 3))
    @test_throws DimensionMismatch MatrixPencil(A, randn(ComplexF64, 3, 3))
    # The trsm strategy is validated both when persisted and on every ENV read.
    @test_throws ErrorException set_trsm_strategy!("bogus")
    withenv("KAPSEUDO_TRSM" => "bogus") do
        @test_throws ErrorException KAPseudospectra.trsm_strategy()
    end
end

@testset "scaled UniformScaling pencil" begin
    # MatrixPencil(A, cI) must keep the scale (generalized pencil), matching the
    # explicit dense-B path.
    A = randn(ComplexF64, 4, 4)
    gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (3, 3))
    P3 = MatrixPencil(A, 3I)
    @test !KAPseudospectra.b_is_identity(P3)
    @test ℂsvdpsa(zg, P3) ≈ ℂsvdpsa(zg, A, Matrix{ComplexF64}(3I, size(A))) rtol = 1e-12
end

@testset "findmaxbatchihl sanity" begin
    # CPU path uses Sys.free_memory(); should be >> any reasonable single-pencil
    # workspace, so the returned batch should be huge but finite.
    n_cpu = findmaxbatchihl(CPU(), ComplexF64, 32, 5)
    @test n_cpu > 0
    @test isfinite(n_cpu)

    if isdefined(Main, :AMDGPU) && AMDGPU.functional()
        n_gpu = findmaxbatchihl(ROCBackend(), ComplexF64, 32, 5)
        @test n_gpu > 0
        @test isfinite(n_gpu)
    end
    if isdefined(Main, :CUDA) && CUDA.functional()
        n_gpu = findmaxbatchihl(CUDABackend(), ComplexF64, 32, 5)
        @test n_gpu > 0
        @test isfinite(n_gpu)
    end
end

@testset "_device_pencil_bytes per pencil shape" begin
    # Must mirror Adapt.adapt_structure exactly: Ac/Bc always materialize dense, Z ships as
    # stored, a standard pencil's B/Bc share one Diagonal identity, a raw pencil ships as-is.
    m = 12
    T = ComplexF64
    bytes = KAPseudospectra._device_pencil_bytes
    S = Matrix{T}(triu(randn(T, m, m)))
    Dm = Diagonal(ones(T, m))

    P_std_diag = KAPseudospectra.SchurMatrixPencil{T, true}(S, S', Dm, Dm, Diagonal(ones(T, m)))
    @test bytes(P_std_diag) == sizeof(T) * (2m^2 + 2m)           # A + Ac + Id + Diagonal Z

    P_std_dense = MatrixPencil(schur(randn(T, m, m)))
    @test bytes(P_std_dense) == sizeof(T) * (3m^2 + m)           # dense Schur Z joins

    P_gen = MatrixPencil(schur(randn(T, m, m), randn(T, m, m) + T(5) * I))
    @test bytes(P_gen) == sizeof(T) * 5m^2                       # A + Ac + B + Bc + Z

    P_raw = KAPseudospectra.MatrixPencil{T}(randn(T, m, m), Matrix{T}(I, m, m))
    @test bytes(P_raw) == sizeof(T) * 2m^2

    # Pencil-aware batch planning consumes these bytes; sanity on the host path.
    n_P = findmaxbatchihl(CPU(), P_std_diag, 5)
    @test n_P > 0 && isfinite(n_P)
end

@testset "_device_column_partition invariants" begin
    # `_ihlpsa_fanout` zips blocks against devices, so the partition must always yield exactly
    # min(ndev, ncols) balanced blocks covering 1:ncols exactly once. Default is round-robin
    # (strided) so clustered hard points spread across devices; KAPSEUDO_STRIDED=0 gives
    # contiguous bands. Both modes must satisfy the invariants (host-side — there is no GPU CI).
    # A partition yielding fewer blocks than devices (e.g. 9 cols / 4 devs → 3 blocks) would
    # BoundsError the device loop.
    for stride in ("1", "0"), ncols in 1:20, ndev in 1:6
        blocks = withenv(() -> _device_column_partition(ncols, ndev),
            "KAPSEUDO_STRIDED" => stride)
        @test blocks isa Vector{StepRange{Int, Int}}
        @test length(blocks) == min(ndev, ncols)
        # coverage + disjointness as a SET (strided blocks are interleaved, not
        # ordered, so compare sorted)
        @test sort(reduce(vcat, collect.(blocks))) == collect(1:ncols)
        # balance: block sizes differ by at most 1
        @test maximum(length.(blocks)) - minimum(length.(blocks)) <= 1
    end

    # Pinned cases. Strided (default): device b takes columns b, b+nblocks, …
    @test withenv(() -> _device_column_partition(1, 4), "KAPSEUDO_STRIDED" => "1") ==
          [1:1:1]
    @test withenv(() -> _device_column_partition(9, 4), "KAPSEUDO_STRIDED" => "1") ==
          [1:4:9, 2:4:9, 3:4:9, 4:4:9]
    @test withenv(() -> _device_column_partition(2, 4), "KAPSEUDO_STRIDED" => "1") ==
          [1:2:2, 2:2:2]
    # Contiguous: balanced bands, exactly min(ndev, ncols) of them.
    @test withenv(() -> _device_column_partition(9, 4), "KAPSEUDO_STRIDED" => "0") ==
          [1:1:3, 4:1:5, 6:1:7, 8:1:9]
    @test isempty(_device_column_partition(0, 4))
    @test_throws ArgumentError _device_column_partition(-1, 4)
    @test_throws ArgumentError _device_column_partition(4, 0)
end

# ihlsrg! non-finite handling: a point that converges early and keeps iterating (fixed driver)
# can break down — β underflows to 0 on eltypes whose range is narrower than their precision
# (Float32-limb MultiFloats), and the tridiagonal's TAIL goes Inf/NaN. The finite leading block
# is the converged estimate, so ihlsrg! must truncate to it; only a point dead from iteration 1
# pins to eps. (Type-generic behavior — tested here at Float64.)
@testset "ihlsrg! truncates to the finite Lanczos prefix on breakdown" begin
    nit, g = 5, 3
    α = fill(4.0 + 0im, nit, g)
    β = fill(1.0 + 0im, nit + 1, g)
    α[4:end, 2] .= Inf                 # breakdown after 3 clean iterations for point 2
    β[4:end, 2] .= NaN
    α[1, 3] = NaN                      # dead from the start for point 3
    zv = fill(1.0 + 0im, g)
    sr = zeros(g)
    KAPseudospectra.ihlsrg!(sr, zv, 1.0, 0.0, α, β)
    @test all(isfinite, sr)
    @test sr[2] ≈ sr[1] rtol = 0.2     # prefix estimate (depth 3 vs 5), NOT the eps pin
    @test sr[3] == eps(Float64)
end

# _eigmax_tridiag's generic path is Gershgorin-bounded Sturm bisection (iterative eig solvers
# can NaN in exotic arithmetic, e.g. GLA's QL at Float32-limb MultiFloat precision). The
# bisection must agree with the Float64 LAPACK path on ordinary input.
@testset "_eigmax_tridiag bisection matches LAPACK" begin
    rng = Random.Xoshiro(7)
    for n in (1, 2, 3, 8, 24)
        d = randn(rng, n)
        e = randn(rng, max(n - 1, 0))
        λ = KAPseudospectra._eigmax_tridiag(d, e)                        # Float64 → LAPACK
        λ32 = KAPseudospectra._eigmax_tridiag(Float32.(d), Float32.(e))  # generic → bisection
        @test isfinite(λ32)
        @test isapprox(Float64(λ32), λ, rtol = 2e-5)
    end
end

# _tridiag_top_lastcomp gets |last component| of the λmax eigenvector by inverse iteration (the
# Ritz residual β_{k+1}·|s_k| stopping test of the adaptive driver). It must match the dense
# LAPACK eigenvector on ordinary input. (The MultiFloat-precision robustness — where a QL
# eigensolver NaNs — is checked in test_multifloats.jl under the `multifloats` gate.) Random
# tridiagonals have a simple spectrum, so |s_k| is well-defined.
@testset "_tridiag_top_lastcomp matches dense eigenvector" begin
    rng = Random.Xoshiro(11)
    for n in (2, 3, 8, 24)
        d = randn(rng, n)
        e = randn(rng, n - 1)
        F = eigen(SymTridiagonal(d, e))
        sk_ref = abs(F.vectors[end, argmax(F.values)])
        λ = KAPseudospectra._eigmax_tridiag(d, e)
        @test isapprox(KAPseudospectra._tridiag_top_lastcomp(d, e, λ), sk_ref, atol = 1e-8)
    end
end
