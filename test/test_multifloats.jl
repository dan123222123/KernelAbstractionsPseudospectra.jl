# Extended-precision (MultiFloats) tests. Opt-in via `"multifloats" in ARGS` in runtests.jl —
# MultiFloats / GenericSchur / GenericLinearAlgebra are test deps but only loaded under that gate
# (a BigFloat dense-SVD oracle is slower than the rest of the suite). Two parts:
#
#  1. test_multifloats_accuracy() — the inverse-Lanczos ihlpsa at Float64x2 must track an
#     INDEPENDENT high-precision oracle (BigFloat dense SVD via ℂsvdpsa, ~77 digits) better than
#     plain Float64 does. CPU-runnable (the extended-precision path goes through the shuffle-free
#     column solve).
#
#  2. test_multifloats_tiled_shuffle(backend) — GPU-only (where warp_trsm_safe): the per-limb
#     `_trsm_shfl` override (MultiFloatsPseudospectra) must make the tiled panel solve BITWISE-match
#     the column solve for Complex{Float64x2}. Skipped on CPU / stock oneAPI (no usable shuffle).

using MultiFloats, GenericSchur, GenericLinearAlgebra, Random
using MatrixDepot                # chebspec test matrix

# Local median (avoid a Statistics dependency for one call).
function _median(v)
    (s = sort(vec(v)); n = length(s);
        iseven(n) ? (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2 : s[(n + 1) ÷ 2])
end

# (1) Accuracy vs a BigFloat oracle that shares neither algorithm (dense SVD vs inverse-Lanczos)
# nor precision (MPFR ~77 digits vs Float64x2 ~32). Small grid: each oracle point is a full
# BigFloat SVD, so keep m/grid modest for CI.
function test_multifloats_accuracy()
    @testset "MultiFloats ihlpsa accuracy vs BigFloat oracle" begin
        setprecision(BigFloat, 256)                       # ≈ 77 decimal digits
        M, G, NIT = 12, 11, 12
        HP = Complex{Float64x2}
        Aref = MatrixDepot.chebspec(Float64, M)           # standard non-normal example
        mat(::Type{T}) where {T <: Complex} = T.(Aref)
        eigs = eigvals(mat(ComplexF64))
        pad = 1.5
        region = ((minimum(real, eigs) - pad, maximum(real, eigs) + pad),
            (minimum(imag, eigs) - pad, maximum(imag, eigs) + pad))
        grid(::Type{T}) where {T <: Complex} = qgrid(T, region[1], region[2], (G, G))[3]
        run_ihl(::Type{T}) where {T <: Complex} = ihlpsa(CPU(), grid(T), MatrixPencil(schur(mat(T))), NIT)

        s_oracle = Float64.(ℂsvdpsa(grid(Complex{BigFloat}), mat(Complex{BigFloat})))
        s64 = run_ihl(ComplexF64)
        smf = Float64.(run_ihl(HP))

        # Relative error, but floor the denominator: near an eigenvalue σ_min(z) → 0, so a bare
        # |Δ|/σ would blow up for any solver — guard it so the test can't flake on a grid point that
        # lands in the ill-conditioned band.
        denom = max.(s_oracle, eps(Float64))
        e64 = abs.(s64 .- s_oracle) ./ denom
        emf = abs.(smf .- s_oracle) ./ denom

        # Extended precision tracks the independent oracle far better than Float64, which loses
        # accuracy in the ill-conditioned band near the spectrum.
        @test _median(emf) < _median(e64)
        @test maximum(emf) < 1e-10           # Float64x2 follows the ~77-digit oracle closely
        @test maximum(emf) < maximum(e64)    # and beats Float64 at its worst point
    end
end

# (2) The per-limb shuffle override must reproduce the column solve bitwise on GPU. The tiled panel
# solve broadcasts each pivot with `_trsm_shfl` (the override), and for a single ≤32 panel the panel
# solve IS the full triangular solve — same `_pdiv` / pencil / ascending-column update order as the
# column kernel, so it must be BITWISE identical. Exercise the override by comparing the tiled panel
# kernels to column for m ≤ 32. Only runs where the warp shuffle is usable for wide types (CUDA /
# AMDGPU / Metal-opt-in); skipped on CPU and stock oneAPI (its KernelIntrinsics @shfl is a TODO stub).
function test_multifloats_tiled_shuffle(backend)
    (KernelAbstractions.isgpu(backend) && KAPseudospectra.warp_trsm_safe(backend, true)) ||
        return
    @testset "MultiFloats per-limb tiled shuffle (bitwise vs column) -- $(backend)" begin
        T = Complex{Float64x2}
        # Panel width follows the hardware warp, as `_tiled_trsm!` does — the panel is exactly one
        # shuffle's reach. The column kernels below stay at a fixed 32: their `wgs` is an unrelated
        # knob, and holding it fixed is what keeps the bitwise comparison meaningful.
        wp = KAPseudospectra.warp_width(backend)
        for m in (16, 31, 32)            # single panel (m ≤ wp): the tiled panel solve is the full solve
            g = 4
            # Build well-conditioned triangular pencils by promoting Float64 randoms (MultiFloats
            # has no native randn); same dominant-diagonal trick as _rand_*tri keeps cond O(1).
            up(s) = (Mr = s .* triu(randn(ComplexF64, m, m), 1);
                [Mr[i, i] = 1 + s * randn(ComplexF64) for i in 1:m]; T.(Mr))
            lo(s) = (Mr = s .* tril(randn(ComplexF64, m, m), -1);
                [Mr[i, i] = 1 + s * randn(ComplexF64) for i in 1:m]; T.(Mr))
            s = 1 / (2m + 1)
            Au, Bu, Al, Bl = up(s), up(s), lo(s), lo(s)
            zv = T.(2 .+ 0.3 .* randn(ComplexF64, g))
            b0 = reduce(hcat, [T.(randn(ComplexF64, m)) for _ in 1:g])

            # forward (lower-tri): tiled panel (one panel = full solve) vs column, bitwise identical
            bt = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._tiled_panel_forward(backend, wp, (wp, g))(
                bt, _to(backend, zv), _to(backend, Al), _to(backend, Bl), 0, m)
            KernelAbstractions.synchronize(backend)
            bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._batched_column_oriented_forward_solve_pencil(backend, 32, (32, g))(
                bc, _to(backend, zv), _to(backend, Al), _to(backend, Bl))
            KernelAbstractions.synchronize(backend)
            @test _from(bt.data) == _from(bc.data)

            # backward (upper-tri): exercises the same per-limb _trsm_shfl in the descending panel solve
            bt = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._tiled_panel_backward(backend, wp, (wp, g))(
                bt, _to(backend, zv), _to(backend, Au), _to(backend, Bu), 0, m)
            KernelAbstractions.synchronize(backend)
            bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._batched_column_oriented_backward_solve_pencil(backend, 32, (32, g))(
                bc, _to(backend, zv), _to(backend, Au), _to(backend, Bu))
            KernelAbstractions.synchronize(backend)
            @test _from(bt.data) == _from(bc.data)
        end
    end
end

# (3) Wide-fit. A generic (B≠I) Complex{Float64x2} pencil needs TWO 32×32 trailing tiles (32 B/elt ⇒
# 64 KB), which overflows the 48 KB shared-memory limit at a full 32-wide tile — `tiled_tiles_fit`
# checks the narrowest candidate, so it tiles at a narrow `tile_cols` instead. Confirm it is
# tile-able and the full multi-panel tiled driver matches the column driver to round-off (m > 32 so
# the trailing update actually runs). GPU + usable wide shuffle only.
function test_multifloats_tiled_generic(backend)
    (KernelAbstractions.isgpu(backend) && KAPseudospectra.warp_trsm_safe(backend, true)) ||
        return
    @testset "MultiFloats wide generic tiled (TC-aware fit, vs column) -- $(backend)" begin
        T = Complex{Float64x2}
        m, g = 64, 64
        s = 1 / (2m + 1)
        up() = (Mr = s .* triu(randn(ComplexF64, m, m), 1);
            [Mr[i, i] = 1 + s * randn(ComplexF64) for i in 1:m]; T.(Mr))
        A, B = up(), up()        # upper-tri ⇒ Ac=A', Bc=B' are lower-tri for the forward sweep
        P = KAPseudospectra.SchurMatrixPencil{T, false}(
            A, collect(A'), B, collect(B'), Matrix{T}(I, m, m))
        @test KAPseudospectra.tiled_tiles_fit(backend, P)            # narrow TC fits → tiled, not column
        @test KAPseudospectra.tile_cols(backend, P) in KAPseudospectra._TILECOLS_CANDIDATES
        Pdev = _to(backend, P)
        zv = _to(backend, T.(2 .+ 0.3 .* randn(ComplexF64, g)))
        b0 = reduce(hcat, [T.(randn(ComplexF64, m)) for _ in 1:g])
        wgs = KAPseudospectra.column_wgs(backend, Pdev)
        bt = VectorOfSimilarVectors(_to(backend, copy(b0)))
        KAPseudospectra._tiled_trsm!(backend, bt, zv, Pdev, wgs);
        KernelAbstractions.synchronize(backend)
        bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
        KAPseudospectra._column_trsm!(backend, bc, zv, Pdev, wgs);
        KernelAbstractions.synchronize(backend)
        xt, xc = _from(bt.data), _from(bc.data)
        @test maximum(Float64.(abs.(xt .- xc))) / maximum(Float64.(abs.(xc))) < 1e-25
    end
end

# (4) Float32-limb RANGE: `ihlsrg!`'s eig work-type promotion cannot widen a Float32-limb
# MultiFloat past Float32's ~3.4e38 range (it out-precisions Float64, so promote_type keeps
# e.g. Float32x4), and the host eig's internal squarings can overflow on large-but-finite
# Lanczos tridiag entries near a true eigenvalue — returning NaN σ from finite input. ihlsrg!
# normalizes the tridiagonal before the eig (λmax is scale-equivariant) and pins any non-finite
# result; σ must stay finite and correct to leading order through the full representable range.
function test_multifloats_f32limb_range()
    @testset "Float32-limb range: ihlsrg! finite on huge tridiag entries" begin
        T = Complex{Float32x4}
        to64(x) = sum(Float64, x._limbs)
        nit, g = 6, 5
        for scale in (1.0f20, 1.0f30, 3.0f37)
            α = fill(T(scale), nit, g)            # diag ≈ scale ⇒ λmax ≈ 1.24·scale
            β = fill(T(scale / 8), nit + 1, g)    # rows 2:end-1 are the off-diagonals
            zv = [T(x) for x in range(0.5, 2.0, length = g)]
            sr = zeros(Float32x4, g)
            KAPseudospectra.ihlsrg!(sr, zv, 1.0, 0.0, α, β)
            @test all(isfinite, sr)
            @test all(>(0), sr)
            # σ = 1/√λmax up to the O(1) tridiag factor: σ·√scale ∈ (0.5, 1.5)
            @test all(s -> 0.5 < to64(s * sqrt(Float32x4(scale))) < 1.5, sr)
        end
    end
end

# (5) The adaptive driver's certified Ritz-residual stop at extended precision. Two checks:
#   • the |s_k| helper (`_tridiag_top_lastcomp`, inverse iteration) at Float32-limb precision — same
#     eigenvector as Float64, via inverse iteration (not an eigendecomposition) at precision where a
#     QL/QR eig NaNs, matching `_eigmax_tridiag`'s Sturm-bisection;
#   • end-to-end: adaptive Float64x2 at a tight rtol reaches the ~32-digit arithmetic floor, so it
#     agrees with the deep fixed control (the residual bound is what lets it converge that far — a
#     successive-change stop stalls short near the small-gap points).
function test_multifloats_residual_stop()
    @testset "MultiFloats adaptive residual stop (wide |s_k| + precision floor)" begin
        rng = Random.Xoshiro(23)
        for n in (3, 8, 24)
            d = randn(rng, n)
            e = randn(rng, n - 1)
            F = eigen(SymTridiagonal(d, e))
            sk_ref = abs(F.vectors[end, argmax(F.values)])
            d2, e2 = Float32x2.(d), Float32x2.(e)
            sk2 = KAPseudospectra._tridiag_top_lastcomp(
                d2, e2, KAPseudospectra._eigmax_tridiag(d2, e2))
            @test isfinite(sk2)
            @test isapprox(sk2, Float32x2(sk_ref), atol = 1e-6)
        end

        T = Complex{Float64x2}
        m = 24
        A = T.(MatrixDepot.chebspec(Float64, m))
        P = MatrixPencil(schur(A))
        eigs = eigvals(ComplexF64.(A))
        pad = 1.5
        region = ((minimum(real, eigs) - pad, maximum(real, eigs) + pad),
            (minimum(imag, eigs) - pad, maximum(imag, eigs) + pad))
        zg = qgrid(T, region[1], region[2], (8, 8))[3]
        x₀ = KAPseudospectra._adaptive_x₀(T, m, 0xBEEF)
        cap = 32 * ceil(Int, log2(m))
        s_adp, _ = KAPseudospectra._ihlpsa_adaptive(CPU(), zg, P; x₀ = x₀, rtol = 1e-25, nit_max = cap)
        s_fix = ihlpsa(CPU(), zg, P, cap; x₀ = x₀)      # deep fixed control at the floor
        to64(x) = sum(Float64, x._limbs)
        @test all(isfinite, s_adp)
        @test maximum(to64.(abs.(s_adp .- s_fix))) < 1e-18     # both reach the Float64x2 floor
    end

    # Generalized (B ≠ I) MultiFloat pencil: BigFloat QZ reduce-and-round (GenericSchur ext).
    @testset "MultiFloats generalized pencil (BigFloat QZ)" begin
        T = Complex{Float64x2}
        m = 24
        A64 = ComplexF64.(Matrix(MatrixDepot.grcar(Float64, m)))
        B64 = ComplexF64.(MatrixDepot.kms(Float64, m))
        P = MatrixPencil(T.(A64), T.(B64))
        @test P isa KAPseudospectra.SchurMatrixPencil{T, false}
        @test !KAPseudospectra.b_is_identity(P)
        @test istriu(P.A) && istriu(P.B)
        # reduction fidelity, bounded by the ComplexF64 reference
        λ = ComplexF64.(diag(P.A) ./ diag(P.B))
        λref = eigvals(A64, B64)
        @test maximum(minimum(abs.(λref .- x)) for x in λ) < 1e-12
        # σ through the full solve vs the dense-SVD oracle on the promoted pencil
        zg = qgrid(T, (-1, 3), (-3, 3), (6, 6))[3]
        σ = ihlpsa(CPU(), zg, P, 60)
        σref = ℂsvdpsa(qgrid(ComplexF64, (-1, 3), (-3, 3), (6, 6))[3],
            MatrixPencil(schur(A64, B64)))
        to64(x) = sum(Float64, x._limbs)
        @test maximum(abs.(to64.(real.(σ)) .- σref) ./ max.(σref, eps())) < 1e-10
        # real-matrix convenience method complexifies both operands
        Pr = MatrixPencil(Float64x2.(real.(A64)), Float64x2.(real.(B64)); bits = 128)
        @test !KAPseudospectra.b_is_identity(Pr)
    end
end
