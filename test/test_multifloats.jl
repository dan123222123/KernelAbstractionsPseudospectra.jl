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

using MultiFloats, GenericSchur, GenericLinearAlgebra
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
        for m in (16, 31, 32)            # single panel (≤32): the tiled panel solve is the full solve
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
            KATRSM._tiled_panel_forward(backend, 32, (32, g))(
                bt, _to(backend, zv), _to(backend, Al), _to(backend, Bl), 0, m)
            KernelAbstractions.synchronize(backend)
            bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._batched_column_oriented_forward_solve_pencil(backend, 32, (32, g))(
                bc, _to(backend, zv), _to(backend, Al), _to(backend, Bl))
            KernelAbstractions.synchronize(backend)
            @test _from(bt.data) == _from(bc.data)

            # backward (upper-tri): exercises the same per-limb _trsm_shfl in the descending panel solve
            bt = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._tiled_panel_backward(backend, 32, (32, g))(
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
# 64 KB) which overflows the 48 KB shared-memory limit — so the OLD `tiled_tiles_fit` (a 32×32 check)
# dropped it to `column`. With the TC-aware fit it tiles at a narrow `TC` instead. Confirm it is now
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
        @test KAPseudospectra.tiled_tc(backend, P) in KAPseudospectra._TC_CANDIDATES
        Pdev = _to(backend, P)
        zv = _to(backend, T.(2 .+ 0.3 .* randn(ComplexF64, g)))
        b0 = reduce(hcat, [T.(randn(ComplexF64, m)) for _ in 1:g])
        wgs = KAPseudospectra.default_wgs(backend, m)
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
