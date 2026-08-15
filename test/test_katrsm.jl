# Correctness tests for KATRSM triangular-solve kernels, checked against the LAPACK
# reference (`LowerTriangular(L) \ b` / `UpperTriangular(U) \ b`, dispatching to trtrs).
# Pencil variants are checked against `LowerTriangular(z*B - A) \ b` / `UpperTriangular(z*B - A) \ b`.
#
# "KATRSM core (CPU host functions)" always runs on include and exercises the pure-Julia
# host loops in trsm_pencil_core.jl (CPU-only by definition, no KernelAbstractions dispatch).
# test_katrsm_kernels(backend) exercises every KA kernel/wrapper KATRSM exposes; runtests.jl
# calls it for CPU() and for each available GPU backend.
#
# All tolerances are 32 * eps(T): the test matrices are diagonally dominant with cond(M) =
# O(1), and pencil shifts are biased so cond(z*B - A) = O(1) too, so a backward-stable solve
# must agree with LAPACK to a small constant multiple of eps(T) regardless of backend.

using Test, KernelAbstractionsPseudospectra, KernelAbstractions, LinearAlgebra
using ArraysOfArrays
using Adapt
using Random

const KATRSM = KernelAbstractionsPseudospectra.KATRSM

# Strictly diagonally dominant triangular matrices: diag ≈ 1, off-diagonals
# scaled by 1/(2m+1) so each row's off-diagonal sum is ≤ 1/2. Condition number
# is O(1) independent of m, so two backward-stable triangular solves should
# agree elementwise to a small constant times eps(T).
function _rand_uppertri(T, m; rng = Random.default_rng())
    s = T(1) / T(2 * m + 1)
    M = s * triu(randn(rng, T, m, m), 1)
    for i in 1:m
        M[i, i] = one(T) + s * randn(rng, T)
    end
    M
end
function _rand_lowertri(T, m; rng = Random.default_rng())
    s = T(1) / T(2 * m + 1)
    M = s * tril(randn(rng, T, m, m), -1)
    for i in 1:m
        M[i, i] = one(T) + s * randn(rng, T)
    end
    M
end

# A few × eps(T): F32 ≈ 1.2e-7, F64 ≈ 2.2e-16.
_tol(::Type{ComplexF32}) = 32 * eps(Float32)
_tol(::Type{ComplexF64}) = 32 * eps(Float64)

# Move CPU array(s) onto the backend's device array type. CPU() ↦ Array.
_to(backend, x) = adapt(KernelAbstractionsPseudospectra.get_bgarray(backend), x)
# Pull device array(s) back to host for comparison.
_from(x) = adapt(Array, x)

@testset "KATRSM core (CPU host functions)" begin
    Random.seed!(0xACE5)

    @testset "pencil core" begin
        for T in (ComplexF32, ComplexF64), m in (1, 8, 16)

            rtol = _tol(T)
            Au = _rand_uppertri(T, m);
            Bu = _rand_uppertri(T, m)
            Al = _rand_lowertri(T, m);
            Bl = _rand_lowertri(T, m)
            b = randn(T, m)
            # |z| ≈ 2.06 keeps z*B - A's diagonal magnitude well above zero
            z = T(2 + 0.5im)

            xref = UpperTriangular(z * Bu .- Au) \ b
            x = copy(b);
            KATRSM.backward_solve_pencil!(x, z, Au, Bu)
            @test isapprox(x, xref; rtol = rtol)

            yref = LowerTriangular(z * Bl .- Al) \ b
            y = copy(b);
            KATRSM.forward_solve_pencil!(y, z, Al, Bl)
            @test isapprox(y, yref; rtol = rtol)
        end
    end

    # The column solve's `wgs` resolves through a piecewise-constant schedule over `m`
    # (`tune_trsm_wgs!` persists it); these cover the pure host-side parse/lookup, which is
    # where a malformed or hand-edited preference would otherwise surface as a runtime error.
    @testset "wgs schedule" begin
        parse_s = KernelAbstractionsPseudospectra._parse_wgs_schedule
        from_s = KernelAbstractionsPseudospectra._wgs_from_schedule

        sched = parse_s("32:32,512:64,1024:128")
        @test sched == [(32, 32), (512, 64), (1024, 128)]
        @test KernelAbstractionsPseudospectra._format_wgs_schedule(sched) == "32:32,512:64,1024:128"
        @test parse_s(" 512 : 64 , 32:32 ") == [(32, 32), (512, 64)]   # whitespace + reordering
        @test parse_s("128:64") == [(128, 64)]                          # m-independent optimum

        @test from_s(sched, 16) == 32        # below the first threshold → the schedule's floor
        @test from_s(sched, 32) == 32
        @test from_s(sched, 511) == 32       # between thresholds → the lower one
        @test from_s(sched, 512) == 64
        @test from_s(sched, 4096) == 128     # above the last → the top entry

        # Malformed values must degrade to `_auto_wgs`, never throw on a solve.
        for bad in ("", "  ", "32", "32:", ":64", "a:64", "32:b", "32:64:128", "0:64", "32:0")
            @test parse_s(bad) === nothing
        end
    end

    @testset "wgs resolution" begin
        P = MatrixPencil(schur(_rand_uppertri(ComplexF64, 16)))
        @test KernelAbstractionsPseudospectra._auto_wgs(CPU(), 1024) == 1        # CPU is always serial per point
        @test KernelAbstractionsPseudospectra.column_wgs(CPU(), P) == 1
        withenv("KAPSEUDO_TRSM_WGS" => "128") do
            @test KernelAbstractionsPseudospectra.column_wgs(CPU(), P) == 128    # env wins over everything
        end
        withenv("KAPSEUDO_TRSM_WGS" => "0") do
            @test_throws ErrorException KernelAbstractionsPseudospectra.column_wgs(CPU(), P)
        end
    end
end

# Backend-parameterized KA kernel tests; called once per available backend from runtests.jl.
function test_katrsm_kernels(backend; types = (ComplexF32, ComplexF64))
    @testset "KATRSM kernels -- $(backend)" begin
        Random.seed!(0xACE5)

        @testset "batched pencil naive (KA wrappers)" begin
            for T in types, m in (8, 16)

                rtol = _tol(T)
                g = 5
                Au = _rand_uppertri(T, m);
                Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m);
                Bl = _rand_lowertri(T, m)
                # bias z away from 1 so z*B - A stays well-conditioned
                zv = T(2) .+ T(0.3) * randn(T, g)
                b0_mat = reduce(hcat, [randn(T, m) for _ in 1:g])

                bv_dev = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                Au_d = _to(backend, Au);
                Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al);
                Bl_d = _to(backend, Bl)
                zv_d = _to(backend, zv)

                # The wrapper's `Vector.(xV)` already copies device data → host.
                xv = KATRSM.batched_backward_solve_pencil(bv_dev, zv_d, Au_d, Bu_d)
                for i in 1:g
                    xref = UpperTriangular(zv[i] * Bu .- Au) \ b0_mat[:, i]
                    @test isapprox(xv[i], xref; rtol = rtol)
                end

                yv = KATRSM.batched_forward_solve_pencil(bv_dev, zv_d, Al_d, Bl_d)
                for i in 1:g
                    yref = LowerTriangular(zv[i] * Bl .- Al) \ b0_mat[:, i]
                    @test isapprox(yv[i], yref; rtol = rtol)
                end
            end
        end

        @testset "batched column-oriented pencil (KA, workgroup parallel)" begin
            # wgs=1 exercises the degenerate single-thread path; wgs=2,4 split
            # the column-update loop across multiple workitems.
            for T in types, m in (8, 16), wgs in (1, 2, 4)
                rtol = _tol(T)
                g = 4
                Au = _rand_uppertri(T, m);
                Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m);
                Bl = _rand_lowertri(T, m)
                zv = T(2) .+ T(0.3) * randn(T, g)
                b0_mat = reduce(hcat, [randn(T, m) for _ in 1:g])

                Au_d = _to(backend, Au);
                Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al);
                Bl_d = _to(backend, Bl)
                zv_d = _to(backend, zv)

                bv = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                KATRSM._batched_column_oriented_backward_solve_pencil(backend, wgs, (
                    wgs, g))(bv, zv_d, Au_d, Bu_d)
                KernelAbstractions.synchronize(backend)
                bv_h = _from(bv.data)
                for i in 1:g
                    xref = UpperTriangular(zv[i] * Bu .- Au) \ b0_mat[:, i]
                    @test isapprox(bv_h[:, i], xref; rtol = rtol)
                end

                bv = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                KATRSM._batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(
                    bv, zv_d, Al_d, Bl_d)
                KernelAbstractions.synchronize(backend)
                bv_h = _from(bv.data)
                for i in 1:g
                    yref = LowerTriangular(zv[i] * Bl .- Al) \ b0_mat[:, i]
                    @test isapprox(bv_h[:, i], yref; rtol = rtol)
                end
            end
        end

        @testset "batched B=I eye kernels (column)" begin
            # For a standard pencil (B = I) the eye column solve drops the B read: M = zI − A,
            # so the pivot is (z − A[j,j]) and off-diagonal updates are (−A[i,j]). They must agree
            # with LAPACK on M = zI − {L,U} and with the generic kernels run with B = the materialized
            # identity, to round-off (they compute the shorter z−A / −A expressions, so NVPTX FMA
            # contraction can differ by ~1 ULP — not bit-for-bit). Column eye runs on any backend.
            # This is the path `_column_trsm!` takes for a StandardSchurMatrixPencil.
            for T in types, m in (8, 16, 31, 32, 64)

                rtol = _tol(T)
                g = 4
                L = _rand_lowertri(T, m);
                U = _rand_uppertri(T, m)   # Ac/A play the triangular role
                Id = Matrix{T}(I, m, m)
                zv = T(2) .+ T(0.3) * randn(T, g)
                b0 = reduce(hcat, [randn(T, m) for _ in 1:g])
                L_d = _to(backend, L);
                U_d = _to(backend, U)
                Id_d = _to(backend, Id);
                zv_d = _to(backend, zv)

                # column eye vs generic-with-identity (and vs LAPACK). wgs=1 single-thread, wgs=4 split.
                for wgs in (1, 4)
                    be = VectorOfSimilarVectors(_to(backend, copy(b0)))
                    KATRSM._batched_column_oriented_forward_solve_eye(backend, wgs, (
                        wgs, g))(be, zv_d, L_d)
                    bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
                    KATRSM._batched_column_oriented_forward_solve_pencil(backend, wgs, (
                        wgs, g))(bc, zv_d, L_d, Id_d)
                    KernelAbstractions.synchronize(backend)
                    beh = _from(be.data);
                    bch = _from(bc.data)
                    for i in 1:g
                        @test isapprox(beh[:, i], LowerTriangular(zv[i] * I - L) \ b0[:, i]; rtol = rtol)
                        @test isapprox(beh[:, i], bch[:, i]; rtol = rtol)   # ~round-off vs generic (FMA order)
                    end

                    be = VectorOfSimilarVectors(_to(backend, copy(b0)))
                    KATRSM._batched_column_oriented_backward_solve_eye(backend, wgs, (
                        wgs, g))(be, zv_d, U_d)
                    bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
                    KATRSM._batched_column_oriented_backward_solve_pencil(backend, wgs, (
                        wgs, g))(bc, zv_d, U_d, Id_d)
                    KernelAbstractions.synchronize(backend)
                    beh = _from(be.data);
                    bch = _from(bc.data)
                    for i in 1:g
                        @test isapprox(beh[:, i], UpperTriangular(zv[i] * I - U) \ b0[:, i]; rtol = rtol)
                        @test isapprox(beh[:, i], bch[:, i]; rtol = rtol)   # ~round-off vs generic (FMA order)
                    end
                end
            end
        end
    end
end

# End-to-end consistency of the GPU `tiled` trsm strategy (KAPSEUDO_TRSM=tiled): it must match the
# shuffle-free `column` baseline to element-type tolerance, across sizes incl. partial last panels
# (m not a multiple of 32). `tiled` self-gates to `column` where the shuffle/tiles aren't usable, so
# it's safe to request on any backend (on a shuffle-unsafe backend it simply is column here).
function test_trsm_strategies(backend; types = (ComplexF32, ComplexF64))
    KernelAbstractions.isgpu(backend) || return
    @testset "trsm strategy consistency -- $(backend)" begin
        # Run `tiled` on pencil `P` and require it to match the `column` baseline to tolerance.
        function check(P, T)
            _, _, zg = qgrid(T, (-3, 3), (-3, 3), (40, 40))
            tol = real(T) === Float32 ? 1e-4 : 1e-10
            σc = withenv(() -> ihlpsa(backend, zg, P, 10), "KAPSEUDO_TRSM" => "column")
            σt = withenv(() -> ihlpsa(backend, zg, P, 10), "KAPSEUDO_TRSM" => "tiled")
            # `tiled-gemm`: shared panel solve, `mul!` trailing for ComplexF32/F64 B=I; for B≠I (and
            # MultiFloats / unsupported backends) it self-gates to the `tiled` trailing kernel. Either
            # way it must match the `column` baseline to tolerance.
            σg = withenv(() -> ihlpsa(backend, zg, P, 10), "KAPSEUDO_TRSM" => "tiled-gemm")
            @test maximum(abs.(σt .- σc)) / maximum(abs.(σc)) < tol
            @test maximum(abs.(σg .- σc)) / maximum(abs.(σc)) < tol
        end

        for T in types
            # Standard (B=I) pencil. Sizes span power-of-two panels and partial last panels (m not a
            # multiple of 32), and small m < 32 (the tiled panel solve is intrinsically 32-wide and
            # masks lanes past a partial last panel — exercised here end-to-end via `tiled`).
            for m in (8, 16, 31, 32, 64, 100, 128, 256, 300)
                rng = Random.seed!(2024)
                check(MatrixPencil(schur(randn(rng, T, m, m))), T)
            end
            # Generalized (B≠I) pencil: the ONLY coverage of the two-tile generic tiled trailing
            # kernels (`_tiled_trailing`, both sweeps) and the B≠I tiled/column paths — every
            # other case here is B=I, so `b_is_identity` selects the sB-free `*_eye` kernels. m>32 so
            # the trailing update actually runs; 100 adds a partial last panel. B is diagonally
            # dominant so z*B−A stays well-conditioned across the grid.
            for m in (64, 100)
                rng = Random.seed!(2025)
                A = randn(rng, T, m, m)
                B = randn(rng, T, m, m) + T(5) * I
                check(MatrixPencil(A, B), T)
            end
        end
    end
end

# Unit coverage for the tiled TRAILING kernels across the whole knob grid the tuner may persist.
# `tune_trsm_tiled!` picks one (TC, W, gt) out of `_TILECOLS_CANDIDATES × _BLOCKWARPS_CANDIDATES × _WARPGRIDPTS_CANDIDATES`
# per device+type, but `test_trsm_strategies` above only ever runs whichever combination the
# resolvers happen to return — so every OTHER combination the tuner might select would otherwise
# ship unverified (tuned full-rate-FP64 hardware persists TC=8/W=8/gt=4, none of which is the
# zero-setup default).
#
# Not gated on `warp_trsm_safe`: that gate exists for the panel solve's warp shuffle
# and these kernels use none, so they run even where the `tiled` STRATEGY self-gates to `column`
# (Intel, Metal) — which is what makes this runnable off a CUDA box. CPU is excluded because KA's
# barrier transform cannot compile them: per-workitem values don't survive the `@synchronize`
# phase split, so the CPU codegen raises `UndefVarError`.
#
# Reference is a dense GEMM over the panel, matching the kernel formulas in trsm_tiled_kernels.jl:
# B=I gives `x[rows] += A[rows,panel]·x[panel]`, generic gives `x[rows] -= (z·B − A)[rows,panel]·x[panel]`.
function test_tiled_trailing_kernels(backend; types = (ComplexF32, ComplexF64))
    KernelAbstractions.isgpu(backend) || return
    wrows = KernelAbstractionsPseudospectra.tile_warp_rows(backend)
    @testset "tiled trailing kernels over the tuner's knob grid -- $(backend)" begin
        # (m, poff): partial row tile, partial panel, and poff=0 (backward no-op boundary).
        for T in types, (m, poff) in ((128, 32), (100, 64), (37, 0))
            rtol = _tol(T)
            psize = min(32, m - poff)
            rbase = poff + psize
            pan = (poff + 1):(poff + psize)
            g = 70                       # not a multiple of any W*gt ⇒ partial grid tile
            rng = Random.Xoshiro(0x7A11 + m + poff)
            A = randn(rng, T, m, m)
            Bm = randn(rng, T, m, m) + T(5) * I
            zv = T(2) .+ T(3 // 10) .* randn(rng, T, g)
            X0 = reduce(hcat, [randn(rng, T, m) for _ in 1:g])
            Ad, Bd, zd = _to(backend, A), _to(backend, Bm), _to(backend, zv)

            for rows in ((rbase + 1):m, 1:poff)
                isempty(rows) && continue
                rtiles = cld(length(rows), wrows)
                ref_eye = copy(X0)
                ref_eye[rows, :] .+= A[rows, pan] * X0[pan, :]
                ref_gen = copy(X0)
                for j in 1:g
                    ref_gen[rows, j] .-= (zv[j] .* Bm[rows, pan] .- A[rows, pan]) * X0[pan, j]
                end

                for TC in KernelAbstractionsPseudospectra._TILECOLS_CANDIDATES,
                    W in KernelAbstractionsPseudospectra._BLOCKWARPS_CANDIDATES,
                    gt in KernelAbstractionsPseudospectra._WARPGRIDPTS_CANDIDATES

                    ndr = (wrows * W * rtiles, cld(g, W * gt))
                    for (eye, ref) in ((true, ref_eye), (false, ref_gen))
                        # Same shared-memory admission the tuner applies before probing a TC.
                        ntiles = eye ? 1 : 2
                        ntiles * wrows * TC * sizeof(T) <=
                        KernelAbstractionsPseudospectra.device_smem_bytes(backend) || continue
                        X = _to(backend, copy(X0))
                        bV = VectorOfSimilarVectors(X)
                        if eye
                            KATRSM._tiled_trailing_eye(backend, (wrows * W, 1))(
                                bV, Ad, poff, psize, rows, gt, Val(wrows), Val(TC);
                                ndrange = ndr)
                        else
                            KATRSM._tiled_trailing(backend, (wrows * W, 1))(
                                bV, zd, Ad, Bd, poff, psize, rows, gt, Val(wrows), Val(TC);
                                ndrange = ndr)
                        end
                        KernelAbstractions.synchronize(backend)
                        @test isapprox(Array(X), ref; rtol = rtol)
                    end
                end
            end
        end
    end
end
