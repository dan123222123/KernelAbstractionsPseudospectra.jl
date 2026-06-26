# Correctness tests for KATRSM triangular-solve kernels.
#
# Each algorithmic variant in KATRSM is checked against the LAPACK reference
# obtained by solving with `LowerTriangular(L) \ b` / `UpperTriangular(U) \ b`
# (which dispatch to LAPACK trtrs). Pencil variants are checked against
# `LowerTriangular(z*B - A) \ b` / `UpperTriangular(z*B - A) \ b`.
#
# Layout
# ------
#  * "KATRSM core (CPU host functions)" testset — always runs on include.
#    Exercises the pure-Julia host loops in trsm_core.jl / trsm_pencil_core.jl
#    (forward_solve!, backward_solve!, forward_solve_pencil!, …) — these don't
#    dispatch through KernelAbstractions and are CPU-only by definition.
#  * test_katrsm_kernels(backend) — a function over a KernelAbstractions
#    backend. Exercises every KA kernel and wrapper KATRSM exposes (naive
#    batched, column-oriented batched, blkco single-block, blkco multi-block
#    partition wrappers) using device-resident inputs. runtests.jl calls this
#    for CPU() unconditionally and for each available GPU backend.
#
# All tolerances are 32 * eps(T): the test matrices are diagonally dominant
# with cond(M) = O(1) and pencil shifts are biased so cond(z*B - A) = O(1)
# too, so a backward-stable solve must agree with LAPACK to a small constant
# multiple of eps(T) regardless of backend.

using Test, KAPseudospectra, KernelAbstractions, LinearAlgebra
using ArraysOfArrays
using Adapt
using Random

const KATRSM = KAPseudospectra.KATRSM

# Strictly diagonally dominant triangular matrices: diag ≈ 1, off-diagonals
# scaled by 1/(2m+1) so each row's off-diagonal sum is ≤ 1/2. Condition number
# is O(1) independent of m, so two backward-stable triangular solves should
# agree elementwise to a small constant times eps(T).
function _rand_uppertri(T, m; rng=Random.default_rng())
    s = T(1) / T(2 * m + 1)
    M = s * triu(randn(rng, T, m, m), 1)
    for i = 1:m
        M[i, i] = one(T) + s * randn(rng, T)
    end
    M
end
function _rand_lowertri(T, m; rng=Random.default_rng())
    s = T(1) / T(2 * m + 1)
    M = s * tril(randn(rng, T, m, m), -1)
    for i = 1:m
        M[i, i] = one(T) + s * randn(rng, T)
    end
    M
end

# A few × eps(T): F32 ≈ 1.2e-7, F64 ≈ 2.2e-16.
_tol(::Type{ComplexF32}) = 32 * eps(Float32)
_tol(::Type{ComplexF64}) = 32 * eps(Float64)

# Move CPU array(s) onto the backend's device array type. CPU() ↦ Array.
_to(backend, x) = adapt(KAPseudospectra.get_bgarray(backend), x)
# Pull device array(s) back to host for comparison.
_from(x) = adapt(Array, x)

# CPU-only host-function tests. These don't go through KernelAbstractions.
@testset "KATRSM core (CPU host functions)" begin
    Random.seed!(0xACE5)

    @testset "non-pencil core" begin
        for T in (ComplexF32, ComplexF64), m in (1, 8, 16)
            rtol = _tol(T)
            U = _rand_uppertri(T, m)
            L = _rand_lowertri(T, m)
            b = randn(T, m)

            xref = UpperTriangular(U) \ b
            x = copy(b); KATRSM.backward_solve!(x, U)
            @test isapprox(x, xref; rtol=rtol)
            x = copy(b); KATRSM.column_oriented_backward_solve!(x, U)
            @test isapprox(x, xref; rtol=rtol)

            yref = LowerTriangular(L) \ b
            y = copy(b); KATRSM.forward_solve!(y, L)
            @test isapprox(y, yref; rtol=rtol)
            y = copy(b); KATRSM.column_oriented_forward_solve!(y, L)
            @test isapprox(y, yref; rtol=rtol)
        end
    end

    @testset "pencil core" begin
        for T in (ComplexF32, ComplexF64), m in (1, 8, 16)
            rtol = _tol(T)
            Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
            Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
            b = randn(T, m)
            # |z| ≈ 2.06 keeps z*B - A's diagonal magnitude well above zero
            z = T(2 + 0.5im)

            xref = UpperTriangular(z * Bu .- Au) \ b
            x = copy(b); KATRSM.backward_solve_pencil!(x, z, Au, Bu)
            @test isapprox(x, xref; rtol=rtol)
            x = copy(b); KATRSM.column_oriented_backward_solve_pencil!(x, z, Au, Bu)
            @test isapprox(x, xref; rtol=rtol)

            yref = LowerTriangular(z * Bl .- Al) \ b
            y = copy(b); KATRSM.forward_solve_pencil!(y, z, Al, Bl)
            @test isapprox(y, yref; rtol=rtol)
            y = copy(b); KATRSM.column_oriented_forward_solve_pencil!(y, z, Al, Bl)
            @test isapprox(y, yref; rtol=rtol)
        end
    end

    # Naive batched KA kernels for non-pencil triangular solves. These take a
    # Vector{Matrix} / Vector{Vector}, which is host-resident pointer storage
    # and does not adapt to a GPU; they're CPU-only by construction (and not
    # exercised by ihlpsa's production paths, which use the pencil variants).
    @testset "batched non-pencil naive (KA, CPU)" begin
        backend = CPU()
        for T in (ComplexF32, ComplexF64), m in (8, 16)
            rtol = _tol(T)
            g = 5
            UV = [_rand_uppertri(T, m) for _ = 1:g]
            LV = [_rand_lowertri(T, m) for _ = 1:g]
            bV0 = [randn(T, m) for _ = 1:g]

            xV = deepcopy(bV0)
            KATRSM._batched_backward_solve(backend, 1)(xV, UV, ndrange=g)
            KernelAbstractions.synchronize(backend)
            for i = 1:g
                @test isapprox(xV[i], UpperTriangular(UV[i]) \ bV0[i]; rtol=rtol)
            end

            yV = deepcopy(bV0)
            KATRSM._batched_forward_solve(backend, 1)(yV, LV, ndrange=g)
            KernelAbstractions.synchronize(backend)
            for i = 1:g
                @test isapprox(yV[i], LowerTriangular(LV[i]) \ bV0[i]; rtol=rtol)
            end
        end
    end
end

# Backend-parameterized KA kernel tests. Called once per available backend
# from runtests.jl.
function test_katrsm_kernels(backend; types=(ComplexF32, ComplexF64))
    @testset "KATRSM kernels -- $(backend)" begin
        Random.seed!(0xACE5)

        @testset "batched pencil naive (KA wrappers)" begin
            for T in types, m in (8, 16)
                rtol = _tol(T)
                g = 5
                Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
                # bias z away from 1 so z*B - A stays well-conditioned
                zv = T(2) .+ T(0.3) * randn(T, g)
                b0_mat = reduce(hcat, [randn(T, m) for _ = 1:g])

                # Move to backend
                bv_dev = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                Au_d = _to(backend, Au); Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al); Bl_d = _to(backend, Bl)
                zv_d = _to(backend, zv)

                # The wrapper's `Vector.(xV)` already copies device data → host.
                xv = KATRSM.batched_backward_solve_pencil(bv_dev, zv_d, Au_d, Bu_d)
                for i = 1:g
                    xref = UpperTriangular(zv[i] * Bu .- Au) \ b0_mat[:, i]
                    @test isapprox(xv[i], xref; rtol=rtol)
                end

                yv = KATRSM.batched_forward_solve_pencil(bv_dev, zv_d, Al_d, Bl_d)
                for i = 1:g
                    yref = LowerTriangular(zv[i] * Bl .- Al) \ b0_mat[:, i]
                    @test isapprox(yv[i], yref; rtol=rtol)
                end
            end
        end

        @testset "batched column-oriented pencil (KA, workgroup parallel)" begin
            # wgs=1 exercises the degenerate single-thread path; wgs=2,4 split
            # the column-update loop across multiple workitems.
            for T in types, m in (8, 16), wgs in (1, 2, 4)
                rtol = _tol(T)
                g = 4
                Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
                zv = T(2) .+ T(0.3) * randn(T, g)
                b0_mat = reduce(hcat, [randn(T, m) for _ = 1:g])

                Au_d = _to(backend, Au); Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al); Bl_d = _to(backend, Bl)
                zv_d = _to(backend, zv)

                bv = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                KATRSM._batched_column_oriented_backward_solve_pencil(backend, wgs, (wgs, g))(bv, zv_d, Au_d, Bu_d)
                KernelAbstractions.synchronize(backend)
                bv_h = _from(bv.data)
                for i = 1:g
                    xref = UpperTriangular(zv[i] * Bu .- Au) \ b0_mat[:, i]
                    @test isapprox(bv_h[:, i], xref; rtol=rtol)
                end

                bv = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                KATRSM._batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bv, zv_d, Al_d, Bl_d)
                KernelAbstractions.synchronize(backend)
                bv_h = _from(bv.data)
                for i = 1:g
                    yref = LowerTriangular(zv[i] * Bl .- Al) \ b0_mat[:, i]
                    @test isapprox(bv_h[:, i], yref; rtol=rtol)
                end
            end
        end

        @testset "batched warp-register pencil (KA, shuffle, GPU only)" begin
            # The warp-register kernels use warp shuffles (@shfl), so they run on GPU
            # backends only. They must agree with LAPACK and be BITWISE-identical to the
            # column-oriented kernel (same pencil/_pdiv/update order). wgs is fixed at the
            # warp width 32; R = cld(m, 32) is the per-lane register-slot count.
            # Skipped where the backend's warp shuffle isn't usable for these solves: on
            # oneAPI that needs the KernelIntrinsics shuffle backend + a pinned SIMD32 width
            # (see `warp_trsm_safe`) — stock oneAPI routes ihlpsa to the column solve, so the
            # warp kernels aren't part of its supported path and would just `## TODO`-stub out.
            if KernelAbstractions.isgpu(backend) && KAPseudospectra.warp_trsm_safe(backend, false)
                wgs = 32
                for T in types, m in (8, 16, 31, 32, 33, 64, 96)
                    rtol = _tol(T)
                    g = 4
                    R = cld(m, wgs)
                    Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
                    Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
                    zv = T(2) .+ T(0.3) * randn(T, g)
                    b0_mat = reduce(hcat, [randn(T, m) for _ = 1:g])

                    Au_d = _to(backend, Au); Bu_d = _to(backend, Bu)
                    Al_d = _to(backend, Al); Bl_d = _to(backend, Bl)
                    zv_d = _to(backend, zv)

                    # forward (lower-triangular)
                    bw = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                    KATRSM._batched_warp_forward_solve_pencil(backend, wgs, (wgs, g))(bw, zv_d, Al_d, Bl_d, Val(R))
                    KernelAbstractions.synchronize(backend)
                    bc = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                    KATRSM._batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bc, zv_d, Al_d, Bl_d)
                    KernelAbstractions.synchronize(backend)
                    bw_h = _from(bw.data); bc_h = _from(bc.data)
                    for i = 1:g
                        yref = LowerTriangular(zv[i] * Bl .- Al) \ b0_mat[:, i]
                        @test isapprox(bw_h[:, i], yref; rtol=rtol)
                        @test bw_h[:, i] == bc_h[:, i]      # bitwise vs column-oriented
                    end

                    # backward (upper-triangular)
                    bw = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                    KATRSM._batched_warp_backward_solve_pencil(backend, wgs, (wgs, g))(bw, zv_d, Au_d, Bu_d, Val(R))
                    KernelAbstractions.synchronize(backend)
                    bc = VectorOfSimilarVectors(_to(backend, copy(b0_mat)))
                    KATRSM._batched_column_oriented_backward_solve_pencil(backend, wgs, (wgs, g))(bc, zv_d, Au_d, Bu_d)
                    KernelAbstractions.synchronize(backend)
                    bw_h = _from(bw.data); bc_h = _from(bc.data)
                    for i = 1:g
                        xref = UpperTriangular(zv[i] * Bu .- Au) \ b0_mat[:, i]
                        @test isapprox(bw_h[:, i], xref; rtol=rtol)
                        @test bw_h[:, i] == bc_h[:, i]
                    end
                end
            end
        end

        @testset "blkco non-pencil kernels (KA, single block)" begin
            for T in types, m in (8, 16)
                rtol = _tol(T)
                U = _rand_uppertri(T, m)
                L = _rand_lowertri(T, m)
                b0 = randn(T, m)
                xref = UpperTriangular(U) \ b0
                yref = LowerTriangular(L) \ b0

                U_d = _to(backend, U); L_d = _to(backend, L)

                for kfn in (KATRSM._blkco_backward_solve_sm3v1,
                            KATRSM._blkco_backward_solve_sm3v2,
                            KATRSM._blkco_backward_solve_sm2,
                            KATRSM._blkco_backward_solve_sm1)
                    d = _to(backend, copy(b0))
                    kfn(backend, m)(d, U_d, ndrange=m)
                    KernelAbstractions.synchronize(backend)
                    @test isapprox(_from(d), xref; rtol=rtol)
                end

                d = _to(backend, copy(b0))
                KATRSM._blkco_forward_solve_sm3(backend, m)(d, L_d, ndrange=m)
                KernelAbstractions.synchronize(backend)
                @test isapprox(_from(d), yref; rtol=rtol)
            end
        end

        @testset "blkco pencil kernels (KA, single block)" begin
            for T in types, m in (8, 16)
                rtol = _tol(T)
                Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
                z = T(2 + 0.7im)
                b0 = randn(T, m)

                Au_d = _to(backend, Au); Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al); Bl_d = _to(backend, Bl)

                d = _to(backend, copy(b0))
                KATRSM._blkco_backward_solve_pencil(backend, m)(d, z, Au_d, Bu_d, ndrange=m)
                KernelAbstractions.synchronize(backend)
                @test isapprox(_from(d), UpperTriangular(z * Bu .- Au) \ b0; rtol=rtol)

                d = _to(backend, copy(b0))
                KATRSM._blkco_forward_solve_pencil(backend, m)(d, z, Al_d, Bl_d, ndrange=m)
                KernelAbstractions.synchronize(backend)
                @test isapprox(_from(d), LowerTriangular(z * Bl .- Al) \ b0; rtol=rtol)
            end
        end

        @testset "blkco pencil wrappers (multi-block partition)" begin
            for T in types, m in (16, 64), nblkcols in (8, 16)
                nblkcols > m && continue
                rtol = _tol(T)
                Au = _rand_uppertri(T, m); Bu = _rand_uppertri(T, m)
                Al = _rand_lowertri(T, m); Bl = _rand_lowertri(T, m)
                z = T(2 + 0.6im)
                b0 = randn(T, m)

                Au_d = _to(backend, Au); Bu_d = _to(backend, Bu)
                Al_d = _to(backend, Al); Bl_d = _to(backend, Bl)

                d = _to(backend, copy(b0))
                KATRSM.blkco_backward_solve_pencil!(d, z, Au_d, Bu_d; nblkcols)
                KernelAbstractions.synchronize(backend)
                @test isapprox(_from(d), UpperTriangular(z * Bu .- Au) \ b0; rtol=rtol)

                d = _to(backend, copy(b0))
                KATRSM.blkco_forward_solve_pencil!(d, z, Al_d, Bl_d; nblkcols)
                KernelAbstractions.synchronize(backend)
                @test isapprox(_from(d), LowerTriangular(z * Bl .- Al) \ b0; rtol=rtol)
            end
        end
    end
end

# End-to-end consistency of the GPU trsm strategies (KAPSEUDO_TRSM): the per-warp `warp`
# and `tiled` solves, and `auto`, must match the shuffle-free `column` baseline to element-
# type tolerance, across sizes incl. partial last panels (m not a multiple of 32).
function test_trsm_strategies(backend; types=(ComplexF32, ComplexF64))
    KernelAbstractions.isgpu(backend) || return
    @testset "trsm strategy consistency -- $(backend)" begin
        # Sizes span power-of-two panels and partial last panels (m not a multiple of 32). 64/256
        # were folded in from the former bench/tiled_check.jl so its coverage isn't lost. 512 (the
        # R=16 warp-compile cliff) is left to bench/warp_trsm_bench.jl to keep CI compile time sane.
        for T in types, m in (32, 64, 100, 128, 256, 300)
            rng = Random.seed!(2024)
            P = MatrixPencil(schur(randn(rng, T, m, m)))
            _, _, zg = qgrid(T, (-3, 3), (-3, 3), (40, 40))
            σc = withenv(() -> ihlpsa(backend, zg, P, 10), "KAPSEUDO_TRSM" => "column")
            tol = real(T) === Float32 ? 1e-4 : 1e-10
            # `auto` is always safe (it routes warp/tiled→column on backends where the shuffle
            # isn't usable, e.g. stock oneAPI). Only force explicit `warp`/`tiled` where they're
            # actually correct (warp_trsm_safe) — otherwise they'd run the stub shuffle and fail.
            strategies = KAPseudospectra.warp_trsm_safe(backend, false) ? ("warp", "tiled", "auto") : ("auto",)
            for strat in strategies
                σ = withenv(() -> ihlpsa(backend, zg, P, 10), "KAPSEUDO_TRSM" => strat)
                @test maximum(abs.(σ .- σc)) / maximum(abs.(σc)) < tol
            end
        end
    end
end
