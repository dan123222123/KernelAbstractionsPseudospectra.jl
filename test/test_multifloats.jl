# Extended-precision (MultiFloats) tests. Opt-in via `"multifloats" in ARGS` in runtests.jl —
# MultiFloats / GenericSchur / GenericLinearAlgebra are test deps but only loaded under that gate
# (a BigFloat dense-SVD oracle is slower than the rest of the suite). Two parts:
#
#  1. test_multifloats_accuracy() — the inverse-Lanczos ihlpsa at Float64x2 must track an
#     INDEPENDENT high-precision oracle (BigFloat dense SVD via ℂsvdpsa, ~77 digits) better than
#     plain Float64 does. Migrated from examples/ihlpsa_multifloats_accuracy.jl (now a real test).
#     CPU-runnable (the extended-precision path goes through the shuffle-free column solve).
#
#  2. test_multifloats_warp_shuffle(backend) — GPU-only (where warp_trsm_safe): the per-limb
#     `_trsm_shfl` override (KAPseudospectraMultiFloatsExt) must make the warp solve BITWISE-match
#     the column solve for Complex{Float64x2}. Skipped on CPU / stock oneAPI (no usable shuffle).

using MultiFloats, GenericSchur, GenericLinearAlgebra
using MatrixDepot                # chebspec test matrix

# Local median (avoid a Statistics dependency for one call).
_median(v) = (s = sort(vec(v)); n = length(s); iseven(n) ? (s[n÷2] + s[n÷2+1]) / 2 : s[(n+1)÷2])

# (1) Accuracy vs a BigFloat oracle that shares neither algorithm (dense SVD vs inverse-Lanczos)
# nor precision (MPFR ~77 digits vs Float64x2 ~32). Small grid: each oracle point is a full
# BigFloat SVD, so keep m/grid modest for CI.
function test_multifloats_accuracy()
    @testset "MultiFloats ihlpsa accuracy vs BigFloat oracle" begin
        setprecision(BigFloat, 256)                       # ≈ 77 decimal digits
        M, G, NIT = 12, 11, 12
        HP = Complex{Float64x2}
        Aref = MatrixDepot.chebspec(Float64, M)           # standard non-normal example
        mat(::Type{T}) where {T<:Complex} = T.(Aref)
        eigs = eigvals(mat(ComplexF64))
        pad = 1.5
        region = ((minimum(real, eigs) - pad, maximum(real, eigs) + pad),
                  (minimum(imag, eigs) - pad, maximum(imag, eigs) + pad))
        grid(::Type{T}) where {T<:Complex} = qgrid(T, region[1], region[2], (G, G))[3]
        run_ihl(::Type{T}) where {T<:Complex} = ihlpsa(CPU(), grid(T), MatrixPencil(schur(mat(T))), NIT)

        s_oracle = Float64.(ℂsvdpsa(grid(Complex{BigFloat}), mat(Complex{BigFloat})))
        s64 = run_ihl(ComplexF64)
        smf = Float64.(run_ihl(HP))

        e64 = abs.(s64 .- s_oracle) ./ s_oracle
        emf = abs.(smf .- s_oracle) ./ s_oracle

        # Extended precision tracks the independent oracle far better than Float64, which loses
        # accuracy in the ill-conditioned band near the spectrum.
        @test _median(emf) < _median(e64)
        @test maximum(emf) < 1e-10           # Float64x2 follows the ~77-digit oracle closely
        @test maximum(emf) < maximum(e64)    # and beats Float64 at its worst point
    end
end

# (2) The per-limb shuffle override must reproduce the column solve bitwise on GPU. Only runs
# where the warp shuffle is usable for wide types (CUDA / AMDGPU / Metal-opt-in); skipped on CPU
# and stock oneAPI (its KernelIntrinsics @shfl for oneAPI is a TODO stub).
function test_multifloats_warp_shuffle(backend)
    (KernelAbstractions.isgpu(backend) && KAPseudospectra.warp_trsm_safe(backend, true)) || return
    @testset "MultiFloats per-limb warp shuffle (bitwise vs column) -- $(backend)" begin
        T = Complex{Float64x2}
        for m in (16, 32, 33, 64)
            g = 4
            wgs = 32
            R = cld(m, wgs)
            # Build well-conditioned triangular pencils by promoting Float64 randoms (MultiFloats
            # has no native randn); same dominant-diagonal trick as _rand_*tri keeps cond O(1).
            up(s) = (Mr = s .* triu(randn(ComplexF64, m, m), 1); [Mr[i, i] = 1 + s * randn(ComplexF64) for i in 1:m]; T.(Mr))
            lo(s) = (Mr = s .* tril(randn(ComplexF64, m, m), -1); [Mr[i, i] = 1 + s * randn(ComplexF64) for i in 1:m]; T.(Mr))
            s = 1 / (2m + 1)
            Au, Bu, Al, Bl = up(s), up(s), lo(s), lo(s)
            zv = T.(2 .+ 0.3 .* randn(ComplexF64, g))
            b0 = reduce(hcat, [T.(randn(ComplexF64, m)) for _ in 1:g])

            # forward (lower-tri): warp vs column, must be bitwise identical
            bw = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._batched_warp_forward_solve_pencil(backend, wgs, (wgs, g))(bw, _to(backend, zv), _to(backend, Al), _to(backend, Bl), Val(R))
            KernelAbstractions.synchronize(backend)
            bc = VectorOfSimilarVectors(_to(backend, copy(b0)))
            KATRSM._batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bc, _to(backend, zv), _to(backend, Al), _to(backend, Bl))
            KernelAbstractions.synchronize(backend)
            @test _from(bw.data) == _from(bc.data)
        end
    end
end
