# Focused correctness smoke test for the warp-register pencil solves on CUDA.
# Compares against LAPACK and against the column-oriented (old) GPU kernel.
#   unset LD_LIBRARY_PATH; julia --project=. bench/warp_smoke.jl
using KAPseudospectra
using KAPseudospectra.KATRSM
using KernelAbstractions
using LinearAlgebra, Random
using ArraysOfArrays
using Adapt
using CUDA

const KA = KernelAbstractions
backend = CUDABackend()
_to(x) = adapt(KAPseudospectra.get_bgarray(backend), x)
_from(x) = adapt(Array, x)
_tol(::Type{ComplexF32}) = 64 * eps(Float32)
_tol(::Type{ComplexF64}) = 64 * eps(Float64)

function rand_uppertri(T, m; rng)
    s = T(1) / T(2m + 1); M = s * triu(randn(rng, T, m, m), 1)
    for i in 1:m; M[i, i] = one(T) + s * randn(rng, T); end; M
end
function rand_lowertri(T, m; rng)
    s = T(1) / T(2m + 1); M = s * tril(randn(rng, T, m, m), -1)
    for i in 1:m; M[i, i] = one(T) + s * randn(rng, T); end; M
end

allok = true
for T in (ComplexF32, ComplexF64), m in (8, 16, 31, 32, 33, 64, 96, 257)
    rng = Random.seed!(0xACE5)
    rtol = _tol(T); g = 4; wgs = 32; R = cld(m, wgs)
    Au = rand_uppertri(T, m; rng); Bu = rand_uppertri(T, m; rng)
    Al = rand_lowertri(T, m; rng); Bl = rand_lowertri(T, m; rng)
    zv = T(2) .+ T(0.3) * randn(rng, T, g)
    b0 = reduce(hcat, [randn(rng, T, m) for _ in 1:g])
    Au_d, Bu_d, Al_d, Bl_d, zv_d = _to(Au), _to(Bu), _to(Al), _to(Bl), _to(zv)

    # forward (lower-tri)
    bv = VectorOfSimilarVectors(_to(copy(b0)))
    _batched_warp_forward_solve_pencil(backend, wgs, (wgs, g))(bv, zv_d, Al_d, Bl_d, Val(R))
    KA.synchronize(backend); fwd = _from(bv.data)
    # backward (upper-tri)
    bv2 = VectorOfSimilarVectors(_to(copy(b0)))
    _batched_warp_backward_solve_pencil(backend, wgs, (wgs, g))(bv2, zv_d, Au_d, Bu_d, Val(R))
    KA.synchronize(backend); bwd = _from(bv2.data)

    # old column-oriented for bitwise comparison
    bvc = VectorOfSimilarVectors(_to(copy(b0)))
    _batched_column_oriented_forward_solve_pencil(backend, wgs, (wgs, g))(bvc, zv_d, Al_d, Bl_d)
    KA.synchronize(backend); fwd_old = _from(bvc.data)

    okf = okb = bit = true
    for i in 1:g
        yref = LowerTriangular(zv[i] * Bl .- Al) \ b0[:, i]
        xref = UpperTriangular(zv[i] * Bu .- Au) \ b0[:, i]
        okf &= isapprox(fwd[:, i], yref; rtol)
        okb &= isapprox(bwd[:, i], xref; rtol)
        bit &= fwd[:, i] == fwd_old[:, i]
    end
    global allok &= okf & okb
    println(rpad("$T m=$m R=$R", 26),
            "fwd=", okf ? "ok " : "FAIL ",
            "bwd=", okb ? "ok " : "FAIL ",
            "bitwise-vs-old=", bit ? "yes" : "no")
end
println(allok ? "ALL CORRECTNESS CHECKS PASSED" : "SOME CHECKS FAILED")
