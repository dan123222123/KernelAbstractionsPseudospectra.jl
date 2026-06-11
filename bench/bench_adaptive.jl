# Benchmark: fixed-`nit` `ihlpsa` vs adaptive `ihlpsa` (per-point hybrid).
#
# Reports wall-time and the converged `nit` for the Grcar matrix — a classic
# pseudospectra example with highly heterogeneous per-grid-point convergence
# (near-eigenvalue points converge in 1–2 iters; mid-pseudospectrum points need
# tens). The adaptive driver retires each grid point at its own converged depth,
# so its win over a hand-picked fixed `nit` grows with grid heterogeneity and m.
#
# Usage:
#   julia --project=test bench/bench_adaptive.jl            # CPU, m ∈ {64,128,256}
#   julia --project=test -e 'include("bench/bench_adaptive.jl"); bench(; ms=(64,128,256,512))'
# On a GPU box: bench(CUDABackend()) / bench(ROCBackend()) after `using CUDA`/`AMDGPU`.
# FP64-less GPUs (Intel iGPU, Apple) must use F32: bench(oneAPIBackend(); T=ComplexF32).

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using Printf

# Grcar matrix: Toeplitz with 1 on the diagonal and first `k` superdiagonals,
# -1 on the subdiagonal.
function grcar(::Type{T}, m, k=3) where {T}
    A = zeros(T, m, m)
    for i in 1:m
        A[i, i] = one(T)
        for j in 1:k
            i + j <= m && (A[i, i+j] = one(T))
        end
        i + 1 <= m && (A[i+1, i] = -one(T))
    end
    return A
end

function bench(backend=CPU(); ms=(64, 128, 256), region=((-1, 3), (-3, 3)), gp=(40, 40),
               T=ComplexF64)
    @printf("Grcar, %d×%d grid over real%s × imag%s, backend=%s, eltype=%s\n",
            gp[1], gp[2], region[1], region[2], backend, T)
    for m in ms
        A = grcar(T, m)
        P = MatrixPencil(A)
        _, _, zg = qgrid(T, region[1], region[2], gp)
        nit_fixed = 4 * ceil(Int, log2(m))

        # Warm up both paths (compilation + device init) before timing.
        ihlpsa(backend, zg, P, 2)
        ihlpsa(backend, zg, P; nit_max=nit_fixed)

        t_fixed = @elapsed ihlpsa(backend, zg, P, nit_fixed)
        n_adp = 0
        t_adp = @elapsed ((_, n_adp) = ihlpsa(backend, zg, P; nit_max=nit_fixed))

        @printf("  m=%4d   fixed(nit=%2d) %7.3fs   adaptive(nit=%2d) %7.3fs (%.2f×)\n",
                m, nit_fixed, t_fixed, n_adp, t_adp, t_fixed / t_adp)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench()
end
