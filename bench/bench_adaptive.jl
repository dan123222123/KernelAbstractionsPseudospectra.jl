# Benchmark: fixed-`nit` `ihlpsa` vs Tier-1 `ihlpsa_adaptive`.
#
# Reports wall-time and the converged `nit` for the Grcar matrix — a classic
# pseudospectra example with highly heterogeneous per-grid-point convergence
# (near-eigenvalue points converge in 1–2 iters; mid-pseudospectrum points need
# tens). This is the baseline that quantifies the Tier-1 win ("stop guessing
# nit") and motivates Tier 2 (per-point compaction, the real wall-clock speedup).
#
# NOTE: Tier 1 re-runs Lanczos from iteration 1 each chunk and still pays the
# slowest-point cost on the whole grid, so its raw wall-time vs a single
# well-chosen fixed-`nit` run is roughly break-even — the value is not having to
# pick `nit` by hand. The dramatic speedup is a Tier 2 deliverable.
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

        # Warm up all paths (compilation + device init) before timing.
        ihlpsa(backend, zg, P, 2)
        ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed)
        ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, compact=true)
        ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, resumable=true)
        ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, compact=true, resumable=true)

        t_fixed = @elapsed ihlpsa(backend, zg, P, nit_fixed)
        n_glob = 0
        t_glob = @elapsed ((_, n_glob) = ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed))
        n_comp = 0
        t_comp = @elapsed ((_, n_comp) = ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, compact=true))
        n_res = 0
        t_res = @elapsed ((_, n_res) = ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, resumable=true))
        n_hyb = 0
        t_hyb = @elapsed ((_, n_hyb) = ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, compact=true, resumable=true))

        @printf("  m=%4d   fixed(nit=%2d) %7.3fs   global(nit=%2d) %7.3fs (%.2f×)   compact(nit=%2d) %7.3fs (%.2f×)   resumable(nit=%2d) %7.3fs (%.2f×)   hybrid(nit=%2d) %7.3fs (%.2f×)\n",
                m, nit_fixed, t_fixed, n_glob, t_glob, t_fixed / t_glob, n_comp, t_comp, t_fixed / t_comp,
                n_res, t_res, t_fixed / t_res, n_hyb, t_hyb, t_fixed / t_hyb)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench()
end
