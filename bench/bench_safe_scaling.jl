# Conservative scaling + correctness check on AMDGPU.
#
# Goals:
#   - Stay well under the iGPU's effective working set (target <500 MB
#     active state) to avoid GPU hangs at large m.
#   - Verify CPU vs AMDGPU agreement at every (m, T) pair.
#   - Sweep m ∈ {64, 128, 256, 512} on a small 20×20 grid, for both
#     ComplexF32 and ComplexF64.
#
# Usage: julia --project=test --threads=auto bench/bench_safe_scaling.jl

using KAPseudospectra
using KernelAbstractions
using LinearAlgebra
using AMDGPU
using MatrixDepot
using Statistics
using Printf
using Random

@assert AMDGPU.functional()
println("AMDGPU device: ", AMDGPU.device())
println("AMDGPU free / total: ", round(AMDGPU.free()/1024^3, digits=2), " / ",
        round(AMDGPU.total()/1024^3, digits=2), " GB")
println("Threads: ", Threads.nthreads())
flush(stdout)

const GRID_N = 20  # 400 points — small but enough to exercise the kernels

# Same convention as test/test_consistency.jl so CPU vs GPU is bit-comparable
# modulo reduction order.
function _seeded_x₀(::Type{T}, m, seed) where {T<:Complex}
    rng = MersenneTwister(seed)
    x = randn(rng, T, m)
    x ./ norm(x)
end

function run_pair(::Type{T}, m) where {T<:Complex}
    nit = 2 * ceil(Int, log2(m))  # over-converge to make CPU/GPU comparable
    A = Matrix{T}(MatrixDepot.matrixdepot("grcar", m))
    x₀ = _seeded_x₀(T, m, 0xACED)

    @printf("\n--- m=%d, grid=%d×%d, nit=%d, T=%s ---\n",
            m, GRID_N, GRID_N, nit, T)
    flush(stdout)

    t_schur = @elapsed F = schur(A)
    P = MatrixPencil(F)
    gx, gy, zg = qgrid(T, (-2.0, 2.0), (-2.0, 2.0), (GRID_N, GRID_N))
    @printf("  schur:        %7.3f s\n", t_schur)
    flush(stdout)

    # CPU reference
    t_cpu = @elapsed s_cpu = ihlpsa(CPU(), zg, P, nit; x₀=x₀)
    @assert all(isfinite, s_cpu)
    @assert size(s_cpu) == (GRID_N, GRID_N)
    @assert minimum(s_cpu) > 0
    @printf("  ihlpsa(CPU):  %7.3f s   range=[%.3e, %.3e]\n",
            t_cpu, minimum(s_cpu), maximum(s_cpu))
    flush(stdout)

    # AMDGPU
    GC.gc(); AMDGPU.HIP.reclaim()
    free_before = AMDGPU.free()
    # warmup
    s_gpu = ihlpsa(ROCBackend(), zg, P, nit; x₀=x₀)
    GC.gc(); AMDGPU.HIP.reclaim()
    free_after_warm = AMDGPU.free()
    times = Float64[]
    for _ in 1:3
        GC.gc(); AMDGPU.HIP.reclaim()
        t = @elapsed ihlpsa(ROCBackend(), zg, P, nit; x₀=x₀)
        push!(times, t)
    end
    t_gpu = minimum(times)
    @assert all(isfinite, s_gpu)
    @assert size(s_gpu) == (GRID_N, GRID_N)
    @assert minimum(s_gpu) > 0

    # Agreement
    diff_abs = maximum(abs.(s_cpu .- s_gpu))
    diff_rel = maximum(abs.(s_cpu .- s_gpu) ./ abs.(s_cpu))

    @printf("  ihlpsa(ROC):  %7.3f s   range=[%.3e, %.3e]\n",
            t_gpu, minimum(s_gpu), maximum(s_gpu))
    @printf("  speedup:      %7.2fx (CPU/GPU)\n", t_cpu / t_gpu)
    @printf("  free Δ during warmup: %.1f MB\n",
            (free_before - free_after_warm) / 1024^2)
    @printf("  CPU vs GPU max abs diff: %.3e   max rel diff: %.3e\n",
            diff_abs, diff_rel)
    flush(stdout)

    # Tolerances match test/test_consistency.jl cross-backend test.
    tol = T <: Complex{Float64} ? 1e-10 : 1e-3
    status = diff_rel <= tol ? "PASS" : "FAIL"
    @printf("  agreement: %s (rtol=%.0e)\n", status, tol)
    flush(stdout)
end

for T in (ComplexF32, ComplexF64)
    for m in (64, 128, 256, 512)
        run_pair(T, m)
    end
end

println("\ndone.")
