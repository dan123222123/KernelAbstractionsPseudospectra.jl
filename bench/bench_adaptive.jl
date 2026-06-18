# Benchmark: fixed-`nit` `ihlpsa` vs adaptive `ihlpsa` (per-point hybrid), single device.
#
# Reports wall-time and the converged `nit` for the Grcar matrix — a classic
# pseudospectra example with highly heterogeneous per-grid-point convergence
# (near-eigenvalue points converge in 1–2 iters; mid-pseudospectrum points need
# tens). The adaptive driver retires each grid point at its own converged depth,
# so its win over a hand-picked fixed `nit` grows with grid heterogeneity and m.
#
# This is the single-device showcase (the fixed-vs-adaptive race); multi-GPU
# scaling lives in bench/multigpu_bench.jl.
#
# Usage:
#   julia --project=test bench/bench_adaptive.jl            # CPU, m ∈ {64,128,256}
#   julia --project=test -e 'include("bench/bench_adaptive.jl"); bench(; ms=(64,128,256,512))'
# On a GPU box: bench(CUDABackend()) / bench(ROCBackend()) after `using CUDA`/`AMDGPU`
# (restricted to the first device). FP64-less GPUs (Intel iGPU, Apple) need F32:
# bench(oneAPIBackend(); T=ComplexF32).

include(joinpath(@__DIR__, "bench_common.jl"))   # deps + shared helpers (grcar_pencil, …)

function bench(backend=CPU(); ms=(64, 128, 256), region=((-1, 3), (-3, 3)), gp=(64, 64),
               T=ComplexF64, devs=missing)
    # Single-device showcase: on a GPU, restrict to the first device so this measures
    # the per-device fixed-vs-adaptive win (multi-GPU scaling is multigpu_bench.jl).
    if ismissing(devs) && KernelAbstractions.isgpu(backend)
        devs = [first(KAPseudospectra.devices(backend))]
    end
    @printf("Grcar, %d×%d grid over real%s × imag%s, backend=%s, eltype=%s\n",
            gp[1], gp[2], region[1], region[2], backend, T)
    for m in ms
        P = grcar_pencil(T, m)
        _, _, zg = qgrid(T, region[1], region[2], gp)
        nit_fixed = 4 * ceil(Int, log2(m))

        # Warm up both paths (compilation + device init) before timing.
        ihlpsa(backend, zg, P, 2; devs)
        ihlpsa(backend, zg, P; nit_max=nit_fixed, devs)

        t_fixed = @elapsed ihlpsa(backend, zg, P, nit_fixed; devs)
        # Public `ihlpsa(...; …)` returns only σ; call the internal driver for the
        # convergence depth to report alongside the timing.
        local n_adp
        t_adp = @elapsed begin
            _, nit_grid = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; nit_max=nit_fixed, devs)
            n_adp = maximum(nit_grid)
        end

        @printf("  m=%4d   fixed(nit=%2d) %7.3fs   adaptive(nit=%2d) %7.3fs (%.2f×)\n",
                m, nit_fixed, t_fixed, n_adp, t_adp, t_fixed / t_adp)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench()
end
