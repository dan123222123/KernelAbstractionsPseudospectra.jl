# Precompile workloads: the CPU workload runs here; the GPU extensions call the two
# helpers from their own @compile_workload blocks.

# Shared precompile body for the GPU extensions: exercises the fixed and adaptive `ihlpsa`
# paths, for both a B=I and a true B≠I pencil, on one device, so each extension's
# `@compile_workload` traces the backend-specialized method instances.
function _precompile_ihlpsa(backend, dev, Ts)
    for T in Ts
        _, _, zg = qgrid(T, (-4, 4), (-4, 4), (100, 100))
        P = MatrixPencil(schur(randn(T, 32, 32)))   # B = I (single matrix)
        ihlpsa(backend, zg, P, 5; devs = [dev])   # fixed-nit path
        ihlpsa(backend, zg, P; devs = [dev])      # adaptive (per-point hybrid)
        # True matrix pencil (B ≠ I): also trace the generalized-Schur construction
        # path so a first MatrixPencil(A, B) call isn't a cold compile.
        Pg = MatrixPencil(randn(T, 32, 32), randn(T, 32, 32) + T(5) * I)
        ihlpsa(backend, zg, Pg, 5; devs = [dev])
    end
    return nothing
end

# Opt-in (PRECOMPILE_GPU_KERNELS) extension of the GPU precompile workload: exercises the
# `tiled` solve so its CodeInstances get created during precompilation. Guarded: a flaky
# precompile-worker GPU *execution* is tolerated since the kernel still *compiles* (the
# CodeInstance is what we need cached), so a failure degrades gracefully.
function _precompile_gpu_kernels(backend, dev, Ts)
    try
        withenv("KAPSEUDO_TRSM" => "tiled") do
            _precompile_ihlpsa(backend, dev, Ts)
        end
    catch err
        @debug "tiled precompile skipped" exception = err
    end
    return nothing
end

@setup_workload begin
    using LinearAlgebra
    using KernelAbstractions
    @compile_workload begin
        for T in [ComplexF32, ComplexF64]
            m = 16
            g = 10
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            ℂsvdpsa(zg, P)
            ℝsvdpsa(zg, P)
            ihlpsa(CPU(), zg, P, m)      # fixed-nit path
            ihlpsa(CPU(), zg, P)         # adaptive (per-point hybrid)
        end
    end
end
