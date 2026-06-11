module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools

# Device-interface overrides for the CUDA backend.
#
# These are defined UNCONDITIONALLY (not under `if CUDA.functional()`). They only
# dispatch on the backend type and never touch hardware at definition time, so they
# are safe to bake into the precompiled image even on a machine with no usable
# device. Guarding the *definitions* on `CUDA.functional()` is a trap: the
# precompile worker process frequently cannot probe the GPU, so functional()
# returns false there, an empty image is baked, and the methods are then missing at
# runtime even though the device works interactively. Only the GPU workload below
# (which actually runs kernels) needs the functional() guard.
KAPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
KAPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
KAPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
KAPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
KAPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
KAPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()

## precompile gpu code (only when a device is actually usable)
if CUDA.functional()
    @setup_workload begin
        using LinearAlgebra
        for T in [ComplexF32, ComplexF64]
            m = 32
            g = 100
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            @compile_workload begin
                d = [collect(CUDA.devices())[1]]
                ihlpsa(CUDABackend(), zg, P, 5; devs=d)   # fixed-nit path
                ihlpsa(CUDABackend(), zg, P; devs=d)      # adaptive (per-point hybrid)
            end
        end
    end
end

end
