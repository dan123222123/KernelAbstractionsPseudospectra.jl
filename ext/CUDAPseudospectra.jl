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
        @compile_workload begin
            KAPseudospectra._precompile_ihlpsa(CUDABackend(), collect(CUDA.devices())[1],
                [ComplexF32, ComplexF64])
        end
    end
end

end
