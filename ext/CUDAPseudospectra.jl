module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools

# Device-interface overrides for the CUDA backend. Defined unconditionally (not under
# `if CUDA.functional()`) so precompile bakes them even when the worker can't probe
# the GPU; only the workload below needs the functional() guard.
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
