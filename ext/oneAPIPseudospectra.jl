module oneAPIPseudospectra

using KAPseudospectra, oneAPI, PrecompileTools

# Device-interface overrides for the Intel/oneAPI backend.
#
# These are defined UNCONDITIONALLY (not under `if oneAPI.functional()`). They
# only dispatch on the backend type and never touch hardware at definition time,
# so they are safe to bake into the precompiled image even on a machine with no
# usable device. Guarding the *definitions* on `oneAPI.functional()` is a trap:
# the precompile worker process frequently cannot probe the GPU, so functional()
# returns false there, an empty image is baked, and the methods are then missing
# at runtime even though the device works interactively. Only the GPU workload
# below (which actually runs kernels) needs the functional() guard.
KAPseudospectra.device(B::oneAPI.oneAPIBackend) = oneAPI.device()
KAPseudospectra.device!(B::oneAPI.oneAPIBackend, dev) = oneAPI.device!(dev)
KAPseudospectra.devices(B::oneAPI.oneAPIBackend) = oneAPI.devices()
KAPseudospectra.get_bgarray(B::oneAPI.oneAPIBackend) = oneAPI.oneArray
# Level Zero exposes only *total* device memory, and on an integrated GPU that
# memory is shared system RAM. There is no free-memory query, so we report free
# system RAM (matching the CPU default).
KAPseudospectra.device_bytes_available(B::oneAPI.oneAPIBackend) = (Sys.free_memory() |> Int)
# oneAPI has no memory-pool reclaim (unlike CUDA.reclaim / AMDGPU.HIP.reclaim);
# fall back to GC, as the Metal extension notes it must.
KAPseudospectra.device_reclaim(B::oneAPI.oneAPIBackend) = GC.gc()
# Most Intel iGPUs lack native FP64; oneAPI then errors with "Double type is not
# supported on this platform" on any Float64 kernel. Gate F64 grid points (and the
# F64 precompile workload below) on the device's FP64 capability flag.
function KAPseudospectra.supports_fp64(B::oneAPI.oneAPIBackend)
    f = oneAPI.oneL0.module_properties(oneAPI.device()).fp64flags
    return (f & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP64) != 0
end

## precompile gpu code (only when a device is actually usable)
if oneAPI.functional()
    @setup_workload begin
        using LinearAlgebra
        # Only precompile the element types the device can actually run, or the
        # F64 workload would throw at precompile time on an FP64-less iGPU.
        Ts = KAPseudospectra.supports_fp64(oneAPIBackend()) ? [ComplexF32, ComplexF64] : [ComplexF32]
        for T in Ts
            m = 32
            g = 100
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            @compile_workload begin
                d = [collect(oneAPI.devices())[1]]
                ihlpsa(oneAPIBackend(), zg, P, 5; devs=d)   # fixed-nit path
                ihlpsa(oneAPIBackend(), zg, P; devs=d)      # adaptive (per-point hybrid)
            end
        end
    end
end

end
