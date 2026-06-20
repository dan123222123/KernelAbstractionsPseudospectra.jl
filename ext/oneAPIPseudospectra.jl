module oneAPIPseudospectra

using KAPseudospectra, oneAPI, PrecompileTools

# Device-interface overrides for the Intel/oneAPI backend. Defined unconditionally (not
# under `if oneAPI.functional()`) so precompile bakes them even when the worker can't
# probe the GPU; only the workload below needs the functional() guard.
KAPseudospectra.device(B::oneAPI.oneAPIBackend) = oneAPI.device()
KAPseudospectra.device!(B::oneAPI.oneAPIBackend, dev) = oneAPI.device!(dev)
KAPseudospectra.devices(B::oneAPI.oneAPIBackend) = oneAPI.devices()
KAPseudospectra.get_bgarray(B::oneAPI.oneAPIBackend) = oneAPI.oneArray
# No Level Zero free-memory query; report free system RAM (shared with the iGPU).
KAPseudospectra.device_bytes_available(B::oneAPI.oneAPIBackend) = (Sys.free_memory() |> Int)
# No memory-pool reclaim on oneAPI; fall back to GC.
KAPseudospectra.device_reclaim(B::oneAPI.oneAPIBackend) = GC.gc()
# oneAPI.jl statically declares `supports_float64` false for every backend, even
# FP64-capable Arc/Max parts. Override the package's `supports_fp64` hook with a
# device-accurate Level-Zero query so F64 auto-enables exactly where the hardware
# supports it (FP64-less iGPUs stay F32-only).
function KAPseudospectra.supports_fp64(B::oneAPI.oneAPIBackend)
    f = oneAPI.oneL0.module_properties(oneAPI.device()).fp64flags
    return (f & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP64) != 0
end

## precompile gpu code (only when a device is actually usable)
if oneAPI.functional()
    @setup_workload begin
        # Only precompile the element types the device can actually run, or the
        # F64 workload would throw at precompile time on an FP64-less iGPU. Such iGPUs
        # also need `KAPseudospectra.set_pdiv_accurate(false)` so the F32 solves don't
        # emit uncompilable `double` (see KATRSM).
        Ts = KAPseudospectra.supports_fp64(oneAPIBackend()) ? [ComplexF32, ComplexF64] : [ComplexF32]
        @compile_workload begin
            KAPseudospectra._precompile_ihlpsa(oneAPIBackend(), collect(oneAPI.devices())[1], Ts)
        end
    end
end

end
