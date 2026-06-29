module oneAPIPseudospectra

using KAPseudospectra, oneAPI, PrecompileTools
import KernelIntrinsics

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
# Per-device queries for the trsm routing. Warp width is pinned to 32 by the SIMD32 override
# (set_intel_force_simd32!); the @localmem budget is the device's max shared-local-memory (queried
# via Level Zero — e.g. 64 KB on Intel UHD Graphics, vs the conservative 48 KB default).
KAPseudospectra.warp_width(B::oneAPI.oneAPIBackend) = 32
KAPseudospectra.device_smem_bytes(B::oneAPI.oneAPIBackend) =
    Int(oneAPI.oneL0.compute_properties(oneAPI.device()).maxSharedLocalMemory)
# oneAPI.jl statically declares `supports_float64` false for every backend, even
# FP64-capable Arc/Max parts. Override the package's `supports_fp64` hook with a
# device-accurate Level-Zero query so F64 auto-enables exactly where the hardware
# supports it (FP64-less iGPUs stay F32-only).
function KAPseudospectra.supports_fp64(B::oneAPI.oneAPIBackend)
    f = oneAPI.oneL0.module_properties(oneAPI.device()).fp64flags
    return (f & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP64) != 0
end

# The `tiled` strategy may use the tiled (shuffle) solve on oneAPI ONLY when it is actually correct,
# which needs BOTH: (1) the KernelIntrinsics oneAPI shuffle backend — the standard 0.1.8 ships only a
# `## TODO` stub, so `@shfl` silently miscompiles there; and (2) a SIMD width pinned to the warp
# width (else a 32-lane workgroup spans several Intel subgroups). Both arrive together via the opt-in
# `set_intel_force_simd32!` + the patched KernelIntrinsics. Without them `warp_trsm_safe` is false, so
# `tiled` self-gates to the shuffle-free `column` solve — correct on stock releases, just without the
# tiled speedup.
# MultiFloats (`wide`) always stay on `column`: their tiled path crashes the SPIR-V translator on
# oneAPI (the per-limb shuffle kernel + reqd-sub-group-size); column is correct (and only ~1.2x
# slower) for extended precision here.
KAPseudospectra.warp_trsm_safe(::oneAPI.oneAPIBackend, wide::Bool) =
    !wide &&
    KAPseudospectra.intel_force_simd32() &&
    Base.get_extension(KernelIntrinsics, :KernelIntrinsicsoneAPIExt) !== nothing

# The SIMD-width pin for the tiled fast path comes from the main-module `__init__`
# setting `IGC_ForceOCLSIMDWidth` (when `set_intel_force_simd32!` is enabled). We deliberately
# do NOT use oneAPI's `reqd_subgroup_size!` (the per-kernel `intel_reqd_sub_group_size`
# execution mode) here: applied globally it crashes the SPIR-V translator on the MultiFloat
# kernels (whereas the env var, a driver-level flag with no per-kernel metadata, pins both the
# IEEE tiled kernels and the MultiFloat column kernels fine). reqd-sub-group-size is the
# cleaner long-term mechanism but needs per-kernel application (IEEE tiled kernels only) first.

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
