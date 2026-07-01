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
# Intel SLM is per-subslice, and a subslice runs ~one workgroup's SLM at a time, so the per-"SM"
# occupancy budget ≈ the per-workgroup SLM (no separate per-SM figure to query). Used only for the
# no-probe analytic `tiled_tc` default — the timed `tune_trsm_tc!` probe is the accurate path here.
KAPseudospectra.device_smem_per_sm(B::oneAPI.oneAPIBackend) =
    Int(oneAPI.oneL0.compute_properties(oneAPI.device()).maxSharedLocalMemory)
# oneAPI.jl statically declares `supports_float64` false for every backend, even
# FP64-capable Arc/Max parts. Override the package's `supports_fp64` hook with a
# device-accurate Level-Zero query so F64 auto-enables exactly where the hardware
# supports it (FP64-less iGPUs stay F32-only).
function KAPseudospectra.supports_fp64(B::oneAPI.oneAPIBackend)
    f = oneAPI.oneL0.module_properties(oneAPI.device()).fp64flags
    return (f & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP64) != 0
end

# `tiled` is correct on oneAPI only with BOTH the KernelIntrinsics oneAPI shuffle backend (the
# stock release ships only a stub) AND a SIMD width pinned to the warp width — both arrive via the
# opt-in `set_intel_force_simd32!` + patched KernelIntrinsics. MultiFloats (`wide`) always stay on
# `column`: their tiled path crashes the SPIR-V translator on oneAPI. See DESIGN_TRSM.md "Open
# items" for why we pin SIMD32 via env var rather than `reqd_subgroup_size!`.
KAPseudospectra.warp_trsm_safe(::oneAPI.oneAPIBackend, wide::Bool) =
    !wide &&
    KAPseudospectra.intel_force_simd32() &&
    Base.get_extension(KernelIntrinsics, :KernelIntrinsicsoneAPIExt) !== nothing

# `tiled-gemm`'s `mul!` trailing has no reliable fast complex GEMM on oneMKL here — keep oneAPI on the
# regular `tiled` trailing kernel (`tiled-gemm` → `tiled`).
KAPseudospectra.tiled_gemm_safe(::oneAPI.oneAPIBackend, ::Type) = false

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
