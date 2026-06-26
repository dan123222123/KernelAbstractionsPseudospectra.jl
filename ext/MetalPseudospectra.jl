module MetalPseudospectra

using KAPseudospectra, Metal, PrecompileTools

# Device-interface overrides for the Apple/Metal backend (single GPU, unified memory,
# so the multi-device plumbing collapses to one device). Defined unconditionally (not
# under `if Metal.functional()`) so precompile bakes them even off-device; only the
# workload below needs the functional() guard.
#
# NOTE: untested on hardware (developed without an Apple machine). The API is
# pinned to Metal.jl ≥ 1.x: `MetalBackend`, `MtlArray`, `Metal.device()`,
# `Metal.device!(::MTLDevice)`, and the `MTLDevice` properties
# `recommendedMaxWorkingSetSize`/`currentAllocatedSize`.
KAPseudospectra.device(B::Metal.MetalBackend) = Metal.device()
KAPseudospectra.device!(B::Metal.MetalBackend, dev) = Metal.device!(dev)
# Metal has no `devices()` enumerator — Apple Silicon shows a single GPU.
KAPseudospectra.devices(B::Metal.MetalBackend) = [Metal.device()]
KAPseudospectra.get_bgarray(B::Metal.MetalBackend) = Metal.MtlArray
# Free working-set bytes ≈ recommended max working set − currently allocated.
# Apple GPUs use unified memory, so this is effectively a slice of system RAM.
function KAPseudospectra.device_bytes_available(B::Metal.MetalBackend)
    dev = Metal.device()
    Int(dev.recommendedMaxWorkingSetSize) - Int(dev.currentAllocatedSize)
end
# Metal has no explicit memory-pool reclaim; fall back to GC.
KAPseudospectra.device_reclaim(B::Metal.MetalBackend) = GC.gc()
# Per-device queries for the trsm routing. Apple SIMD-groups are 32-wide; threadgroup memory is
# the @localmem budget for the tiled solve. NOTE: untested on hardware.
KAPseudospectra.warp_width(B::Metal.MetalBackend) = 32
KAPseudospectra.device_smem_bytes(B::Metal.MetalBackend) = Int(Metal.device().maxThreadgroupMemoryLength)
# Make the warp/tiled fast path opt-in on Metal (like oneAPI), defaulting to the always-correct
# column solve under `auto`. Metal's shuffles ARE correct, so this is a policy gate (modest
# speedup vs the column default) rather than a correctness one — flip it with set_metal_warp_trsm!.
KAPseudospectra.warp_trsm_safe(::Metal.MetalBackend, wide::Bool) =
    !wide && KAPseudospectra.metal_warp_trsm()
# No `supports_fp64` override needed: the default defers to `supports_float64`, which
# Metal declares `false` (no `double` in MSL), so F64 paths skip Metal — F32 only. Metal
# also needs `KAPseudospectra.set_pdiv_accurate(false)` so the F32 solves compile (see KATRSM).

## precompile gpu code (only when a device is actually usable)
if Metal.functional()
    @setup_workload begin
        # Apple GPUs have no FP64 — ComplexF32 only.
        @compile_workload begin
            KAPseudospectra._precompile_ihlpsa(MetalBackend(), Metal.device(), [ComplexF32])
        end
    end
end

end
