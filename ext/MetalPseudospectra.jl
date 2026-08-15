module MetalPseudospectra

using KernelAbstractionsPseudospectra, Metal, PrecompileTools

# Device-interface overrides for the Apple/Metal backend (single GPU, unified memory, so the
# multi-device plumbing collapses to one device). Defined unconditionally (not under
# `if Metal.functional()`) so precompile bakes them even off-device.
#
# NOTE: untested on hardware. Pinned to Metal.jl ≥ 1.x:
# `MetalBackend`, `MtlArray`, `Metal.device()`, `Metal.device!(::MTLDevice)`, and the
# MTLDevice properties `recommendedMaxWorkingSetSize`/`currentAllocatedSize`/
# `maxThreadgroupMemoryLength`.
KernelAbstractionsPseudospectra.device(B::Metal.MetalBackend) = Metal.device()
KernelAbstractionsPseudospectra.device!(B::Metal.MetalBackend, dev) = Metal.device!(dev)
# Metal has no `devices()` enumerator — Apple Silicon shows a single GPU.
KernelAbstractionsPseudospectra.devices(B::Metal.MetalBackend) = [Metal.device()]
KernelAbstractionsPseudospectra.get_bgarray(B::Metal.MetalBackend) = Metal.MtlArray
# Free working-set bytes ≈ recommended max working set − currently allocated.
# Apple GPUs use unified memory, so this is effectively a slice of system RAM.
function KernelAbstractionsPseudospectra.device_bytes_available(B::Metal.MetalBackend)
    dev = Metal.device()
    # Allocation can exceed the recommended max under memory pressure — floor at 0.
    max(0, Int(dev.recommendedMaxWorkingSetSize) - Int(dev.currentAllocatedSize))
end
# Metal has no explicit memory-pool reclaim; fall back to GC.
KernelAbstractionsPseudospectra.device_reclaim(B::Metal.MetalBackend) = GC.gc()
# Per-device queries for the trsm routing. Apple SIMD-groups are 32-wide; threadgroup memory
# is the @localmem budget for the tiled solve. NOTE: untested on hardware.
KernelAbstractionsPseudospectra.warp_width(B::Metal.MetalBackend) = 32
function KernelAbstractionsPseudospectra.device_smem_bytes(B::Metal.MetalBackend)
    Int(Metal.device().maxThreadgroupMemoryLength)
end
# Apple GPUs don't expose a per-core total threadgroup-memory figure; the per-threadgroup
# length is a conservative LOWER bound on the per-"SM" budget. Feeds only the no-probe
# analytic `tile_cols` default.
function KernelAbstractionsPseudospectra.device_smem_per_sm(B::Metal.MetalBackend)
    Int(Metal.device().maxThreadgroupMemoryLength)
end
# Tiled fast path is opt-in on Metal (POLICY gate, not correctness — Metal's IEEE-F32
# shuffles are correct). Off by default until `set_metal_warp_trsm!` opts in. `!wide` is
# moot here: no FP64 on Apple GPUs, so Float64-based MultiFloats can't reach Metal through
# any path.
function KernelAbstractionsPseudospectra.warp_trsm_safe(::Metal.MetalBackend, wide::Bool)
    !wide && KernelAbstractionsPseudospectra.metal_warp_trsm()
end
# `tiled-gemm`'s `mul!` trailing has no reliable complex-GEMM path via MPS here — keep Metal
# on the regular `tiled` trailing kernel (`tiled-gemm` → `tiled`).
KernelAbstractionsPseudospectra.tiled_gemm_safe(::Metal.MetalBackend, ::Type) = false
# No `supports_fp64` override needed: the default defers to `supports_float64`, which Metal
# declares `false` (no `double` in MSL), so F64 paths skip Metal — F32 only. Metal also
# needs `KernelAbstractionsPseudospectra.set_pdiv_accurate!(false)` so the F32 solves compile (see KATRSM).

## precompile gpu code (only when a device is actually usable)
if Metal.functional()
    @setup_workload begin
        # Apple GPUs have no FP64 — ComplexF32 only.
        @compile_workload begin
            KernelAbstractionsPseudospectra._precompile_ihlpsa(MetalBackend(), Metal.device(), [ComplexF32])
        end
    end
end

end
