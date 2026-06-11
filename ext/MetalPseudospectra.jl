module MetalPseudospectra

using KAPseudospectra, Metal, PrecompileTools

# Device-interface overrides for the Apple/Metal backend.
#
# Implemented against Metal.jl's `MetalBackend` (KernelAbstractions). Apple
# M-series systems expose a single GPU with unified memory, so the multi-device
# plumbing collapses to one device.
#
# Defined UNCONDITIONALLY (not under `if Metal.functional()`/`if Sys.isapple()`):
# they only dispatch on the backend type and never touch hardware at definition
# time, so they are safe to bake into the precompiled image even off-device.
# Guarding the *definitions* is a trap — the precompile worker may evaluate the
# guard as false, bake an empty image, and leave the overrides missing at runtime.
# Only the GPU workload below needs the functional() guard.
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
# No `supports_fp64` override needed: the KAPseudospectra default defers to
# `KernelAbstractions.supports_float64`, which Metal declares `false` (no `double`
# in Metal Shading Language). So the F64 grid/test/precompile paths skip Metal and
# only ComplexF32 runs — same gating as the FP64-less Intel path.

## precompile gpu code (only when a device is actually usable)
if Metal.functional()
    @setup_workload begin
        using LinearAlgebra
        # Apple GPUs have no FP64 — ComplexF32 only.
        for T in [ComplexF32]
            m = 32
            g = 100
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            @compile_workload begin
                ihlpsa(MetalBackend(), zg, P, 5; devs=[Metal.device()])   # fixed-nit path
                ihlpsa(MetalBackend(), zg, P; devs=[Metal.device()])       # adaptive (per-point hybrid)
            end
        end
    end
end

end
