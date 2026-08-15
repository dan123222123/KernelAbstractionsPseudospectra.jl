module AMDGPUPseudospectra

using KernelAbstractionsPseudospectra, AMDGPU, PrecompileTools

# Device-interface overrides for the AMDGPU/ROCm backend. Defined unconditionally (not
# under `if AMDGPU.functional()`) so precompile bakes them even off-device.
KernelAbstractionsPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
KernelAbstractionsPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
KernelAbstractionsPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
KernelAbstractionsPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
KernelAbstractionsPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.free()
KernelAbstractionsPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()
# Per-device queries for the trsm routing. warpSize is 64 on CDNA / 32 on RDNA, and the tiled
# solve takes its panel width from here — hardcoding 32 would idle half the lanes on wave64.
# Tiled shuffle path is only exercised on AMD hardware via opt-in CI.
function KernelAbstractionsPseudospectra.warp_width(B::AMDGPU.ROCBackend)
    Int(AMDGPU.HIP.properties(AMDGPU.device()).warpSize)
end
function KernelAbstractionsPseudospectra.device_smem_bytes(B::AMDGPU.ROCBackend)
    Int(AMDGPU.HIP.properties(AMDGPU.device()).sharedMemPerBlock)
end
# Per-CU LDS, feeding only the no-probe analytic `tile_cols` default (`tune_trsm_tiled!` is
# authoritative). Falls back to ~2× the per-block LDS if AMDGPU.jl doesn't expose the per-MP
# field.
function KernelAbstractionsPseudospectra.device_smem_per_sm(B::AMDGPU.ROCBackend)
    let p = AMDGPU.HIP.properties(AMDGPU.device())
        hasproperty(p, :maxSharedMemoryPerMultiProcessor) ?
        Int(p.maxSharedMemoryPerMultiProcessor) : 2 * Int(p.sharedMemPerBlock)
    end
end

## precompile gpu code (only when a device is actually usable)
if AMDGPU.functional()
    @setup_workload begin
        @compile_workload begin
            KernelAbstractionsPseudospectra._precompile_ihlpsa(ROCBackend(), collect(AMDGPU.devices())[1],
                [ComplexF32, ComplexF64])
        end
    end
end

end
