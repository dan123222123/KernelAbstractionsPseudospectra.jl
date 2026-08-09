module AMDGPUPseudospectra

using KAPseudospectra, AMDGPU, PrecompileTools

# Device-interface overrides for the AMDGPU/ROCm backend. Defined unconditionally (not
# under `if AMDGPU.functional()`) so precompile bakes them even off-device.
KAPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
KAPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
KAPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
KAPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
KAPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.free()
KAPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()
# Per-device queries for the trsm routing. warpSize is 64 on CDNA / 32 on RDNA, and the tiled
# solve takes its panel width from here — hardcoding 32 would idle half the lanes on wave64.
# Tiled shuffle path is only exercised on AMD hardware via opt-in CI.
function KAPseudospectra.warp_width(B::AMDGPU.ROCBackend)
    Int(AMDGPU.HIP.properties(AMDGPU.device()).warpSize)
end
function KAPseudospectra.device_smem_bytes(B::AMDGPU.ROCBackend)
    Int(AMDGPU.HIP.properties(AMDGPU.device()).sharedMemPerBlock)
end
# Per-CU LDS, feeding only the no-probe analytic `tile_cols` default (`tune_trsm_tiled!` is
# authoritative). Falls back to ~2× the per-block LDS if AMDGPU.jl doesn't expose the per-MP
# field.
function KAPseudospectra.device_smem_per_sm(B::AMDGPU.ROCBackend)
    let p = AMDGPU.HIP.properties(AMDGPU.device())
        hasproperty(p, :maxSharedMemoryPerMultiProcessor) ?
        Int(p.maxSharedMemoryPerMultiProcessor) : 2 * Int(p.sharedMemPerBlock)
    end
end

## precompile gpu code (only when a device is actually usable)
if AMDGPU.functional()
    @setup_workload begin
        @compile_workload begin
            KAPseudospectra._precompile_ihlpsa(ROCBackend(), collect(AMDGPU.devices())[1],
                [ComplexF32, ComplexF64])
        end
    end
end

end
