module AMDGPUPseudospectra

using KAPseudospectra, AMDGPU, PrecompileTools

# Device-interface overrides for the AMDGPU/ROCm backend. Defined unconditionally (not
# under `if AMDGPU.functional()`) so precompile bakes them even when the worker can't
# probe the GPU; only the workload below needs the functional() guard.
KAPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
KAPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
KAPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
KAPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
KAPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.free()
KAPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()
# Per-device queries for the trsm routing. warpSize is 64 on CDNA / 32 on RDNA; `warp_width` sets the
# column solve's workgroup size (the tiled solve's panel width is a fixed 32 regardless). NOTE: the
# AMD tiled shuffle path is only exercised on AMD hardware (opt-in CI) — validate there.
KAPseudospectra.warp_width(B::AMDGPU.ROCBackend) = Int(AMDGPU.HIP.properties(AMDGPU.device()).warpSize)
KAPseudospectra.device_smem_bytes(B::AMDGPU.ROCBackend) =
    Int(AMDGPU.HIP.properties(AMDGPU.device()).sharedMemPerBlock)

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
