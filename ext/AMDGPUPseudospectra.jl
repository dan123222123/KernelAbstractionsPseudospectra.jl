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
