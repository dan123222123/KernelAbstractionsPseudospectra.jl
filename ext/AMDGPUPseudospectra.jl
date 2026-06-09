module AMDGPUPseudospectra

using KAPseudospectra, AMDGPU, PrecompileTools

# Device-interface overrides for the AMDGPU/ROCm backend.
#
# These are defined UNCONDITIONALLY (not under `if AMDGPU.functional()`). They only
# dispatch on the backend type and never touch hardware at definition time, so they
# are safe to bake into the precompiled image even on a machine with no usable
# device. Guarding the *definitions* on `AMDGPU.functional()` is a trap: the
# precompile worker process frequently cannot probe the GPU, so functional()
# returns false there, an empty image is baked, and the methods are then missing at
# runtime even though the device works interactively. Only the GPU workload below
# (which actually runs kernels) needs the functional() guard.
KAPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
KAPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
KAPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
KAPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
KAPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.free()
KAPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()

## precompile gpu code (only when a device is actually usable)
if AMDGPU.functional()
    @setup_workload begin
        using LinearAlgebra
        for T in [ComplexF32, ComplexF64]
            m = 32
            g = 100
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            @compile_workload begin
                ihlpsa(ROCBackend(), zg, P, 5; devs=[collect(AMDGPU.devices())[1]])
            end
        end
    end
end

end
