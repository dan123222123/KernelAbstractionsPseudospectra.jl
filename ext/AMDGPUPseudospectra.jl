module AMDGPUPseudospectra

using KAPseudospectra, AMDGPU, PrecompileTools

if AMDGPU.functional()

    @eval begin
        KAPseudospectra.device(B::AMDGPU.ROCBackend) = AMDGPU.device()
        KAPseudospectra.device!(B::AMDGPU.ROCBackend, dev) = AMDGPU.device!(dev)
        KAPseudospectra.devices(B::AMDGPU.ROCBackend) = AMDGPU.devices()
        KAPseudospectra.get_bgarray(B::AMDGPU.ROCBackend) = AMDGPU.ROCArray
        KAPseudospectra.device_bytes_available(B::AMDGPU.ROCBackend) = AMDGPU.free()
        KAPseudospectra.device_reclaim(B::AMDGPU.ROCBackend) = AMDGPU.HIP.reclaim()
        ## precompile gpu code
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

end