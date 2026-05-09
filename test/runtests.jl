using Test, KAPseudospectra

if isempty(ARGS) || "all" in ARGS
    all_tests = true
else
    all_tests = false
end

include("eigtool_core.jl")

# --- parter16 reference (eigtool .mat file) ---

@testset "svdpsa parter16" begin
    @testset "F32" begin
        @test testsvdpsa(tdir * "F32parter16.mat", 1e-6)
    end
    @testset "F64" begin
        @test testsvdpsa(tdir * "F64parter16.mat", 1e-14)
    end
end

function test_ihlpsa_parter16(backend)
    @testset "ihlpsa parter16 -- $(backend)" begin
        @testset "F32" begin
            @test testihlpsa(tdir * "F32parter16.mat", backend, 1e-6)
        end
        @testset "F64" begin
            @test testihlpsa(tdir * "F64parter16.mat", backend, 1e-14)
        end
    end
end

# --- internal-consistency, edge-cases, ℝsvdpsa, KATRSM kernels ---

include("test_consistency.jl")
include("test_realsvdpsa.jl")
include("test_edge_cases.jl")
include("test_katrsm.jl")

# --- backend dispatch ---

if all_tests || "cpu" in ARGS
    using KernelAbstractions
    test_ihlpsa_parter16(CPU())
    test_katrsm_kernels(CPU())
end

if all_tests || "cuda" in ARGS
    using CUDA
    if CUDA.functional()
        test_ihlpsa_parter16(CUDABackend())
        test_cross_backend(CUDABackend())
        test_katrsm_kernels(CUDABackend())
    end
end

if all_tests || "amdgpu" in ARGS
    using AMDGPU
    if AMDGPU.functional()
        test_ihlpsa_parter16(ROCBackend())
        test_cross_backend(ROCBackend())
        test_katrsm_kernels(ROCBackend())
    end
end
