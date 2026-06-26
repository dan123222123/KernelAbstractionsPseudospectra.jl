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

function test_ihlpsa_parter16(backend; types=(ComplexF32, ComplexF64))
    @testset "ihlpsa parter16 -- $(backend)" begin
        if ComplexF32 in types
            @testset "F32" begin
                @test testihlpsa(tdir * "F32parter16.mat", backend, 1e-6)
                # adaptive (per-point hybrid) vs the same eigtool reference; stops
                # at the adaptive rtol (measured 3.9e-7 normed — matches fixed run).
                @test testihlpsa_adaptive(tdir * "F32parter16.mat", backend, 1e-5)
            end
        end
        if ComplexF64 in types
            @testset "F64" begin
                @test testihlpsa(tdir * "F64parter16.mat", backend, 1e-14)
                # adaptive stops at the default F64 rtol=1e-6 (not at eps), so it
                # lands ~6.1e-11 normed vs the eigtool reference — orders of
                # magnitude inside plotting accuracy. Tighten `rtol` for more.
                @test testihlpsa_adaptive(tdir * "F64parter16.mat", backend, 1e-10)
            end
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

# CUDA is opt-in ("cuda" only — NOT part of `all`): CI runs CPU-only, so pulling in the whole
# CUDA/GPUCompiler/LLVM stack there is pure precompile overhead and the GPU tests can't run on a
# CPU runner anyway (the core kernels are still compiled by the CPU run). Add it on a CUDA machine
# (`julia --project=test -e 'using Pkg; Pkg.add("CUDA")'`) and run `test/runtests.jl cuda`, or use
# a GPU CI runner. Same opt-in pattern as AMDGPU / oneAPI / Metal.
if "cuda" in ARGS
    using CUDA
    if CUDA.functional()
        test_ihlpsa_parter16(CUDABackend())
        test_cross_backend(CUDABackend())
        test_adaptive_backend(CUDABackend())
        test_katrsm_kernels(CUDABackend())
        test_trsm_strategies(CUDABackend())
    end
end

# AMDGPU is opt-in ("amdgpu" only — NOT part of `all`): current AMDGPU releases require
# KernelIntrinsics 1.x, which is incompatible with the CUDA stack's 0.1.x in a shared test
# environment, so it can't be a default test dep. Add it on AMD hardware first
# (`julia --project=test -e 'using Pkg; Pkg.add("AMDGPU")'`) and run `test/runtests.jl amdgpu`.
if "amdgpu" in ARGS
    using AMDGPU
    if AMDGPU.functional()
        test_ihlpsa_parter16(ROCBackend())
        test_cross_backend(ROCBackend())
        test_adaptive_backend(ROCBackend())
        test_katrsm_kernels(ROCBackend())
        test_trsm_strategies(ROCBackend())
    end
end

# oneAPI is opt-in ("oneapi" only — NOT part of `all`): it's only useful on Intel
# GPUs (absent on CI), so it's not a dep of the test env. Add it on Intel hardware
# first (`julia --project=test -e 'using Pkg; Pkg.add("oneAPI")'`) and run
# `test/runtests.jl oneapi`.
if "oneapi" in ARGS
    using oneAPI
    if oneAPI.functional()
        # Most Intel iGPUs lack native FP64; restrict to ComplexF32 there, and
        # auto-enable ComplexF64 on FP64-capable Intel GPUs (Arc, Data Center Max).
        Ts = KAPseudospectra.supports_fp64(oneAPIBackend()) ? (ComplexF32, ComplexF64) : (ComplexF32,)
        test_ihlpsa_parter16(oneAPIBackend(); types=Ts)
        test_cross_backend(oneAPIBackend(); types=Ts)
        test_adaptive_backend(oneAPIBackend(); types=Ts)
        test_katrsm_kernels(oneAPIBackend(); types=Ts)
        test_trsm_strategies(oneAPIBackend(); types=Ts)
    end
end

# Metal is opt-in ("metal" only — NOT part of `all`): it can only be installed and
# run on Apple silicon, so add it to the test env there first
# (`julia --project=test -e 'using Pkg; Pkg.add("Metal")'`) and run
# `test/runtests.jl metal`. Apple GPUs have no FP64, so this is ComplexF32-only.
if "metal" in ARGS
    using Metal
    if Metal.functional()
        Ts = KAPseudospectra.supports_fp64(MetalBackend()) ? (ComplexF32, ComplexF64) : (ComplexF32,)
        test_ihlpsa_parter16(MetalBackend(); types=Ts)
        test_cross_backend(MetalBackend(); types=Ts)
        test_adaptive_backend(MetalBackend(); types=Ts)
        test_katrsm_kernels(MetalBackend(); types=Ts)
        test_trsm_strategies(MetalBackend(); types=Ts)
    end
end
