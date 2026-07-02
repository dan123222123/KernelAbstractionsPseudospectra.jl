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

# Extended-precision (MultiFloats) tests are opt-in ("multifloats" in ARGS — NOT part of `all`),
# same bring-your-own pattern as the GPU backends below: add the deps, then run
# `test/runtests.jl multifloats`:
#   julia --project=test -e 'using Pkg; Pkg.add(["MultiFloats","GenericSchur","GenericLinearAlgebra"])'
# (MultiFloats is a weak dep via the MultiFloatsPseudospectra extension; GenericSchur and
# GenericLinearAlgebra supply the generic Schur + tridiagonal eigen the extended path needs, and
# are not package deps.) The accuracy oracle runs on CPU; the per-limb tiled-shuffle kernel test
# runs from the GPU blocks below (self-gates to backends where the shuffle is usable).
if "multifloats" in ARGS
    include("test_multifloats.jl")
    test_multifloats_accuracy()
end

# --- backend dispatch ---

if all_tests || "cpu" in ARGS
    using KernelAbstractions
    test_ihlpsa_parter16(CPU())
    test_katrsm_kernels(CPU())
end

# GPU backends are opt-in ("cuda"/"amdgpu"/"oneapi"/"metal" in ARGS — NOT part of `all`): CI is
# CPU-only, so pulling in a GPU stack there is pure precompile overhead with nothing to run against
# (core kernels are still compiled by the CPU run above). Add the package first
# (`julia --project=test -e 'using Pkg; Pkg.add("X")'`) then run `test/runtests.jl X`.
if "cuda" in ARGS
    using CUDA
    if CUDA.functional()
        test_ihlpsa_parter16(CUDABackend())
        test_cross_backend(CUDABackend())
        test_adaptive_backend(CUDABackend())
        test_katrsm_kernels(CUDABackend())
        test_trsm_strategies(CUDABackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_shuffle(CUDABackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_generic(CUDABackend())
    end
end

# AMDGPU needs KernelIntrinsics 1.x, which conflicts with CUDA's 0.1.x in a shared test env.
if "amdgpu" in ARGS
    using AMDGPU
    if AMDGPU.functional()
        test_ihlpsa_parter16(ROCBackend())
        test_cross_backend(ROCBackend())
        test_adaptive_backend(ROCBackend())
        test_katrsm_kernels(ROCBackend())
        test_trsm_strategies(ROCBackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_shuffle(ROCBackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_generic(ROCBackend())
    end
end

# oneAPI is Intel-only.
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
        ("multifloats" in ARGS) && test_multifloats_tiled_shuffle(oneAPIBackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_generic(oneAPIBackend())
    end
end

# Metal is Apple-only, no FP64.
if "metal" in ARGS
    using Metal
    if Metal.functional()
        Ts = KAPseudospectra.supports_fp64(MetalBackend()) ? (ComplexF32, ComplexF64) : (ComplexF32,)
        test_ihlpsa_parter16(MetalBackend(); types=Ts)
        test_cross_backend(MetalBackend(); types=Ts)
        test_adaptive_backend(MetalBackend(); types=Ts)
        test_katrsm_kernels(MetalBackend(); types=Ts)
        test_trsm_strategies(MetalBackend(); types=Ts)
        ("multifloats" in ARGS) && test_multifloats_tiled_shuffle(MetalBackend())
        ("multifloats" in ARGS) && test_multifloats_tiled_generic(MetalBackend())
    end
end
