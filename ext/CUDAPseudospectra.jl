module CUDAPseudospectra

using KernelAbstractionsPseudospectra, CUDA, PrecompileTools

# Device-interface overrides for the CUDA backend. Defined unconditionally (not under
# `if CUDA.functional()`) so precompile bakes them even off-device.
#
# No CUDA-specific triangular solve here — column and tiled solves run through the portable
# KernelAbstractions + KernelIntrinsics kernels (src/KATRSM/), same as every other
# backend.
KernelAbstractionsPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
KernelAbstractionsPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
KernelAbstractionsPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
KernelAbstractionsPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
KernelAbstractionsPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
KernelAbstractionsPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()
# Per-device queries for the trsm routing.
KernelAbstractionsPseudospectra.warp_width(B::CUDA.CUDABackend) = Int(CUDA.warpsize(CUDA.device()))   # 32 on all current NVIDIA HW
function KernelAbstractionsPseudospectra.device_smem_bytes(B::CUDA.CUDABackend)
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
end
function KernelAbstractionsPseudospectra.device_smem_per_sm(B::CUDA.CUDABackend)
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
end

## precompile gpu code (only when a device is actually usable)
if CUDA.functional()
    @setup_workload begin
        @compile_workload begin
            # Precompile via the shuffle-free "column" solve. Tiled kernels JIT-compile on
            # first use instead: CUDA kernel PTX doesn't survive precompile→runtime anyway,
            # and tiled shuffle kernels are least reliable in the headless worker.
            dev = collect(CUDA.devices())[1]
            withenv("KAPSEUDO_TRSM" => "column") do
                KernelAbstractionsPseudospectra._precompile_ihlpsa(CUDABackend(), dev, [
                    ComplexF32, ComplexF64])
            end
            # Opt-in (`enable_gpu_kernel_cache!()`): also compiles warp/tiled kernels,
            # persisted via GPUCompiler's disk cache. Needs Julia ≥ 1.13
            # (JuliaLang/julia#60747) for cross-session reuse; launches are individually
            # guarded so unsupported stacks don't break precompilation.
            KernelAbstractionsPseudospectra.PRECOMPILE_GPU_KERNELS &&
                KernelAbstractionsPseudospectra._precompile_gpu_kernels(CUDABackend(), dev, [
                    ComplexF32, ComplexF64])
        end
    end
end

end
