module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools

# Device-interface overrides for the CUDA backend. Defined unconditionally (not under
# `if CUDA.functional()`) so precompile bakes them even when the worker can't probe
# the GPU; only the workload below needs the functional() guard.
#
# NOTE: there is no CUDA-specific triangular solve here — the column and tiled solves run through
# the portable KernelAbstractions + KernelIntrinsics kernels (src/KATRSM.jl) on CUDA like every
# other backend; that portable interface is the package's value. See DESIGN_TRSM.md.
KAPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
KAPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
KAPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
KAPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
KAPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
KAPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()
# Per-device queries for the trsm routing (replace the hardcoded warp width / 48 KB smem proxy).
KAPseudospectra.warp_width(B::CUDA.CUDABackend) = Int(CUDA.warpsize(CUDA.device()))   # 32 on all current NVIDIA HW
function KAPseudospectra.device_smem_bytes(B::CUDA.CUDABackend)
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
end
function KAPseudospectra.device_smem_per_sm(B::CUDA.CUDABackend)
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
end

## precompile gpu code (only when a device is actually usable)
if CUDA.functional()
    @setup_workload begin
        @compile_workload begin
            # Precompile via the shuffle-free "column" solve. The tiled kernels are
            # JIT-compiled on first use instead: CUDA kernel PTX does not survive the
            # precompile→runtime process boundary anyway (only host-side method instances
            # are cached, which the column path exercises identically), and executing the
            # tiled shuffle kernels inside the headless precompile worker is where GPU
            # execution is least reliable.
            dev = collect(CUDA.devices())[1]
            withenv("KAPSEUDO_TRSM" => "column") do
                KAPseudospectra._precompile_ihlpsa(CUDABackend(), dev, [
                    ComplexF32, ComplexF64])
            end
            # Opt-in (`enable_gpu_kernel_cache!()`): also compile the warp/tiled kernels so their
            # code persists across sessions via GPUCompiler's disk cache. Needs Julia ≥ 1.13
            # (JuliaLang/julia#60747) for cross-session reuse; the launches are individually
            # guarded so this never breaks precompilation on stacks where it isn't yet supported.
            KAPseudospectra.PRECOMPILE_GPU_KERNELS &&
                KAPseudospectra._precompile_gpu_kernels(CUDABackend(), dev, [
                    ComplexF32, ComplexF64])
        end
    end
end

end
