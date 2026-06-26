module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools

# Device-interface overrides for the CUDA backend. Defined unconditionally (not under
# `if CUDA.functional()`) so precompile bakes them even when the worker can't probe
# the GPU; only the workload below needs the functional() guard.
#
# NOTE: there is no CUDA-specific triangular solve here — the warp-register and tiled solves
# run through the portable KernelAbstractions + KernelIntrinsics kernels (src/KATRSM.jl) on
# CUDA like every other backend. An earlier hand-rolled `@cuda` + `CUDA.shfl_sync` warp solve
# (opt-in via KAPSEUDO_CUDA_NATIVE) was removed: the package's value is the minimal portable
# interface, and the `auto` strategy already routes around the KA+KI R=16 codegen cliff by
# switching to `tiled` at m≥512. See DESIGN_TRSM.md and git history for the diagnosis.
KAPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
KAPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
KAPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
KAPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
KAPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
KAPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()
# Per-device queries for the trsm routing (replace the hardcoded warp width / 48 KB smem proxy).
KAPseudospectra.warp_width(B::CUDA.CUDABackend) = Int(CUDA.warpsize(CUDA.device()))   # 32 on all current NVIDIA HW
KAPseudospectra.device_smem_bytes(B::CUDA.CUDABackend) =
    Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))

## precompile gpu code (only when a device is actually usable)
if CUDA.functional()
    @setup_workload begin
        @compile_workload begin
            # Precompile via the shuffle-free "column" solve. The warp/tiled kernels are
            # JIT-compiled on first use instead: CUDA kernel PTX does not survive the
            # precompile→runtime process boundary anyway (only host-side method instances
            # are cached, which the column path exercises identically), and executing the
            # warp/tiled shuffle kernels inside the headless precompile worker is where GPU
            # execution is least reliable.
            withenv("KAPSEUDO_TRSM" => "column") do
                KAPseudospectra._precompile_ihlpsa(CUDABackend(), collect(CUDA.devices())[1],
                    [ComplexF32, ComplexF64])
            end
        end
    end
end

end
