# Exercised during the sysimage build (PackageCompiler `precompile_execution_file`) so that
# KAPseudospectra's solve paths and the GPU host-side machinery are traced into the image.
# The big win — preloading CUDA/GPUCompiler/LLVM/KAPseudospectra — comes from listing them in
# `bake`; this file additionally traces concrete `ihlpsa` method instances. Kept small and
# fully guarded so a missing optional dep (KernelAbstractions / a GPU backend) just skips that
# part instead of failing the build. Broaden the types/sizes to match your typical workloads.

using KAPseudospectra

# CPU path (KernelAbstractions provides CPU()).
try
    @eval using KernelAbstractions, LinearAlgebra, Random
    for T in (ComplexF64, ComplexF32)
        _, _, zg = qgrid(T, (-2, 2), (-2, 2), (16, 16))
        P = MatrixPencil(schur(randn(T, 48, 48)))
        ihlpsa(CPU(), zg, P, 6)   # fixed-nit
        ihlpsa(CPU(), zg, P)      # adaptive
    end
catch err
    @warn "CPU precompile workload skipped" exception = err
end

# GPU path (only if a CUDA device is usable in the build environment).
try
    @eval using CUDA, LinearAlgebra, Random
    if CUDA.functional()
        for T in (ComplexF64, ComplexF32)
            _, _, zg = qgrid(T, (-2, 2), (-2, 2), (16, 16))
            P = MatrixPencil(schur(randn(T, 64, 64)))
            ihlpsa(CUDABackend(), zg, P, 6)
        end
    end
catch err
    @warn "CUDA precompile workload skipped" exception = err
end
