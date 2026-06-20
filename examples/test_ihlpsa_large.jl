##
using KAPseudospectra
using KernelAbstractions
using LinearAlgebra, MatrixDepot, Plots, LaTeXStrings   # GR backend (Plots' default)

# Large-scale timing demo. Runs on CPU out of the box, but the sizes below
# (n up to 2^12) really want an accelerator — add a backend to the examples
# project (see README.md) and uncomment one of the lines below.
backend = CPU()
#using CUDA;   backend = CUDABackend()
#using AMDGPU; backend = ROCBackend()
#using Metal;  backend = MetalBackend()    # Apple GPUs (Float32 only — no FP64)
#using oneAPI; backend = oneAPIBackend()   # Intel GPUs (Float32 only on FP64-less iGPUs)

pltdir = (@__DIR__) * "/test_large_results/"
mkpath(pltdir)
##

##
open(pltdir * "timing", "w") do f
    T = ComplexF32
    g = 300
    maxnit = 8
    gx, gy, zg = qgrid(T, (-1, 1), (-1, 1), (g, g))
    #for n in [2^m for m = 8:14] # ambitious range...best for multi-device computations
    for n in [2^m for m = 8:12]
        A = MatrixDepot.golub(T, n)
        # note the first run takes longer to run than subsequent ones -- likely an issue with precompilation
        timschur = @elapsed P = MatrixPencil(schur(A))
        timsrg = @elapsed begin
            srg = ihlpsa(backend, zg, P, maxnit)
            KernelAbstractions.synchronize(backend)   # so GPU timing isn't just launch latency
        end
        write(f, "$(n),$(timschur),$(timsrg)\n")
        flush(f)
        tv = -6:0.2:-1
        tl = [L"10^{%$i}" for i in tv]
        levels = tv
        plt = plot(size=(1000, 1000))
        color = :darkrainbow
        clabels = false
        contour!(gx, gy, log10.(srg); color, colorbar_ticks=(tv, tl), levels, line=(1, :solid), clabels)
        display(plt)
        savefig(pltdir * "golub$(n).svg")
    end
end
##
