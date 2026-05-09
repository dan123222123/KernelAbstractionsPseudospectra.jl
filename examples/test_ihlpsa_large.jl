##
using KAPseudospectra
using CUDA # accelerator needed unless you want to wait forever...
using LinearAlgebra, MatrixDepot, Plots, LaTeXStrings
pyplot()
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
        timsrg = CUDA.@elapsed srg = ihlpsa(CUDABackend(), zg, P, maxnit)
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
