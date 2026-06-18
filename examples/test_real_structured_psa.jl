##
using KAPseudospectra
using LinearAlgebra
using KernelAbstractions
using Plots          # GR backend (Plots' default — no extra deps)
using LaTeXStrings
##

##
T = ComplexF64
n = 5
g = 400
nit = 3
A = diagm( # Demmel matrix
    0 => ones(T, n) * -1,
    1 => ones(T, n - 1) * -10,
    2 => ones(T, n - 2) * -100,
    3 => ones(T, n - 3) * -1000,
    4 => ones(T, n - 4) * -10000
)
gx, gy, zg = qgrid(T, (-3, 1), (-2, 2), (g, g))
P = MatrixPencil(schur(A))
srg = ihlpsa(CPU(), zg, P, nit)
ssrg = ℝsvdpsa(zg, P)
##

## # plot structured and unstructured psa on the same figure
tv = -10:1:-2
tl = [L"10^{%$i}" for i in tv]
levels = tv
plt = plot(size=(1000, 1000))
color = :darkrainbow
clabels = false
contour!(gx, gy, log10.(srg); color, colorbar_ticks=(tv, tl), levels, line=(1, :dashdot), clabels)
contour!(gx, gy, log10.(ssrg); color, colorbar_ticks=(tv, tl), levels, line=(1, :solid), clabels)
scatter!(eigvals(A), markershape=:diamond, label="")
##