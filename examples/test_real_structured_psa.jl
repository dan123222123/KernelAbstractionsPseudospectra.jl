##
using KernelAbstractionsPseudospectra
using LinearAlgebra
using KernelAbstractions
using Plots          # GR backend (Plots' default — no extra deps)
##

##
T = ComplexF64
n = 5
A = diagm( # Demmel matrix
    0 => ones(T, n) * -1,
    1 => ones(T, n - 1) * -10,
    2 => ones(T, n - 2) * -100,
    3 => ones(T, n - 3) * -1000,
    4 => ones(T, n - 4) * -10000
)
gx, gy, zg = qgrid(T, (-3, 1), (-2, 2), (400, 400))
P = MatrixPencil(schur(A))
srg = ihlpsa(CPU(), zg, P, 3)   # fixed nit=3
ssrg = ℝsvdpsa(zg, P)
##

## structured (solid) vs unstructured (dash-dot) σ_ε, eigenvalues overlaid
psaplot(gx, gy, srg; levels = -10:1:-2, line = (1, :dashdot), size = (1000, 1000))
psaplot!(gx, gy, ssrg, eigvals(A); levels = -10:1:-2, line = (1, :solid))
##
