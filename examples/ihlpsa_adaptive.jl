## Adaptive-nit ihlpsa demo
#
# Shows the *adaptive depth* form of `ihlpsa` (omit the positional `nit`): each
# grid point runs inverse-Lanczos until its σ converges, retiring early instead
# of lockstepping every point to a fixed depth.
#
# Defaults to CPU(); uncomment a backend line below to run on a GPU (needs
# adding that backend package first — see "Running on a GPU" in examples/README.md).
using LinearAlgebra, MatrixDepot
using KernelAbstractions
using KAPseudospectra
##

## choose your backend
backend = CPU()
#
#using CUDA;   backend = CUDABackend()
#using AMDGPU; backend = ROCBackend()
#using Metal;  backend = MetalBackend()    # Apple GPUs (Float32 only — no FP64)
#using oneAPI; backend = oneAPIBackend()   # Intel GPUs (Float32 only on FP64-less iGPUs)

# trsm kernel workgroup size: 256 for CPU/CUDA, 16 for AMDGPU, 32 for Intel (one subgroup)
wgs = 256
##

## problem — use ComplexF32 so this runs as-is on an FP64-less iGPU (Intel UHD).
T = ComplexF32
n = 16
A = MatrixDepot.parter(T, n)
gx, gy, zg = qgrid(T, (-2, 5), (-4.5, 4.5), (1000, 1000))
P = MatrixPencil(schur(A))
##

## 1) drop-in adaptive — same call as fixed-nit but omit `nit`
srg = ihlpsa(backend, zg, P; wgs)
@info "adaptive σ" extrema = extrema(srg)
##

## 2) read the convergence depth two ways
ihlpsa(backend, zg, P; wgs, verbose=true)   # (a) verbose=true logs the deepest depth reached
# (b) the un-exported driver returns (σ, nit_grid): nit_grid[i] is the depth
#     at which grid point i retired
srg_adp, nit_grid = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs)
@info "adaptive driver" deepest = maximum(nit_grid) shallowest = minimum(nit_grid) mean_depth = sum(nit_grid) / length(nit_grid)
##

## 3) accuracy check — adaptive vs an over-converged fixed-nit control.
# Run fixed-nit well past where adaptive stopped; the two should agree to ~rtol.
nit_over = 8 * ceil(Int, log2(n))                 # the adaptive nit_max default
srg_fixed = ihlpsa(backend, zg, P, nit_over; wgs)
rel = maximum(abs, srg_adp .- srg_fixed) / maximum(abs, srg_fixed)
@info "adaptive vs over-converged fixed" deepest = maximum(nit_grid) nit_over max_rel_diff = rel
##

## 4) tighten the tolerance — costs more depth, sharper σ
srg_tight, nit_grid_tight = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, rtol=1e-6)
@info "tighter rtol" deepest_default = maximum(nit_grid) deepest_tight = maximum(nit_grid_tight)
##

## 5) plot — σ (left) vs per-point Lanczos depth (right); top row default rtol, bottom row rtol=1e-6
using Plots

# σ pseudospectra contour (eigenvalues overlaid) via the KAPseudospectra recipe
σcontour(srg, title) = psaplot(gx, gy, srg, eigvals(A); levels=-3:0.25:0, title=title)
# per-point adaptive Lanczos depth map
depthmap(nit, title) = heatmap(gx, gy, nit; color=:viridis, title=title)

plt = plot(
    σcontour(srg, "σ — default rtol"), depthmap(nit_grid, "depth — default rtol"),
    σcontour(srg_tight, "σ — rtol=1e-6"), depthmap(nit_grid_tight, "depth — rtol=1e-6"),
    layout=(2, 2), size=(1600, 1300),
)
display(plt)
##
