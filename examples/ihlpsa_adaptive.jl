## Adaptive-nit ihlpsa demo
#
# Shows the *adaptive depth* form of `ihlpsa` (omit the positional `nit`): each
# grid point runs inverse-Lanczos until its σ converges, retiring early instead
# of lockstepping every point to a fixed depth. Run blocks interactively (VS
# Code / Pluto-style `##` cells) or `julia --project=examples ihlpsa_adaptive.jl`.
#
# To evaluate a GPU backend (e.g. oneAPI on an Intel iGPU) WITHOUT touching any
# Project.toml in this repo, use a throwaway named environment — see the bottom
# of this file for the exact recipe.
using LinearAlgebra, MatrixDepot
using KernelAbstractions
using KAPseudospectra
##

## choose your backend
#backend = CPU()
#
#using CUDA;   backend = CUDABackend()
#using AMDGPU; backend = ROCBackend()
#using Metal;  backend = MetalBackend()    # Apple GPUs (Float32 only — no FP64)
using oneAPI; backend = oneAPIBackend()   # Intel GPUs (Float32 only on FP64-less iGPUs)

# trsm kernel workgroup size: 256 for CPU/CUDA, 16 for AMDGPU, 32 for Intel (one subgroup)
wgs = 32
##

## problem — use ComplexF32 so this runs as-is on an FP64-less iGPU (Intel UHD).
T = ComplexF32
n = 16
g = 1000
A = MatrixDepot.parter(T, n)
gx, gy, zg = qgrid(T, (-2, 5), (-4.5, 4.5), (g, g))
P = MatrixPencil(schur(A))
##

## 1) drop-in adaptive — same call as fixed-nit but omit `nit`
srg = ihlpsa(backend, zg, P; wgs)
@info "adaptive σ" extrema = extrema(srg)
##

## 2) read the convergence depth two ways
# (a) verbose=true logs the deepest depth reached
ihlpsa(backend, zg, P; wgs, verbose=true)
# (b) the un-exported driver returns (σ, nit_grid): nit_grid[i] is the depth at
#     which grid point i retired — a same-shape map of per-point Lanczos cost.
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

## 5) plot — σ (left) vs per-point Lanczos depth (right), default tolerance on
# top, tighter tolerance (rtol=1e-6) on the bottom. The σ panels should look
# essentially identical; the depth panels show the extra iterations the tighter
# tolerance buys — concentrated in the hard region near the spectrum, while easy
# regions still retire shallow. Needs Plots from examples/Project.toml.
using Plots, LaTeXStrings

tv = -3:0.25:0
tl = [L"10^{%$i}" for i in tv]

# σ pseudospectra contour with the eigenvalues overlaid
function σcontour(srg, title)
    p = contour(gx, gy, log10.(srg); color=:darkrainbow, colorbar_ticks=(tv, tl),
        levels=tv, line=(1, :solid), clabels=false, title=title)
    scatter!(p, eigvals(A), markershape=:diamond, label="")
    return p
end
# per-point adaptive Lanczos depth map
depthmap(nit, title) = heatmap(gx, gy, nit; color=:viridis, title=title)

plt = plot(
    σcontour(srg, "σ — default rtol"), depthmap(nit_grid, "depth — default rtol"),
    σcontour(srg_tight, "σ — rtol=1e-6"), depthmap(nit_grid_tight, "depth — rtol=1e-6"),
    layout=(2, 2), size=(1600, 1300),
)
display(plt)
##

# ──────────────────────────────────────────────────────────────────────────────
# Evaluating oneAPI.jl in a TEMPORARY environment (no Project.toml pollution)
# ──────────────────────────────────────────────────────────────────────────────
# This repo's Project.toml lists oneAPI only as a *weakdep* (extension trigger),
# and examples/Project.toml lists it as a regular dep. To benchmark oneAPI on a
# fresh, throwaway env instead — keeping both Project.tomls untouched and
# uninstantiated — use a NAMED shared environment that lives outside the repo
# under ~/.julia/environments/:
#
#   $ julia --project=@kaps-oneapi
#   julia> using Pkg
#   julia> Pkg.develop(path="/home/dfolescu/version_control/math/KAPseudospectra.jl-private")
#   julia> Pkg.add("oneAPI")          # pulls the oneAPI driver + Float32 iGPU support
#   julia> include("examples/ihlpsa_adaptive.jl")   # after uncommenting oneAPI above
#
# `dev`-ing the package (rather than `add`) means your local source edits are
# picked up live, and the oneAPI extension (ext/oneAPIPseudospectra.jl) loads
# automatically once `using oneAPI` runs. The env is reusable across sessions but
# disposable — `rm -rf ~/.julia/environments/kaps-oneapi` when done.
#
# Want it fully ephemeral (gone on exit)? Replace the first two lines with:
#   julia> using Pkg; Pkg.activate(; temp=true)
#   julia> Pkg.develop(path="…"); Pkg.add("oneAPI")
