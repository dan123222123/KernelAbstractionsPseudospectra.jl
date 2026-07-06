## Pseudospectra of a Loewner matrix pencil
#
# Reproduces Example 1 of
#
#   M. Embree & A. C. Ioniţă, "Pseudospectra of Loewner Matrix Pencils",
#   in Realization and Model Reduction of Dynamical Systems (2022),
#   https://doi.org/10.1007/978-3-030-95157-3_4
#
# SISO system (n = 2): E = I, A = [-1.1 1; 1 -1.1], B = [1; 0], C = [0 1],
# with σ(A) = {-0.1, -2.1}. Four Loewner pencils are built from four
# interpolation-point choices (paper Table 2); all recover the same
# eigenvalues but with visibly different pseudospectra (paper Fig. 1).
#
# NOTE: the paper's eq. (14) uses B = [0;1] (H = the (2,2) entry of (sI−A)⁻¹),
# but its Table 2 / Fig. 1 only reproduce with the off-diagonal entry, i.e.
# B = [1;0] as used here — same poles {-0.1,-2.1}, different residues.
#
# Tiny n = 2 pencils: dense O(n³) SVD, runs on CPU out of the box (see README.md).
using LinearAlgebra
using KAPseudospectra
##

## the SISO system and its scalar transfer function H(z) = C(zI − A)⁻¹B
T = ComplexF64
A = T[-1.1 1; 1 -1.1]
B = T[1; 0]                                       # see NOTE above: off-diagonal H
C = T[0 1]
H(z) = (C * ((z * I - A) \ B))[1]

# SISO Loewner / shifted-Loewner from right points λ and left points μ
# (directions all 1, so wⱼ = H(λⱼ), vᵢ = H(μᵢ)); see paper eq. (1).
function loewner(λ, μ)
    L = T[(H(μ[i]) - H(λ[j])) / (μ[i] - λ[j]) for i in eachindex(μ), j in eachindex(λ)]
    Lₛ = T[(μ[i] * H(μ[i]) - λ[j] * H(λ[j])) / (μ[i] - λ[j])
           for i in eachindex(μ), j in eachindex(λ)]
    return L, Lₛ
end

# four interpolation-point choices (paper Table 2): right points λ, left points μ.
cases = [
    ("a", [0.0, 1.0], [im, -im]),
    ("b", [0.25, 0.75], [2im, -2im]),
    ("c", [0.40, 0.60], [4im, -4im]),
    ("d", [8.0, 9.0], [10.0 + 0im, 11.0 + 0im])
]
##

## compute σ_ε^{(1,1)}(𝕃ₛ, 𝕃) for each case: σ_min(z𝕃 − 𝕃ₛ)/(1+|z|) = ℂsvdpsa(zg, P, 1, 1)
g = 1000
gx, gy, zg = qgrid(T, (-3.0, 1.0), (-1.5, 1.5), (g, g))

results = map(cases) do (name, λ, μ)
    L, Lₛ = loewner(λ, μ)
    P = MatrixPencil(Lₛ, L)                      # the pencil z𝕃 − 𝕃ₛ
    psa = ℂsvdpsa(zg, P, 1, 1)                   # σ_ε^{(1,1)} = σ_min(z𝕃 − 𝕃ₛ)/(1+|z|)
    ev = eigvals(Lₛ, L)                          # recovered poles (should = σ(A))
    sv = svdvals(L)                              # singular values of 𝕃 (cf. Table 2)
    @info "case $name" poles=round.(sort(real(ev)), digits = 4) s1=sv[1] s2=sv[2]
    (; name, psa, ev)
end
##

## reproduce paper Fig. 1 — a 2×2 grid of pseudospectra, log10 ε contours, with
## the recovered eigenvalues overlaid.
using Plots                                      # GR backend (Plots' default)

# Match the paper: line-contour boundaries, one shared log10 ε scale across all
# four panels (a single `clims`), one colorbar drawn once in a fifth layout slot.
fields = [log10.(r.psa) for r in results]
finite_extrema(f) = extrema(x for x in f if isfinite(x))
gmin = floor(minimum(first(finite_extrema(f)) for f in fields) / 0.5) * 0.5
gmax = ceil(maximum(last(finite_extrema(f)) for f in fields) / 0.5) * 0.5
glevels = collect(gmin:0.5:gmax)
clims = (gmin, gmax)

function panel(r, fld)
    p = contour(gx, gy, fld; levels = glevels, color = :darkrainbow, clims = clims,
        colorbar = false, clabels = false, title = "($(r.name))", aspect_ratio = :equal,
        xlims = (-3, 1), ylims = (-1.5, 1.5))
    vline!(p, [0.0]; color = :gray, label = "")       # imaginary axis (stability boundary)
    scatter!(
        p, real(r.ev), imag(r.ev); markershape = :circle, mc = :black, ms = 4, label = "")
    return p
end
panels = [panel(r, f) for (r, f) in zip(results, fields)]
# a NaN heatmap carries the shared color scale → one colorbar, no visible plot
cbar = heatmap(fill(NaN, 2, 2); clims = clims, color = :darkrainbow, colorbar = true,
    framestyle = :none, ticks = false, title = "log₁₀ ε")
plt = plot(panels..., cbar; layout = (@layout [grid(2, 2) c{0.07w}]), size = (1350, 1000),
    plot_title = "σ_ε^{(1,1)} of the Loewner pencil z𝕃 − 𝕃ₛ  (Embree & Ioniţă, Fig. 1)")
display(plt)
##
