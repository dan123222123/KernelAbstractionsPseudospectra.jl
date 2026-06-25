### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses `@bind` for interactivity. When running this notebook outside of
# Pluto, the following 'mock version' of `@bind` gives bound variables a default value.
macro bind(def, element)
    #= https://github.com/fonsp/Pluto.jl =#
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a0000001-0000-0000-0000-000000000001
md"""
# Pseudospectra Explorer
Interactive inverse-Lanczos pseudospectra (`KAPseudospectra.ihlpsa`), GPU-accelerated when a
CUDA device is available. Pick a matrix and resolution; the contour shows `log₁₀ σ_min(zI − A)`.
"""

# ╔═╡ a0000002-0000-0000-0000-000000000002
# Use the project Pluto was launched in (must have KAPseudospectra + Plots + PlutoUI + MatrixDepot,
# and CUDA for the GPU path). The explicit Pkg.activate tells Pluto to NOT manage packages itself.
begin
    import Pkg
    Pkg.activate(Base.active_project() === nothing ? dirname(@__DIR__) : Base.active_project())
    using KAPseudospectra, KernelAbstractions, PlutoUI, Plots, MatrixDepot, LinearAlgebra, Random
    gpu = false
    try
        @eval using CUDA
        gpu = CUDA.functional()
    catch
    end
    md"**Backend:** $(gpu ? "CUDA GPU 🚀" : "CPU")"
end

# ╔═╡ a0000003-0000-0000-0000-000000000003
md"""
**Matrix** $(@bind which Select(["chebspec" => "Chebyshev spectral (non-normal)",
                                 "grcar" => "Grcar",
                                 "random" => "Random dense"]))
**Size m** $(@bind m Slider(16:16:512; default=128, show_value=true))
**Grid (n×n)** $(@bind G Slider(20:10:200; default=80, show_value=true))
**Lanczos depth** $(@bind nit Slider(2:2:20; default=8, show_value=true))
"""

# ╔═╡ a0000004-0000-0000-0000-000000000004
A = let
    Random.seed!(42)
    if which == "chebspec"
        ComplexF64.(MatrixDepot.chebspec(Float64, m))
    elseif which == "grcar"                       # standard non-normal Grcar matrix
        G = Matrix{ComplexF64}(I, m, m)
        for j in 1:m-1; G[j+1, j] = -1; end       # subdiagonal
        for k in 1:3, j in 1:m-k; G[j, j+k] = 1; end   # 3 superdiagonals
        G
    else
        randn(ComplexF64, m, m)
    end
end;

# ╔═╡ a0000005-0000-0000-0000-000000000005
result = let
    eigs = eigvals(A)
    pad = 0.5 + 0.2 * (maximum(abs, eigs))
    rx = (minimum(real, eigs) - pad, maximum(real, eigs) + pad)
    ry = (minimum(imag, eigs) - pad, maximum(imag, eigs) + pad)
    gx, gy, zg = qgrid(ComplexF64, rx, ry, (G, G))
    P = MatrixPencil(schur(A))
    backend = gpu ? CUDA.CUDABackend() : CPU()
    t = @elapsed (srg = ihlpsa(backend, zg, P, nit))
    (; gx, gy, srg, eigs, t)
end;

# ╔═╡ a0000006-0000-0000-0000-000000000006
let
    r = result
    lev = collect(-16:1:0)
    p = contourf(r.gx, r.gy, clamp.(log10.(r.srg'), -16, 0);
        levels=lev, color=:turbo, aspect_ratio=1,
        xlabel="Re z", ylabel="Im z",
        title="log₁₀ σ_min  —  $(which), m=$(m)  ($(round(r.t*1000, digits=1)) ms)")
    scatter!(p, real.(r.eigs), imag.(r.eigs); m=(:white, :diamond, 3), label="eigenvalues")
    p
end

# ╔═╡ Cell order:
# ╟─a0000001-0000-0000-0000-000000000001
# ╟─a0000002-0000-0000-0000-000000000002
# ╟─a0000003-0000-0000-0000-000000000003
# ╠═a0000004-0000-0000-0000-000000000004
# ╠═a0000005-0000-0000-0000-000000000005
# ╠═a0000006-0000-0000-0000-000000000006
