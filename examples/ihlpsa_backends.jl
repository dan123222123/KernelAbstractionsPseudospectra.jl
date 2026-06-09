##
using LinearAlgebra, MatrixDepot
using Plots, LaTeXStrings
# GR has some issues with colorbar ticks/lables -- use pyplot to get a nicer figure (needs PyPlot.jl)
pyplot()
##

## choose your backend
using KernelAbstractions
backend = CPU()
#
#using CUDA
#backend = CUDABackend()
#
#using AMDGPU
#backend = ROCBackend()
#
#using Metal             # Apple GPUs (Float32 only — Metal has no FP64)
#backend = MetalBackend()
#
#using oneAPI            # Intel GPUs (Float32 only on FP64-less iGPUs)
#backend = oneAPIBackend()
using KAPseudospectra
# workgroup size of the trsm kernels -- will bake this into the package extensions at some point TODO
wgs = 256 # good for multi-threaded CPU and CUDA
#wgs = 16 # good for AMDGPU
#wgs = 32 # good for Intel (one subgroup); try 16 if slower
##

##
T = ComplexF32
n = 16
g = 300
nit = 8
A = MatrixDepot.parter(T, n)
gx, gy, zg = qgrid(T, (-2, 5), (-4.5, 4.5), (g, g))
P = MatrixPencil(schur(A))
srg = ihlpsa(backend, zg, P, nit; wgs)
#
tv = -3:0.25:0
tl = [L"10^{%$i}" for i in tv]
levels = tv
plt = plot(size=(1000, 1000))
color = :darkrainbow
clabels = false
contour!(gx, gy, log10.(srg); color, colorbar_ticks=(tv, tl), levels, line=(1, :solid), clabels)
scatter!(eigvals(A), markershape=:diamond, label="")
##