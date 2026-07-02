##
using LinearAlgebra, MatrixDepot
using Plots   # GR backend (Plots' default — no extra deps)
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
#using Metal             # Apple GPUs (Float32 only — no FP64)
#backend = MetalBackend()
#
#using oneAPI            # Intel GPUs (Float32 only on FP64-less iGPUs)
#backend = oneAPIBackend()
using KAPseudospectra
# trsm kernel workgroup size: 256 for CPU/CUDA, 16 for AMDGPU, 32 for Intel (one subgroup)
##

##
T = ComplexF32
A = MatrixDepot.parter(T, 16)
gx, gy, zg = qgrid(T, (-2, 5), (-4.5, 4.5), (300, 300))
P = MatrixPencil(schur(A))
srg = ihlpsa(backend, zg, P, 8; wgs=256)
# Adaptive nit (omit the positional `nit`): retires each grid point at its own
# converged depth. Pass `verbose=true` to log the depth reached. Drop-in:
#srg = ihlpsa(backend, zg, P; wgs=256)
#
psaplot(gx, gy, srg, eigvals(A); levels=-3:0.25:0, size=(1000, 1000))
##