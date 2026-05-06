# KAPseudospectra.jl

```@meta
CurrentModule = KAPseudospectra
```

**Accelerated Pseudospectral Calculations with KernelAbstractions**

KAPseudospectra.jl is a Julia package for computing pseudospectra (resolvent norms) of matrices and matrix pencils using GPU acceleration through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

## Features

- **Multiple computational methods**:
  - SVD-based methods for small to medium matrices (CPU, multi-threaded)
  - Inverse Hermitian Lanczos for large matrices (GPU-accelerated)

- **Structured and unstructured pseudospectra**:
  - Complex (unstructured) perturbations via [`ℂsvdpsa`](@ref)
  - Real (structured) perturbations via [`ℝsvdpsa`](@ref)

- **Multi-backend support** via KernelAbstractions.jl:
  - CPU (multi-threaded)
  - NVIDIA CUDA
  - AMD ROCm (AMDGPU)
  - Apple Metal

- **Multi-device execution**: Automatic work distribution across multiple GPUs

- **Matrix pencils**: Support for generalized eigenvalue problems (A, B)

## Installation

```julia
using Pkg
Pkg.add("KAPseudospectra")
```

For GPU support, install the appropriate backend:

```julia
# NVIDIA CUDA
Pkg.add("CUDA")

# AMD ROCm
Pkg.add("AMDGPU")

# Apple Metal
Pkg.add("Metal")
```

## Quick Start

### Basic Usage

```julia
using KAPseudospectra
using LinearAlgebra

# Create a matrix
m = 100
A = randn(ComplexF64, m, m)

# Generate complex grid
gx, gy, zg = qgrid(ComplexF64, (-3, 3), (-3, 3), (200, 200))

# Compute pseudospectra using SVD
P = MatrixPencil(schur(A))
psa = ℂsvdpsa(zg, P)

# Visualize
using Plots
contour(gx, gy, psa, levels=10 .^ (-6:0.5:0))
scatter!(real(eigvals(A)), imag(eigvals(A)), marker=:x, color=:red)
```

### GPU Acceleration

```julia
using CUDA
using KernelAbstractions

# Large matrix
m = 2000
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))

# Fine grid
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (400, 400))

# Compute on GPU
psa = ihlpsa(CUDABackend(), zg, P, progress=true)
```

### Multi-GPU

```julia
using CUDA

# Automatic distribution across all available GPUs
backend = CUDABackend()
psa = ihlpsa(backend, zg, P, progress=true)

# Manual device selection
devices = CUDA.devices()[1:2]  # Use first 2 GPUs
psa = ihlpsa(backend, zg, P, devs=devices)
```

## Package Contents

### [Pseudospectra Theory](@ref)
Mathematical background on pseudospectra, computational methods, and applications.

### [Standard Usage Example](@ref)
Comprehensive examples demonstrating basic usage, visualization, and different computational methods.

### [Multi-GPU Computation](@ref)
Advanced examples showing multi-device computation, performance optimization, and large-scale workflows.

### [API Reference](@ref)
Complete documentation of all exported functions and types.

## Performance Guidelines

| Matrix Size | Grid Size | Recommended Method | Backend |
|:------------|:----------|:-------------------|:--------|
| m < 200     | Any       | `ℂsvdpsa` / `ℝsvdpsa` | CPU |
| 200 ≤ m < 500 | ≤ 200×200 | `ℂsvdpsa` / `ℝsvdpsa` | CPU |
| 200 ≤ m < 500 | > 200×200 | `ihlpsa` | GPU |
| m ≥ 500     | Any       | `ihlpsa` | GPU |
| m ≥ 2000    | > 300×300 | `ihlpsa` | Multi-GPU |

## Method Comparison

| Feature | SVD Methods | Inverse Lanczos |
|:--------|:------------|:----------------|
| **Accuracy** | Exact | Iterative approximation |
| **Complexity per point** | O(m³) | O(m² × log m) |
| **GPU Support** | No | Yes |
| **Multi-GPU** | No | Yes |
| **Best for** | m < 500 | m ≥ 500 |
| **Real pseudospectra** | ✓ (`ℝsvdpsa`) | ✗ |

## Citation

If you use KAPseudospectra.jl in your research, please cite:

```bibtex
@software{kapseudospectra,
  title = {KAPseudospectra.jl: GPU-Accelerated Pseudospectral Computations},
  author = {Dan Folescu},
  year = {2025},
  url = {https://github.com/dan123222123/KAPseudospectra.jl}
}
```

## Related Packages

- [EigTool](http://www.comlab.ox.ac.uk/pseudospectra/eigtool/) - MATLAB GUI for pseudospectra (inspiration for this package)
- [Pseudospectra.jl](https://github.com/RalphAS/Pseudospectra.jl) - Pure Julia pseudospectra package
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) - Backend abstraction layer

## References

1. Trefethen, L.N. & Embree, M. (2005). *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*. Princeton University Press.

2. Lui, S.H. (1997). "Computation of pseudospectra by continuation." *SIAM Journal on Scientific Computing*, 18(2), 565-573.

3. Wright, T.G. & Trefethen, L.N. (2001). "Large-scale computation of pseudospectra using ARPACK and eigs." *SIAM Journal on Scientific Computing*, 23(2), 591-605.

4. Qiu, L., Bernhardsson, B., Rantzer, A., Davison, E.J., Young, P.M., & Doyle, J.C. (1995). "A formula for computation of the real stability radius." *Automatica*, 31(6), 879-890.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/dan123222123/KAPseudospectra.jl).

## License

This package is released under the MIT License.
