# Standard Usage Example

```@meta
CurrentModule = KAPseudospectra
```

This page demonstrates basic usage of KAPseudospectra.jl for computing and visualizing pseudospectra.

## Setup

First, load the required packages:

```julia
using KAPseudospectra
using LinearAlgebra
using Plots
using LaTeXStrings
```

## Example 1: Grcar Matrix

The Grcar matrix is a classic example of a nonnormal matrix with interesting pseudospectra.

### Matrix Construction

```julia
# Grcar matrix of size m×m
function grcar(m)
    A = diagm(0 => ones(m), 1 => ones(m-1), 2 => ones(m-2), 3 => ones(m-3))
    A -= diagm(-1 => ones(m-1))
    return A
end

m = 100
A = ComplexF64.(grcar(m))
```

### Computing Pseudospectra

Create a complex grid and compute pseudospectra:

```julia
# Define grid in complex plane
gx, gy, zg = qgrid(ComplexF64, (-1, 4), (-2.5, 2.5), (200, 200))

# Convert to Schur form for efficiency
P = MatrixPencil(schur(A))

# Compute unstructured pseudospectra using SVD
psa_complex = ℂsvdpsa(zg, P)

# Compute structured (real) pseudospectra
psa_real = ℝsvdpsa(zg, P)
```

### Visualization

```julia
# Plot unstructured pseudospectra
contour(gx, gy, psa_complex,
    levels = 10 .^ (-6:0.5:0),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)",
    title = "Unstructured Pseudospectra: Grcar($m)",
    aspect_ratio = :equal,
    color = :viridis
)

# Add eigenvalues
scatter!(real(eigvals(A)), imag(eigvals(A)),
    marker = :circle,
    markersize = 3,
    color = :red,
    label = "Eigenvalues"
)

savefig("grcar_unstructured.png")
```

Compare structured vs unstructured:

```julia
# Side-by-side comparison
p1 = contour(gx, gy, psa_complex,
    levels = 10 .^ (-6:0.5:0),
    title = "Unstructured",
    aspect_ratio = :equal
)
scatter!(real(eigvals(A)), imag(eigvals(A)),
    marker = :x, markersize = 2, color = :red, label = ""
)

p2 = contour(gx, gy, psa_real,
    levels = 10 .^ (-6:0.5:0),
    title = "Structured (Real)",
    aspect_ratio = :equal
)
scatter!(real(eigvals(A)), imag(eigvals(A)),
    marker = :x, markersize = 2, color = :red, label = ""
)

plot(p1, p2, layout = (1, 2), size = (800, 400))
savefig("grcar_comparison.png")
```

## Example 2: Random Matrix

Demonstration with a random nonnormal matrix:

```julia
# Create random nonnormal matrix
m = 50
A = randn(ComplexF64, m, m)

# Grid covering region around eigenvalues
gx, gy, zg = qgrid(ComplexF64, (-3, 3), (-3, 3), (150, 150))

# Compute pseudospectra
P = MatrixPencil(schur(A))
psa = ℂsvdpsa(zg, P)

# Visualize
contourf(gx, gy, psa,
    levels = 10 .^ (-4:0.5:1),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)",
    title = "Random Matrix Pseudospectra",
    color = :thermal
)
scatter!(real(eigvals(A)), imag(eigvals(A)),
    marker = :circle, markersize = 3, color = :white, label = "Eigenvalues"
)
```

## Example 3: Using Inverse Lanczos (CPU)

For larger matrices, the IHL method is more efficient:

```julia
m = 500
A = randn(ComplexF64, m, m)

# Create grid
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (100, 100))

# Convert to Schur form
P = MatrixPencil(schur(A))

# Compute using IHL on CPU with progress bar
using KernelAbstractions
psa = ihlpsa(CPU(), zg, P, progress=true)

# Visualize
contour(gx, gy, psa,
    levels = 10 .^ (-5:0.5:0),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)",
    title = "IHL Pseudospectra (m=$m)",
    color = :viridis,
    aspect_ratio = :equal
)
scatter!(real(eigvals(A)), imag(eigvals(A)),
    marker = :+, markersize = 2, color = :red, label = ""
)
```

## Example 4: Generalized Eigenvalue Problem

Computing pseudospectra for a matrix pencil (A, B):

```julia
m = 80
A = randn(ComplexF64, m, m)
B = randn(ComplexF64, m, m)

# Create grid
gx, gy, zg = qgrid(ComplexF64, (-2, 2), (-2, 2), (150, 150))

# Generalized Schur form
P = MatrixPencil(schur(A, B))

# Compute with equal weighting on A and B perturbations
γ, δ = 0.5, 0.5
psa = ℂsvdpsa(zg, P, γ, δ)

# Visualize
contour(gx, gy, psa,
    levels = 10 .^ (-5:0.5:0),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)",
    title = L"Pencil Pseudospectra: $\gamma = \delta = 0.5$",
    color = :plasma,
    aspect_ratio = :equal
)
scatter!(real(eigvals(A, B)), imag(eigvals(A, B)),
    marker = :x, markersize = 3, color = :white, label = "Eigenvalues"
)
```

## Performance Tips

1. **Use Schur form**: Always convert to Schur form via `MatrixPencil(schur(...))` for better performance
2. **Choose the right method**:
   - SVD methods (`ℂsvdpsa`, `ℝsvdpsa`): Best for m < 500, CPU-only
   - IHL method (`ihlpsa`): Best for m ≥ 500, supports GPU acceleration
3. **Grid resolution**: Start with coarse grids (50×50) for exploration, refine as needed
4. **Multi-threading**: SVD methods automatically use all CPU threads

## Next Steps

- For GPU acceleration and large-scale problems, see [Multi-GPU Computation](@ref)
- For complete API documentation, see [API Reference](@ref)
- For theoretical background, see [Pseudospectra Theory](@ref)
