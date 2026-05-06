module KAPseudospectra

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa!, ℝsvdpsa!
export ℂsvdpsa, ℝsvdpsa

include("ihlpsa.jl")
export ihlpsa

using GridArrays

"""
    qgrid(T, tx, ty, gp)

Generate a rectangular grid in the complex plane for pseudospectra computation.

Creates equispaced grids along real and imaginary axes and combines them into a complex-valued
grid suitable for evaluating pseudospectra.

# Arguments
- `T`: Element type for the grid (e.g., `ComplexF64`, `ComplexF32`)
- `tx`: Tuple `(xmin, xmax)` defining the real axis range
- `ty`: Tuple `(ymin, ymax)` defining the imaginary axis range
- `gp`: Tuple `(nx, ny)` specifying number of grid points in each dimension

# Returns
A 3-tuple containing:
- `gx::EquispacedGrid`: Grid along real axis (length `nx`)
- `gy::EquispacedGrid`: Grid along imaginary axis (length `ny`)
- `zg::Matrix{T}`: Complex grid matrix (size: `nx × ny`), where `zg[i,j] = gx[i] + im*gy[j]`

# Examples
```julia
# Create 100×100 grid covering [-4,4] + i[-4,4]
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (100, 100))

# Grid dimensions
size(zg)  # (100, 100)

# Corner values
zg[1, 1]      # -4.0 - 4.0im (bottom-left)
zg[end, end]  # 4.0 + 4.0im (top-right)

# Use with pseudospectra functions
using LinearAlgebra
A = randn(ComplexF64, 50, 50)
P = MatrixPencil(schur(A))
psa = ℂsvdpsa(zg, P)  # psa is 100×100, suitable for heatmap(gx, gy, psa)
```

# Notes
- The grid is oriented for standard plotting: first dimension (columns) = real axis,
  second dimension (rows) = imaginary axis
- Pseudospectra functions return transposed matrices for direct plotting compatibility
- Uses `GridArrays.jl` for efficient grid construction

See also: [`ℂsvdpsa`](@ref), [`ihlpsa`](@ref), [`MatrixPencil`](@ref)
"""
function qgrid(T, tx, ty, gp)
    gx = EquispacedGrid(gp[1], tx...)
    gy = EquispacedGrid(gp[2], ty...)
    grid = ProductGrid(gx, gy * 1im)
    return gx, gy, Matrix{T}(sum.(collect(grid)))
end
export qgrid

## precompile gpu code
using PrecompileTools

@setup_workload begin
    using LinearAlgebra
    using KernelAbstractions
    @compile_workload begin
        for T in [ComplexF32, ComplexF64]
            m = 16
            g = 10
            gx, gy, zg = qgrid(T, (-4, 4), (-4, 4), (g, g))
            A = randn(T, m, m)
            P = MatrixPencil(schur(A))
            ℂsvdpsa(zg, P)
            ℝsvdpsa(zg, P)
            ihlpsa(CPU(), zg, P)
        end
    end
end

end
