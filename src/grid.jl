"""
    qgrid(T, tx, ty, gp) -> (gx, gy, zg)

Build a 2D grid of complex shifts of element type `T` over the rectangle
`tx = (x_min, x_max)`, `ty = (y_min, y_max)` with `gp = (nx, ny)` points per axis.
Returns real coordinate vectors `gx`/`gy` along the real and imaginary axes and the
shift matrix `zg[i, j] = gx[i] + im * gy[j]`, ready for [`ihlpsa`](@ref) /
[`ℂsvdpsa`](@ref) and plotting via [`psaplot`](@ref).
"""
function qgrid(T, tx, ty, gp)
    nx, ny = gp
    gx = range(tx[1], tx[2], length = nx)
    gy = range(ty[1], ty[2], length = ny)
    zg = [T(complex(x, y)) for x in gx, y in gy]
    return collect(gx), collect(gy), zg
end

"""
    psaplot(gx, gy, σ[, eigenvalues]; levels, kwargs...)
    psaplot!(gx, gy, σ[, eigenvalues]; ...)

Plot recipe for a pseudospectra field: contours of `log10.(σ)` over the grid `(gx, gy)` (as
returned by [`qgrid`](@ref) and [`ihlpsa`](@ref)/[`ℂsvdpsa`](@ref)), on a `log₁₀ σ` colorbar and,
if a fourth argument is given, the `eigenvalues` overlaid as diamonds. Pass `levels` and any Plots
attribute (`size`, `line`, `seriestype=:contourf`, …) as keywords.

Requires a plotting stack: the recipe lives in an extension that activates once `RecipesBase` (pulled
in by `Plots`) is loaded. Example: `using Plots; psaplot(gx, gy, srg, eigvals(A); levels=-6:0)`.
"""
function psaplot end
function psaplot! end
