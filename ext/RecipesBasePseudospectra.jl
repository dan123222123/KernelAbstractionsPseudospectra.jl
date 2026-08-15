module RecipesBasePseudospectra

# Plots recipe for pseudospectra fields: contours of log10(σ) over the shift grid, with an
# optional eigenvalue overlay. Activates once RecipesBase (pulled in by Plots) is loaded, so
# the core package carries no plotting dependency. Implements `KernelAbstractionsPseudospectra.psaplot[!]`.

using RecipesBase
import KernelAbstractionsPseudospectra: psaplot, psaplot!

# Carries the positional args (gx, gy, σ[, eigenvalues]) into the recipe.
struct PsaPlot
    args::Tuple
end
psaplot(args...; kw...) = RecipesBase.plot(PsaPlot(args); kw...)
psaplot!(args...; kw...) = RecipesBase.plot!(PsaPlot(args); kw...)

@recipe function f(h::PsaPlot)
    n = length(h.args)
    n in (3, 4) ||
        throw(ArgumentError("psaplot(gx, gy, σ[, eigenvalues]); got $n positional args"))
    gx, gy, σ = h.args[1], h.args[2], h.args[3]
    logσ = log10.(σ)

    seriestype --> :contour
    color --> :darkrainbow
    clabels --> false
    aspect_ratio --> :equal
    xguide --> "Re z"
    yguide --> "Im z"
    # GR ignores colorbar_ticks on a contour colorbar; relabel via the axis title instead.
    colorbar_title --> "log₁₀ σ"

    # A range `levels` trips some Plots contour paths; materialize it to a vector.
    get(plotattributes, :levels, nothing) isa AbstractRange &&
        (levels := collect(plotattributes[:levels]))

    @series begin
        gx, gy, logσ
    end

    if n == 4
        ev = h.args[4]
        @series begin
            seriestype := :scatter
            markershape --> :diamond
            markercolor --> :white
            markersize --> 3
            label := ""
            (real.(ev), imag.(ev))
        end
    end
end

end
