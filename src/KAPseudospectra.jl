module KAPseudospectra

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa!, ℝsvdpsa!
export ℂsvdpsa, ℝsvdpsa

include("ihlpsa.jl")
export ihlpsa

using GridArrays
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
