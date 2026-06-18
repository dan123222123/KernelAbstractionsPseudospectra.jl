module KAPseudospectra

include("core.jl")
export MatrixPencil

include("svdpsa.jl")
export ℂsvdpsa, ℝsvdpsa

include("ihlpsa.jl")
export ihlpsa

# Build a 2D grid of complex shifts. Returns (gx, gy, zg) where gx and gy are
# real ranges along the real and imaginary axes and zg[i,j] = gx[i] + im*gy[j].
function qgrid(T, tx, ty, gp)
    nx, ny = gp
    gx = range(tx[1], tx[2], length=nx)
    gy = range(ty[1], ty[2], length=ny)
    zg = [T(complex(x, y)) for x in gx, y in gy]
    return collect(gx), collect(gy), zg
end
export qgrid

# Shared precompile body for the GPU extensions. Builds a small problem for each
# element type in `Ts` and exercises the fixed and adaptive `ihlpsa` paths (both the
# B=I and a true B≠I pencil) on one device. Called from inside each extension's
# `@compile_workload` so the backend-specialized method instances are traced and
# cached. `LinearAlgebra` (for `schur`) is in module scope via `svdpsa.jl`.
function _precompile_ihlpsa(backend, dev, Ts)
    for T in Ts
        _, _, zg = qgrid(T, (-4, 4), (-4, 4), (100, 100))
        P = MatrixPencil(schur(randn(T, 32, 32)))   # B = I (single matrix)
        ihlpsa(backend, zg, P, 5; devs=[dev])   # fixed-nit path
        ihlpsa(backend, zg, P; devs=[dev])      # adaptive (per-point hybrid)
        # True matrix pencil (B ≠ I): also trace the generalized-Schur construction
        # path so a first MatrixPencil(A, B) call isn't a cold compile.
        Pg = MatrixPencil(randn(T, 32, 32), randn(T, 32, 32) + T(5) * I)
        ihlpsa(backend, zg, Pg, 5; devs=[dev])
    end
    return nothing
end

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
            ihlpsa(CPU(), zg, P, m)      # fixed-nit path
            ihlpsa(CPU(), zg, P)         # adaptive (per-point hybrid)
        end
    end
end

end
