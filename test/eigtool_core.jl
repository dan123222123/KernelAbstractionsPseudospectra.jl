using LinearAlgebra
using GridArrays
using MAT

const global tdir = (@__DIR__) * "/"

# read psa matfile and compute eigtool grid
function readvars(fname)
    vars = matread(fname)
    A = complex(vars["A"])
    gx = vec(vars["gx"])
    gy = vec(vars["gy"])
    srgref = vars["srg"]
    grid = ProductGrid(gx, gy * 1im)
    zg = Matrix{eltype(A)}(sum.(collect(grid))) # matrix
    return srgref, A, zg
end

# test svdpsa against eigtool
function testsvdpsa(fname, tol)
    srgref, A, zg = readvars(fname)
    srg = ℂsvdpsa(zg, A, I, 1, 0)
    println("Normed Error for ℂsvdpsa is $(norm(abs.(srgref .- srg))/norm(srgref))")
    return (norm(abs.(srgref .- srg)) / norm(srgref)) < tol
end

# test ihlpsa against eigtool. Fixed-nit runs to full depth (nit = size(A, 1)), so it
# reproduces the eigtool reference to the same tight tol as ℂsvdpsa (1e-6 F32 /
# 1e-14 F64) — the looser tol below is only the rtol-bounded adaptive test, by design.
function testihlpsa(fname, backend, tol)
    srgref, A, zg = readvars(fname)
    P = MatrixPencil(schur(Matrix{complex(eltype(A))}(A)))
    srg = ihlpsa(backend, zg, P, size(A, 1))
    println("Normed Error for ihlpsa on $(backend) is $(norm(abs.(srgref .- srg))/norm(srgref))")
    return (norm(abs.(srgref .- srg)) / norm(srgref)) < tol
end

# test adaptive ihlpsa against eigtool. Stops at the adaptive rtol rather than running
# to full convergence, so `tol` here is the rtol-bounded accuracy (≈ stopping rtol) —
# looser than the fixed-nit/eigtool reference, by design. kwargs forward adaptive
# options (rtol, nit_chunk, nit_max, …).
function testihlpsa_adaptive(fname, backend, tol; kwargs...)
    srgref, A, zg = readvars(fname)
    P = MatrixPencil(schur(Matrix{complex(eltype(A))}(A)))
    # Public `ihlpsa(…; …)` returns only σ; call the internal driver for the
    # convergence depth we want to report alongside the error.
    srg, nit_grid = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; kwargs...)
    println("Normed Error for ihlpsa adaptive on $(backend) is $(norm(abs.(srgref .- srg))/norm(srgref)) (nit=$(maximum(nit_grid)))")
    return (norm(abs.(srgref .- srg)) / norm(srgref)) < tol
end
