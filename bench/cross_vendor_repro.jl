# Cross-vendor numerical reproducibility: same deterministic pencil/grid/x₀, same fixed nit —
# compared across CUDA / AMDGPU / oneAPI by cross_vendor_compare.jl once each vendor has run
# this script. FIXED-driver-only: the adaptive driver could retire a borderline point at a
# different chunk per vendor, conflating arithmetic differences with algorithmic ones; the
# fixed driver runs the identical computational graph everywhere. x₀ is built here from an
# explicitly-seeded MersenneTwister and passed via `ihlpsa(...; x₀=...)` — a plain host
# Vector, byte-identical regardless of backend. ComplexF32 because oneAPI devices may lack
# native FP64.
#
# Usage:  julia --project=bench bench/cross_vendor_repro.jl cuda      # or amdgpu | oneapi
include(joinpath(@__DIR__, "bench_common.jl"))
include(joinpath(@__DIR__, "cross_vendor_common.jl"))
using DelimitedFiles

backend = select_backend(ARGS)
const T = ComplexF32
const M = 32
const GRIDN = 32
const NIT = 24
const SEED = 0x78766572          # "xver", arbitrary but fixed
const RESULTS = results_dir()

foreach(println, repro_stamp(backend))

P = MatrixPencil(MatrixDepot.grcar(T, M))                # deterministic; no seeding needed
_, _, zg = qgrid(T, (-3, 3), (-3, 3), (GRIDN, GRIDN))     # pure linspace; no RNG involved
x = randn(MersenneTwister(SEED), T, M)
x0 = x ./ norm(x)                                         # host Vector{T}: identical on every vendor

zpd = pinned_zpd(backend, T, M, NIT; ngrid = GRIDN * GRIDN)
σ = ihlpsa(backend, zg, P, NIT; x₀ = x0, zpd)

tag = backend_tag(backend)
outpath = cross_vendor_sigma_path(RESULTS, tag)
writedlm(outpath, σ, ',')
println("wrote ", outpath, " (", size(σ), " grid, m=", M, ", nit=", NIT, ", backend=", tag, ")")
