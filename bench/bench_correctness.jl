# End-to-end correctness against a BigFloat dense-SVD oracle. For each precision the
# converged ihlpsa σ field is compared to the TRUE σ_min(zI − A) at 256-bit BigFloat
# (GenericLinearAlgebra), on a small grid. Two pencils per cell:
#   promoted — Float64 LAPACK Schur factor promoted limb-exactly (the suite default);
#              MultiFloat rungs floor at ~1e-15.
#   generic  — full-precision Schur (bench_common `hiprec_schur`), lifting MultiFloat rungs to
#              T's arithmetic floor end-to-end.
# CPU-only (BigFloat SVD is O(m³)/point) — sized for a many-core host.
#
#   julia --project=bench bench/bench_correctness.jl
include(joinpath(@__DIR__, "bench_common.jl"))
using DelimitedFiles, Statistics

setprecision(BigFloat, HIPREC_BITS)   # BENCH_HIPREC_BITS — the oracle and the Schur reduction
                                      # must agree, so there is one knob for both
const MS = env_ints("BENCH_MS", (64, 128, 256))
const GRIDN = env_int("BENCH_CORR_GRIDN", 12)
const TOKS = eltype_tokens(get(ENV, "BENCH_ELTYPES", "f32,f32x2,f64,f32x4,f64x2,f64x4"))
const RESULTS = results_dir()

# MultiFloat → BigFloat exactly via the limb sum. The bare BigFloat/Float64 constructors route
# through Float64(::MultiFloat{Float32,N}), which is undefined (base ≠ output), so a Float32-limb
# rung (f32x2/f32x4) would throw; the limb sum works for every base and is exact at 256 bits.
bigval(x::MultiFloats.MultiFloat) = sum(BigFloat, x._limbs)
bigval(x::Real) = BigFloat(x)
bigcplx(z::Complex) = complex(bigval(real(z)), bigval(imag(z)))

foreach(println, repro_stamp(CPU()))
@printf("matrix=%s  grid=%d²  BigFloat=%d bits  eltypes=%s\n",
    BENCH_MATRIX, GRIDN, precision(BigFloat), join(TOKS, ","))
csv = open_csv(joinpath(RESULTS, "bench_correctness.csv"),
    "eltype,m,gridn,mode,eps,max_err,median_err")
@printf("%-6s %-6s %-9s %-11s %-11s\n", "tok", "m", "mode", "max_err", "median_err")

for m in MS
    Ac = bench_matrix(ComplexF64, m)
    F = schur(Ac)
    Abig = Complex{BigFloat}.(Ac)                  # exact ground-truth matrix
    for tok in TOKS
        T = ELTYPES[tok]
        Tr = real(T)
        zg = bench_grid(T, GRIDN)
        σref = ℂsvdpsa(bigcplx.(zg), Abig)             # BigFloat truth on the solver's exact grid
        Iₘ = Diagonal(ones(T, m))
        # Standard (B = I), not via bench_pencil: subject is the REDUCTION's accuracy.
        # Exempt from BENCH_PENCIL.
        pen(Z, S) = KernelAbstractionsPseudospectra.SchurMatrixPencil{T, true}(S, collect(S'), Iₘ, Iₘ, Z)
        modes = (("promoted", (T.(F.Z), T.(F.T))),
            ("generic", (Matrix{T}(I, m, m), hiprec_schur(Ac, T))))
        for (mode, (Z, S)) in modes
            σ = converged_solve(CPU(), zg, pen(Z, S))
            e = abs.(bigval.(σ) .- σref)
            mx, md = Float64(maximum(e)), Float64(median(e))
            println(csv, @sprintf("%s,%d,%d,%s,%.3e,%.3e,%.3e",
                tok, m, GRIDN, mode, Float64(bigval(eps(Tr))), mx, md))
            flush(csv)
            @printf("%-6s %-6d %-9s %-11.3e %-11.3e\n", tok, m, mode, mx, md)
        end
    end
end
close_csv(csv, joinpath(RESULTS, "bench_correctness.csv"))
