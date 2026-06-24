# =============================================================================
# Very high precision pseudospectra on the GPU with MultiFloats.jl
# =============================================================================
#
# `ihlpsa` and the KATRSM triangular-solve kernels are generic over the complex element
# type — they use only `+ - * / conj sqrt real abs2`. MultiFloats.jl supplies isbits
# extended-precision floats (`Float64x2` = double-double ≈ 32 digits, `Float64x4` ≈ 64),
# so extended-precision pseudospectra run inside the GPU kernels with no package changes;
# only a generic Schur factorization (GenericSchur) and tridiagonal `eigen`
# (GenericLinearAlgebra) are needed for the dense-matrix path and the σ extraction.
#
# Why it matters: for a strongly non-normal matrix the resolvent (zI − A)⁻¹ is highly
# ill-conditioned near the spectrum, so a Float64 inverse-Lanczos solve loses accuracy and
# reports the wrong σ_min there. Running the solve in double-double keeps the recurrence
# accurate and recovers the correct pseudospectral levels — verified against an independent
# high-precision oracle in the sibling script `ihlpsa_multifloats_accuracy.jl`.
# =============================================================================

using KernelAbstractions
backend = CPU()                       # works everywhere and is the correctness reference
#using CUDA; backend = CUDABackend()  # add your GPU backend (CUDA/AMDGPU/oneAPI) to use it
# Note: MultiFloats is Float64-based, so double-double on FP64-throttled consumer cards
# (e.g. Pascal at 1/32 FP64) is correct but slow — keep M and the grid modest there.

using KAPseudospectra, LinearAlgebra, Random
using MultiFloats
using GenericSchur                    # generic schur(::Matrix{Complex{Float64xN}})
using GenericLinearAlgebra            # generic eigen for the σ-extraction tridiagonal

# On CUDA the NVPTX compiler can contract a*b+c into an FMA, which breaks the error-free
# transforms double-double relies on (MultiFloats #23). Current MultiFloats releases emit
# FMA-safe operators by default, and the GPU correctness gate at the bottom checks this
# empirically. Do not wrap these kernels in `@fastmath`.

## ---- precision --------------------------------------------------------------
const HP = Complex{Float64x2}         # double-double (~32 digits); use Float64x4 for ~64

## ---- problem knobs ----------------------------------------------------------
const M      = 20                     # matrix size
const NIT    = 12                     # fixed Lanczos depth. Inverse-Lanczos targets the
                                      #   EXTREME eigenvalue (1/σ_min²), which it captures in
                                      #   a handful of steps — well below m. Here it's fully
                                      #   converged by ~4–6 (the adaptive run below picks 6 at
                                      #   every point); 12 leaves margin without over-iterating.
const G      = 60                     # grid is G×G
const SEED   = 42                     # start-vector seed (matched across precisions)

const DO_PLOT = true                  # save a contour figure (Plots/GR; set false to skip)

## ---- a strongly non-normal matrix -------------------------------------------
# chebspec: the Chebyshev spectral differentiation operator (MatrixDepot) — a standard,
# strongly non-normal pseudospectra example. It is dense, so the Schur factorization is
# genuinely exercised. Built once in Float64 and promoted, so every precision sees the
# identical matrix.
using MatrixDepot
const AREF = MatrixDepot.chebspec(Float64, M)
matrix(::Type{T}) where {T<:Complex} = T.(AREF)

const EIGS = eigvals(matrix(ComplexF64))
const REGION = let pad = 1.5
    ((minimum(real, EIGS) - pad, maximum(real, EIGS) + pad),
     (minimum(imag, EIGS) - pad, maximum(imag, EIGS) + pad))
end

# Deterministic unit-norm start vector, drawn in Float64 and promoted so every precision
# starts from the same direction (a fair comparison).
make_x0(::Type{T}) where {T<:Complex} = (Random.seed!(SEED); x = T.(randn(ComplexF64, M)); x ./ norm(x))

## ---- one pseudospectra solve at a given precision ---------------------------
function solve(bk, ::Type{T}; nit=NIT, adaptive=false, kw...) where {T<:Complex}
    P = MatrixPencil(schur(matrix(T)))                # dense ⇒ schur via GenericSchur for T
    _, _, zg = qgrid(T, REGION[1], REGION[2], (G, G))
    return adaptive ? ihlpsa(bk, zg, P; x₀=make_x0(T), kw...) :
                      ihlpsa(bk, zg, P, nit; x₀=make_x0(T))
end

gx, gy, _ = qgrid(ComplexF64, REGION[1], REGION[2], (G, G))

## ---- run Float64 vs double-double -------------------------------------------
@info "running Float64 reference …"
srg_f64 = solve(backend, ComplexF64)

@info "running $(HP) (double-double) …"
srg_mf  = Float64.(solve(backend, HP))                # down-convert only for comparison/plot

## ---- report -----------------------------------------------------------------
# Report the relative disagreement field rather than min(σ): where Float64 and double-double
# differ, the ill-conditioned resolvent has degraded the Float64 solve and the double-double
# value is the accurate one. Comparing min(σ) directly would be misleading, since Float64's
# small in-band values are dominated by rounding error rather than genuine resolution.
D = abs.(srg_f64 .- srg_mf) ./ srg_mf
worst = argmax(D)
zworst = gx[worst[2]] + im * gy[worst[1]]             # srg[a,b] ↔ z = gx[b] + im·gy[a]
println()
println("================ Float64 vs $(HP) ================")
println("  agree to <1%                : ", count(<(0.01), D), " / ", length(D))
println("  differ by >10%              : ", count(>(0.1), D), " / ", length(D), "  ← Float64 unreliable")
println("  max relative |F64 − MF| / MF: ", maximum(D))
println("  worst z = ", round(zworst, digits=3), ":  σ_F64 = ", srg_f64[worst], ",  σ_MF = ", srg_mf[worst])

## ---- adaptive depth in extended precision -----------------------------------
# The adaptive driver (omit `nit`) runs in extended precision too. Tighten rtol/atol below
# their Float64 defaults (rtol 1e-6) to converge to full double-double accuracy; each grid
# point then runs only as deep as it needs.
@info "running $(HP) adaptive depth …"
srg_adapt = Float64.(solve(backend, HP; adaptive=true, rtol=1e-25, atol=eps(real(HP))))
Da = abs.(srg_adapt .- srg_mf) ./ srg_mf
println("  adaptive vs fixed nit=$(NIT) ($(HP)): max rel diff = ", maximum(Da))

## ---- GPU correctness gate ---------------------------------------------------
# On the GPU, cross-check the double-double arithmetic against the CPU: the two should agree
# far more closely than either does to the Float64 solve. If GPU-MF instead tracked Float64,
# CUDA FMA contraction would have corrupted the double-double error-free transforms.
if KernelAbstractions.isgpu(backend)
    @info "GPU MultiFloats correctness gate vs CPU …"
    cpu = Float64.(solve(CPU(), HP))
    gate = maximum(abs.(srg_mf .- cpu) ./ cpu)
    println("  GPU-MF vs CPU-MF max rel diff = ", gate,
            "  (should be ≪ the F64-vs-MF band of ~", round(Int, maximum(D) * 100), "%)")
end

## ---- optional plot ----------------------------------------------------------
# Left: the double-double pseudospectra. Right: where Float64 disagrees — a band hugging the
# spectrum, exactly where the ill-conditioned resolvent defeats Float64.
if DO_PLOT
    using Plots
    gr()
    p1 = contourf(gx, gy, clamp.(log10.(srg_mf), -17, 0); levels=collect(-17:1:0), color=:turbo,
                  title="log10 σ_min  ($(HP))", xlabel="Re z", ylabel="Im z", aspect_ratio=1)
    scatter!(p1, real.(EIGS), imag.(EIGS); m=(:diamond, 3, :white), label="")
    p2 = contourf(gx, gy, log10.(clamp.(D, 1e-16, Inf)); levels=collect(-16:1:2), color=:inferno,
                  title="log10 |σ_F64 − σ_MF| / σ_MF", xlabel="Re z", aspect_ratio=1)
    scatter!(p2, real.(EIGS), imag.(EIGS); m=(:diamond, 3, :cyan), label="")
    plt = plot(p1, p2; layout=(1, 2), size=(1500, 720),
               plot_title="KAPseudospectra inverse-Lanczos + GenericSchur, chebspec m=$M")
    savefig(plt, joinpath(@__DIR__, "ihlpsa_multifloats.png"))
    @info "saved figure" path = joinpath(@__DIR__, "ihlpsa_multifloats.png")
end
