# =============================================================================
# Extended-precision pseudospectra on the GPU with MultiFloats.jl
# =============================================================================
# `ihlpsa` and the KATRSM triangular solves are generic over the complex element type, so
# MultiFloats.jl's isbits extended floats run *inside* the GPU kernels with no package
# changes — only a generic Schur factorization (GenericSchur) and tridiagonal eigen
# (GenericLinearAlgebra) are pulled in.
#
# Why it matters: near the spectrum of a strongly non-normal matrix the resolvent
# (zI − A)⁻¹ is ill-conditioned enough that a single-precision inverse-Lanczos solve
# reports the wrong σ_min. Double-single keeps the recurrence accurate and recovers the
# correct pseudospectral levels — verified against a BigFloat oracle in
# test/test_multifloats.jl.
#
# Both precisions here are Float32-based, so the whole example runs on GPUs *without*
# Float64 (Intel iGPUs, Apple Metal) as well as on Float64 hardware. On an FP64-less GPU,
# also run `KAPseudospectra.set_pdiv_accurate(false)` once and restart Julia, so the
# ComplexF32 baseline solve doesn't widen its complex divide through `double`. On a card
# with fast Float64, use ComplexF64 / Complex{Float64x2} for the ~16-vs-32-digit story.
# =============================================================================

using KernelAbstractions
# backend = CPU()                          # portable default; uncomment the GPU backend you have:
# using CUDA;   backend = CUDABackend()
# using AMDGPU; backend = ROCBackend()
using oneAPI; backend = oneAPIBackend()

using KAPseudospectra, LinearAlgebra, Random, MatrixDepot, MultiFloats
using GenericSchur, GenericLinearAlgebra  # generic schur / eigen for the extended types

LP = ComplexF64                          # plain single precision — the baseline that degrades
HP = Complex{Float32x4}                  # double-single (~14 digits) — the accurate solve

# chebspec: a dense, strongly non-normal operator (a standard pseudospectra example).
A    = MatrixDepot.chebspec(Float32, 20)
eigs = eigvals(A)                        # spectrum — sets the plot window and the markers
xlim = (minimum(real, eigs) - 1.5, maximum(real, eigs) + 1.5)
ylim = (minimum(imag, eigs) - 1.5, maximum(imag, eigs) + 1.5)
G    = 60                                # G×G grid

# One pseudospectra solve at complex element type T (schur() dispatches to GenericSchur for
# the extended types). The start vector is deterministic and shared across precisions.
function solve(bk, ::Type{T}; nit=12, adaptive=false, kw...) where {T<:Complex}
    P = MatrixPencil(schur(T.(A)))
    _, _, zg = qgrid(T, xlim, ylim, (G, G))
    Random.seed!(42); x₀ = T.(randn(ComplexF64, size(A, 1))); x₀ ./= norm(x₀)
    return adaptive ? ihlpsa(bk, zg, P; x₀, kw...) : ihlpsa(bk, zg, P, nit; x₀)
end

## ---- single vs double-single ------------------------------------------------
# Both σ fields are kept in their native type (MultiFloats only narrows back to its own base,
# Float32 — not Float64), so the comparison arithmetic below runs in full double-single.
@info "solving in $LP …";  lo = solve(backend, LP)
@info "solving in $HP …";  hi = solve(backend, HP)

# Relative disagreement field: where the two differ, the ill-conditioned resolvent has
# degraded the single-precision solve and the double-single value is the accurate one.
gx, gy, _ = qgrid(ComplexF32, xlim, ylim, (G, G))
D = abs.(lo .- hi) ./ hi
w = argmax(D)                            # hi[a,b] ↔ z = gx[b] + im·gy[a]
println("\n================ $LP vs $HP ================")
println("  agree to <1%   : ", count(<(0.01f0), D), " / ", length(D))
println("  differ by >10% : ", count(>(0.1f0), D), " / ", length(D), "   ← single precision unreliable")
println("  max rel diff   : ", Float32(maximum(D)), "  at z = ", round(gx[w[2]] + im * gy[w[1]], digits=3))

## ---- adaptive depth ---------------------------------------------------------
# Omit `nit`: each grid point runs only as deep as its own convergence needs. rtol=1f-12
# drives it to full double-single accuracy (well within Float32x2's ~14 digits).
@info "solving in $HP, adaptive depth …"
hi_adapt = solve(backend, HP; adaptive=true, rtol=1f-12)
println("  adaptive vs fixed nit=12: max rel diff = ", Float32(maximum(abs.(hi_adapt .- hi) ./ hi)))

## ---- GPU correctness gate ---------------------------------------------------
# On the GPU, double-single must track the CPU far more closely than either tracks the
# single-precision solve. (If FMA contraction had corrupted the error-free transforms, the
# GPU result would collapse toward single-precision accuracy.) Do not @fastmath these.
if KernelAbstractions.isgpu(backend)
    @info "GPU vs CPU double-single gate …"
    cpu = solve(CPU(), HP)
    println("  GPU-HP vs CPU-HP max rel diff = ", Float32(maximum(abs.(hi .- cpu) ./ cpu)))
end

## ---- optional plot ----------------------------------------------------------
# Left: the double-single pseudospectra. Right: where single precision disagrees — a band
# hugging the spectrum, exactly where the ill-conditioned resolvent defeats Float32.
if true
    using Plots; gr()
    # Float32.(): Plots needs a plain float, not a MultiFloats type — narrow at the boundary
    p1 = contourf(gx, gy, clamp.(log10.(Float32.(hi)), -17, 0); levels=collect(-17:0), color=:turbo,
                  title="log10 σ_min  ($HP)", xlabel="Re z", ylabel="Im z", aspect_ratio=1)
    p2 = contourf(gx, gy, log10.(clamp.(Float32.(D), 1f-16, Inf32)); levels=collect(-16:2), color=:inferno,
                  title="log10 |σ_$LP − σ_$HP| / σ_$HP", xlabel="Re z", aspect_ratio=1)
    scatter!(p1, real.(eigs), imag.(eigs); m=(:diamond, 3, :white), label="")
    scatter!(p2, real.(eigs), imag.(eigs); m=(:diamond, 3, :cyan), label="")
    plt = plot(p1, p2; layout=(1, 2), size=(1500, 720),
               plot_title="KAPseudospectra inverse-Lanczos + GenericSchur, chebspec m=$(size(A, 1))")
    savefig(plt, joinpath(@__DIR__, "ihlpsa_multifloats.png"))
    @info "saved figure" path = joinpath(@__DIR__, "ihlpsa_multifloats.png")
end
