# extended-precision ihlpsa vs an independent BigFloat oracle.
#
# The same pseudospectra computed three ways on chebspec (a standard strongly non-normal example):
# 1) inverse-Lanczos at ComplexF64; 2) inverse-Lanczos at Complex{Float64x2}; and, 3) a dense-SVD
# oracle (`ℂsvdpsa`) in BigFloat at 256 bits (~77 digits).
using KernelAbstractionsPseudospectra, KernelAbstractions, LinearAlgebra, MatrixDepot, MultiFloats
using GenericSchur, GenericLinearAlgebra  # generic schur / eigen for the extended types

m, G, nit = 20, 21, 12
setprecision(BigFloat, 256)

A = MatrixDepot.chebspec(Float64, m)
eigs = eigvals(complex.(A))
pad = 1.5
xlim = (minimum(real, eigs) - pad, maximum(real, eigs) + pad)
ylim = (minimum(imag, eigs) - pad, maximum(imag, eigs) + pad)
grid(::Type{T}) where {T <: Complex} = qgrid(T, xlim, ylim, (G, G))[3]

solve(::Type{T}) where {T <: Complex} = ihlpsa(CPU(), grid(T), MatrixPencil(schur(T.(A))), nit)

@info "oracle: dense BigFloat SVD on the $(G)×$(G) grid …"
σ_oracle = ℂsvdpsa(grid(Complex{BigFloat}), Complex{BigFloat}.(A))
@info "inverse-Lanczos at ComplexF64 and Complex{Float64x2} …"
σ64 = solve(ComplexF64)
σmf = solve(Complex{Float64x2})

# Relative error in BigFloat, denominator floored: σ_min(z) → 0 at the eigenvalues, where
# any solver's *relative* accuracy necessarily degrades (absolute accuracy ~ eps·‖A‖).
relerr(σ) = abs.(BigFloat.(σ) .- σ_oracle) ./ max.(σ_oracle, eps(BigFloat))
median(v) = (s = sort(vec(v)); n = length(s);
    iseven(n) ? (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2 : s[(n + 1) ÷ 2])

e64, emf = relerr(σ64), relerr(σmf)
println("\nchebspec m=$m, $(G)×$(G) grid, nit=$nit — relative error vs the BigFloat oracle")
println("  precision            median      worst       points >1% off")
for (name, e) in ("Float64  (~16 dig)" => e64, "Float64x2 (~32 dig)" => emf)
    println("  $name  $(Float64(median(e)))  $(Float64(maximum(e)))  ",
        count(>(big"0.01"), e), " / ", length(e))
end
w = argmax(e64)
println("\nworst Float64 point: true σ = $(Float64(σ_oracle[w])) — Float64 reports ",
    "$(σ64[w]), Float64x2 reports $(Float64(BigFloat(σmf[w])))")
