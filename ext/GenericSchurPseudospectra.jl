module GenericSchurPseudospectra

# High-precision Schur reduction for extended-precision (MultiFloats) pencils — LAPACK's
# `schur` is IEEE-only, so we reduce at BigFloat via GenericSchur and round the triangular
# factor to the working type.
#
# `bigfloat_schur_factor` is a named primitive, not a `schur` overload: overloading
# `LinearAlgebra.schur` for `Complex{<:MultiFloat}` would be type piracy, and the rounded
# `Z = I` factor doesn't satisfy the Schur-object contract (`A ≈ Z·T·Z'`).

using LinearAlgebra
using MultiFloats: MultiFloat
import GenericSchur
import KernelAbstractionsPseudospectra
using PrecompileTools: @setup_workload, @compile_workload

# Call GenericSchur's drivers by name (`gschur!`/`ggschur!`), immune to any co-loaded
# generic-Schur package (notably GLA, whose complex Schur is broken). The
# `Complex{BigFloat}.(…)` copies are fresh, so the in-place `!` is safe.
function KernelAbstractionsPseudospectra.bigfloat_schur_factor(A::AbstractMatrix; bits::Integer = 256)
    setprecision(BigFloat, bits) do
        Matrix(GenericSchur.gschur!(Complex{BigFloat}.(A)).T)
    end
end

function KernelAbstractionsPseudospectra.bigfloat_qz_factor(A::AbstractMatrix, B::AbstractMatrix;
        bits::Integer = 256)
    setprecision(BigFloat, bits) do
        F = GenericSchur.ggschur!(Complex{BigFloat}.(A), Complex{BigFloat}.(B))
        (Matrix(F.S), Matrix(F.T))
    end
end

# Pencil for a complex MultiFloat matrix, standard (B = I) or generalized. Strictly more
# specific than the core `MatrixPencil(::AbstractMatrix{<:Complex})`, so it wins for
# `Complex{<:MultiFloat}`. Both cases reduce at BigFloat, round the triangular factor(s) to
# the working type, and ship a rounded-identity `Z`.
# The B ≠ I reduction is O(hours) at m ≳ 512; cache `bigfloat_qz_factor`'s output if a size
# is revisited.
function KernelAbstractionsPseudospectra.MatrixPencil(A::AbstractMatrix{Complex{MF}},
        B::Union{AbstractMatrix{Complex{MF}}, UniformScaling} = I;
        bits::Integer = 256) where {MF <: MultiFloat}
    m = LinearAlgebra.checksquare(A)
    if B isa UniformScaling
        isone(B.λ) || throw(ArgumentError(
            "a scaled UniformScaling B is not supported for MultiFloat pencils — pass cI as a dense matrix"))
        S = Complex{MF}.(KernelAbstractionsPseudospectra.bigfloat_schur_factor(A; bits))
        Iₘ = Diagonal(ones(Complex{MF}, m))
        return KernelAbstractionsPseudospectra.SchurMatrixPencil{Complex{MF}, true}(
            S, collect(S'), Iₘ, Iₘ, Matrix{Complex{MF}}(I, m, m))
    end
    LinearAlgebra.checksquare(B) == m || throw(DimensionMismatch("A and B sizes differ"))
    Sb, Tb = KernelAbstractionsPseudospectra.bigfloat_qz_factor(A, B; bits)
    S, T = Complex{MF}.(Sb), Complex{MF}.(Tb)
    KernelAbstractionsPseudospectra.SchurMatrixPencil{Complex{MF}, false}(
        S, collect(S'), T, collect(T'), Matrix{Complex{MF}}(I, m, m))
end

# Real MultiFloat input (grcar and the other test matrices are real)
function KernelAbstractionsPseudospectra.MatrixPencil(A::AbstractMatrix{MF}; kwargs...) where {MF <: MultiFloat}
    KernelAbstractionsPseudospectra.MatrixPencil(Complex{MF}.(A); kwargs...)
end
function KernelAbstractionsPseudospectra.MatrixPencil(A::AbstractMatrix{MF}, B::AbstractMatrix{MF};
        kwargs...) where {MF <: MultiFloat}
    KernelAbstractionsPseudospectra.MatrixPencil(Complex{MF}.(A), Complex{MF}.(B); kwargs...)
end

# Precompile the BigFloat-reduce-and-round path (a tiny low-precision matrix compiles the
# same methods a real large-m call uses) so a first `MatrixPencil(::MultiFloat)` isn't a
# cold compile.
@setup_workload begin
    A = Complex{MultiFloat{Float64, 2}}.(reshape(1:36, 6, 6) .+ im .* reshape(36:-1:1, 6, 6))
    @compile_workload begin
        KernelAbstractionsPseudospectra.MatrixPencil(A; bits = 64)
    end
end

end
