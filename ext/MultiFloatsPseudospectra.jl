module MultiFloatsPseudospectra

# Per-limb warp-shuffle broadcast for MultiFloats. The core `_trsm_shfl` fallback shuffles a
# `MultiFloat` as one wide composite via a single `@shfl`, which MISCOMPILES (returns
# garbage) for multi-limb values. This extension overrides `_trsm_shfl` to shuffle each
# hardware-float limb separately and reconstruct the value, which is exact.

using MultiFloats: MultiFloat
using KernelIntrinsics: @shfl, Idx
import KAPseudospectra

# Shuffle each limb as a hardware float, then rebuild the MultiFloat. `@generated` so the
# per-limb shuffles are fully unrolled (N is a compile-time type parameter).
@inline @generated function KAPseudospectra.KATRSM._trsm_shfl(
        v::MultiFloat{
            T, N}, src) where {T, N}
    limbs = [:(@shfl(Idx, v._limbs[$k], src)) for k in 1:N]
    :(MultiFloat{T, N}(($(limbs...),)))
end

@inline KAPseudospectra.KATRSM._trsm_shfl(v::Complex{<:MultiFloat}, src) = Complex(
    KAPseudospectra.KATRSM._trsm_shfl(real(v), src),
    KAPseudospectra.KATRSM._trsm_shfl(imag(v), src))

end
