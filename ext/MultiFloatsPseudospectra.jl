module MultiFloatsPseudospectra

# Per-limb warp-shuffle broadcast for MultiFloats. The tiled panel solve broadcasts pivots with
# `KATRSM._trsm_shfl`. The core fallback shuffles the whole value with a single `@shfl`, which is
# exact for IEEE hardware floats but MISCOMPILES for a multi-limb `MultiFloat` shuffled as one wide
# composite inside the solve loop (it returns garbage). Shuffling each underlying hardware-float limb
# separately and reconstructing the value is exact, so this extension overrides `_trsm_shfl` for
# MultiFloat element types.

using MultiFloats: MultiFloat
using KernelIntrinsics: @shfl, Idx
import KAPseudospectra

# Shuffle each limb as a hardware float, then rebuild the MultiFloat. `@generated` so the
# per-limb shuffles are fully unrolled (N is a compile-time type parameter).
@inline @generated function KAPseudospectra.KATRSM._trsm_shfl(v::MultiFloat{T,N}, src) where {T,N}
    limbs = [:(@shfl(Idx, v._limbs[$k], src)) for k in 1:N]
    :(MultiFloat{T,N}(($(limbs...),)))
end

@inline KAPseudospectra.KATRSM._trsm_shfl(v::Complex{<:MultiFloat}, src) =
    Complex(KAPseudospectra.KATRSM._trsm_shfl(real(v), src),
            KAPseudospectra.KATRSM._trsm_shfl(imag(v), src))

end
