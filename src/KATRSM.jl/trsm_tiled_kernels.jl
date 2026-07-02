# Tiled / blocked batched pencil triangular solves (KernelAbstractions + KernelIntrinsics).
# See DESIGN_TRSM.md §"Tiled" for the algorithm and design rationale.
# Block layout (forward / lower-triangular sweep; the backward sweep is the mirror image):
#
#     columns:   1 … koff        koff+1 … koff+plen      koff+plen+1 … m
#              ┌──────────────┬──────────────────────┬──────────────────────┐
#   rows<panel │     done     │     (zero — upper     │      (zero)           │
#              │              │       triangle)       │                      │
#              ├──────────────┼──────────────────────┼──────────────────────┤
#   panel k    │              │  ◣ triangular tile    │      (zero)           │
#              │              │  (_tiled_panel_*)     │                      │
#              ├──────────────┼──────────────────────┼──────────────────────┤
#   trailing i │              │  A,B[i, panel] ⇒ sA,sB│      (not yet         │
#              │              │  (_tiled_trailing_*)  │       solved)         │
#              └──────────────┴──────────────────────┴──────────────────────┘
#
# The structurally-zero triangle of A/B is never loaded into sA/sB or multiplied.
#
# Trailing-update workgroups use a flat 1-D group index decoded into (row-tile, grid-pt-tile)
# to avoid relying on KA 2-D group indexing.

using KernelIntrinsics: @shfl, Idx

# Warp-shuffle broadcast of each solved pivot lane→lane (also synchronizes the warp). For IEEE
# hardware floats this is a single `@shfl`. Multi-limb types (MultiFloats) are miscompiled when
# shuffled as one wide composite and silently return garbage — MultiFloatsPseudospectra overrides
# this with a per-limb shuffle instead.
@inline _trsm_shfl(v, src) = @shfl(Idx, v, src)

# ---- diagonal panel solves (one warp per grid point) ----

# Shared body for generic (B≠I) and B=I "eye" panel solves; `veye::Val` selects the pencil element
# via `_piv_elem`/`_offd_elem` (eye wrappers pass `B = nothing`). One warp per grid point: `bl` holds
# this lane's RHS entry in a register; the owner lane (lane==jj) broadcasts its solved pivot via
# `@shfl`, other lanes emit zero. `valid` masks lanes past a partial last panel (plen<32).
@inline function _panel_fwd_body!(b, z, A, B, koff, plen, lane, veye)   # lower-tri (ascending)
    ET = eltype(A)
    j0 = koff + lane
    valid = lane <= plen
    bl = valid ? b[j0] : zero(ET)
    for jj = 1:plen
        j = koff + jj
        piv = (lane == jj) ? _pdiv(bl, @inline _piv_elem(veye, j, z, A, B)) : zero(ET)
        xj = _trsm_shfl(piv, jj)
        if lane == jj
            bl = xj
        elseif (lane > jj) & valid                       # rows below the pivot
            bl -= xj * @inline _offd_elem(veye, j0, j, z, A, B)
        end
    end
    valid && (b[j0] = bl)
end
@inline function _panel_bwd_body!(b, z, A, B, koff, plen, lane, veye)   # upper-tri (descending)
    ET = eltype(A)
    j0 = koff + lane
    valid = lane <= plen
    bl = valid ? b[j0] : zero(ET)
    for jj = plen:-1:1
        j = koff + jj
        piv = (lane == jj) ? _pdiv(bl, @inline _piv_elem(veye, j, z, A, B)) : zero(ET)
        xj = _trsm_shfl(piv, jj)
        if lane == jj
            bl = xj
        elseif (lane < jj) & valid                       # rows above the pivot
            bl -= xj * @inline _offd_elem(veye, j0, j, z, A, B)
        end
    end
    valid && (b[j0] = bl)
end
@kernel function _tiled_panel_forward(bv, zv, @Const(A), @Const(B), koff, plen)
    lane = @index(Local); gi = @index(Group)
    @inline _panel_fwd_body!(bv[gi], zv[gi], A, B, koff, plen, lane, Val(false))
end
@kernel function _tiled_panel_backward(bv, zv, @Const(A), @Const(B), koff, plen)
    lane = @index(Local); gi = @index(Group)
    @inline _panel_bwd_body!(bv[gi], zv[gi], A, B, koff, plen, lane, Val(false))
end
@kernel function _tiled_panel_forward_eye(bv, zv, @Const(A), koff, plen)   # B = I: single matrix
    lane = @index(Local); gi = @index(Group)
    @inline _panel_fwd_body!(bv[gi], zv[gi], A, nothing, koff, plen, lane, Val(true))
end
@kernel function _tiled_panel_backward_eye(bv, zv, @Const(A), koff, plen)  # B = I: single matrix
    lane = @index(Local); gi = @index(Group)
    @inline _panel_bwd_body!(bv[gi], zv[gi], A, nothing, koff, plen, lane, Val(true))
end

# ---- tiled trailing updates (shared A,B tile reused across `gt` grid points) ----
# `@localmem` tile is 32 rows (fixed, one full warp) × TC columns; TC is a compile-time `Val` knob
# decoupled from the 32-wide panel solve, chosen per device+type by `tiled_tc` (src/ihlpsa_trsm.jl).
# See DESIGN_TRSM.md "Trailing-tile width and occupancy" for the tuning rationale.
# A panel wider than TC is subtracted in ⌈plen/TC⌉ column sub-tiles (loop over `c0`); A,B DRAM
# traffic is unchanged. `@synchronize` brackets each sub-tile so the shared tile can be reloaded —
# both barriers must be reached uniformly by every thread (the `c0` loop count depends only on
# plen/TC, so it is).

# Forward: subtract panel k from trailing rows rbase+1 … m.
@kernel function _tiled_trailing_forward(bv, zv, @Const(A), @Const(B),
                                         koff, plen, rbase, m, gt, rtiles, ::Val{TC}, ::Val{W}) where {TC,W}
    li = @index(Local)                # 1..32W
    grp = @index(Group)               # 1..rtiles*ggrid
    @uniform ET = eltype(A)
    t = (li - 1) % 32 + 1            # lane / row within tile
    w = (li - 1) ÷ 32 + 1           # warp 1..W (W warps share one tile)
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1     # each grid-block covers W*gt grid points
    sA = @localmem ET (32, TC)
    sB = @localmem ET (32, TC)
    i = rbase + (bi - 1) * 32 + t
    g = length(zv)
    gp0 = (bg - 1) * (W * gt) + (w - 1) * gt
    for c0 = 0:TC:plen-1
        clen = min(TC, plen - c0)
        jj = w                        # warp w loads columns jj ≡ w (mod W)
        while jj <= clen
            j = koff + c0 + jj
            sA[t, jj] = i <= m ? A[i, j] : zero(ET)
            sB[t, jj] = i <= m ? B[i, j] : zero(ET)
            jj += W
        end
        @synchronize()
        if i <= m
            for gg = 1:gt
                gp = gp0 + gg
                if gp <= g
                    z = zv[gp]
                    b = bv[gp]
                    acc = b[i]
                    for jj2 = 1:clen
                        acc -= (z * sB[t, jj2] - sA[t, jj2]) * b[koff + c0 + jj2]
                    end
                    b[i] = acc
                end
            end
        end
        @synchronize()
    end
end

# Backward: subtract panel k from rows above it, 1 … koff.
@kernel function _tiled_trailing_backward(bv, zv, @Const(A), @Const(B),
                                          koff, plen, gt, rtiles, ::Val{TC}, ::Val{W}) where {TC,W}
    li = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    t = (li - 1) % 32 + 1
    w = (li - 1) ÷ 32 + 1
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, TC)
    sB = @localmem ET (32, TC)
    i = (bi - 1) * 32 + t
    g = length(zv)
    gp0 = (bg - 1) * (W * gt) + (w - 1) * gt
    for c0 = 0:TC:plen-1
        clen = min(TC, plen - c0)
        jj = w
        while jj <= clen
            j = koff + c0 + jj
            sA[t, jj] = i <= koff ? A[i, j] : zero(ET)
            sB[t, jj] = i <= koff ? B[i, j] : zero(ET)
            jj += W
        end
        @synchronize()
        if i <= koff
            for gg = 1:gt
                gp = gp0 + gg
                if gp <= g
                    z = zv[gp]
                    b = bv[gp]
                    acc = b[i]
                    for jj2 = 1:clen
                        acc -= (z * sB[t, jj2] - sA[t, jj2]) * b[koff + c0 + jj2]
                    end
                    b[i] = acc
                end
            end
        end
        @synchronize()
    end
end

# ---- B = I trailing updates (one tile, no z) ----
# For a standard pencil (B = I) the trailing rows i are strictly off the panel columns j, so
# B[i,j] = 0 and the pencil reduces to M[i,j] = -A[i,j]: `acc -= M[i,j]·b[j]` ⇒ `acc += A[i,j]·b[j]`.
# Dropping the B tile halves shared memory and removes z from the trailing update.
#
# Separate kernels rather than a runtime `if eye` / `Val{eye}` body: `@localmem` is a *static*
# allocation, so any unified body touching `sB` would reserve both tiles regardless of the flag.
# Dispatch happens at the host call site (the `eye` branch in `_tiled_trsm!`, src/ihlpsa_trsm.jl),
# which picks the right kernel symbol.

@kernel function _tiled_trailing_forward_eye(bv, @Const(A), koff, plen, rbase, m, gt, rtiles, ::Val{TC}, ::Val{W}) where {TC,W}
    li = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    t = (li - 1) % 32 + 1
    w = (li - 1) ÷ 32 + 1
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, TC)
    i = rbase + (bi - 1) * 32 + t
    g = length(bv)
    gp0 = (bg - 1) * (W * gt) + (w - 1) * gt
    for c0 = 0:TC:plen-1
        clen = min(TC, plen - c0)
        jj = w
        while jj <= clen
            sA[t, jj] = i <= m ? A[i, koff + c0 + jj] : zero(ET)
            jj += W
        end
        @synchronize()
        if i <= m
            for gg = 1:gt
                gp = gp0 + gg
                if gp <= g
                    b = bv[gp]
                    acc = b[i]
                    for jj2 = 1:clen
                        acc += sA[t, jj2] * b[koff + c0 + jj2]
                    end
                    b[i] = acc
                end
            end
        end
        @synchronize()
    end
end

@kernel function _tiled_trailing_backward_eye(bv, @Const(A), koff, plen, gt, rtiles, ::Val{TC}, ::Val{W}) where {TC,W}
    li = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    t = (li - 1) % 32 + 1
    w = (li - 1) ÷ 32 + 1
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, TC)
    i = (bi - 1) * 32 + t
    g = length(bv)
    gp0 = (bg - 1) * (W * gt) + (w - 1) * gt
    for c0 = 0:TC:plen-1
        clen = min(TC, plen - c0)
        jj = w
        while jj <= clen
            sA[t, jj] = i <= koff ? A[i, koff + c0 + jj] : zero(ET)
            jj += W
        end
        @synchronize()
        if i <= koff
            for gg = 1:gt
                gp = gp0 + gg
                if gp <= g
                    b = bv[gp]
                    acc = b[i]
                    for jj2 = 1:clen
                        acc += sA[t, jj2] * b[koff + c0 + jj2]
                    end
                    b[i] = acc
                end
            end
        end
        @synchronize()
    end
end
