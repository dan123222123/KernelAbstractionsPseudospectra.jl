# Tiled / blocked batched pencil triangular solves. Block layout (forward sweep; backward is the
# mirror image):
#
#     columns:   1 … poff        poff+1 … poff+psize      poff+psize+1 … m
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
# The structurally-zero triangle of A/B is never loaded into sA/sB or multiplied. Trailing-update
# launches are 2-D over workgroups: `ndrange = (wp·nwarps·rtiles, ggrid)` with a `(wp·nwarps, 1)`
# workgroup, where `wp = warp_width(backend)` (64 on CDNA, 32 elsewhere).

using KernelAbstractions
using KernelIntrinsics: @shfl, Idx

# Warp-shuffle broadcast of a solved pivot lane→lane (also synchronizes the warp). Multi-limb
# types (MultiFloats) are miscompiled when shuffled as one wide composite and silently return
# garbage — MultiFloatsPseudospectra overrides this with a per-limb shuffle instead.
@inline _trsm_shfl(v, src) = @shfl(Idx, v, src)

# ---- diagonal panel solves (one warp per grid point) ----

# Shared body for generic (B≠I) and eye (B=I) panel solves; `veye::Val` selects the pencil
# element via `_piv_elem`/`_offd_elem` (eye wrappers pass `B = nothing`). One warp per grid point:
# `bl` holds this lane's RHS entry in a register; the owner lane broadcasts its solved pivot via
# `@shfl`. `valid` masks lanes past a partial last panel (psize < the warp width).
@inline function _panel_fwd_body!(b, z, A, B, poff, psize, lane, veye)   # lower-tri (ascending)
    ET = eltype(A)
    j0 = poff + lane
    valid = lane <= psize
    bl = valid ? b[j0] : zero(ET)
    for jj in 1:psize
        j = poff + jj
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
@inline function _panel_bwd_body!(b, z, A, B, poff, psize, lane, veye)   # upper-tri (descending)
    ET = eltype(A)
    j0 = poff + lane
    valid = lane <= psize
    bl = valid ? b[j0] : zero(ET)
    for jj in psize:-1:1
        j = poff + jj
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
@kernel function _tiled_panel_forward(bv, zv, @Const(A), @Const(B), poff, psize)
    lane = @index(Local);
    gi = @index(Group)
    @inline _panel_fwd_body!(bv[gi], zv[gi], A, B, poff, psize, lane, Val(false))
end
@kernel function _tiled_panel_backward(bv, zv, @Const(A), @Const(B), poff, psize)
    lane = @index(Local);
    gi = @index(Group)
    @inline _panel_bwd_body!(bv[gi], zv[gi], A, B, poff, psize, lane, Val(false))
end
@kernel function _tiled_panel_forward_eye(bv, zv, @Const(A), poff, psize)   # B = I: single matrix
    lane = @index(Local);
    gi = @index(Group)
    @inline _panel_fwd_body!(bv[gi], zv[gi], A, nothing, poff, psize, lane, Val(true))
end
@kernel function _tiled_panel_backward_eye(bv, zv, @Const(A), poff, psize)  # B = I: single matrix
    lane = @index(Local);
    gi = @index(Group)
    @inline _panel_bwd_body!(bv[gi], zv[gi], A, nothing, poff, psize, lane, Val(true))
end

# ---- tiled trailing updates (shared A,B tile reused across `gridpts` grid points) ----
# @localmem tile is NumTileWarpRows × NumTileCols (compile-time Val knobs):
#   NumTileWarpRows  trailing rows staged per tile (one warp's worth; sourced from warp_width).
#   NumTileCols      panel columns staged per tile (from tile_cols); a wider panel is subtracted
#                    in ⌈psize/NumTileCols⌉ column sub-tiles (the c0 loop).
#
# @synchronize brackets each sub-tile reload; both barriers must be reached uniformly by every
# thread (c0's trip count depends only on psize/NumTileCols). Forward/backward sweeps share one
# kernel via a runtime `rows` window (forward (rbase+1):m, backward 1:poff). nwarps is recovered
# from @groupsize() rather than passed, since the launch's workgroupsize already fixes it
# statically — passing it again could silently disagree with the launch geometry. @synchronize
# must stay lexically in the kernel body (KA's CPU backend splits kernels only at that scope).

# Splits the flat local index into (lane, warp) and places the block's grid-point window. A
# block's nwarps warps all cover the same rows and share one tile, differing only in grid points:
# warp `w` takes the w-th `gridpts`-sized slice.
@inline function _trailing_ids(li, bg, gridpts, warprows, nwarps)
    t = (li - 1) % warprows + 1         # lane = row within the tile
    w = (li - 1) ÷ warprows + 1         # warp 1..nwarps
    return t, w, (bg - 1) * (nwarps * gridpts) + (w - 1) * gridpts
end

# Cooperative sub-tile load: warp `w` takes columns w, w+nwarps, w+2·nwarps, … Rows past the sweep's
# last row store zero, so the accumulate below needs no second mask. `sB::Nothing` is the B = I
# case, dispatched away at compile time.
@inline function _trailing_load!(sA, sB::Nothing, A, B, t, i, inrange, poff, c0, clen, w, nwarps)
    ET = eltype(A)
    jj = w
    while jj <= clen
        sA[t, jj] = inrange ? A[i, poff + c0 + jj] : zero(ET)
        jj += nwarps
    end
end
@inline function _trailing_load!(sA, sB, A, B, t, i, inrange, poff, c0, clen, w, nwarps)
    ET = eltype(A)
    jj = w
    while jj <= clen
        j = poff + c0 + jj
        sA[t, jj] = inrange ? A[i, j] : zero(ET)
        sB[t, jj] = inrange ? B[i, j] : zero(ET)
        jj += nwarps
    end
end

# Row `i` of one grid point against the shared tile; `b[i]` is read and written once per sub-tile.
@inline function _trailing_row!(b, z, sA, sB::Nothing, t, i, poff, c0, clen)
    acc = b[i]
    for jj in 1:clen
        acc += sA[t, jj] * b[poff + c0 + jj]
    end
    b[i] = acc
end
@inline function _trailing_row!(b, z, sA, sB, t, i, poff, c0, clen)
    acc = b[i]
    for jj in 1:clen
        acc -= (z * sB[t, jj] - sA[t, jj]) * b[poff + c0 + jj]
    end
    b[i] = acc
end

# Subtract panel k from the row window `rows`: forward passes (rbase+1):m, backward 1:poff.
@kernel function _tiled_trailing(bv, zv, @Const(A), @Const(B), poff, psize, rows, gridpts,
        ::Val{NumTileWarpRows}, ::Val{NumTileCols}) where {NumTileWarpRows, NumTileCols}
    li = @index(Local)                # 1 .. NumTileWarpRows*nwarps
    bi, bg = @index(Group, NTuple)    # (row tile, grid-point tile)
    @uniform ET = eltype(A)
    nwarps = @groupsize()[1] ÷ NumTileWarpRows   # warps/block, static via the launch's workgroupsize
    ilim = last(rows)
    t, w, gp0 = @inline _trailing_ids(li, bg, gridpts, NumTileWarpRows, nwarps)
    sA = @localmem ET (NumTileWarpRows, NumTileCols)
    sB = @localmem ET (NumTileWarpRows, NumTileCols)
    i = first(rows) - 1 + (bi - 1) * NumTileWarpRows + t
    zpd = length(bv)                  # grid points in this batch (the driver's zpd)
    for c0 in 0:NumTileCols:(psize - 1)
        clen = min(NumTileCols, psize - c0)
        @inline _trailing_load!(sA, sB, A, B, t, i, i <= ilim, poff, c0, clen, w, nwarps)
        @synchronize()
        if i <= ilim
            for gg in 1:gridpts
                gp = gp0 + gg
                gp <= zpd &&
                    @inline _trailing_row!(bv[gp], zv[gp], sA, sB, t, i, poff, c0, clen)
            end
        end
        @synchronize()
    end
end

# ---- B = I trailing update (one tile, no z) ----
# B=I: trailing rows i are off the panel columns j, so B[i,j]=0 and M[i,j]=-A[i,j], i.e.
# `acc -= M[i,j]·b[j]` ⇒ `acc += A[i,j]·b[j]`.
#
# A separate kernel rather than a runtime `if eye` / `Val{eye}` body: `@localmem` is a *static*
# allocation, so any unified body touching `sB` would reserve both tiles regardless of the flag.
# Selected by the `eye` branch in `_tiled_trsm!` (src/ihlpsa_trsm.jl).
@kernel function _tiled_trailing_eye(bv, @Const(A), poff, psize, rows, gridpts,
        ::Val{NumTileWarpRows}, ::Val{NumTileCols}) where {NumTileWarpRows, NumTileCols}
    li = @index(Local)
    bi, bg = @index(Group, NTuple)
    @uniform ET = eltype(A)
    nwarps = @groupsize()[1] ÷ NumTileWarpRows
    ilim = last(rows)
    t, w, gp0 = @inline _trailing_ids(li, bg, gridpts, NumTileWarpRows, nwarps)
    sA = @localmem ET (NumTileWarpRows, NumTileCols)
    i = first(rows) - 1 + (bi - 1) * NumTileWarpRows + t
    zpd = length(bv)                  # grid points in this batch (the driver's zpd)
    for c0 in 0:NumTileCols:(psize - 1)
        clen = min(NumTileCols, psize - c0)
        @inline _trailing_load!(
            sA, nothing, A, nothing, t, i, i <= ilim, poff, c0, clen, w, nwarps)
        @synchronize()
        if i <= ilim
            for gg in 1:gridpts
                gp = gp0 + gg
                gp <= zpd &&
                    @inline _trailing_row!(
                        bv[gp], nothing, sA, nothing, t, i, poff, c0, clen)
            end
        end
        @synchronize()
    end
end
