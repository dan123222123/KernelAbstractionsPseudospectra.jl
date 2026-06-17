# Tiled / blocked batched pencil triangular solves (KernelAbstractions + KernelIntrinsics).
#
# For large m the per-grid-point warp solves become bandwidth-bound: every warp re-streams
# the whole m×m A,B pencil from DRAM with zero reuse across grid points (A,B are shared; only
# z varies). The tiled solve fixes this with a right-looking blocked algorithm, panel width
# = warp size (32):
#   for each panel k:
#     (1) PANEL SOLVE — solve the ≤32×32 diagonal block for every grid point (one warp per
#         grid point, pivots broadcast by `@shfl`, same numerics as the register-warp kernel);
#     (2) TRAILING UPDATE — a tiled GEMM that subtracts panel k's contribution from the
#         trailing rows. Each workgroup loads the A,B[row-tile, panel-k] tile into `@localmem`
#         ONCE and reuses it across `gt` grid points, turning the streaming into compute-bound
#         work. The pencil M = zB − A is kept as separate A,B tiles + a per-grid-point z combine
#         so the shared tiles are z-independent and reused across the batch.
#
# Trailing-update workgroups use a flat 1-D group index decoded into (row-tile, grid-pt-tile)
# to avoid relying on KA 2-D group indexing. Numerics match the column-oriented / register-warp
# kernels (same `_pdiv`, `zBAij`, update order).

using KernelIntrinsics: @shfl, Idx

# ---- diagonal panel solves (one warp per grid point) ----

@kernel function _tiled_panel_forward(bv, zv, @Const(A), @Const(B), koff, plen)   # lower-tri block
    lane = @index(Local)
    gi = @index(Group)
    @uniform ET = eltype(A)
    b = bv[gi]
    z = zv[gi]
    j0 = koff + lane
    valid = lane <= plen
    bl = valid ? b[j0] : zero(ET)
    for jj = 1:plen
        j = koff + jj
        piv = (lane == jj) ? _pdiv(bl, @inline zBAij(j, j, z, A, B)) : zero(ET)
        xj = @shfl(Idx, piv, jj)
        if lane == jj
            bl = xj
        elseif (lane > jj) & valid
            bl -= xj * @inline zBAij(j0, j, z, A, B)
        end
    end
    valid && (b[j0] = bl)
end

@kernel function _tiled_panel_backward(bv, zv, @Const(A), @Const(B), koff, plen)  # upper-tri block
    lane = @index(Local)
    gi = @index(Group)
    @uniform ET = eltype(A)
    b = bv[gi]
    z = zv[gi]
    j0 = koff + lane
    valid = lane <= plen
    bl = valid ? b[j0] : zero(ET)
    for jj = plen:-1:1
        j = koff + jj
        piv = (lane == jj) ? _pdiv(bl, @inline zBAij(j, j, z, A, B)) : zero(ET)
        xj = @shfl(Idx, piv, jj)
        if lane == jj
            bl = xj
        elseif (lane < jj) & valid
            bl -= xj * @inline zBAij(j0, j, z, A, B)
        end
    end
    valid && (b[j0] = bl)
end

# ---- tiled trailing updates (shared A,B tile reused across `gt` grid points) ----

# Forward: subtract panel k from trailing rows rbase+1 … m.
@kernel function _tiled_trailing_forward(bv, zv, @Const(A), @Const(B),
                                         koff, plen, rbase, m, gt, rtiles)
    t = @index(Local)                 # 1..32 → row within tile
    grp = @index(Group)               # 1..rtiles*ggrid
    @uniform ET = eltype(A)
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, 32)
    sB = @localmem ET (32, 32)
    i = rbase + (bi - 1) * 32 + t
    for jj = 1:plen
        j = koff + jj
        sA[t, jj] = i <= m ? A[i, j] : zero(ET)
        sB[t, jj] = i <= m ? B[i, j] : zero(ET)
    end
    @synchronize()
    if i <= m
        g = length(zv)
        gp0 = (bg - 1) * gt
        for gg = 1:gt
            gp = gp0 + gg
            if gp <= g
                z = zv[gp]
                b = bv[gp]
                acc = b[i]
                for jj = 1:plen
                    acc -= (z * sB[t, jj] - sA[t, jj]) * b[koff + jj]
                end
                b[i] = acc
            end
        end
    end
end

# Backward: subtract panel k from rows above it, 1 … koff.
@kernel function _tiled_trailing_backward(bv, zv, @Const(A), @Const(B),
                                          koff, plen, gt, rtiles)
    t = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, 32)
    sB = @localmem ET (32, 32)
    i = (bi - 1) * 32 + t
    for jj = 1:plen
        j = koff + jj
        sA[t, jj] = i <= koff ? A[i, j] : zero(ET)
        sB[t, jj] = i <= koff ? B[i, j] : zero(ET)
    end
    @synchronize()
    if i <= koff
        g = length(zv)
        gp0 = (bg - 1) * gt
        for gg = 1:gt
            gp = gp0 + gg
            if gp <= g
                z = zv[gp]
                b = bv[gp]
                acc = b[i]
                for jj = 1:plen
                    acc -= (z * sB[t, jj] - sA[t, jj]) * b[koff + jj]
                end
                b[i] = acc
            end
        end
    end
end
