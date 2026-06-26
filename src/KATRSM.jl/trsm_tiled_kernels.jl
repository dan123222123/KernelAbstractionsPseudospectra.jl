# Tiled / blocked batched pencil triangular solves (KernelAbstractions + KernelIntrinsics).
# See also DESIGN_TRSM.md §"Tiled" for the bandwidth motivation, the crossover with the
# register-warp solve, and the end-to-end speedups.
#
# For large m the per-grid-point warp solves become bandwidth-bound: every warp re-streams
# the whole m×m A,B pencil from DRAM with zero reuse across grid points (A,B are shared; only
# z varies). The tiled solve fixes this with a right-looking blocked algorithm, panel width
# = warp size (32):
#   for each panel k:
#     (1) PANEL SOLVE — solve the ≤32×32 *triangular* diagonal tile for every grid point
#         (lower-triangular for the forward sweep, upper-triangular for the backward sweep;
#         one warp per grid point, pivots broadcast by `@shfl`, same numerics as the
#         register-warp kernel);
#     (2) TRAILING UPDATE — a tiled GEMM that subtracts panel k's contribution from the
#         trailing rows. Each workgroup loads the A,B[row-tile, panel-k] tile into `@localmem`
#         ONCE and reuses it across `gt` grid points, turning the streaming into compute-bound
#         work. The pencil M = zB − A is kept as separate A,B tiles + a per-grid-point z combine
#         so the shared tiles are z-independent and reused across the batch.
#
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
# Note on the "dead" zero triangle of A/B: the blocking keeps every trailing-tile index in
# the FILLED part of the triangle by construction — for the forward sweep the trailing rows
# are i > koff+plen against panel columns j ≤ koff+plen, so i > j (filled subdiagonal); the
# backward sweep loads rows i ≤ koff against columns j > koff, so i < j (filled superdiagonal).
# The structurally-zero entries therefore exist as device storage but are NEVER loaded into
# sA/sB or multiplied — dead storage, not dead computation.
#
# Trailing-update workgroups use a flat 1-D group index decoded into (row-tile, grid-pt-tile)
# to avoid relying on KA 2-D group indexing. Numerics match the column-oriented / register-warp
# kernels (same `_pdiv`, `zBAij`, update order).

using KernelIntrinsics: @shfl, Idx

# ---- diagonal panel solves (one warp per grid point) ----

# Solve the lower-triangular diagonal tile of panel k for one grid point per workgroup.
# `koff` is the panel's column offset; `lane` is the 1-based KA local index, so `j0 = koff+lane`
# is the row this lane owns within the panel. `bl` holds that row's RHS entry in a register and
# is updated in place. Columns are solved in ascending order (lower-triangular forward substitution).
@kernel function _tiled_panel_forward(bv, zv, @Const(A), @Const(B), koff, plen)   # lower-triangular diagonal tile
    lane = @index(Local)
    gi = @index(Group)
    @uniform ET = eltype(A)
    b = bv[gi]
    z = zv[gi]
    j0 = koff + lane
    valid = lane <= plen                       # mask off lanes past a partial last panel (plen < 32)
    bl = valid ? b[j0] : zero(ET)
    for jj = 1:plen
        j = koff + jj
        # Only the owner lane (lane == jj) computes the real pivot x_j; every other lane emits
        # zero, so `@shfl(Idx, piv, jj)` broadcasts the single non-zero value from lane jj to all.
        piv = (lane == jj) ? _pdiv(bl, @inline zBAij(j, j, z, A, B)) : zero(ET)
        xj = _trsm_shfl(piv, jj)
        if lane == jj
            bl = xj                            # owner stores its solved value
        elseif (lane > jj) & valid
            bl -= xj * @inline zBAij(j0, j, z, A, B)   # rows BELOW the pivot get the lower-tri update
        end
    end
    valid && (b[j0] = bl)                       # `& valid` guard: never write back an out-of-range row
end

# Mirror of `_tiled_panel_forward` for the upper-triangular diagonal tile (backward substitution).
# Columns are solved in DESCENDING order, and each solved column updates the rows ABOVE it
# (`lane < jj`) — the dual of the forward kernel's `lane > jj`.
@kernel function _tiled_panel_backward(bv, zv, @Const(A), @Const(B), koff, plen)  # upper-triangular diagonal tile
    lane = @index(Local)
    gi = @index(Group)
    @uniform ET = eltype(A)
    b = bv[gi]
    z = zv[gi]
    j0 = koff + lane
    valid = lane <= plen                       # mask off lanes past a partial last panel
    bl = valid ? b[j0] : zero(ET)
    for jj = plen:-1:1
        j = koff + jj
        # Owner lane jj computes the pivot; others emit zero; the shuffle broadcasts from lane jj.
        piv = (lane == jj) ? _pdiv(bl, @inline zBAij(j, j, z, A, B)) : zero(ET)
        xj = _trsm_shfl(piv, jj)
        if lane == jj
            bl = xj
        elseif (lane < jj) & valid
            bl -= xj * @inline zBAij(j0, j, z, A, B)   # rows ABOVE the pivot get the upper-tri update
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

# ---- B = I trailing updates (one tile, no z) ----
# For a standard pencil (B = I) the trailing rows i are strictly off the panel columns j, so
# B[i,j] = 0 and the pencil reduces to M[i,j] = -A[i,j]: `acc -= M[i,j]·b[j]` ⇒ `acc += A[i,j]·b[j]`.
# Dropping the B tile halves shared memory (better occupancy; a wide element type's single 32-KB
# tile now fits 48 KB) and removes z from the trailing update.
#
# Why these are SEPARATE kernels rather than one kernel with a runtime `if eye` branch (or a
# `Val`-tagged body): `@localmem` is a *static* allocation in a GPU kernel, so a unified body
# that conditionally touches `sB` would still reserve both tiles regardless of the flag —
# defeating the entire point of the eye path (the halved shared-memory footprint). The B=I-vs-
# pencil choice is therefore made by dispatch at the host call site (see the `eye` branch in
# `_tiled_trsm!`, src/ihlpsa_trsm.jl), which picks the right kernel symbol; a `Val{eye}` tag would
# compile to these same two bodies and save no lines.

@kernel function _tiled_trailing_forward_eye(bv, @Const(A), koff, plen, rbase, m, gt, rtiles)
    t = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, 32)
    i = rbase + (bi - 1) * 32 + t
    for jj = 1:plen
        sA[t, jj] = i <= m ? A[i, koff + jj] : zero(ET)
    end
    @synchronize()
    if i <= m
        g = length(bv)
        gp0 = (bg - 1) * gt
        for gg = 1:gt
            gp = gp0 + gg
            if gp <= g
                b = bv[gp]
                acc = b[i]
                for jj = 1:plen
                    acc += sA[t, jj] * b[koff + jj]
                end
                b[i] = acc
            end
        end
    end
end

@kernel function _tiled_trailing_backward_eye(bv, @Const(A), koff, plen, gt, rtiles)
    t = @index(Local)
    grp = @index(Group)
    @uniform ET = eltype(A)
    bi = (grp - 1) % rtiles + 1
    bg = (grp - 1) ÷ rtiles + 1
    sA = @localmem ET (32, 32)
    i = (bi - 1) * 32 + t
    for jj = 1:plen
        sA[t, jj] = i <= koff ? A[i, koff + jj] : zero(ET)
    end
    @synchronize()
    if i <= koff
        g = length(bv)
        gp0 = (bg - 1) * gt
        for gg = 1:gt
            gp = gp0 + gg
            if gp <= g
                b = bv[gp]
                acc = b[i]
                for jj = 1:plen
                    acc += sA[t, jj] * b[koff + jj]
                end
                b[i] = acc
            end
        end
    end
end
