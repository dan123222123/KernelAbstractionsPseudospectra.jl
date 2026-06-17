# Warp-cooperative, register-blocked batched pencil triangular solves.
#
# One warp (ws = 32 lanes) solves one grid point. The RHS `b` lives entirely in
# registers: lane ℓ (1-based) owns rows ℓ, ℓ+ws, ℓ+2ws, … → R = cld(m, ws) register
# slots `bl_1 … bl_R`. The pivot for each column is broadcast lane→lane with a warp
# shuffle (`@shfl(Idx, …)`), which also synchronizes the warp — so there are NO block
# barriers and `b` never round-trips through global memory (vs the column-oriented
# kernels, which keep `b` in global and do ~2m `@synchronize()` per solve).
#
# R must be a compile-time constant for the register array, so each solve is a
# `@generated` device function specialized on `Val{R}`: the generator emits the fully
# unrolled panel loops with the register variables `bl_p` named by literal slot index.
# m (= matrix size) is fixed for an entire ihlpsa run, so this specializes once per
# problem — it does NOT recompile on the per-iteration survivor count g (that stays the
# dynamic ndrange; see commit 32fb8df).
#
# Numerics match the column-oriented reference exactly: same pencil M = zB − A built on
# the fly via `zBAij`, same `_pdiv`, same per-column / per-row update order, so results
# are intended to be bitwise-identical to `_batched_column_oriented_*_solve_pencil`.

using KernelIntrinsics: @shfl, Idx

_blsym(s) = Symbol("bl_", s)

# Forward (lower-triangular): panels ascend p = 1…R; within a panel columns ascend; each
# solved column updates rows BELOW it (same-panel lanes > jj, and panels q > p).
@inline @generated function _warp_reg_forward_solve_pencil!(bg, z, A, B, lane, ws, ::Val{R}) where {R}
    body = Expr(:block, :(m = size(A, 1)), :(ET = eltype(bg)))
    # load owned rows into registers bl_1 … bl_R (out-of-range slots padded with 0)
    for r in 1:R
        bl = _blsym(r)
        push!(body.args, quote
            local ir = lane + $(r - 1) * ws
            $bl = ir <= m ? bg[ir] : zero(ET)
        end)
    end
    for p in 1:R
        blp = _blsym(p)
        # diagonal 32×32 block solve (columns j = (p-1)ws + jj, ascending)
        push!(body.args, quote
            for jj = 1:ws
                local j = $(p - 1) * ws + jj
                if j <= m                                   # warp-uniform → shuffle safe
                    local piv = (lane == jj) ? _pdiv($blp, @inline zBAij(j, j, z, A, B)) : zero(ET)
                    local xj = @shfl(Idx, piv, jj)          # broadcast pivot from lane jj
                    if lane == jj
                        $blp = xj
                    elseif lane > jj
                        local i = $(p - 1) * ws + lane
                        if i <= m
                            $blp = $blp - xj * @inline zBAij(i, j, z, A, B)
                        end
                    end
                end
            end
        end)
        # off-diagonal: panel p's solved columns update rows in panels q > p
        for q in (p+1):R
            blq = _blsym(q)
            push!(body.args, quote
                for jj = 1:ws
                    local j = $(p - 1) * ws + jj
                    if j <= m
                        local xj = @shfl(Idx, $blp, jj)
                        local i = $(q - 1) * ws + lane
                        if i <= m
                            $blq = $blq - xj * @inline zBAij(i, j, z, A, B)
                        end
                    end
                end
            end)
        end
    end
    # write registers back to global once
    for r in 1:R
        bl = _blsym(r)
        push!(body.args, quote
            local ir = lane + $(r - 1) * ws
            if ir <= m
                bg[ir] = $bl
            end
        end)
    end
    push!(body.args, :(return nothing))
    body
end

# Backward (upper-triangular): mirror — panels descend p = R…1; within a panel columns
# descend; each solved column updates rows ABOVE it (same-panel lanes < jj, panels q < p).
@inline @generated function _warp_reg_backward_solve_pencil!(bg, z, A, B, lane, ws, ::Val{R}) where {R}
    body = Expr(:block, :(m = size(A, 1)), :(ET = eltype(bg)))
    for r in 1:R
        bl = _blsym(r)
        push!(body.args, quote
            local ir = lane + $(r - 1) * ws
            $bl = ir <= m ? bg[ir] : zero(ET)
        end)
    end
    for p in R:-1:1
        blp = _blsym(p)
        # diagonal block solve (columns descending)
        push!(body.args, quote
            for jj = ws:-1:1
                local j = $(p - 1) * ws + jj
                if j <= m
                    local piv = (lane == jj) ? _pdiv($blp, @inline zBAij(j, j, z, A, B)) : zero(ET)
                    local xj = @shfl(Idx, piv, jj)
                    if lane == jj
                        $blp = xj
                    elseif lane < jj
                        local i = $(p - 1) * ws + lane
                        if i <= m
                            $blp = $blp - xj * @inline zBAij(i, j, z, A, B)
                        end
                    end
                end
            end
        end)
        # off-diagonal: panel p's solved columns update rows in panels q < p (all above)
        for q in 1:(p-1)
            blq = _blsym(q)
            push!(body.args, quote
                for jj = ws:-1:1
                    local j = $(p - 1) * ws + jj
                    if j <= m
                        local xj = @shfl(Idx, $blp, jj)
                        local i = $(q - 1) * ws + lane
                        if i <= m
                            $blq = $blq - xj * @inline zBAij(i, j, z, A, B)
                        end
                    end
                end
            end)
        end
    end
    for r in 1:R
        bl = _blsym(r)
        push!(body.args, quote
            local ir = lane + $(r - 1) * ws
            if ir <= m
                bg[ir] = $bl
            end
        end)
    end
    push!(body.args, :(return nothing))
    body
end

# KA kernel wrappers: one workgroup (= one warp) per grid point, ndrange = (ws, g).
@kernel function _batched_warp_forward_solve_pencil(bv, zv, @Const(A), @Const(B), ::Val{R}) where {R}
    @uniform ws = @groupsize()[1]
    lane = @index(Local)
    gi = @index(Group)
    @inline _warp_reg_forward_solve_pencil!(bv[gi], zv[gi], A, B, lane, ws, Val(R))
end
@kernel function _batched_warp_backward_solve_pencil(bv, zv, @Const(A), @Const(B), ::Val{R}) where {R}
    @uniform ws = @groupsize()[1]
    lane = @index(Local)
    gi = @index(Group)
    @inline _warp_reg_backward_solve_pencil!(bv[gi], zv[gi], A, B, lane, ws, Val(R))
end
