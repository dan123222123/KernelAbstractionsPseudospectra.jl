using KernelAbstractions

# CPU backend kernels behave oddly with @synchronize inside arbitrary control flow —
# see https://github.com/JuliaGPU/KernelAbstractions.jl/issues/330 if something unexplainable happens.

# KA kernels for forward/backward solves of a matrix pencil, built on the fly (never materialized)
# to limit device memory.

include("trsm_pencil_core.jl")

## BATCHED ##

# One workitem per pencil (batch dimension = number of workitems).
@kernel function _batched_forward_solve_pencil(bV, zv, A, B)
    I = @index(Global, Linear)
    if I <= length(zv)
        b = bV[I]
        z = zv[I]
        @inline forward_solve_pencil!(b, z, A, B)
    end
end
@kernel function _batched_backward_solve_pencil(bV, zv, A, B)
    I = @index(Global, Linear)
    if I <= length(zv)
        b = bV[I]
        z = zv[I]
        @inline backward_solve_pencil!(b, z, A, B)
    end
end
# One workgroup per pencil (not one workitem): off-diagonal updates in a solved column are
# mutually independent, so the workgroup does them in parallel, cycling if smaller than m.
#
# Generic (B≠I) and eye (B=I) kernels share only the per-element formula, via
# `_piv_elem`/`_offd_elem` (`Val{false}` ⇒ `zBAij`; `Val{true}` ⇒ the B-free reduction). Eye
# kernels pass `B = nothing`.
#
# The loop body is duplicated in each @kernel rather than factored into a shared helper: KA's CPU
# backend can only split a kernel at @synchronize points LEXICALLY inside the @kernel body —
# hiding them in an inlined helper raises "@synchronize used outside kernel" on CPU. Dispatched
# from `_column_trsm!` per `b_is_identity(P)`.
@kernel function _batched_column_oriented_forward_solve_pencil(bv, zv, @Const(A), @Const(B))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local);
    gi = @index(Group)
    for j in 1:1:m
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], @inline _piv_elem(Val(false), j, zv[gi], A, B))
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j + i
        while I <= m
            bv[gi][I] -= sbj[1] * @inline _offd_elem(Val(false), I, j, zv[gi], A, B)
            I += BLKSIZE
        end
        @synchronize()
    end
end
@kernel function _batched_column_oriented_backward_solve_pencil(bv, zv, @Const(A), @Const(B))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local);
    gi = @index(Group)
    for j in m:-1:1
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], @inline _piv_elem(Val(false), j, zv[gi], A, B))
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            bv[gi][I] -= sbj[1] * @inline _offd_elem(Val(false), I, j, zv[gi], A, B)
            I -= BLKSIZE
        end
        @synchronize()
    end
end
@kernel function _batched_column_oriented_forward_solve_eye(bv, zv, @Const(A))   # B = I: single matrix
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local);
    gi = @index(Group)
    for j in 1:1:m
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], @inline _piv_elem(Val(true), j, zv[gi], A, nothing))
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j + i
        while I <= m
            bv[gi][I] -= sbj[1] * @inline _offd_elem(Val(true), I, j, zv[gi], A, nothing)
            I += BLKSIZE
        end
        @synchronize()
    end
end
@kernel function _batched_column_oriented_backward_solve_eye(bv, zv, @Const(A))  # B = I: single matrix
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local);
    gi = @index(Group)
    for j in m:-1:1
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], @inline _piv_elem(Val(true), j, zv[gi], A, nothing))
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            bv[gi][I] -= sbj[1] * @inline _offd_elem(Val(true), I, j, zv[gi], A, nothing)
            I -= BLKSIZE
        end
        @synchronize()
    end
end
