using KernelAbstractions

# note, cpu kernels exhibit odd behavior when using synchronize inside of arbitrary control flows
# see https://github.com/JuliaGPU/KernelAbstractions.jl/issues/330 if unexplainable behavior happens

# KA kernels for forward and backward solves of a matrix pencil
# the matrix pencil is constructed "on the fly" to limit device memory usage as much as possible

include("trsm_pencil_core.jl")

## BATCHED ##

# "naive" trsm
# each pencil is solved at the WORKITEM level, with the number of workITEMS equal to the batch dimension
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
# "column-oriented" trsm
# when a diagonal element is solved for, all operations in the corresponding column are independent of one another and can be done in parallel by a workgroup
# if the max workgroup size is less than size(P,1), workitems cycle down/up the column to complete all independent operations
# each pencil is solved at the WORKGROUP level, with the number of workgroups equal to the batch dimension
# Generic (B≠I) and B=I "eye" forward/backward each share ONE body: `veye::Val` selects the pencil
# element via `_piv_elem`/`_offd_elem`. For the eye case those never read B, so the `_eye` wrappers
# take a SINGLE matrix and pass `B = nothing` — skipping the identity-B DRAM read (~half the matrix
# bandwidth for B=I). The eye result matches the generic to round-off (~1 ULP via FMA order, since
# the eye computes the shorter z−A / −A); the generic routes through `zBAij`, AST-identical to the
# former inline `(z*B)−A`, so it stays bitwise. The lane-0 pivot lives in @localmem and is broadcast
# by the @synchronize block barrier. Wrappers dispatched from `_column_trsm!` per `b_is_identity(P)`.
@inline function _col_fwd_body!(b, z, A, B, veye, BLKSIZE, m, i)
    sbj = @localmem eltype(A) 1
    for j = 1:1:m
        if i == 1
            sbj[1] = _pdiv(b[j], @inline _piv_elem(veye, j, z, A, B))
            b[j] = sbj[1]
        end
        @synchronize()
        I = j + i
        while I <= m
            b[I] -= sbj[1] * @inline _offd_elem(veye, I, j, z, A, B)
            I += BLKSIZE
        end
        @synchronize()
    end
end
@inline function _col_bwd_body!(b, z, A, B, veye, BLKSIZE, m, i)
    sbj = @localmem eltype(A) 1
    for j = m:-1:1
        if i == 1
            sbj[1] = _pdiv(b[j], @inline _piv_elem(veye, j, z, A, B))
            b[j] = sbj[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            b[I] -= sbj[1] * @inline _offd_elem(veye, I, j, z, A, B)
            I -= BLKSIZE
        end
        @synchronize()
    end
end
@kernel function _batched_column_oriented_forward_solve_pencil(bv, zv, @Const(A), @Const(B))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    i = @index(Local); gi = @index(Group)
    @inline _col_fwd_body!(bv[gi], zv[gi], A, B, Val(false), BLKSIZE, m, i)
end
@kernel function _batched_column_oriented_backward_solve_pencil(bv, zv, @Const(A), @Const(B))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    i = @index(Local); gi = @index(Group)
    @inline _col_bwd_body!(bv[gi], zv[gi], A, B, Val(false), BLKSIZE, m, i)
end
@kernel function _batched_column_oriented_forward_solve_eye(bv, zv, @Const(A))   # B = I: single matrix
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    i = @index(Local); gi = @index(Group)
    @inline _col_fwd_body!(bv[gi], zv[gi], A, nothing, Val(true), BLKSIZE, m, i)
end
@kernel function _batched_column_oriented_backward_solve_eye(bv, zv, @Const(A))  # B = I: single matrix
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    i = @index(Local); gi = @index(Group)
    @inline _col_bwd_body!(bv[gi], zv[gi], A, nothing, Val(true), BLKSIZE, m, i)
end

# put pencil versions of sm3 blkco kernels
# will end up using twice the shared memory to store A and B, and compute zB - A in shared memory
# computing this outside of the kernel might make sense to limit sm usage, but would likely need higher-order matrix pencils to see a performance improvement...

@kernel function _blkco_forward_solve_pencil(d, z, @Const(A), @Const(B))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(A) BLKSIZE
    sA = @localmem eltype(A) (BLKSIZE, BLKSIZE)
    sM = @localmem eltype(A) (BLKSIZE, BLKSIZE)

    i = @index(Local)

    sd[i] = d[i]
    for j = 1:MSIZE
        sA[i, j] = A[i, j]
        sM[i, j] = B[i, j]
    end
    @synchronize()
    for j = 1:MSIZE
        sM[i, j] = @inline zBAij(i, j, z, sA, sM)
    end
    @synchronize()
    for j = 1:MSIZE
        if i == j
            sd[i] = _pdiv(sd[i], sM[i, i])
        end
        @synchronize()
        if i > j && i <= size(A, 1)
            sd[i] -= sd[j] * sM[i, j]
        end
        @synchronize()
    end
    @synchronize()
    d[i] = sd[i]
end

@kernel function _blkco_backward_solve_pencil(d, z, @Const(A), @Const(B))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(A) BLKSIZE
    sA = @localmem eltype(A) (BLKSIZE, BLKSIZE)
    sM = @localmem eltype(A) (BLKSIZE, BLKSIZE)

    i = @index(Local)

    sd[i] = d[i]
    for j = MSIZE:-1:1
        sA[i, j] = A[i, j]
        sM[i, j] = B[i, j]
    end
    @synchronize()
    for j = MSIZE:-1:1
        sM[i, j] = @inline zBAij(i, j, z, sA, sM)
    end
    @synchronize()
    for j = MSIZE:-1:1
        if i == j
            sd[i] = _pdiv(sd[i], sM[i, i])
        end
        @synchronize()
        if i < j
            sd[i] -= sd[j] * sM[i, j]
        end
        @synchronize()
    end
    @synchronize()
    d[i] = sd[i]
end
