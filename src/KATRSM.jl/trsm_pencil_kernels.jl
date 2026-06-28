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
@kernel function _batched_column_oriented_forward_solve_pencil(bv, zv, @Const(A), @Const(B))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end

    sbj = @localmem eltype(A) 1

    i = @index(Local)
    gi = @index(Group)

    for j = 1:1:m
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], zv[gi] * B[j, j] - A[j, j])
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j + i
        while I <= m
            bv[gi][I] -= sbj[1] * (zv[gi] * B[I, j] - A[I, j])
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

    i = @index(Local)
    gi = @index(Group)

    for j = m:-1:1
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], zv[gi] * B[j, j] - A[j, j])
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            bv[gi][I] -= sbj[1] * (zv[gi] * B[I, j] - A[I, j])
            I -= BLKSIZE
        end
        @synchronize()
    end
end

# B = I ("eye") column-oriented solves. For a standard pencil M = zI − A the pivot is (z − A[j,j])
# and the off-diagonal update is (−A[i,j]): the z·B term vanishes (B[i,j]=0 off the diagonal, =1 on
# it). These skip the Bc/B read entirely — ~half the matrix-data DRAM traffic for B=I, the dominant
# bandwidth cost. Numerically equivalent to the generic kernels for B=I and backward-stable vs
# LAPACK; they match to round-off (~1 ULP, not bit-for-bit: computing the shorter z−A / −A
# expressions lets NVPTX contract FMAs differently). Dispatched from `_column_trsm!` when `b_is_identity(P)`.
@kernel function _batched_column_oriented_forward_solve_eye(bv, zv, @Const(A))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local)
    gi = @index(Group)
    for j = 1:1:m
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], zv[gi] - A[j, j])
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j + i
        while I <= m
            bv[gi][I] -= sbj[1] * (-A[I, j])
            I += BLKSIZE
        end
        @synchronize()
    end
end
@kernel function _batched_column_oriented_backward_solve_eye(bv, zv, @Const(A))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        m = size(A, 1)
    end
    sbj = @localmem eltype(A) 1
    i = @index(Local)
    gi = @index(Group)
    for j = m:-1:1
        if i == 1
            sbj[1] = _pdiv(bv[gi][j], zv[gi] - A[j, j])
            bv[gi][j] = sbj[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            bv[gi][I] -= sbj[1] * (-A[I, j])
            I -= BLKSIZE
        end
        @synchronize()
    end
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
