using KernelAbstractions

# KA kernels for forward and backward solves of a square matrix

## naive ##

include("trsm_core.jl")

@kernel function _batched_backward_solve(bV, MV)
    I = @index(Global, Linear)
    if I <= length(bV)
        M = MV[I]
        b = bV[I]
        @inline backward_solve!(b, M)
    end
end

@kernel function _batched_forward_solve(bV, MV)
    I = @index(Global, Linear)
    if I <= length(bV)
        M = MV[I]
        b = bV[I]
        @inline forward_solve!(b, M)
    end
end

## "column-oriented" ##

@kernel function _column_oriented_backward_solve_sm1(b, @Const(M))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end
    sd = @localmem eltype(M) (1,)
    i = @index(Local)
    for j = MSIZE:-1:1
        if i == 1
            sd[1] = _pdiv(b[j], M[j, j])
            b[j] = sd[1]
        end
        @synchronize()
        I = j - i
        while I >= 1
            b[I] -= sd[1] * M[I, j]
            I -= BLKSIZE
        end
        @synchronize()
    end
end

## blocked, "column-oriented" ##

# see Hogg2013 for some ideas about the implementation, but note this is NOT an exact reimplementation
# "sm*" denotes levels of shared memory usage in the kernels -- 1 being the most "gloabl" and 3 being the most "local"
# v1 and v2 denote different ways of organizing the matrix in shared memory -- to evaluate which approach yields coallesed acceses

# for all blkco kernels we assume that:
# 1) M is square/upper triangular;
# 2) size(M,1) is <= the block size

### backward solve

# both matrix and rhs go into shared memory
# note, on NVIDIA A100 max 32/64 bit complex matrices limit workgroup size to 64/32 bits
@kernel function _blkco_backward_solve_sm3v1(d, @Const(M))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(M) BLKSIZE
    sM = @localmem eltype(M) (BLKSIZE, BLKSIZE)

    i = @index(Local)

    sd[i] = d[i]
    # load M into shared memory "backwards" to prevent cache misses later
    for j = MSIZE:-1:1
        sM[i, j] = M[i, j]
    end
    @synchronize()
    # solve M column-wise
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
@kernel function _blkco_backward_solve_sm3v2(d, @Const(M))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(M) BLKSIZE
    sM = @localmem eltype(M) (BLKSIZE, BLKSIZE)

    i = @index(Local)

    sd[i] = d[i]
    # load M into shared memory "backwards" to prevent cache misses later
    for j = MSIZE:-1:1
        sM[i, MSIZE-j+1] = M[i, j]
    end
    @synchronize()
    # solve M column-wise
    for j = MSIZE:-1:1
        if i == j
            sd[i] = _pdiv(sd[i], sM[i, MSIZE-j+1])
        end
        @synchronize()
        if i < j
            sd[i] -= sd[j] * sM[i, MSIZE-j+1]
        end
        @synchronize()
    end
    @synchronize()
    d[i] = sd[i]
end
# only rhs goes into shared memory
@kernel function _blkco_backward_solve_sm2(d, @Const(M))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(M) BLKSIZE

    i = @index(Local)

    sd[i] = d[i]
    @synchronize()
    # solve M column-wise
    for j = MSIZE:-1:1
        if i == j
            sd[i] = _pdiv(sd[i], M[i, i])
        end
        @synchronize()
        if i < j
            sd[i] -= sd[j] * M[i, j]
        end
        @synchronize()
    end
    @synchronize()
    d[i] = sd[i]
end
# only the current column pivot goes into shared memory
@kernel function _blkco_backward_solve_sm1(d, @Const(M))

    @uniform begin
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(M) (1,)

    i = @index(Local)

    # solve M column-wise
    for j = MSIZE:-1:1
        if i == j
            sd[1] = _pdiv(d[i], M[i, i])
            d[i] = sd[1]
        end
        @synchronize()
        if i < j
            d[i] -= sd[1] * M[i, j]
        end
        @synchronize()
    end
end

### forward solve

@kernel function _blkco_forward_solve_sm3(d, @Const(M))

    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = @ndrange()[1]
    end

    # shared memory
    sd = @localmem eltype(M) BLKSIZE
    sM = @localmem eltype(M) (BLKSIZE, BLKSIZE)

    i = @index(Local)

    sd[i] = d[i]
    for j = 1:MSIZE
        sM[i, j] = M[i, j]
    end
    @synchronize()
    for j = 1:MSIZE
        if i == j
            sd[i] = _pdiv(sd[i], sM[i, i])
        end
        @synchronize()
        if i > j && i <= size(M, 1)
            sd[i] -= sd[j] * sM[i, j]
        end
        @synchronize()
    end
    @synchronize()
    d[i] = sd[i]
end

## full hogg ##

# full d and M are given
# we first do a block diagonal solve and then re-use the same threads to do a matrix multiply
@kernel function _blkco_backward_solve(d, @Const(M))
    @uniform begin
        BLKSIZE = @groupsize()[1]
        MSIZE = size(M, 1) # assumed that M is square!
    end
    # diagonal and sub-matrix in shared memory
    sd = @localmem eltype(M) BLKSIZE
    sM = @localmem eltype(M) (BLKSIZE, BLKSIZE)

    # solve will be fully "block-local"
    i = @index(Local)
    ri = BLKSIZE - i + 1 # reversed thread indices for shared memory accesses

    if i <= MSIZE # only do work if within valid indicies of M

        # start at the bottom of d/M because this is backward solve
        # note that effective indices of threads are now reversed
        I = MSIZE - i + 1
        chunkstart = MSIZE - BLKSIZE
        chunkend = MSIZE

        #while I >= 1 # move effective thread indices "up" the matrix

        # first step -- block solve

        # load current diagonal and chunk of M into shared memory
        sd[ri] = d[I]
        for j = 1:(min(BLKSIZE, (chunkend - chunkstart)))
            sM[ri, j] = M[I, chunkstart+j]
        end
        @synchronize()
        #if i == 1 # check shared memory ordering
        #    @print("\n")
        #    for k = 1:BLKSIZE
        #        for l = 1:BLKSIZE
        #            @print("$(sM[k, l]) ")
        #        end
        #        #@print("$(sd[k])\n")
        #        @print("\n")
        #    end
        #end

        ## TODO ##

        ## solve M column-wise
        #for j = chunkend:-1:chunkstart
        #    if I == j
        #        sd[ri] /= sM[i, i]
        #    end
        #    @synchronize()
        #    if i < j
        #        sd[i] -= sd[j] * sM[i, j]
        #    end
        #    @synchronize()
        #end
        #@synchronize()
        #d[i] = sd[i]

        #end

    end

end