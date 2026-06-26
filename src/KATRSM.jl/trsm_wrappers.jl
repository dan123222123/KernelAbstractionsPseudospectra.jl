# note these wrappers are generally non-blocking
include("trsm_kernels.jl")

## naive ##
function forward_solve(b::AbstractVector{T}, M) where {T<:Number}
    m, n = size(M)
    @assert m == n
    for k = 1:m
        @assert !iszero(M[k, k])
    end
    x = deepcopy(b)
    forward_solve!(x, M)
    return x
end
function batched_back_solve!(bV, MV)
    backend = get_backend(bV.data)
    @assert get_backend(MV.data) == backend
    _batched_back_solve(backend)(bV, MV, ndrange=length(bV))
end
function batched_back_solve(bV, MV)
    @assert length(bV) == length(MV)
    # check that M ∈ MV is square and non-singular
    for M in MV
        m, n = size(M)
        @assert m == n
        all(x -> !iszero(x), diag(M))
    end
    # launch
    xV = deepcopy(bV)
    batched_back_solve!(xV, MV)
    return xV
end

## blocked, "column-oriented" ##

function blkco_backward_solve!(b, M; nblkcols=32, blkbsk=_blk_backward_solve_sm3v1)
    # sanity checks
    m, n = size(M)
    #@assert m == n
    #@assert length(b) == m
    backend = get_backend(b)
    #@assert backend == get_backend(M)
    # compile dynamic ndrange kernels
    blkbs = blkbsk(backend, nblkcols)
    # partition the diagonal indicies of M into <=stride batches
    diagonal_partitions = reverse.(Iterators.partition(m:-1:1, nblkcols))
    for cols in diagonal_partitions
        @views blkbs(b[cols], M[cols, cols], ndrange=length(cols))
        mvrows = 1:(cols[1]-1)
        if !isempty(mvrows)
            @views b[mvrows] .-= (M[mvrows, cols] * b[cols])
        end
    end
end
function blkco_forward_solve!(b, M; nblkcols=32, blkfsk=_blk_forward_solve_sm3)
    # sanity checks
    m, n = size(M)
    @assert m == n
    @assert length(b) == m
    backend = get_backend(b)
    @assert backend == get_backend(M)
    # compile dynamic ndrange kernels
    blkfs = blkfsk(backend, nblkcols)
    # partition the diagonal indicies of M into <=stride batches
    diagonal_partitions = Iterators.partition(1:m, nblkcols)
    for cols in diagonal_partitions
        @views blkfs(b[cols], M[cols, cols], ndrange=length(cols))
        mvrows = (cols[end]+1):m
        if !isempty(mvrows)
            @views b[mvrows] .-= (M[mvrows, cols] * b[cols])
        end
    end
end
