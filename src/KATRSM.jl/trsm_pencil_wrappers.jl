include("trsm_pencil_kernels.jl")
using KernelAbstractions

function batched_forward_solve_pencil!(bV, zv, A, B)
    backend = get_backend(A)
    _batched_forward_solve_pencil(backend, 1)(bV, zv, A, B, ndrange=length(zv))
end

function batched_forward_solve_pencil(bV, zv, A, B)
    @assert length(bV) == length(zv)
    # check that A,B are the same dimension and square
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    # launch
    xV = deepcopy(bV)
    batched_forward_solve_pencil!(xV, zv, A, B)
    synchronize(get_backend(A))
    return Vector.(xV)
end

function batched_backward_solve_pencil!(bV, zv, A, B)
    backend = get_backend(A)
    _batched_backward_solve_pencil(backend, 1)(bV, zv, A, B, ndrange=length(zv))
end

function batched_backward_solve_pencil(bV, zv, A, B)
    @assert length(bV) == length(zv)
    # check that A,B are the same dimension and square
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    # launch
    xV = deepcopy(bV)
    batched_backward_solve_pencil!(xV, zv, A, B)
    synchronize(get_backend(A))
    return Vector.(xV)
end

function batched_column_oriented_forward_solve_pencil!(bv, zv, A, B, wgs=64)
    backend = get_backend(A)
    g = length(zv)
    _batched_column_oriented_forward_solve_pencil(backend, wgs)(bv, zv, A, B, ndrange=(wgs, g))
end

function batched_column_oriented_forward_solve_pencil(bv, zv, A, B, wgs=64)
    backend = get_backend(A)
    @assert get_backend(bv.data) == backend
    @assert get_backend(zv) == backend
    @assert get_backend(B) == backend
    xv = deepcopy(bv)
    batched_column_oriented_forward_solve_pencil!(xv, zv, A, B, wgs)
    return xv
end

function blkco_backward_solve_pencil!(b, z, A, B; nblkcols=16, blkbsk=_blkco_backward_solve_pencil)
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    @assert length(b) == m
    backend = get_backend(b)
    @assert backend == get_backend(A)
    @assert backend == get_backend(B)
    blkbs = blkbsk(backend, nblkcols)
    diagonal_partitions = reverse.(Iterators.partition(m:-1:1, nblkcols))
    for cols in diagonal_partitions
        @views blkbs(b[cols], z, A[cols, cols], B[cols, cols], ndrange=length(cols))
        mvrows = 1:(cols[1]-1)
        if !isempty(mvrows)
            # b_cols is materialized (not @views) so the GEMV rhs is a
            # contiguous device vector — AMDGPU's complex `gemv!` has no
            # method for SubArray-of-ROCArray rhs.
            b_cols = b[cols]
            @views b[mvrows] .-= ((z .* B[mvrows, cols] .- A[mvrows, cols]) * b_cols)
        end
    end
end
function blkco_forward_solve_pencil!(b, z, A, B; nblkcols=16, blkfsk=_blkco_forward_solve_pencil)
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    @assert length(b) == m
    backend = get_backend(b)
    @assert backend == get_backend(A)
    @assert backend == get_backend(B)
    blkfs = blkfsk(backend, nblkcols)
    diagonal_partitions = Iterators.partition(1:m, nblkcols)
    for cols in diagonal_partitions
        @views blkfs(b[cols], z, A[cols, cols], B[cols, cols], ndrange=length(cols))
        mvrows = (cols[end]+1):m
        if !isempty(mvrows)
            b_cols = b[cols]
            @views b[mvrows] .-= ((z * B[mvrows, cols] - A[mvrows, cols]) * b_cols)
        end
    end
end

function batched_blkco_backward_solve_pencil!(bv, zv, A, B; nblkcols=16, blkbsk=_blkco_backward_solve_pencil)
    @assert length(zv) == length(bv)
    for i in eachindex(bv)
        @views blkco_backward_solve_pencil!(bv[i], zv[i], A, B; nblkcols, blkbsk)
    end
end
function batched_blkco_forward_solve_pencil!(bv, zv, A, B; nblkcols=16, blkfsk=_blkco_forward_solve_pencil)
    @assert length(zv) == length(bv)
    for i in eachindex(bv)
        @views blkco_forward_solve_pencil!(bv[i], zv[i], A, B; nblkcols, blkfsk)
    end
end