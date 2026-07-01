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
