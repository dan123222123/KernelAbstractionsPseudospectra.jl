include("trsm_pencil_kernels.jl")
using KernelAbstractions

function batched_forward_solve_pencil!(bV, zv, A, B)
    backend = get_backend(A)
    _batched_forward_solve_pencil(backend, 1)(bV, zv, A, B, ndrange = length(zv))
end

function batched_forward_solve_pencil(bV, zv, A, B)
    @assert length(bV) == length(zv)
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    xV = deepcopy(bV)
    batched_forward_solve_pencil!(xV, zv, A, B)
    synchronize(get_backend(A))
    return Vector.(xV)
end

function batched_backward_solve_pencil!(bV, zv, A, B)
    backend = get_backend(A)
    _batched_backward_solve_pencil(backend, 1)(bV, zv, A, B, ndrange = length(zv))
end

function batched_backward_solve_pencil(bV, zv, A, B)
    @assert length(bV) == length(zv)
    @assert size(A) == size(B)
    m, n = size(A)
    @assert m == n
    xV = deepcopy(bV)
    batched_backward_solve_pencil!(xV, zv, A, B)
    synchronize(get_backend(A))
    return Vector.(xV)
end
