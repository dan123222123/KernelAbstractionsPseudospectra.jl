module KATRSM

# Precision-preserving complex division for the triangular/pencil solves.
#
# Julia's Base widens `Complex{Float16}`/`Complex{Float32}` division to a wider
# float for accuracy (e.g. `ComplexF32 / ComplexF32` is evaluated in Float64),
# which emits `double` operations. GPUs without native FP64 — most Intel iGPUs —
# cannot compile those and fail with "unsupported use of double value". The naive
# form `x·conj(d)/abs2(d)` keeps everything in the input precision; for the
# well-conditioned (Schur-triangular) systems these solves target it is accurate
# to a small multiple of eps(T). Float64 keeps Base's robust division (it does not
# widen), so CPU/CUDA/AMDGPU F64 results are unchanged.
@inline _pdiv(x::Complex{T}, d::Complex{T}) where {T<:Union{Float16,Float32}} = (x * conj(d)) / abs2(d)
@inline _pdiv(x, d) = x / d

include("trsm_wrappers.jl")
include("trsm_pencil_wrappers.jl")

export _batched_backward_solve_pencil, _batched_column_oriented_backward_solve_pencil
export _batched_forward_solve_pencil, _batched_column_oriented_forward_solve_pencil

end