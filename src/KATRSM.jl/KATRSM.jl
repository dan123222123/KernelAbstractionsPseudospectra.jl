module KATRSM

using Preferences

# Complex division for the triangular/pencil solves. The default uses Base division
# (`x / d`), which widens `Complex{Float16/Float32}` through `double` — the most
# accurate option. FP64-less GPUs (most Intel iGPUs, Apple Metal) can't compile that
# `double`; on those, set the `pdiv_accurate` preference to false
# (`set_pdiv_accurate(false)`) so `_pdiv` keeps the divide in the input precision
# (`x·conj(d)/abs2(d)`) — a few eps less accurate on these well-conditioned
# Schur-triangular systems, but the only form FP64-less hardware can run. Float64 is
# unaffected. Baked at load time, so changing it recompiles the kernels (restart Julia).
const PDIV_ACCURATE = @load_preference("pdiv_accurate", true)

if PDIV_ACCURATE
    @inline _pdiv(x, d) = x / d
else
    @inline _pdiv(x::Complex{T}, d::Complex{T}) where {T<:Union{Float16,Float32}} = (x * conj(d)) / abs2(d)
    @inline _pdiv(x, d) = x / d
end

"""
    set_pdiv_accurate(flag::Bool)

Choose how the KATRSM triangular solves divide in Float16/Float32. `true` (default)
uses Base's division (widens through `double`; most accurate). `false` keeps the
divide in the input precision (`x·conj(d)/abs2(d)`) — slightly less accurate, but the
only form FP64-less GPUs (Intel iGPUs, Apple Metal) can compile. Writes the
`pdiv_accurate` preference to LocalPreferences.toml and triggers recompilation, so the
change takes effect after restarting Julia. (Float64 results are identical either way.)
"""
function set_pdiv_accurate(flag::Bool)
    @set_preferences!("pdiv_accurate" => flag)
    @info "KATRSM pdiv_accurate = $flag written to LocalPreferences.toml; restart Julia for it to take effect."
    return flag
end

include("trsm_wrappers.jl")
include("trsm_pencil_wrappers.jl")
include("trsm_warp_kernels.jl")
include("trsm_tiled_kernels.jl")

export _batched_backward_solve_pencil, _batched_column_oriented_backward_solve_pencil
export _batched_forward_solve_pencil, _batched_column_oriented_forward_solve_pencil
export set_pdiv_accurate
export _batched_warp_forward_solve_pencil, _batched_warp_backward_solve_pencil
export _tiled_panel_forward, _tiled_panel_backward
export _tiled_trailing_forward, _tiled_trailing_backward

end