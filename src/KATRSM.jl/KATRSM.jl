module KATRSM

using Preferences

# Complex division for the triangular/pencil solves. Base widens `Complex{Float16/
# Float32}` division through `double` for accuracy, which FP64-less GPUs (most Intel
# iGPUs) can't compile. The default ("fast") `_pdiv` keeps the divide in the input
# precision (`x·conj(d)/abs2(d)`) — accurate to a few eps on these well-conditioned
# Schur-triangular systems, and the only form FP64-less devices can run. Set the
# `pdiv_accurate` preference (`set_pdiv_accurate(true)`) to use Base's widening
# division instead: more accurate, but needs FP64 (not for FP64-less iGPUs/Metal).
# Float64 is unaffected either way. The choice is baked at load time, so changing it
# recompiles the kernels (restart Julia).
const PDIV_ACCURATE = @load_preference("pdiv_accurate", false)

if PDIV_ACCURATE
    @inline _pdiv(x, d) = x / d
else
    @inline _pdiv(x::Complex{T}, d::Complex{T}) where {T<:Union{Float16,Float32}} = (x * conj(d)) / abs2(d)
    @inline _pdiv(x, d) = x / d
end

"""
    set_pdiv_accurate(flag::Bool)

Choose how the KATRSM triangular solves divide in Float16/Float32. `false` (default)
keeps the divide in the input precision (compiles on FP64-less GPUs); `true` uses
Base's widening division — more accurate, but requires FP64 (not for FP64-less Intel
iGPUs or Apple Metal). Writes the `pdiv_accurate` preference to LocalPreferences.toml
and triggers recompilation, so the change takes effect after restarting Julia.
(Float64 results are identical either way.)
"""
function set_pdiv_accurate(flag::Bool)
    @set_preferences!("pdiv_accurate" => flag)
    @info "KATRSM pdiv_accurate = $flag written to LocalPreferences.toml; restart Julia for it to take effect."
    return flag
end

include("trsm_wrappers.jl")
include("trsm_pencil_wrappers.jl")

export _batched_backward_solve_pencil, _batched_column_oriented_backward_solve_pencil
export _batched_forward_solve_pencil, _batched_column_oriented_forward_solve_pencil
export set_pdiv_accurate

end