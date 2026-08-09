module KATRSM

using Preferences

# Complex division for triangular/pencil solves. Default widens Complex{Float16/Float32}
# through Float64 for accuracy; FP64-less GPUs (Intel iGPUs, Apple Metal) can't compile that, so
# pdiv_accurate=false divides in the input precision instead (less accurate, but the only form
# such hardware can run). Float64 unaffected. Baked in at load time: needs a Julia restart.
const PDIV_ACCURATE = @load_preference("pdiv_accurate", true)

if PDIV_ACCURATE
    @inline _pdiv(x, d) = x / d
else
    @inline _pdiv(x::Complex{T}, d::Complex{T}) where {T <: Union{Float16, Float32}} = (x *
                                                                                        conj(d)) /
                                                                                       abs2(d)
    @inline _pdiv(x, d) = x / d
end

"""
    set_pdiv_accurate!(flag::Bool)

Choose how KATRSM's triangular solves divide in Float16/Float32. `true` (default) uses
Base division (most accurate). `false` divides in the input precision instead — the
only form FP64-less GPUs (Intel iGPUs, Apple Metal) can compile. Writes the
`pdiv_accurate` preference to LocalPreferences.toml; restart Julia for it to take
effect. Float64 results are identical either way.
"""
function set_pdiv_accurate!(flag::Bool)
    @set_preferences!("pdiv_accurate" => flag)
    @info "KATRSM pdiv_accurate = $flag written to LocalPreferences.toml; restart Julia for it to take effect."
    return flag
end

include("trsm_pencil_wrappers.jl")
include("trsm_tiled_kernels.jl")

# Underscore kernel entry points are internal; consumers import them explicitly
# (`using .KATRSM: _tiled_panel_forward, ...`) rather than via a blanket export.
export set_pdiv_accurate!

end
