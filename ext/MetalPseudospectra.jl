# TODO: Metal extension is INCOMPLETE and currently UNREGISTERED in Project.toml.
#
# To enable the Metal backend, this file needs to dispatch the full set of
# device-interface methods that ihlpsa expects (see ext/AMDGPUPseudospectra.jl
# or ext/CUDAPseudospectra.jl for the reference shape):
#
#   KAPseudospectra.device(::Metal.MtlBackend)
#   KAPseudospectra.device!(::Metal.MtlBackend, dev)
#   KAPseudospectra.devices(::Metal.MtlBackend)            ← present below
#   KAPseudospectra.get_bgarray(::Metal.MtlBackend)        ← present below
#   KAPseudospectra.device_bytes_available(::Metal.MtlBackend)
#   KAPseudospectra.device_reclaim(::Metal.MtlBackend)
#
# Only the last two require non-obvious Metal.jl API calls (recommended
# working set size; manual GC.gc() since Metal has no explicit reclaim).
#
# Until those are added (and tested on Apple silicon), this file is dead and
# Project.toml does NOT register it as an extension. Re-add the
#   Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
# line under [weakdeps] and
#   MetalPseudospectra = "Metal"
# under [extensions] when the dispatch is complete.

module MetalPseudospectra

using KAPseudospectra, Sys, Metal

if Sys.isapple()
    @eval KAPseudospectra.devices(B::Metal.MtlBackend) = Metal.devices()
    @eval KAPseudospectra.get_bgarray(B::Metal.MtlBackend) = Metal.MtlArray
end

end
