# General per-backend device interface: where arrays live, the device handle, free memory,
# reclaim, and the FP64-capability query. CPU defaults here; each GPU extension
# (ext/*Pseudospectra.jl) overrides these for its backend. This is the broad backend abstraction
# shared by the ihlpsa fixed/adaptive drivers and the bench scripts — it is NOT specific to the
# triangular solves, so it lives at module level rather than inside ihlpsa. (The trsm-specific
# device-routing hooks — warp_width, device_smem_bytes, warp_trsm_safe — stay with the trsm logic
# in ihlpsa_trsm.jl.)

# Device-operations interface: CPU defaults here; each GPU extension overrides these
# (array type, device handle access, free-memory query, reclaim).
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

# Whether `backend`'s device can run Float64/ComplexF64 kernels. Default defers to
# `KernelAbstractions.supports_float64`; overridable so the oneAPI extension can substitute a
# device-accurate FP64 query (see its note). F64 paths skip unsupported devices.
supports_fp64(B) = KernelAbstractions.supports_float64(B)
