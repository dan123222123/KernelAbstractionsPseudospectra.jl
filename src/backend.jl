# Per-backend device interface: the hooks the GPU extensions (ext/*Pseudospectra.jl) specialize for
# their backend, with CPU/default implementations here. This is the single place the backend
# abstraction lives — array type, device handle, free memory, reclaim, FP64 capability, and the
# device-property queries the trsm solves route on (warp width, shared-memory budget, warp-shuffle
# safety). The trsm LOGIC that consumes these (default_wgs, tiled_tiles_fit, trsmIHL, the solve
# drivers) stays in ihlpsa_trsm.jl.

# ── device / array / memory operations (CPU defaults; each GPU extension overrides) ──
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

# ── device-property queries the trsm solves route on (defaults here; extensions query hardware) ──

# Warp/subgroup width (one shuffle domain). Sets the column solve's workgroup size (`default_wgs`);
# the tiled solve's panel width is a fixed 32 regardless. The default is 32, but this is a per-backend
# QUERY hook, not a baked constant: each GPU extension overrides it with the actual hardware value —
# `CUDA.warpsize` (32), `AMDGPU` warpSize (32 on RDNA, 64 on CDNA), 32 for Metal SIMD-groups, 32 for
# Intel under the SIMD32 pin (`set_intel_force_simd32!`). KernelAbstractions has no portable
# pre-launch query, so the value comes from each backend's API.
warp_width(backend) = 32

# Shared-memory bytes per workgroup available to the tiled solve's `@localmem` tiles. Default a
# conservative 48 KB (the static-shared-memory limit on Volta/Turing/early-Ampere); each GPU
# extension overrides it with the real device query so the tiled-vs-column routing uses the actual
# per-device budget instead of an assumed constant.
device_smem_bytes(backend) = 48 * 1024

# Shared-memory bytes per SM (multiprocessor) — used only to estimate the tiled trailing kernel's
# resident-blocks-per-SM when picking the trailing-tile width `TC` (`tiled_tc`). A conservative 64 KB
# default; each GPU extension overrides it with the real per-SM query. (Distinct from
# `device_smem_bytes`, which is the per-BLOCK limit.)
device_smem_per_sm(backend) = 64 * 1024

# Whether the tiled solve (which broadcasts pivots with warp shuffles) is correct under the `tiled`
# strategy on this backend. True for CUDA / AMDGPU / Metal: fixed warp/wavefront/SIMD width + a
# hardware shuffle. On oneAPI it needs BOTH the KernelIntrinsics oneAPI shuffle backend AND a pinned
# SIMD width, so that extension overrides this; without them `tiled` falls back to the shuffle-free
# `column` solve — correct, just not the fast path. Metal also gates it behind an opt-in preference.
# `wide` is true for non-IEEE element types (MultiFloats / BigFloat); the oneAPI override keeps those
# on `column`.
warp_trsm_safe(backend, wide) = true
