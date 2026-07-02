# Per-backend device interface: the hooks the GPU extensions (ext/*Pseudospectra.jl) specialize,
# with CPU/default implementations here — array type, device handle, memory, FP64 capability, and the
# device-property queries the trsm solves route on. The trsm logic that consumes these lives in
# ihlpsa_trsm.jl. Per-backend values and the routing they drive: see the package documentation.

# ── device / array / memory operations (CPU defaults; each GPU extension overrides) ──
get_bgarray(B::CPU) = Array
device(B::CPU) = CPU()
devices(B::CPU) = CPU()
device!(B::CPU, dev) = CPU()
device_bytes_available(B::CPU) = (Sys.free_memory() |> Int)
device_reclaim(B::CPU) = GC.gc()

# Whether the device can run Float64/ComplexF64 kernels; F64 paths skip devices that can't. Defaults
# to KernelAbstractions.supports_float64; overridable (the oneAPI extension substitutes its own query).
supports_fp64(B) = KernelAbstractions.supports_float64(B)

# ── device-property queries the trsm solves route on (defaults here; extensions query hardware) ──

# Warp/subgroup width — sets the column solve's workgroup size. Extensions override with the hardware value.
warp_width(backend) = 32

# Shared memory per workgroup for the tiled solve's `@localmem` tiles. Extensions override with the device query.
device_smem_bytes(backend) = 48 * 1024

# Shared memory per SM — used to size the trailing-tile width (`tiled_tc`). Extensions override with the device query.
device_smem_per_sm(backend) = 64 * 1024

# Whether the tiled solve's warp-shuffle pivot broadcast is correct on this backend+type (`wide` is
# true for non-IEEE element types); `tiled` self-gates to `column` when false.
warp_trsm_safe(backend, wide) = true

# Whether tiled-gemm's `mul!` trailing update hits a fast vendor complex GEMM for `T`; tiled-gemm
# self-gates to the regular tiled kernel when false.
tiled_gemm_safe(backend, ::Type{T}) where {T} = real(T) <: Base.IEEEFloat
