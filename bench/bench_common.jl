# Shared helpers for the KAPseudospectra benchmark scripts (bench_adaptive.jl,
# multigpu_bench.jl, nit_chunk_sweep.jl, and future benches). Pull it in with
#   include(joinpath(@__DIR__, "bench_common.jl"))
# It brings the common deps into scope and defines the logging / timing / device /
# test-matrix helpers each bench would otherwise re-implement.

using KAPseudospectra, KernelAbstractions
using LinearAlgebra, MatrixDepot, Printf, Dates

# Short HH:MM:SS stamp for progress lines.
clock() = Dates.format(Dates.now(), "HH:MM:SS")

# Open `path` and return `(logln, io)`: `logln(args...)` tees one line to stdout and
# the file. Close `io` when the bench finishes.
function bench_logger(path)
    io = open(path, "w")
    logln = (args...) -> (s = string(args...); println(s); println(io, s); flush(io); s)
    return logln, io
end

# best-of-`reps` wall-clock seconds for `f` (assumes its kernels are already warmed up).
function bestof(f; reps=2)
    best = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed f()
        best = min(best, t)
    end
    best
end

# Reclaim memory on EVERY device of `backend` so no config inherits another's
# allocator state (the drivers fan out across all devices, so reclaim all of them, not
# just the current one). Uses the package's backend-agnostic device interface; CPU has
# nothing per-device to reclaim.
function reclaim_all(backend)
    if KernelAbstractions.isgpu(backend)
        for d in KAPseudospectra.devices(backend)
            KAPseudospectra.device!(backend, d)
            KAPseudospectra.device_reclaim(backend)
        end
    else
        KAPseudospectra.device_reclaim(backend)
    end
    GC.gc(); GC.gc()
end

# Backend from `args[1]` (cuda|amdgpu|oneapi|metal|cpu), loading the matching GPU
# package on demand; defaults to `default` when no arg is given.
function select_backend(args=ARGS; default="cpu")
    which = isempty(args) ? default : lowercase(first(args))
    which == "cpu" && return CPU()
    if which == "cuda"
        @eval Main using CUDA
        return Base.invokelatest(() -> Main.CUDA.CUDABackend())
    elseif which == "amdgpu"
        @eval Main using AMDGPU
        return Base.invokelatest(() -> Main.AMDGPU.ROCBackend())
    elseif which == "oneapi"
        @eval Main using oneAPI
        return Base.invokelatest(() -> Main.oneAPI.oneAPIBackend())
    elseif which == "metal"
        @eval Main using Metal
        return Base.invokelatest(() -> Main.Metal.MetalBackend())
    end
    error("unknown backend $(which); use cuda|amdgpu|oneapi|metal|cpu")
end

# MatrixPencil builders from MatrixDepot test matrices (both factor through schur via
# the user-facing MatrixPencil(A) constructor).
grcar_pencil(T, m) = MatrixPencil(MatrixDepot.grcar(T, m))
golub_pencil(T, n) = MatrixPencil(MatrixDepot.golub(T, n))
