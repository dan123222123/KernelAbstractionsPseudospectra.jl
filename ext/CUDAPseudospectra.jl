module CUDAPseudospectra

using KAPseudospectra, CUDA, PrecompileTools
using ArraysOfArrays: flatview

const _pdiv = KAPseudospectra.KATRSM._pdiv
const zBAij = KAPseudospectra.KATRSM.zBAij
const FULLMASK = 0xffffffff % UInt32

# Hand-rolled CUDA warp-register pencil solves. Same algorithm as the portable KA +
# KernelIntrinsics kernels (one warp / grid point, RHS in registers, pivot broadcast by
# warp shuffle, no block barriers) but lowered straight through `@cuda` + `CUDA.shfl_sync`.
# Motivation: the KA+KI path hits a codegen regression at R=16 (m≈512) where it runs SLOWER
# than the baseline column kernel; the identical algorithm here does not (native register
# use grows smoothly 71→224 for R=4→32 with only 32 B spill). Bitwise-identical to the KA
# kernels — same `_pdiv`, same `zBAij`, same per-column / per-row update order.
# `b` is the m×g flatview; lane = threadIdx (1-based), gi = blockIdx (grid point).
@generated function _warp_cuda_fwd!(b, z, A, B, ::Val{R}) where {R}      # lower-tri, M = z̄B−A
    bl(s) = Symbol("bl_", s)
    q = Expr(:block, :(lane = Int(threadIdx().x)), :(gi = Int(blockIdx().x)),
             :(m = size(A, 1)), :(ws = 32), :(ET = eltype(b)), :(zz = conj(z[gi])))
    for r in 1:R
        push!(q.args, quote
            local ir = lane + $(r - 1) * ws
            $(bl(r)) = ir <= m ? b[ir, gi] : zero(ET)
        end)
    end
    for p in 1:R
        blp = bl(p)
        push!(q.args, quote
            for jj = 1:ws
                local j = $(p - 1) * ws + jj
                if j <= m
                    local piv = (lane == jj) ? _pdiv($blp, zBAij(j, j, zz, A, B)) : zero(ET)
                    local xj = CUDA.shfl_sync(FULLMASK, piv, jj)
                    if lane == jj
                        $blp = xj
                    elseif lane > jj
                        local i = $(p - 1) * ws + lane
                        i <= m && ($blp = $blp - xj * zBAij(i, j, zz, A, B))
                    end
                end
            end
        end)
        for qq in (p+1):R
            blq = bl(qq)
            push!(q.args, quote
                for jj = 1:ws
                    local j = $(p - 1) * ws + jj
                    if j <= m
                        local xj = CUDA.shfl_sync(FULLMASK, $blp, jj)
                        local i = $(qq - 1) * ws + lane
                        i <= m && ($blq = $blq - xj * zBAij(i, j, zz, A, B))
                    end
                end
            end)
        end
    end
    for r in 1:R
        push!(q.args, quote
            local ir = lane + $(r - 1) * ws
            ir <= m && (b[ir, gi] = $(bl(r)))
        end)
    end
    push!(q.args, :(return nothing))
    q
end
@generated function _warp_cuda_bwd!(b, z, A, B, ::Val{R}) where {R}      # upper-tri, M = zB−A
    bl(s) = Symbol("bl_", s)
    q = Expr(:block, :(lane = Int(threadIdx().x)), :(gi = Int(blockIdx().x)),
             :(m = size(A, 1)), :(ws = 32), :(ET = eltype(b)), :(zz = z[gi]))
    for r in 1:R
        push!(q.args, quote
            local ir = lane + $(r - 1) * ws
            $(bl(r)) = ir <= m ? b[ir, gi] : zero(ET)
        end)
    end
    for p in R:-1:1
        blp = bl(p)
        push!(q.args, quote
            for jj = ws:-1:1
                local j = $(p - 1) * ws + jj
                if j <= m
                    local piv = (lane == jj) ? _pdiv($blp, zBAij(j, j, zz, A, B)) : zero(ET)
                    local xj = CUDA.shfl_sync(FULLMASK, piv, jj)
                    if lane == jj
                        $blp = xj
                    elseif lane < jj
                        local i = $(p - 1) * ws + lane
                        i <= m && ($blp = $blp - xj * zBAij(i, j, zz, A, B))
                    end
                end
            end
        end)
        for qq in 1:(p-1)
            blq = bl(qq)
            push!(q.args, quote
                for jj = ws:-1:1
                    local j = $(p - 1) * ws + jj
                    if j <= m
                        local xj = CUDA.shfl_sync(FULLMASK, $blp, jj)
                        local i = $(qq - 1) * ws + lane
                        i <= m && ($blq = $blq - xj * zBAij(i, j, zz, A, B))
                    end
                end
            end)
        end
    end
    for r in 1:R
        push!(q.args, quote
            local ir = lane + $(r - 1) * ws
            ir <= m && (b[ir, gi] = $(bl(r)))
        end)
    end
    push!(q.args, :(return nothing))
    q
end

# CUDA override of the register-warp solve. OPT-IN via KAPSEUDO_CUDA_NATIVE=1 — the portable
# KA + KernelIntrinsics path is the default. The native kernels run the identical algorithm
# straight through `@cuda` + `CUDA.shfl_sync`, which avoids a KA+KI codegen regression at
# R=16 (m≈512). (Note: the "auto" strategy routes m≥512 to the tiled solve anyway, so this
# override mainly matters when "warp" is forced at large m.)
function KAPseudospectra._warp_trsm!(backend::CUDA.CUDABackend, bV, zv, P, wgs)
    if get(ENV, "KAPSEUDO_CUDA_NATIVE", "0") != "1"
        return KAPseudospectra._warp_trsm_ka!(backend, bV, zv, P, wgs)
    end
    g = length(zv)
    R = cld(size(P, 1), wgs)
    bm = flatview(bV)
    @cuda threads=32 blocks=g _warp_cuda_fwd!(bm, zv, P.Ac, P.Bc, Val(R))
    @cuda threads=32 blocks=g _warp_cuda_bwd!(bm, zv, P.A, P.B, Val(R))
end

# Device-interface overrides for the CUDA backend. Defined unconditionally (not under
# `if CUDA.functional()`) so precompile bakes them even when the worker can't probe
# the GPU; only the workload below needs the functional() guard.
KAPseudospectra.device(B::CUDA.CUDABackend) = CUDA.device()
KAPseudospectra.device!(B::CUDA.CUDABackend, dev) = CUDA.device!(dev)
KAPseudospectra.devices(B::CUDA.CUDABackend) = CUDA.devices()
KAPseudospectra.get_bgarray(B::CUDA.CUDABackend) = CUDA.CuArray
KAPseudospectra.device_bytes_available(B::CUDA.CUDABackend) = CUDA.free_memory()
KAPseudospectra.device_reclaim(B::CUDA.CUDABackend) = CUDA.reclaim()

## precompile gpu code (only when a device is actually usable)
if CUDA.functional()
    @setup_workload begin
        @compile_workload begin
            # Precompile via the shuffle-free "column" solve. The warp/tiled kernels are
            # JIT-compiled on first use instead: CUDA kernel PTX does not survive the
            # precompile→runtime process boundary anyway (only host-side method instances
            # are cached, which the column path exercises identically), and executing the
            # warp/tiled shuffle kernels inside the headless precompile worker is where GPU
            # execution is least reliable.
            withenv("KAPSEUDO_TRSM" => "column") do
                KAPseudospectra._precompile_ihlpsa(CUDABackend(), collect(CUDA.devices())[1],
                    [ComplexF32, ComplexF64])
            end
        end
    end
end

end
