# F64 accuracy of adaptive ihlpsa vs a dense-SVD oracle (Grcar).
#
# Oracle: at each grid point z, σ_min(zI − A) computed by a dense LAPACK SVD in
# Float64 (the ground truth). Compared against:
#   - adaptive ihlpsa (F64, default rtol=1e-6)  -> stops at the stopping tol
#   - fixed deep-nit ihlpsa (F64)               -> a second Lanczos reference
#
# F64 on a 1080 Ti is ~1/32 rate, so small m and grid. Errors are stratified by σ
# magnitude: DESIGN.md predicts contour-relevant points (σ ≳ 1e-8) land ~1e-8…1e-7,
# with relative accuracy inherently decaying as σ → 0 (absolute accuracy ~eps·‖A‖).
#
# Usage:  unset LD_LIBRARY_PATH; JULIA_NUM_THREADS=auto \
#           julia --project=test bench/f64_accuracy_vs_oracle.jl

using KAPseudospectra
using CUDA
using LinearAlgebra, Printf, Statistics, Dates

const RESULTS = joinpath(@__DIR__, "results_adaptive")
mkpath(RESULTS)
const T      = ComplexF64
const REGION = ((-1, 3), (-3, 3))
const G      = 90                       # 90x90 = 8100 points (90/6 = 15)

logio = open(joinpath(RESULTS, "f64_accuracy_log.txt"), "w")
clock() = Dates.format(Dates.now(), "HH:MM:SS")
logln(args...) = (s = string(args...); println(s); println(logio, s); flush(logio))

function grcar(::Type{S}, m, k=3) where {S}
    A = zeros(S, m, m)
    for i in 1:m
        A[i, i] = one(S)
        for j in 1:k; i+j <= m && (A[i, i+j] = one(S)); end
        i+1 <= m && (A[i+1, i] = -one(S))
    end
    A
end

# dense oracle: σ_min(zI − A) over the grid, aligned with zg[i,j]
function oracle_grid(A, zg)
    m = size(A, 1)
    Im = Matrix{eltype(A)}(I, m, m)
    σ = similar(zg, real(eltype(A)))
    Threads.@threads for idx in CartesianIndices(zg)
        σ[idx] = minimum(svdvals(zg[idx] * Im - A))
    end
    σ
end

# error stats restricted to a boolean mask
function stats(adp, orc, mask)
    a = adp[mask]; o = orc[mask]
    rel = abs.(a .- o) ./ abs.(o)
    abserr = abs.(a .- o)
    (; n=count(mask), relmax=maximum(rel), relmed=median(rel),
       absmax=maximum(abserr), absmed=median(abserr))
end

logln("="^78)
logln("F64 adaptive vs dense-SVD oracle (Grcar)   ", Dates.now())
logln("backend=CUDABackend  T=", T, "  grid=", G, "x", G, " (", G*G, " pts)  region=", REGION)
logln("="^78)

_, _, zg = qgrid(T, REGION[1], REGION[2], (G, G))

acc_csv = open(joinpath(RESULTS, "f64_accuracy.csv"), "w")
println(acc_csv, "m,method,rtol_or_nit,subset,npts,rel_max,rel_median,abs_max,abs_median")

for m in (64, 128)
    A = grcar(T, m)
    P = MatrixPencil(A)
    nit_deep = 8 * ceil(Int, log2(m))

    logln("\n## m=", m, "   (nit_deep=", nit_deep, ")")
    logln("[", clock(), "] dense oracle (", G*G, " SVDs of ", m, "x", m, ")...")
    torc = @elapsed orc = oracle_grid(A, zg)
    logln("[", clock(), "] oracle done (", @sprintf("%.1f s", torc), ")  σ range = [",
          @sprintf("%.2e", minimum(orc)), ", ", @sprintf("%.2e", maximum(orc)), "]")

    # warmup both GPU paths at this m
    ihlpsa(CUDABackend(), zg, P, 4)
    ihlpsa(CUDABackend(), zg, P; nit_max=nit_deep)

    # internal driver returns (σ, nit_used); public ihlpsa returns only σ
    tadp = @elapsed ((σ_adp_t, nused) = KAPseudospectra._ihlpsa_adaptive(CUDABackend(), zg, P))  # adaptive, default rtol=1e-6
    tfix = @elapsed (σ_fix_t = ihlpsa(CUDABackend(), zg, P, nit_deep))    # deep fixed reference
    σ_adp = permutedims(σ_adp_t)    # back to zg[i,j] alignment (ihlpsa returns permutedims internally)
    σ_fix = permutedims(σ_fix_t)
    logln(@sprintf("[%s] adaptive %.2fs (nit_used=%d, rtol=1e-6)   fixed(nit=%d) %.2fs", clock(), tadp, nused, nit_deep, tfix))

    # masks by σ magnitude (contour-relevant vs deep tail)
    allm   = trues(size(orc))
    contour = orc .>= 1e-8
    deep    = orc .< 1e-8

    for (label, method, knob, mask) in (
            ("all",     "adaptive", "rtol=1e-6", allm),
            ("contour", "adaptive", "rtol=1e-6", contour),
            ("tail",    "adaptive", "rtol=1e-6", deep),
            ("all",     "fixed",    "nit=$nit_deep", allm),
            ("contour", "fixed",    "nit=$nit_deep", contour))
        any(mask) || continue
        adp = method == "adaptive" ? σ_adp : σ_fix
        s = stats(adp, orc, mask)
        logln(@sprintf("    %-8s %-9s n=%-5d  rel: max %.2e  med %.2e   abs: max %.2e  med %.2e",
              label, method, s.n, s.relmax, s.relmed, s.absmax, s.absmed))
        println(acc_csv, @sprintf("%d,%s,%s,%s,%d,%.6e,%.6e,%.6e,%.6e",
                m, method, knob, label, s.n, s.relmax, s.relmed, s.absmax, s.absmed))
        flush(acc_csv)
    end
end
close(acc_csv)
logln("\n", "="^78); logln("DONE. results in ", RESULTS); close(logio)
