# End-to-end time-to-field for the SAME σ deliverable, three ways:
#   gpu     — our converged adaptive ihlpsa (to the precision floor) on the GPU under test
#   cpu     — the same on the host CPU (one own-workspace solve per thread)
#   eigtool — the standard tool (Wright/Embree/Trefethen) via `matlab -batch` + eigtool_psa.m.
#             Probe-and-skip (no MATLAB/checkout, or non-grcar) ⇒ NaN, never fails.
# Fairness sentinels ride in the CSV: σ agreement GPU-vs-CPU and CPU-vs-EigTool on EigTool's
# own returned grid (it iterates to 1e-5 and may mirror-symmetrize, so agreement reads to
# ~5 digits). factor_s is separate because EigTool's timed call includes its own Schur
# reduction: compare factor_s + {cpu,gpu}_s against eigtool_s for totals.
#   julia --project=bench bench/bench_cpu_gpu.jl cuda      # (or amdgpu | oneapi)
include(joinpath(@__DIR__, "bench_common.jl"))
using DelimitedFiles

backend = select_backend(ARGS)
if !KernelAbstractions.isgpu(backend)
    @warn "bench_cpu_gpu compares a GPU backend against the CPU — nothing to do for $(backend)"
    exit(0)
end
devs = KernelAbstractionsPseudospectra.devices(backend)

# ComplexF64: EigTool is double-precision-only. FP64-less devices fall back to F32 (the
# EigTool leg self-skips there).
const T = KernelAbstractionsPseudospectra.supports_fp64(backend) ? ComplexF64 : ComplexF32
const MS = env_ints("BENCH_MS", (64, 128, 256, 512, 1024))
const EIGTOOL_MS = env_ints("BENCH_EIGTOOL_MS", (64, 128, 256, 512))  # MATLAB is the long pole
const REPS = env_int("BENCH_REPS", 3)
const REPS_CPU = env_int("BENCH_REPS_CPU", 2)
const RESULTS = results_dir()

# Size of the depth-grid + σ field dump (depthmap_*.csv). Capped below max(MS): the overlay is
# the BigFloat Schur diagonal, O(m³).
const DEPTHMAP_M = maximum(filter(<=(env_int("BENCH_DEPTHMAP_M", 256)), MS); init = minimum(MS))
const DEPTHMAP_TOK = T === ComplexF64 ? "f64" : "f32"

const MATLAB = Sys.which("matlab")
const EIGTOOL_DIR = get(ENV, "EIGTOOL_PATH", joinpath(@__DIR__, "eigtool"))
const EIGTOOL_OK = MATLAB !== nothing && isfile(joinpath(EIGTOOL_DIR, "eigtool.m")) &&
                   T === ComplexF64 && BENCH_MATRIX == "grcar"
EIGTOOL_OK ||
    @warn "EigTool leg skipped" matlab = MATLAB eigtool_dir = EIGTOOL_DIR eltype = T BENCH_MATRIX

# One headless EigTool run: returns (t_eigtool_s, npts, maxdiff vs our CPU σ) or NaNs on failure.
function eigtool_leg(m, gridn, box, P)
    pre = joinpath(RESULTS, "eigtool_m$(m)")
    ax = "[$(box[1][1]) $(box[1][2]) $(box[2][1]) $(box[2][2])]"
    script = "addpath('$(EIGTOOL_DIR)'); addpath('$(@__DIR__)'); " *
             "eigtool_psa($m, $gridn, $ax, '$pre')"
    try
        run(pipeline(`$MATLAB -batch $script`; stdout = devnull))
        meta = readdlm(pre * "_meta.csv", ',', Float64)
        x = vec(readdlm(pre * "_x.csv", ',', Float64))
        y = vec(readdlm(pre * "_y.csv", ',', Float64))
        sigs = readdlm(pre * "_sigs.csv", ',', Float64)
        t_et = meta[5]
        # Compare on EigTool's own returned grid: ihlpsa returns σ with reversed dims vs its
        # input grid, so a package-native zg (x down rows) already matches EigTool's
        # rows-follow-y layout.
        zg = T.(reshape(x, :, 1) .+ im .* reshape(y, 1, :))
        σ = converged_solve(CPU(), zg, P)
        (t_et, length(x) * length(y), maximum(abs.(Float64.(σ) .- sigs)))
    catch err
        @warn "EigTool run failed for m=$m — NaN columns" err
        (NaN, 0, NaN)
    end
end

foreach(println, repro_stamp(backend))
println("cpu = ", Sys.CPU_NAME, " (", Threads.nthreads(), " julia threads)   gpu = ",
    device_name(backend), "   eigtool = ", EIGTOOL_OK ? MATLAB : "skipped")
csv = open_csv(joinpath(RESULTS, "bench_cpu_gpu.csv"),
    "backend,eltype,m,gridn,grid,max_depth,cpu_threads,factor_s,cpu_s,gpu_s,speedup," *
    "eigtool_s,eigtool_grid,per_point_cpu_s,per_point_gpu_s,per_point_eigtool_s," *
    "maxdiff_gpu_cpu,maxdiff_eigtool_cpu")
@printf("%-6s %-6s %-9s %-10s %-10s %-9s %-11s %-10s\n",
    "m", "grid", "factor", "cpu(s)", "gpu(s)", "gpu/cpu", "eigtool(s)", "et-maxdiff")
for m in MS
    gridn = bench_gridn(backend)
    g = gridn * gridn
    nitmax = converged_nit_max(m)                        # adaptive cap; sizes the α/β workspace
    box = bench_box()
    A = bench_matrix(T, m)
    # Standard (B = I), not via bench_pencil: EigTool computes standard pseudospectra only, so a
    # generalized pencil here would have nothing to compare against. Exempt from BENCH_PENCIL.
    tfac = @elapsed P = MatrixPencil(schur(A))
    zg = bench_grid(T, gridn; box)
    zpd = pinned_zpd(backend, T, m, nitmax; ngrid = g, headroom = zpd_headroom())

    # Time-to-solution of the same converged deliverable on GPU and CPU; σg/σc share the
    # layout, so the sentinel differences directly.
    σg, nitg = KernelAbstractionsPseudospectra._ihlpsa_adaptive(backend, zg, P;              # warm-up + GPU σ + depth
        rtol = converged_rtol(T), nit_max = nitmax, devs, zpd)
    maxdepth = maximum(nitg)

    # σ field + retirement depths + eigenvalue overlay (depthmap_*.csv); both arrays are
    # already in hand from the warm-up solve above, so this costs one write.
    if m == DEPTHMAP_M
        writedlm(joinpath(RESULTS, "depthmap_sigma_$(DEPTHMAP_TOK)_m$(m).csv"),
            to64.(abs.(σg)), ',')
        writedlm(joinpath(RESULTS, "depthmap_nitgrid_$(DEPTHMAP_TOK)_m$(m).csv"), nitg, ',')
        # Float64 eigvals land visibly off the chevron for grcar, so take the (disk-cached)
        # BigFloat Schur diagonal.
        λ = diag(bigfloat_schur(bench_matrix(ComplexF64, m), m))
        writedlm(joinpath(RESULTS, "depthmap_eigvals_$(DEPTHMAP_TOK)_m$(m).csv"),
            [Float64.(real.(λ)) Float64.(imag.(λ))], ',')
    end
    tg = bestof(() -> converged_solve(backend, zg, P; devs, zpd), backend; reps = REPS)
    σc = converged_solve(CPU(), zg, P)                   # warm-up + CPU σ for the sentinel
    tc = bestof(() -> converged_solve(CPU(), zg, P), CPU(); reps = REPS_CPU)
    mdgc = Float64(maximum(abs.(σg .- σc)))

    t_et, g_et, md_et = (EIGTOOL_OK && m in EIGTOOL_MS) ? eigtool_leg(m, gridn, box, P) :
                        (NaN, 0, NaN)

    @printf("%-6d %-6d %-9.3f %-10.4f %-10.4f %-9s %-11s %-10s\n",
        m, gridn, tfac, tc, tg, @sprintf("%.2fx", tc / tg),
        isnan(t_et) ? "—" : @sprintf("%.2f", t_et),
        isnan(md_et) ? "—" : @sprintf("%.1e", md_et))
    println(csv, @sprintf("%s,%s,%d,%d,%d,%d,%d,%.4f,%.6f,%.6f,%.4f,%s,%d,%.6e,%.6e,%s,%.3e,%s",
        backend_tag(backend), T, m, gridn, g, maxdepth, Threads.nthreads(), tfac, tc, tg, tc / tg,
        isnan(t_et) ? "NaN" : @sprintf("%.4f", t_et), g_et, tc / g, tg / g,
        isnan(t_et) ? "NaN" : @sprintf("%.6e", t_et / g_et), mdgc,
        isnan(md_et) ? "NaN" : @sprintf("%.3e", md_et)))
    flush(csv)
end
close_csv(csv, joinpath(RESULTS, "bench_cpu_gpu.csv"))
