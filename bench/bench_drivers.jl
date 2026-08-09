# The driver study: fixed-`nit` ihlpsa vs the adaptive (per-point hybrid) driver,
# on f32 and f64, single device. The only experiment that RACES the two drivers against each
# other; the converged experiments reach the adaptive driver through `converged_solve`, and
# everything else holds default_nit(m) so iteration count is a controlled variable. Adaptive FoM
# uses Σ nit_grid, not g·nit (which would erase the feature). Two modes:
#   (default) — timing: cold first call (TTFP) + hot best-of per driver, one CSV row per
#               (eltype, m, driver).
#   --trace   — CUDA-only (CUDA.@profile): ranked host/device tables, a kernel-family time budget,
#               and per-launch solve durations. CUPTI crashes nondeterministically (run this leg
#               last) and profiler overhead pollutes wall-clock.
#   julia --project=bench bench/bench_drivers.jl cuda [--trace]
include(joinpath(@__DIR__, "bench_common.jl"))
using DelimitedFiles

backend = select_backend(filter(a -> !startswith(a, "--"), ARGS))
gpu = KernelAbstractions.isgpu(backend)
# Single-device showcase: restrict to the first device so this measures the per-device
# fixed-vs-adaptive win (multi-device scaling is bench_multigpu.jl).
devs = gpu ? [first(KAPseudospectra.devices(backend))] : missing

const MS = env_ints("BENCH_MS", (64, 128, 256, 512, 1024))
const REPS = env_int("BENCH_REPS", 3)
const TOKS = runnable_eltypes(backend, eltype_tokens("f32,f64"))
const RESULTS = results_dir()

# --trace: CUDA.@profile the same configs (kernel budget + retirement curve).
const SOLVE_RX = r"solve|tiled_panel|tiled_trailing"
const REC_RX = r"ihl_ttr_q_next|q_next|v2v"
const GATHER_RX = r"qv_gather"

# CUDA.jl renders the tables to the display width; the default 80 columns crops the kernel
# names, the one column the tables exist for.
wide(io) = IOContext(io, :displaysize => (1000, 240), :limit => false)

# @eval'd at runtime, not lowered at include time: a CPU run would die on the unconditional
# Main.CUDA reference just defining trace_mode (CUDA is in Main by now via select_backend).
profiled(f) = @eval Main CUDA.@profile $f()

# res.device is CUDA.jl's raw per-event trace (start, stop, name per launch/copy) — profiler
# internals, so the caller try/catches a schema move. idle = trace span − busy (host-logic time,
# since kernels don't overlap on one stream).
function summary_row(res, m, driver, nit, g, wall)
    dev = res.device
    names = string.(dev.name)
    d = Float64.(dev.stop .- dev.start)
    busy = sum(d)
    tscale = busy > 1e7 ? 1e-9 : 1.0             # normalize if the trace is in ns, not s
    d .*= tscale
    busy *= tscale
    trace = (maximum(dev.stop) - minimum(dev.start)) * tscale
    solve, rec = occursin.(SOLVE_RX, names), occursin.(REC_RX, names)
    gat, cop = occursin.(GATHER_RX, names), startswith.(names, "[")
    solve_s, rec_s, gat_s, cop_s = sum(d[solve]), sum(d[rec]), sum(d[gat]), sum(d[cop])
    nsol = count(solve)
    @sprintf("%d,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%d,%.3f,%.3f,%.3f",
        m, driver, nit, g, wall, trace, busy, trace - busy, solve_s, rec_s, gat_s, cop_s,
        busy - solve_s - rec_s - gat_s - cop_s, nsol,
        1e3 * solve_s / max(nsol, 1), nsol == 0 ? 0.0 : 1e3 * minimum(d[solve]),
        nsol == 0 ? 0.0 : 1e3 * maximum(d[solve]))
end

function launch_rows(io, res, m, driver)
    dev = res.device
    names = string.(dev.name)
    keep = occursin.(SOLVE_RX, names)
    any(keep) || return
    dur = Float64.(dev.stop .- dev.start)
    tscale = sum(dur) > 1e7 ? 1e-9 : 1.0
    t0 = minimum(dev.start)
    ks = findall(keep)
    ks = ks[sortperm(Float64.(dev.start[ks]))]
    for (i, j) in enumerate(ks)
        tag = occursin("forward", names[j]) ? "fwd" :
              occursin("backward", names[j]) ? "bwd" :
              occursin("panel", names[j]) ? "panel" : "trailing"
        println(io, @sprintf("%d,%s,%d,%s,%.6f,%.4f", m, driver, i, tag,
            (dev.start[j] - t0) * tscale, dur[j] * tscale * 1e3))
    end
end

function trace_mode()
    nameof(typeof(backend)) == :CUDABackend ||
        error("--trace is CUDA-only (CUDA.@profile); got $(backend). " *
              "Use bench_kernels.jl --counters with ncu/rocprof/VTune on other vendors.")
    T = ELTYPES[first(runnable_eltypes(backend, eltype_tokens(get(ENV, "BENCH_TRACE_ELTYPE", "f64"))))]
    # Small grid: the budget fractions and the retirement shape don't need the mid tier, and a
    # smaller trace dodges the CUPTI buffer-pool crash more often.
    gridn = env_int("BENCH_TRACE_GRIDN", 128)
    g = gridn * gridn
    foreach(println, repro_stamp(backend))
    sio = open_csv(joinpath(RESULTS, "bench_drivers_trace_summary.csv"),
        "m,driver,nit,gridpts,wall_s,trace_s,busy_s,idle_s,solve_s,recurrence_s," *
        "gather_s,copy_s,other_s,n_solve_launches,solve_avg_ms,solve_min_ms,solve_max_ms")
    lio = open_csv(joinpath(RESULTS, "bench_drivers_trace_launches.csv"),
        "m,driver,idx,kernel,t_start_s,dur_ms")
    for m in env_ints("BENCH_TRACE_MS", (256, 1024))
        (; P, zg, zpd) = bench_setup(backend, T, m; gridn)
        # Converged-fixed baseline: flat depth = the adaptive stop's deepest point, so the fixed
        # band and the adaptive retirement tail are compared at like accuracy.
        (_, ng0) = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; nit_max = converged_nit_max(m), devs, zpd)
        nit = maximum(ng0)
        runs = (("fixed", () -> ihlpsa(backend, zg, P, nit; devs, zpd)),
            ("adaptive", () -> ihlpsa(backend, zg, P; nit_max = converged_nit_max(m), devs, zpd)))
        foreach(r -> r[2](), runs)               # warm both drivers (compile + device init)
        for (driver, f) in runs
            reclaim_all(backend)
            t = @elapsed res = profiled(f)
            path = joinpath(RESULTS, "bench_drivers_trace_m$(m)_$(driver).txt")
            open(path, "w") do io
                foreach(ln -> println(io, ln), repro_stamp(backend))
                println(io, "m=$m  driver=$driver  nit=$nit  grid=$(gridn)×$(gridn)  T=$T  ",
                    "zpd=$zpd  wall_s=", @sprintf("%.3f", t), "  (wall includes profiler overhead)")
                println(io)
                show(wide(io), MIME"text/plain"(), res)
                println(io)
            end
            println("\n===== m=$m  $driver  (wall ", @sprintf("%.3f", t), " s) =====")
            show(wide(stdout), MIME"text/plain"(), res)
            println("\nwrote ", path)
            try
                println(sio, summary_row(res, m, driver, nit, g, t))
                flush(sio)
                launch_rows(lio, res, m, driver)
                flush(lio)
            catch err
                @warn "raw-trace summary failed (profiler schema change?) — tables still written" err
            end
        end
    end
    close(sio)
    close(lio)
end

# default: the fixed-vs-adaptive timing race.
function timing_mode()
    foreach(println, repro_stamp(backend))
    csv = open_csv(joinpath(RESULTS, "bench_drivers.csv"),
        "backend,eltype,m,gridn,grid,driver,nit,iters,zpd,first_call_s,time_s," *
        "per_shift_iter_s,gridpts_per_s,gflops,pencil")
    @printf("%-5s %-6s %-8s %-8s %-9s %-9s %-11s %-9s %-6s\n",
        "tok", "m", "driver", "nit", "1st(s)", "time(s)", "per_it(µs)", "Mpts/s", "GFLOP/s")
    for tok in TOKS
        T = ELTYPES[tok]
        for m in MS
            (; gridn, g, zg, nit, P, eye, zpd) = bench_setup(backend, T, m)
            function row(driver, nitcol, iters, t1, t)
                psi, gf = per_shift_iter(t, iters), gflops(t, m, iters; eye)
                @printf("%-5s %-6d %-8s %-8d %-9.4f %-9.4f %-11.3f %-9.3f %-6.3f\n",
                    tok, m, driver, nitcol, t1, t, psi * 1e6, g / t / 1e6, gf)
                println(csv, @sprintf("%s,%s,%d,%d,%d,%s,%d,%d,%s,%.6f,%.6f,%.6e,%.3f,%.4f,%s",
                    backend_tag(backend), tok, m, gridn, g, driver, nitcol, iters,
                    string(zpd), t1, t, psi, g / t, gf, pencil_label()))
                flush(csv)
            end

            # Adaptive first: the Ritz-residual stop retires each point at its own depth.
            # _ihlpsa_adaptive returns the depth grid — Σ is the adaptive work, max is the flat
            # depth a non-adaptive solver needs everywhere. Cold TTFP accrues here.
            nitmax = converged_nit_max(m)
            local σ, ng
            t1a = @elapsed ((σ, ng) = KAPseudospectra._ihlpsa_adaptive(
                backend, zg, P; nit_max = nitmax, devs, zpd))
            ta = bestof(backend; reps = REPS) do
                KAPseudospectra._ihlpsa_adaptive(backend, zg, P; nit_max = nitmax, devs, zpd)
            end

            # Converged-fixed baseline: flat depth = Dmax (the adaptive stop's deepest point), so
            # every point reaches the certified accuracy.
            Dmax = maximum(ng)
            t1f = @elapsed ihlpsa(backend, zg, P, Dmax; devs, zpd)
            tf = bestof(() -> ihlpsa(backend, zg, P, Dmax; devs, zpd), backend; reps = REPS)
            row("fixed", Dmax, total_iters(g, Dmax), t1f, tf)
            row("adaptive", Dmax, total_iters(ng), t1a, ta)
        end
    end
    close_csv(csv, joinpath(RESULTS, "bench_drivers.csv"))
end

"--trace" in ARGS ? trace_mode() : timing_mode()
