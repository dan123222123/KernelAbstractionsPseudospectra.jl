# Stopping-criteria ablation: certified Ritz-residual retirement (:certified) vs the
# successive-σ-change test (:cauchy), each at nconfirm ∈ {1, 2}. Two legs — replay (one
# deep trajectory scored against a dense-SVD oracle) and live (the actual adaptive driver,
# wall-clock only) — plus CPU-side ramp-spectrum torture rows.
#   julia --project=bench bench/bench_stopping.jl cuda
include(joinpath(@__DIR__, "bench_common.jl"))

backend = select_backend(filter(a -> !startswith(a, "--"), ARGS))
gpu = KernelAbstractions.isgpu(backend)
# Single-device only: multi-device load-balance is bench_multigpu's concern.
devs = gpu ? [first(KAPseudospectra.devices(backend))] : missing
gpu && KAPseudospectra.device!(backend, first(KAPseudospectra.devices(backend)))

using Adapt: adapt

const MS = env_ints("BENCH_MS", (64, 128, 256, 512, 1024))
const REPS = env_int("BENCH_REPS", 3)
const TOKS = runnable_eltypes(backend, eltype_tokens("f32,f64"))
const RESULTS = results_dir()
const CHUNK = 2                       # checkpoint stride — the driver's nit_chunk default
const ORACLE_FULL_M = env_int("BENCH_STOP_ORACLE_FULL_M", 256)
const ORACLE_N = env_int("BENCH_STOP_ORACLE_N", 512)
const POINTS_M = env_int("BENCH_STOP_POINTS_M", 256)   # m at which the per-point CSV is dumped
const RULES = ((:certified, 2), (:certified, 1), (:cauchy, 2), (:cauchy, 1))

# Driver's default cap: compares rules at shipped working-accuracy rtol, not the precision floor.
stop_nitmax(m) = env_int("BENCH_STOP_NITMAX", 8 * max(1, ceil(Int, log2(m))))

nearest_q(sorted, q) = sorted[clamp(round(Int, 1 + q * (length(sorted) - 1)), 1, length(sorted))]

# σ and residual-bound sequences at every checkpoint depth, for every grid point, from one
# deep replay trajectory per batch.
function replay_sequences(backend, P, zv_h, idxbatches, nit_max, x₀)
    T = eltype(zv_h)
    R = real(T)
    bg = KAPseudospectra.get_bgarray(backend)
    ks = collect(CHUNK:CHUNK:nit_max)
    σs = Matrix{R}(undef, length(ks), length(zv_h))
    rs = similar(σs)
    for idxb in idxbatches
        g = length(idxb)
        # Per-batch workspace (not the driver's hoisted one) — idxbatches is a single batch here.
        ihl = adapt(bg, KAPseudospectra.IHLworkspace(P, g, x₀))
        α = adapt(bg, zeros(T, nit_max, g))
        β = adapt(bg, zeros(T, nit_max + 1, g))
        view(ihl.zv, 1:g) .= adapt(bg, zv_h[idxb])
        KAPseudospectra.lockstep_ihl!(α, β, ihl, nit_max, g)
        αh, βh = adapt(Array, α), adapt(Array, β)
        σk = zeros(R, g)
        rk = zeros(R, g)
        for (i, k) in enumerate(ks)
            KAPseudospectra.ihlsrg!(σk, view(zv_h, idxb), 1, 0,
                αh[1:k, :], βh[1:(k + 1), :]; resid = rk)
            σs[i, idxb] .= σk
            rs[i, idxb] .= rk
        end
    end
    return ks, σs, rs
end

# Replay one rule over the checkpoint sequences, mirroring `_sdihlpsa_adaptive` (nconfirm
# streak; cap falls back to the deepest σ), via the package's own predicates.
function replay_rule(criterion, nconfirm, ks, σs, rs, rtol, atol)
    ncp, n = size(σs)
    depth = zeros(Int, n)
    σret = zeros(eltype(σs), n)
    cap = falses(n)
    for j in 1:n
        σprev = σs[1, j]
        streak = 0
        retired = false
        for i in 2:ncp
            ok = criterion === :certified ?
                 KAPseudospectra._adaptive_converged(σs[i, j], rs[i, j], rtol, atol) :
                 KAPseudospectra._cauchy_converged(σs[i, j], σprev, rtol, atol)
            streak = ok ? streak + 1 : 0
            if streak >= nconfirm
                depth[j], σret[j], retired = ks[i], σs[i, j], true
                break
            end
            σprev = σs[i, j]
        end
        retired || ((depth[j], σret[j], cap[j]) = (ks[end], σs[end, j], true))
    end
    return (; depth, σret, cap)
end

# Dense-SVD oracle on the flat shift vector: every point at m ≤ ORACLE_FULL_M, a seeded
# ORACLE_N-subsample above (svdvals is O(m³) per point). Returns (indices, σ_true).
function oracle_sigma(P, zv_h, m)
    sel = if m <= ORACLE_FULL_M || length(zv_h) <= ORACLE_N
        collect(eachindex(zv_h))
    else
        sort!(shuffle(MersenneTwister(0xBEEF), collect(eachindex(zv_h)))[1:ORACLE_N])
    end
    return sel, vec(ℂsvdpsa(reshape(zv_h[sel], :, 1), P))
end

# Error-at-retirement stats vs the oracle; masks σ_true ≲ √eps points (atol floor, where
# relative error is noise).
function err_stats(σret_flat, sel, σt, rtol, R)
    mask = σt .> sqrt(eps(R))
    errs = abs.(σret_flat[sel][mask] .- σt[mask]) ./ σt[mask]
    isempty(errs) && return (n = 0, masked = count(!, mask), viol = 0, maxr = NaN, medr = NaN)
    s = sort(errs)
    (n = length(errs), masked = count(!, mask), viol = count(>(rtol), errs),
        maxr = s[end] / rtol, medr = nearest_q(s, 0.5) / rtol)
end

depth_stats(d) = (s = sort(vec(d)); (mean = sum(s) / length(s), p90 = nearest_q(s, 0.9),
    max = s[end], sum = sum(s)))

function csv_row(csv, tok, m, matrix, gridn, leg, crit, nconfirm, rtol, nitmax, npts,
        retired, hitcap, ds, es, t1, t)
    println(csv, @sprintf("%s,%s,%d,%s,%d,%d,%s,%s,%d,%.1e,%d,%d,%d,%d,%.2f,%d,%d,%d,%d,%d,%d,%.4g,%.4g,%.4f,%.4f,%s",
        backend_tag(backend), tok, m, matrix, gridn, npts, leg, crit, nconfirm, rtol,
        nitmax, npts, retired, hitcap, ds.mean, ds.p90, ds.max, ds.sum,
        es.n, es.masked, es.viol, es.maxr, es.medr, t1, t, pencil_label()))
    flush(csv)
end

foreach(println, repro_stamp(backend))
csvpath = joinpath(RESULTS, "bench_stopping.csv")
csv = open_csv(csvpath,
    "backend,eltype,m,matrix,gridn,grid,leg,rule,nconfirm,rtol,nit_max,npts,retired," *
    "hit_cap,mean_depth,p90_depth,max_depth,sum_depth,oracle_pts,masked_pts,viol," *
    "max_err_over_rtol,med_err_over_rtol,first_call_s,time_s,pencil")
@printf("%-5s %-6s %-7s %-12s %-4s %-9s %-9s %-7s %-9s %-9s\n",
    "tok", "m", "leg", "rule", "nc", "meandep", "Σdepth", "viol", "maxe/rtol", "time(s)")

for tok in TOKS
    T = ELTYPES[tok]
    R = real(T)
    rtol = R(KAPseudospectra._adaptive_default_rtol(T))
    atol = R(KAPseudospectra._adaptive_default_atol(T))
    for m in MS
        gridn = bench_gridn(backend)
        zg = bench_grid(T, gridn)
        nitmax = stop_nitmax(m)
        P = bench_pencil(T, m).P
        zpd = pinned_zpd(backend, T, m, nitmax; ngrid = gridn * gridn,
            headroom = zpd_headroom())
        x₀ = KAPseudospectra._adaptive_x₀(T, size(P, 1), 0x61646170)   # driver default seed
        zv_h, idxbatches = KAPseudospectra._grid_batches(zg, ismissing(zpd) ? length(zg) : zpd)
        sel, σt = oracle_sigma(P, zv_h, m)

        # ---- replay leg: one deep trajectory, every rule on the same data ----
        ks, σs, rs = replay_sequences(backend, P, zv_h, idxbatches, nitmax, x₀)
        replays = Dict{Tuple{Symbol, Int}, Any}()
        for (crit, nc) in RULES
            rr = replay_rule(crit, nc, ks, σs, rs, rtol, atol)
            replays[(crit, nc)] = rr
            ds = depth_stats(rr.depth)
            es = err_stats(rr.σret, sel, σt, rtol, R)
            @printf("%-5s %-6d %-7s %-12s %-4d %-9.2f %-9d %-7d %-9.3g %-9s\n",
                tok, m, "replay", crit, nc, ds.mean, ds.sum, es.viol, es.maxr, "-")
            csv_row(csv, tok, m, BENCH_MATRIX, gridn, "replay", crit, nc, rtol, nitmax,
                length(zv_h), count(!, rr.cap), count(rr.cap), ds, es, NaN, NaN)
        end

        # Per-point dump at POINTS_M: one row per grid point, carrying σ_true and, per rule, the
        # retirement depth and relative error — the per-point distribution the summary rows average.
        if m == POINTS_M && tok === last(TOKS)
            path = joinpath(RESULTS, "bench_stopping_points_$(tok)_m$(m).csv")
            open(path, "w") do io
                hdr = ["z_re", "z_im", "sigma_true"]
                for (crit, nc) in RULES
                    push!(hdr, "depth_$(crit)$(nc)", "err_$(crit)$(nc)")
                end
                println(io, join(hdr, ","))
                for (i, gi) in enumerate(sel)
                    cols = Any[real(zv_h[gi]), imag(zv_h[gi]), σt[i]]
                    for (crit, nc) in RULES
                        rr = replays[(crit, nc)]
                        push!(cols, rr.depth[gi], abs(rr.σret[gi] - σt[i]) / σt[i])
                    end
                    println(io, join(cols, ","))
                end
            end
            println("wrote ", path)
        end

        # ---- live leg: the actual driver per rule (cold + hot best-of) ----
        for (crit, nc) in RULES
            run() = KAPseudospectra._ihlpsa_adaptive(backend, zg, P;
                criterion = crit, nconfirm = nc, nit_max = nitmax, devs, zpd)
            local σL, ngL
            t1 = @elapsed ((σL, ngL) = run())
            t = bestof(run, backend; reps = REPS)
            σflat = vec(permutedims(σL))          # back to zv_h (flattened-grid) order
            ngflat = vec(permutedims(ngL))
            ds = depth_stats(ngflat)
            es = err_stats(σflat, sel, σt, rtol, R)
            # hit_cap counts depth == nit_max: cap fallbacks plus legit final-checkpoint
            # retirements (indistinguishable per point).
            @printf("%-5s %-6d %-7s %-12s %-4d %-9.2f %-9d %-7d %-9.3g %-9.4f\n",
                tok, m, "live", crit, nc, ds.mean, ds.sum, es.viol, es.maxr, t)
            csv_row(csv, tok, m, BENCH_MATRIX, gridn, "live", crit, nc, rtol, nitmax,
                length(zv_h), length(ngflat) - count(==(nitmax), ngflat),
                count(==(nitmax), ngflat), ds, es, t1, t)
        end
    end
end

# ---- torture rows: ramp-spectrum diagonal, CPU replay only (the quasi-continuum regime
# where Cauchy retires on agreement-without-arrival) ----
if get(ENV, "BENCH_STOP_TORTURE", "1") == "1"
    T = ComplexF64
    R = Float64
    rtol = R(KAPseudospectra._adaptive_default_rtol(T))
    atol = R(KAPseudospectra._adaptive_default_atol(T))
    m = env_int("BENCH_STOP_TORTURE_M", 256)
    nitmax = stop_nitmax(m)
    x₀ = KAPseudospectra._adaptive_x₀(T, m, 0x61646170)
    for gap in (1e-2, 1e-3, 1e-4, 1e-5)
        σ1 = 1e-2
        S = Matrix(Diagonal(T.(σ1 .* (1.0 .+ gap .* (0:(m - 1))))))
        P = MatrixPencil(schur(S))
        zg1 = fill(zero(T), 1, 1)
        zv_h, idxb = KAPseudospectra._grid_batches(zg1, 1)
        ks, σs, rs = replay_sequences(CPU(), P, zv_h, idxb, nitmax, x₀)
        for (crit, nc) in RULES
            rr = replay_rule(crit, nc, ks, σs, rs, rtol, atol)
            ds = depth_stats(rr.depth)
            es = err_stats(rr.σret, [1], [σ1], rtol, R)
            @printf("%-5s %-6d %-7s %-12s %-4d %-9.2f %-9d %-7d %-9.3g %-9s\n",
                "f64", m, "ramp", crit, nc, ds.mean, ds.sum, es.viol, es.maxr, "-")
            csv_row(csv, "f64", m, "ramp$(gap)", 1, "torture", crit, nc, rtol, nitmax,
                1, count(!, rr.cap), count(rr.cap), ds, es, NaN, NaN)
        end
    end
end

close_csv(csv, csvpath)
