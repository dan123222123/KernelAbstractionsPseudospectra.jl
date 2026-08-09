# Tune THIS device's trsm knobs and persist them. A prerequisite, not an experiment — see
# README.md in this directory (§ Why tracked profiles).
#
# `tune_trsm!` runs `tune_trsm_tiled!` (picks one (tilecols, blockwarps, warpgridpts) triple
# at m=512) and `tune_trsm_wgs!` (sweeps the column solve's workgroup size over m, persisted
# as a schedule). Each probe sizes its batch from `findmaxbatchihl`.
#
# Device-agnostic: same backend argument as every other script.
#
# MultiFloat rungs are probed too (drop with BENCH_TUNE_TYPES=f32,f64 for IEEE only).
#
# Output is the tracked PROFILE at bench/tuning/$KAPSEUDO_TUNE_KEY.toml (not the gitignored
# LocalPreferences.toml); later runs select it via KAPSEUDO_TUNE_PROFILE. Commit it.
#
#   julia --project=bench bench/tuning/tune_device.jl cuda
#   KAPSEUDO_TUNE_KEY=1080ti julia --project=bench bench/tuning/tune_device.jl cuda
#   BENCH_TUNE_SWEEP_MS=1024,2048 julia --project=bench bench/tuning/tune_device.jl cuda

include(joinpath(@__DIR__, "..", "bench_common.jl"))

# Probing under an active profile would measure the profile's knobs and persist values it shadows.
haskey(ENV, "KAPSEUDO_TUNE_PROFILE") && (@info "unsetting KAPSEUDO_TUNE_PROFILE for the probe";
    delete!(ENV, "KAPSEUDO_TUNE_PROFILE"); KAPseudospectra.reload_tuning!())

backend = select_backend()
KernelAbstractions.isgpu(backend) ||
    error("tune_device is a GPU probe; got $(backend). Nothing to tune on the CPU path.")

# Probe the FIRST device: the knobs are a property of the architecture. Assumes every device
# on the queue is the same model — a heterogeneous box needs one pass per model.
dev = first(KAPseudospectra.devices(backend))
foreach(println, repro_stamp(backend))
@info "tuning device" backend dev device_name(backend)
println("BEFORE: ", tuning_stamp())

profile = joinpath(@__DIR__, get(ENV, "KAPSEUDO_TUNE_KEY", "device") * ".toml")
# BENCH_TUNE_TYPES (not the experiments' BENCH_ELTYPES): which rungs to probe vs. measure are
# separate knobs. `runnable_eltypes` drops rungs this device can't host (no FP64 -> no
# f64/f64xN), so an FP64-less card probes the f32 family only instead of erroring midway.
tune_toks = let toks = Symbol.(split(get(ENV, "BENCH_TUNE_TYPES",
        "f32,f64,f32x2,f32x4,f64x2,f64x4"), ","))
    all(t -> haskey(ELTYPES, t), toks) ||
        error("BENCH_TUNE_TYPES tokens must be in $(keys(ELTYPES)) (got $toks)")
    runnable_eltypes(backend, toks)
end
tune_types = Tuple(ELTYPES[t] for t in tune_toks)
@info "probing element types" tune_types
# Record the contention canary while the probe has the device to itself — the only moment an
# idle baseline is certain; every later run compares against it.
canary = device_canary(backend)
@info "contention canary (idle baseline)" canary_gflops=canary
chosen = KAPseudospectra.tune_trsm!(backend, dev; types = tune_types, profile,
    device = device_name(backend),
    extra = isnan(canary) ? Dict{String, String}() :
            Dict("canary_gflops" => string(round(canary; digits = 1))))
@info "PERSISTED (tiled knobs at the probe's m; wgs as a schedule over m)" chosen.tiled chosen.wgs profile

# Per-type completeness audit: TOML has no `missing`, and the 16-key IEEE check in
# `write_tune_profile` cannot see MultiFloat rungs, so report per probed type which knobs
# landed — a skipped rung (e.g. a tile that can't fit shared memory) is visible in the log
# instead of silently heuristic.
if isfile(profile)
    tbl = let raw = TOML.parsefile(profile)
        get(raw, "KAPseudospectra", raw)
    end
    audit = [(T, e,
                 [k for k in ("trsm_tilecols", "trsm_blockwarps", "trsm_warpgridpts", "trsm_wgs")
                  if !haskey(tbl, "$(k)_$(T)_$(e)")])
             for T in tune_types for e in ("eye", "gen")]
    for (T, e, absent) in audit
        println("tune audit: ", rpad("$T/$e", 44),
            isempty(absent) ? "PRESENT" : "MISSING " * join(absent, ","))
    end
    nmiss = sum(cell -> length(cell[3]), audit)
    nmiss == 0 || @warn "tune audit: $nmiss knob(s) missing across the probed types — " *
                        "those cells will run on heuristic defaults"
else
    @warn "tune audit skipped — no profile was written" profile
end

# Log-only: does the tiled optimum move with m? Nothing is written; records how much the
# probe's single operating point costs at other sizes. Tiled only — wgs was already swept above.
for m in env_ints("BENCH_TUNE_SWEEP_MS", (1024, 2048))
    @info "===== tiled sweep at m=$m (persist=false, log only) ====="
    try
        KAPseudospectra.tune_trsm_tiled!(backend, dev; types = tune_types, m, persist = false)
    catch e
        @warn "sweep at m=$m failed" exception = e
    end
end

println("AFTER:  ", tuning_stamp())
@info "commit $(relpath(profile, dirname(dirname(@__DIR__)))) — an uncommitted re-tune is invisible to " *
      "every run that reads the numbers later"
