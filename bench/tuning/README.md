# Tuning profiles

One file per machine, holding tuned trsm parameters. The producer and the manual calibration studies live here too: `tune_device.jl` (the CI-wired probe), `nit_chunk_sweep.jl` (adaptive-driver chunking crossover) and `multifloat_flop_costs.jl`
(native-FLOP expansion table for the effective-AI model).

The profile is selected via an env variable:

    KAPSEUDO_TUNE_PROFILE=bench/tuning/a100.toml julia --project=bench bench/bench_kernels.jl cuda

Per knob the resolution order is

    KAPSEUDO_TRSM_{TILECOLS,BLOCKWARPS,WARPGRIDPTS,WGS} env scalar  >  profile  >  LocalPreferences.toml  >  heuristic

The tuning profile outranks `LocalPreferences.toml`.

## Why tracked profiles

Every experiment expects a tuned device, and every run states what configured it (`repro_stamp` prints `tuning=TUNED`/`PARTIAL`/`UNTUNED` plus the source file): a speedup quoted against an unconfigured default is a property of the configuration, not of the kernel. PARTIAL deliberately counts as untuned (`is_tuned()`), since a table missing only `trsm_wgs_*` still leaves the whole column solve on its heuristic.

## Regenerating

    julia --project=bench bench/tuning/tune_device.jl cuda   # writes bench/tuning/$KAPSEUDO_TUNE_KEY.toml

In CI this happens only on `BENCH_EXP=tune` (add `BENCH_ONLY_BOX=a100|1080ti` to re-tune one box); every other job reads the committed profile. Download the rewritten TOML from the build's artifacts and COMMIT it — an uncommitted re-tune does not exist.

## What a complete profile contains

The IEEE core is 16 keys: `{trsm_tilecols, trsm_blockwarps, trsm_warpgridpts, trsm_wgs}` × `{ComplexF32, ComplexF64}` × `{eye, gen}`. Fewer than that and some part of the solve is still on its default; `write_tune_profile` stamps a `# PARTIAL` line and warns.

MultiFloat rungs are probed too (opt out with `BENCH_TUNE_TYPES=f32,f64`), adding `…_Complex{MultiFloat{…}}_…` keys — quoted, since they are not bare TOML keys. TOML has no `missing`: an untuned knob is an absent key, indistinguishable in the file from "not applicable on this device", so `tune_device.jl` prints a per-type PRESENT/MISSING audit after the probe — a rung the probe skipped (e.g. a tile that cannot fit shared memory) is visible in the run log.

Values are strings by convention (that is what `Preferences` persists); the readers `tryparse`, so bare integers work too. `trsm_wgs_*` is a piecewise-constant schedule over `m`, `"m₀:wgs₀,m₁:wgs₁"`, read as the `wgs` of the last threshold `≤ m`.

## Files

| Profile | Machine | Box key | Buildkite queue |
| --- | --- | --- | --- |
| `a100.toml` | NVIDIA A100-PCIE-40GB | `a100` | `cuda-a100` |
| `1080ti.toml` | 6× NVIDIA GTX 1080 Ti | `1080ti` | `cuda-6xgtx1080ti` |
