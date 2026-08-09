# Benchmarking KAPseudospectra.jl

`bench/` holds the benchmark **experiments**, all sharing one bench framework in `bench_common.jl`.

## The experiments

| Script | Question | Extra modes |
|---|---|---|
| `bench_drivers.jl` | fixed-`nit` vs adaptive driver (f32 + f64, single device). The only experiment that races the two drivers against each other — the converged experiments reach the adaptive driver through `converged_solve`, and everything else holds `default_nit(m)`. | `--trace` (CUDA-only): `CUDA.@profile` kernel-budget + per-launch retirement CSVs; runs last in CI (CUPTI crash risk) |
| `bench_kernels.jl` | the trsm strategy race (column / tiled / tiled-gemm) × the eltype ladder × m, at BOTH granularities per cell: isolated solve (roofline columns — logical AND effective/CGMA arithmetic intensity, the latter counting native limb FLOPs so MultiFloats climb toward compute-bound; see `tuning/multifloat_flop_costs.jl`) at **fixed `nit`** (exact launch count ⇒ exact analytic AI) and end-to-end **converged** `ihlpsa` (adaptive to the precision floor; f32/f64 dispersion). Column rows also carry a CPU solve time. Correctness is `maxdiff_svd`: each converged strategy vs the dense-SVD ground truth (`ℂsvdpsa`) on a coarse `BENCH_SVD_GRIDN²` sub-grid — because the solve runs to the arithmetic floor, this reports each strategy's true precision floor (all strategies agree there — the tiled-vs-column correctness check), not a shared under-convergence artifact; grcar's honest f32 digit-loss reads as conditioning. | `--counters`: one launch per cell for ncu/nvprof capture |
| `bench_multigpu.jl` | strong scaling (ndev = 1..N) + size sweep, f32 + f64, both drivers (the sanctioned adaptive exception: does the adaptive win survive multi-device load imbalance?). Needs ≥ 2 same-backend devices. | |
| `bench_cpu_gpu.jl` | end-to-end time-to-field of the SAME **converged** deliverable (adaptive to the precision floor): our GPU vs our (package-threaded) CPU vs **EigTool** (`matlab -batch` + `eigtool_psa.m`; probe-and-skip). σ-agreement columns are the fairness sentinels — note EigTool iterates to a 1e-5 tolerance and mirror-symmetrizes real-matrix grids. Also dumps the adaptive depth grid + σ field (`depthmap_*.csv`) at `BENCH_DEPTHMAP_M`. | |
| `bench_schur.jl` | one-time host Schur (`gees`) / QZ (`gges`) factorization vs the **converged** per-point sweep rate (adaptive to the floor); headline is the grid-free break-even `g* = factor_s / per_point_s`. Generalized rows use B = `kms` (deterministic). | |
| `bench_correctness.jl` | end-to-end σ vs a 256-bit **BigFloat dense-SVD oracle** (`ℂsvdpsa`), CPU-only (no CUDA; runs on any box's idle cores): `promoted` (Float64 LAPACK Schur, the ~1e-15 end-to-end bound) vs `generic` (full-precision GenericSchur reduction at BigFloat) — each MultiFloat rung reaches its own arithmetic floor end-to-end. Not part of the GPU perf sweep. | `BENCH_EXP=correctness`; sizes via `BENCH_CORR_MS` / `BENCH_CORR_GRIDN` / `BENCH_CORR_ELTYPES` |
| `bench_batching.jl` | the two batching knobs, GPU-only, **fixed driver only**: (a) isolated column solve at fixed m across batch widths, one batched launch vs a host loop of batch-of-1 launches — the launch-bound baseline nothing else in the suite measures; (b) end-to-end `ihlpsa` across grid batch sizes `zpd`, against the workspace model `findmaxbatchihl` budgets against, flagging the pin it selects. | widths/sizes via `BENCH_BATCH_M` / `BENCH_BATCHES` / `BENCH_UNBATCHED_MAX` / `BENCH_ZPD_M` |
| `bench_stopping.jl` | stopping-criteria ablation: the shipped **certified Ritz-residual** retirement vs the **successive-σ-change (Cauchy)** test it replaced, each at `nconfirm ∈ {1,2}`. A **replay** leg evaluates every rule on identical deep α/β trajectories against a dense-SVD oracle (full grid ≤ 256, seeded subsample above); a **live** leg races the actual driver per rule via the unexported `criterion` kwarg (cold + hot best-of); **ramp-spectrum torture rows** probe the ρ→1 creep regime where Cauchy under-resolves. The nconfirm sweep is an indication of where each setting fails, not a default-picker. | `BENCH_EXP=stopping` (opt-in, not under the umbrella yet); `BENCH_STOP_NITMAX` / `BENCH_STOP_ORACLE_N` / `BENCH_STOP_ORACLE_FULL_M` / `BENCH_STOP_POINTS_M` / `BENCH_STOP_TORTURE(_M)` |
| `bench_baseline.jl` | the missing **external** baseline: cuBLAS `trsm_batched` against explicitly materialized per-shift triangles vs our three strategies, at matched arithmetic. The naive path's compulsory DRAM traffic is Θ(batch·m²) where ours is Θ(m²), so it runs *near peak bandwidth* and still loses. Reports time, GFLOP/s, the traffic model, implied GB/s, and the working-set size that eventually OOMs. CUDA-only (no portable batched TRSM) and needs full-rate FP64. | `--counters`: one launch per leg for an ncu `dram__bytes` capture — the ground truth for the traffic model |
| `bench_golub.jl` | XL pseudospectra **portrait** of the golub matrix (random unit-LU product; det = 1 while the spectral radius grows ≈ 66·m): one adaptive solve across every device, dumping the σ field, depth grid, gees eigenvalues, and grid axes for plotting. Not a benchmark — no reps, timings logged for provenance only. | `BENCH_EXP=golub` (opt-in, hours-scale, not under the umbrella); `BENCH_GOLUB_M` / `BENCH_GOLUB_GRIDN` / `BENCH_GOLUB_HW` / `BENCH_GOLUB_NITMAX` / `BENCH_GOLUB_SEED` / `BENCH_GOLUB_COLS` (row slice `"lo:hi"`, flushed as it retires — survives a step timeout as checkpoints) |

Post-processing: `profiler_summarize.jl` parses an ncu **or** nvprof `--csv` capture
(format auto-detected) of `bench_kernels.jl --counters` into per-kernel-family saturation
tables. `gpu.jl <backend>` runs bench_drivers + bench_kernels in isolated subprocesses (the local
one-command smoke).

Artifact names carry the **box**: every file a CI run produces ends in `_a100` / `_1080ti` before
its extension (`bench_kernels_a100.csv`, `strong_scaling_1080ti.csv`,
`ncu_summary_f64_m256_a100.csv`), stamped by `.buildkite/scripts/lib/tag.sh` as each experiment
uploads — one build can run the same experiment on both cards, and a row's provenance should be
readable off the filename. Running an experiment by hand leaves the plain name.

The `bench_*.jl` scripts are the source of truth (GemmKernels-style: callable anywhere,
they produce their own data); Buildkite merely routes each run to the hardware it needs.
Plotting and analysis are downstream of the CSVs and of
`bundle_<box>_build<N>.tar.gz`; this directory produces data, not figures.

Manual tuning studies live in [`tuning/`](tuning/) alongside the profiles they produce:
`tune_device.jl` (the CI-wired probe), `nit_chunk_sweep.jl` and `multifloat_flop_costs.jl`
(hand-run calibration studies). A correctness **test**, not a benchmark:
`cross_vendor_repro.jl` / `cross_vendor_compare.jl` (`BENCH_EXP=repro`).

## Policy (bench_common.jl)

- **Depth** — two regimes. The **roofline / isolated-solve** (bench_kernels' `iso_*` columns) uses a
  fixed `default_nit(m) = 4⌈log₂ m⌉`: a known, identical trsm launch count is what makes the
  analytic FLOP/byte/AI exact. This is NOT a converged solve — at m ≳ 256 the smallest singular
  value is still under-resolved at that depth — so the **end-to-end time-to-solution** rows
  (bench_kernels' e2e columns, bench_cpu_gpu, bench_schur) and the **SVD sentinel** instead run the adaptive driver to the
  **precision floor** via `converged_solve` (`converged_rtol(T)` per base precision, `nit_max`
  headroom over the plotting default). The adaptive stop is a certified **Ritz residual bound**
  (`½·β_{k+1}·|s_k|/λ ≤ rtol`), so σ reaches the arithmetic floor uniformly in m — a
  successive-change test stalled early near small-gap points. `BENCH_NIT` pins the fixed depth;
  `BENCH_NIT_MAX` the adaptive cap. `bench_drivers` still owns the fixed-vs-adaptive
  comparison itself.
- **Grid** — `bench_gridn(backend)`: GPU sweeps default to **128²**, CPU to 64². The
  reported metrics (AI, GFLOP/s, per-(shift·iteration) time, bench_multigpu scaling efficiency,
  bench_schur g*, saturation) are all grid-free and extrapolate, so the grid is a pure wall-clock
  multiplier on everything except the cpu_gpu field dumps (depth map / σ). `BENCH_GRIDN=<n>`
  overrides; 512 is the publication size (§ Grid sizing).
- **Tuning** — every experiment expects a **tuned device**, and the knobs come from a tracked
  profile: `KAPSEUDO_TUNE_PROFILE=bench/tuning/a100.toml`. Both CI scripts export one; the probe
  runs only on `BENCH_EXP=tune`, which rewrites the profile for committing. Locally, once per
  machine: `julia --project=bench bench/tuning/tune_device.jl cuda`, then commit
  `bench/tuning/<key>.toml`. `repro_stamp` prints `tuning=TUNED`/`PARTIAL`/`UNTUNED` with the
  source file, so a run is self-describing. A speedup quoted against an unconfigured (or
  partially configured) baseline is a property of the configuration, not of the kernel.
  See [`tuning/README.md`](tuning/README.md).
- **Pencil mode** — `BENCH_PENCIL` selects `gen` (default) or `std`. The package's object is
  the pencil `zB − A`; `B = I` is the special case. `gen` is the default and every experiment
  stamps a `pencil` column, so a `gen` run and a `std` run of the same script are directly
  comparable. Two legs are exempt **by construction** and pin `std` inline: bench_cpu_gpu (EigTool
  computes standard pseudospectra only) and bench_correctness (its subject is the reduction's accuracy, and
  a full-precision QZ is impractically slow). Extended-precision `gen` rungs reduce with a
  promoted Float64 QZ for the same reason — sound for throughput, Schur-bounded for accuracy;
  the run warns once.
- **Matrix** — grcar, suite-wide, built in one place: a named non-normal family, deterministic,
  needs m > 4, and matches MATLAB `gallery('grcar')` for the EigTool leg. MultiFloat rungs
  reduce with the full-precision GenericSchur factor (BigFloat, cached per size, rounded to the
  working type) by default — end-to-end accuracy to the arithmetic floor of the eltype;
  `BENCH_FAST_SCHUR=1` opts into the fast Float64-promote factor for the kernel-perf sweeps. The
  shift box is grcar's (`bench_box()`).
- **Eltype ladder** — `BENCH_ELTYPES` tokens `f32 f64 f32x2 f32x4 f64x2 f64x4`; rungs gate
  per device (FP64 support, per-limb shuffle) with a warning per skip. Full-rate-FP64
  hardware owns the f64-limb ladder; reduced-rate-FP64 devices run the f32-limb
  story.

## Setup

The env is CPU-instantiable; add the one backend package for the device under test (the GPU
backends can't co-resolve in one environment):

```sh
julia --project=bench -e 'using Pkg; Pkg.instantiate()'          # CPU-ready
julia --project=bench -e 'using Pkg; Pkg.add("CUDA")'            # or AMDGPU / oneAPI
```

## Running locally

```sh
julia --project=bench bench/gpu.jl cuda            # bench_drivers + bench_kernels, bounded smoke sizes
julia --project=bench bench/bench_kernels.jl cuda  # one experiment, policy sizes
julia --project=bench bench/bench_multigpu.jl cuda 2   # cap to 2 devices
BENCH_GRIDN=512 julia --project=bench bench/bench_drivers.jl cuda # 512² publication fields
```

`gpu.jl` defaults to a **bounded smoke** (BENCH_MS=64,128 · BENCH_GRIDN=64 · BENCH_REPS=2 ·
BENCH_ELTYPES=f32); set `KAPSEUDO_BENCH_FULL=1` for the policy sizes.

## Sizing knobs

| Variable | Meaning | Default |
|---|---|---|
| `BENCH_MS` | matrix sizes m, comma-separated | per experiment (kernels GPU: `16,…,1024`) |
| `BENCH_GRIDN` | grid side length (512 = publication) | 128 GPU / 64 CPU |
| `BENCH_NIT` | flat fixed depth (roofline/isolated-solve) — overrides `4⌈log₂ m⌉` | — |
| `BENCH_NIT_MAX` | adaptive cap for the converged (precision-floor) solves — overrides `20⌈log₂ m⌉` | — |
| `BENCH_ELTYPES` | eltype-ladder rungs (drivers, kernels; also selects `--counters` cells) | exp-specific |
| `BENCH_REPS` / `BENCH_REPS_STATS` | best-of reps / dispersion reps (kernels f32+f64 rungs) | 3 / 15 |
| `BENCH_MS_MF` | kernels MultiFloat-rung sizes | `128,256,512` (GPU) |
| `BENCH_SOLVE_BATCH` | kernels isolated-solve batch (kernel geometry, not a σ-grid) | 4096 |
| `BENCH_CPUGPU_MS` | kernels CPU-row sizes (MultiFloat CPU is the long pole) | `128,256,512` |
| `BENCH_SVD_GRIDN` | kernels `maxdiff_svd` sentinel sub-grid side (dense SVD is O(m³)/point); `0` skips it | `8` |
| `BENCH_REPS_CPU` | cpu_gpu best-of reps for the CPU timing leg | `2` |
| `BENCH_DEPTHMAP_M` | cpu_gpu depth-grid + σ dump size (capped below `max(BENCH_MS)`) | `256` |
| `BENCH_HIPREC_BITS` | BigFloat precision (bits) — the MultiFloat GenericSchur reduction and the correctness oracle (256 resolves f64x4) | `256` |
| `BENCH_MS_SCHUR` | schur sizes (QZ at 8192 is tens of minutes) | `1024,2048,4096` |
| `BENCH_SCHUR_ELTYPE` | schur eltype | `f64` |
| `BENCH_MG_NS` / `BENCH_SS_N` | multigpu size-sweep ns / strong-scaling n | `512,…,4096` / `1024` |
| `KAPSEUDO_STRIDED` | multi-device column assignment: `1` round-robin (load-balances the adaptive driver), `0` contiguous bands | `1` |
| `BENCH_EIGTOOL_MS` | cpu_gpu sizes that also run EigTool | `64,…,512` |
| `BENCH_TRACE_MS` / `BENCH_TRACE_GRIDN` / `BENCH_TRACE_ELTYPE` | drivers `--trace` config | `256,1024` / 128 / `f64` |
| `BENCH_ZPD_HEADROOM` | divisor on the pinned batch where the adaptive driver runs (gather scratch OOM headroom) | 2 |
| `BENCH_PENCIL` | pencil mode: `gen` (generalized `zB − A`) or `std` (`B = I`) | `gen` |
| `BENCH_BASE_MS` / `BENCH_BASE_COHORT` / `BENCH_BASE_NIT` | baseline sizes / shifts per batched call / iterations | `128,…,1024` / `1024` / `8` |
| `KAPSEUDO_BENCH_FULL=1` | gpu.jl: skip the smoke defaults | — |
| `KAPSEUDO_MAX_ZPD` | hard cap on the per-device batch (small-max-alloc GPUs) | — |
| `EIGTOOL_PATH` | EigTool checkout for cpu_gpu | `bench/eigtool` (CI clones it) |

## Grid sizing

CI rides the bare defaults (128² GPU / 64² CPU). Every reported rate is grid-free, so those
numbers extrapolate; `BENCH_GRIDN=512` is only for the runs whose *output is the field itself*
(the cpu_gpu depth map / σ dump, `bench_golub.jl`) or for a headline time-to-solution. 512² is
the point past which a rendered σ-field stops improving at single-column figure sizes, so it is
the suite's publication grid; 1024² only for a full-width filled colormap.

## Running on Buildkite

Everything is selected by **build env vars**, set in **New Build → Environment Variables**
(`.buildkite/pipeline.yml`; the long step bodies live in `.buildkite/scripts/`):

- `BENCH_EXP` — a comma list of sections, or `all` for each box's benchmark umbrella
  (A100: kernels, drivers, cpu_gpu, batching, baseline, schur; 1080 Ti: multigpu, kernels).
  Section names match the `bench_*.jl` scripts minus the prefix: `kernels drivers cpu_gpu
  schur stopping batching baseline multigpu golub nsys counters tune repro correctness`.
  `drivers` includes the `--trace` leg (scheduled last);
  `counters` runs ncu on full-rate-FP64 (Volta+) boxes and nvprof on pre-Volta boxes, one
  capture per (eltype, m).
  Unset ⇒ no jobs run.
- `BENCH_ONLY_BOX=a100|1080ti` — restrict the CUDA jobs to one box, e.g.
  `BENCH_EXP=kernels BENCH_ONLY_BOX=1080ti` to iterate on one box without
  queueing a redundant job on the other box.
- Sizing knobs ride along the same way (`BENCH_MS=512,1024` etc. — the steps use
  `${VAR:-default}`, so build-level env wins), or `BENCH_GRIDN=512` for a publication-grid
  run.

Results upload as per-section build artifacts plus one
`bundle_<box>_build<N>.tar.gz` per box (`bench/results/*`).

## Per-backend notes

- **Full-rate-FP64 devices** (e.g. an A100) — the f64/f64xN rungs are only a fair fight
  here; `counters` captures use `ncu`. MATLAB for the EigTool leg is probed
  (`command -v matlab`, else `module load "$MATLAB_MODULE"` — set `MATLAB_MODULE` on the build
  if your site's module is not the default); the leg self-skips without it.
- **Reduced-rate-FP64 / pre-Volta devices** (e.g. Pascal cards like the GTX 1080 Ti, FP64
  at 1/32 rate) — run the f32-limb ladder + multigpu instead; `counters` captures fall
  back to `nvprof` (no Nsight Compute support pre-Volta). Peak BW comes out of the CUDA
  attribute query fine. The box defaults `BENCH_ELTYPES=f32` for every section but gives
  `kernels` the `f32,f32x2,f32x4` ladder; a build-level `BENCH_ELTYPES` overrides both, and
  `BENCH_ELTYPES_1080TI` overrides the kernels ladder alone.
- **ROCm devices** (e.g. AMD RX 6900 XT) — cross-vendor correctness leg only. AMDGPU.jl
  needs system ROCm 6.0+ and `JULIA_NUM_THREADS=1` (multithread GC bug).
- **FP64-less devices** (e.g. Intel Arc A380) — cross-vendor correctness leg only. No
  native FP64 ⇒ ComplexF32 + `set_pdiv_accurate!(false)`; the tiled solve self-gates
  (stock oneAPI `@shfl` stub).

## Figure of merit

The trsm solve is **latency-bound** (dependent row/panel chains), so the headline is **time
per (shift · iteration)**, with GFLOP/s as the secondary number. The roofline figure exists
to *show* the kernel sitting far below both ceilings (drawn from published per-device peaks
at figure time), and the `--counters` captures show what the logical-flop roofline can't:
rising precision saturates the compute pipes (ncu `sm__throughput` climbs the limb ladder)
even as logical arithmetic intensity falls.
