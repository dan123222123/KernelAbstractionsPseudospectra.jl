# Progress — adaptive `nit` for `ihlpsa`

Tracking changes on branch `adaptive_nit` for the adaptive-iterations feature
(design in `ADAPTIVE_NIT_PLAN.md`). Use this as the source for commit messages.

## Tier 1 — global adaptive `nit` (whole grid, chunked) — DONE

Adds an exported `ihlpsa_adaptive` "auto-nit" driver: runs lockstep
inverse-Lanczos on the whole grid in chunks of `nit_chunk`, stops when **every**
grid point's σ has converged between successive chunks. Returns `(σ, nit_used)`.
The fixed-`nit` `ihlpsa(...)` is left untouched as the non-adaptive regression
control (separate function, not a flag).

Four gaps in the design doc's sketch were fixed:
1. **Fixed `x₀` across chunks (correctness).** Doc looped with `x₀=missing`, so
   each chunk drew a fresh `randn` start (`src/ihlpsa.jl:102`) → comparing σ at
   nit=k vs nit=k+chunk compared two *independent* runs. The driver now
   materializes one deterministic `x₀` up front (`_adaptive_x₀`, seeded random
   when none supplied) and reuses it every chunk, so `eigmax(T_k)` vs
   `eigmax(T_{k+chunk})` is a principled monitor.
2. **Robust convergence criterion.** `_adaptive_converged` uses combined
   `|Δ| ≤ atol + rtol·|σ|` with an explicit eps-sentinel short-circuit (points
   pinned to `eps` at eigenvalues by `ihlsrg!` retire instead of pinning the
   grid), and eltype-branched defaults (F32→1e-4, F64→1e-6).
3. **Multi-device collapse avoided** by passing the whole grid to `ihlpsa` each
   chunk (inherits the existing column fan-out; no `N×1` single-device collapse).
4. **Non-adaptive path preserved**: additive API; the driver only *calls*
   `ihlpsa`.

### Files changed (Tier 1)
- `src/ihlpsa.jl` — added `using Random`; `_adaptive_x₀`, `_adaptive_default_rtol`,
  `_adaptive_default_atol`, `_adaptive_converged`, and the `ihlpsa_adaptive`
  driver (in the `## WRAPPER FUNCTIONS ##` section).
- `src/KAPseudospectra.jl` — export `ihlpsa_adaptive`; added to `@compile_workload`.
- `Project.toml` — added `Random` (stdlib) to `[deps]` (the explicit `using
  Random` for `MersenneTwister` requires it as a direct dependency; the old bare
  `randn` calls resolved transitively).
- `test/test_consistency.jl` — `ihlpsa_adaptive vs ℂsvdpsa` and
  `ihlpsa_adaptive generalized pencil (B≠I)` CPU testsets; backend-parametrized
  `test_adaptive_backend`.
- `test/runtests.jl` — wired `test_adaptive_backend` into the CUDA/AMDGPU
  `functional()` guards.
- `bench/bench_adaptive.jl` — new; Grcar fixed-`nit` vs adaptive comparison.

### Verification (Tier 1)
- `julia --project=test --threads=auto test/runtests.jl cpu` — all pass, no
  regressions. New testsets: `ihlpsa_adaptive vs ℂsvdpsa` 6/6,
  `ihlpsa_adaptive generalized pencil (B≠I)` 4/4 (F32+F64). No `@warn` (early
  stop worked).
- Bench at m=64 Grcar: Tier-1 adaptive ≈ 0.37× (slower) — expected; Tier 1
  re-runs from iter 1 each chunk and still pays the slowest-point cost on the
  whole grid. The wall-clock win is a Tier 2 deliverable.

### Note
`Manifest.toml` is gitignored; the tracked dependency change is in `Project.toml`.
Anyone pulling this branch must re-run `Pkg.instantiate()`.

## Tier 2 — per-point compaction (`compact=true`) — DONE

Adds a `compact::Bool=false` kwarg to `ihlpsa_adaptive`. With `compact=true`, after
each chunk the converged grid points are pruned and only the still-active remainder
is re-launched, so the active set shrinks and total work scales with grid
heterogeneity instead of the slowest point.

The two bottlenecks the design doc's Tier 2 sketch left unsolved are handled:
- **Multi-device collapse.** The doc reshaped the active set to `N×1`, which
  would put the whole active set on one device (and, before the dispatch fix
  below, BoundsError outright). The worker packs the active points into a
  `(nrows × ncols)` sub-grid with `ncols = min(ndev, nact)`: full device spread
  whenever the active set is large enough, and `nrows = 1` with zero padding in
  the convergence tail (`nact < ndev`). `ndev` is
  `length(devices(backend))`/`length(devs)` on GPU, `1` on CPU. (Relies on the
  dispatch fix below; an earlier interim workaround pinned `ncols = ndev`.)
- **Index choreography across the transpose.** Everything is done in flat
  column-major index space of the internal `(nx,ny)` grid. The pack fills
  `vec(zg_sub)` column-major from `zg_flat[idx]` (trailing pad slots replicate an
  active z, results discarded); ihlpsa returns `permutedims(sr_grid)`, so
  `vec(permutedims(S))` recovers column-major order matching `vec(zg_sub)` and the
  first `nact` entries map back through `idx`. Output is
  `permutedims(reshape(σ_out, (nx,ny)))`, matching ihlpsa's convention.

Correctness rests on per-point Lanczos being independent across the batch (same
fixed `x₀`): compaction changes *which* points are computed, never a point's value.
So Tier 2 == Tier 1 within the convergence tolerance — asserted directly in tests.

### Files changed (Tier 2)
- `src/ihlpsa.jl` — `compact` kwarg on `ihlpsa_adaptive`; new
  `_ihlpsa_adaptive_compact` worker.
- `test/test_consistency.jl` — `ihlpsa_adaptive compaction (Tier 2)` CPU testset
  (compact ≈ SVD oracle AND compact ≈ non-compact, standard + B≠I, F32+F64);
  `test_adaptive_backend` extended with a `compact=true` cross-backend check.
- `bench/bench_adaptive.jl` — added a `compact` column (fixed vs global vs compact).

### Verification (Tier 2)
- CPU suite passes, no regressions. `ihlpsa_adaptive compaction (Tier 2)` 8/8. The
  `compact ≈ global` assertion validates the pack/back-map across the transpose.
- Bench (CPU, Grcar, 40×40 grid, `nit_max = 4·ceil(log2 m)`):
  - m=64:  fixed 0.58s · global 1.15s (0.50×) · **compact 0.41s (1.40×)**
  - m=128: fixed 1.93s · global 5.20s (0.37×) · **compact 1.89s (1.02×)**
  Compact beats both fixed and global; the absolute win is modest because Grcar is
  the hard case (much of the grid stays active) and `nit_max` is capped tight. The
  headline takeaway: compaction recovers the chunk-restart overhead that the Tier 1
  global driver wastes (~2.7× faster than global at m=128).

## Multi-device dispatch fix — DONE

Fixes a pre-existing bug in `ihlpsa`'s GPU dispatch (predates the adaptive
work; surfaced while reviewing it). The grid is split across devices by
columns, but the old ceil-based partition

```julia
Iterators.partition(1:ncols, ceil(Integer, ncols / ndev))
```

can yield FEWER blocks than devices (1 col/4 devs → 1 block; 9 cols/4 devs →
blocks of 3,3,3 = 3 blocks), while both device loops (per-device zpd resolution
and the spawn fan-out) iterate ALL of `devs` and index `zgidxbatches[did]` —
a `BoundsError` on any multi-GPU box for many grid widths.

The fix (verified host-side; there is no GPU CI):
- New pure helper `_device_column_partition(ncols, ndev)`: balanced contiguous
  partition into exactly `min(ndev, ncols)` blocks, sizes differing by ≤1.
  Also improves load balance (9 cols/4 devs: 3,2,2,2 on four devices instead of
  3,3,3 on three).
- Both device loops now iterate `zip(devs, zgidxbatches)` — zip truncates to
  the shorter side, so surplus devices are simply never touched (no `device!`,
  no memory query). `devs` is only iterated, never indexed: required because
  `CUDA.devices()` is a non-indexable `DeviceIterator` (AMDGPU → vector,
  oneAPI → collectable list, Metal → `[Metal.device()]`).
- `results` is sized by block count; column order is preserved (ordered
  contiguous blocks + ordered `hcat`). Progress description now reports
  "used/total device(s)". The sequential-before-fan-out zpd resolution
  (process-global `device!`) is preserved.
- Tier-2 compaction workaround relaxed accordingly (see Tier 2 section).
- `test_adaptive_backend` gained a `types` kwarg (mirrors `test_cross_backend`)
  and is now wired into the oneapi and metal runtests blocks with the
  FP64-gated `Ts` — previously it hardcoded ComplexF64 and would fail on
  FP64-less devices (Intel iGPU, Apple).

### Files changed (dispatch fix)
- `src/ihlpsa.jl` — `_device_column_partition` helper; dispatch rewrite
  (zip-based device loops); compact-worker `ncols = min(ndev, nact)`.
- `test/test_edge_cases.jl` — `_device_column_partition invariants` testset:
  sweep ncols ∈ 1:20 × ndev ∈ 1:6 (block count, coverage/order/disjointness,
  balance) + pinned regression cases for the verified old-partition failures.
- `test/test_consistency.jl` — `types` kwarg on `test_adaptive_backend`.
- `test/runtests.jl` — `test_adaptive_backend(...; types=Ts)` in oneapi/metal
  blocks.

### Verification (dispatch fix)
- CPU suite: all pass, `_device_column_partition invariants` 485/485, all
  adaptive testsets unchanged (CPU is `ndev = 1`, where old and new dispatch
  coincide — the multi-device behavior is validated by the partition
  invariants + the zip-truncation reasoning, and by the GPU testsets when run
  on hardware).
- Bench smoke at m=64: behavior unchanged.
- **Intel GPU hardware run** (`test/runtests.jl oneapi`, Raptor Lake-P UHD
  iGPU, FP64-less → F32-only via the `supports_fp64` gate): all pass —
  `ihlpsa parter16` 1/1 (normed error 3.9e-7, same magnitude as CPU),
  `cross-backend` 1/1, **`ihlpsa_adaptive: CPU vs oneAPIBackend` 3/3** (Tier 1
  values + converged-nit parity + Tier 2 compact cross-backend), KATRSM
  kernels 90/90. This exercises the rewritten dispatch's single-device GPU
  path end-to-end and the full adaptive feature on Intel hardware; the
  multi-device path (>1 GPU) remains validated host-side only.
- `bench/bench_adaptive.jl` gained a `T=ComplexF64` kwarg so FP64-less GPUs can
  run it in F32 (`bench(oneAPIBackend(); T=ComplexF32)`).

### First GPU bench numbers (Intel UHD iGPU, F32, Grcar 40×40 grid)
- CPU F32:  m=64 fixed 0.40s · global 0.61s (0.66×) · compact 0.33s (1.21×);
  m=128 fixed 1.84s · global 4.63s (0.40×) · compact 1.62s (1.13×).
- iGPU F32: m=64 fixed 0.33s · global 0.78s (0.43×) · **compact 0.71s (0.47×)**;
  m=128 fixed 0.50s · global 1.54s (0.33×) · **compact 1.16s (0.43×)**.
- Takeaway: on GPU, adaptive (both tiers) is currently a wall-clock LOSS at
  these sizes. The GPU finishes the fixed-`nit` kernel work so fast (m=128:
  0.50s vs CPU 1.84s) that the adaptive drivers' per-chunk fixed costs dominate:
  each chunk is a fresh `ihlpsa` call (device sync, `findmaxbatchihl`'s
  reclaim+memory query, workspace alloc, restart-from-iter-1 redundancy), and
  compaction shrinks batch sizes, cutting GPU occupancy. The feature's GPU value
  today is auto-`nit`, not speed. This is the concrete motivation for Tier 3
  (resumable lockstep: amortize the restart) and for re-benching at larger m /
  denser grids where kernel time dominates launch overhead.

### Follow-up (post-merge)
- Update CLAUDE.md's "Multi-device & batching model" section in the main
  checkout once this branch merges (it describes the old ceil-based split).

## Tier 3 — resumable lockstep (`resumable=true`) — DONE

Eliminates the per-chunk Lanczos restart. Key realization: **no kernel changes
were needed** (the design doc anticipated "kernel-arg plumbing + a stash
struct"). The per-point state between iterations (q_{n-1}, q_n, v, β_{n+1})
already lives in the `IHLworkspace` (`Qv[1]`, `Qv[2]`, `v`) and the `β` array;
`lockstep_ihl!`'s loop body at iteration n reads only those. So:

- `lockstep_ihl!` gained a `start::Integer=1` kwarg — seeds q₁ from x₀ only
  when `start == 1`, otherwise continues from the resident state. Resume
  contract documented at the function: same workspace instance, same α/β
  arrays (sized for the full `nit_max` budget up front), same batch occupancy.
- `_sdihlpsa_adaptive_resumable` mirrors `sdihlpsa`'s batch structure with the
  chunk loop INSIDE each batch: state stays resident, each chunk runs
  `start = nit_done+1 : nit_new`, σ is recomputed from the leading
  coefficients, and the batch stops when ALL its points converge. Total kernel
  work per batch = one fixed-`nit` run at the converged depth — optimal for
  lockstep. (Workspace reuse across batches is safe exactly as in `sdihlpsa`:
  iteration 1 multiplies stale Qv[1] by the fresh batch's β[1] = 0.)
- `_ihlpsa_adaptive_resumable` multi-device dispatcher: same
  `_device_column_partition` + zip pattern as `ihlpsa`; each device converges
  at its OWN depth (devices over easy regions retire early). `nit_used` =
  deepest across devices.
- API: `ihlpsa_adaptive(...; resumable=true)`. Mutually exclusive with
  `compact` (compaction re-packs points between launches, invalidating the
  resident state) — `ArgumentError` if both. Added to `@compile_workload`.

### Files changed (Tier 3)
- `src/ihlpsa.jl` — `start` kwarg on `lockstep_ihl!`;
  `_sdihlpsa_adaptive_resumable`; `_ihlpsa_adaptive_resumable`; `resumable`
  kwarg + exclusivity check on `ihlpsa_adaptive`.
- `src/KAPseudospectra.jl` — resumable call in `@compile_workload`.
- `test/test_consistency.jl` — `ihlpsa_adaptive resumable (Tier 3)` testset:
  on CPU resumable reproduces Tier 1's EXACT convergence path (`nit_used`
  equal, σ ≈ to 1e-12 F64 / 1e-6 F32 — same flop sequence, only bookkeeping
  differs), plus SVD-oracle, B≠I, and kwarg-exclusivity checks;
  `test_adaptive_backend` extended with a resumable cross-backend check.
- `bench/bench_adaptive.jl` — resumable column.

### Verification (Tier 3)
- CPU suite: all pass, Tier 3 testset 9/9 (incl. the strict `n_res == n_global`
  path-identity assertion), zero regressions.
- Intel iGPU (`runtests.jl oneapi`, F32): all pass —
  `ihlpsa_adaptive: CPU vs oneAPIBackend` now 5/5 (resumable values match CPU,
  converged nit within one chunk).
- Bench (F32, Grcar 40×40): **resumable fixes the GPU wall-clock loss**:
  - iGPU m=64:  fixed 0.33s · global 0.41× · compact 0.46× · **resumable 0.32s
    (1.05×)** while auto-finding nit=18 < the 24 budget.
  - iGPU m=128: fixed 0.50s · global 0.31× · compact 0.44× · **resumable 0.52s
    (0.96×)** — the grid needs the full cap (nit=28), so resumable does the
    same kernel work as fixed with only ~4% checkpoint overhead.
  - CPU m=64: resumable 1.24× (best); CPU m=128: compact 1.17× vs resumable
    1.01× — when many points converge early, pruning (Tier 2) wins on CPU;
    when launch overhead dominates (GPU), state residency (Tier 3) wins.
- Net: adaptive `nit` is now effectively FREE on GPU (parity with a
  perfectly-tuned fixed `nit`, without having to know it in advance).

## Stress testing at scale — DONE

Changes:
- The resumable CPU path now honors a user-supplied `zpd` (its batch loop is
  sequential, so CPU batching is safe — unlike `sdihlpsa`, whose ThreadsX
  batches would race on the shared workspace). Enables memory-capping and
  multi-batch testing on CPU.
- New permanent test: forced multi-batch resumable (`zpd=37` → 4 batches incl.
  a partial one) vs single-batch (Tier 3 testset now 11/11).

Findings (Grcar, F32):
- **Multi-batch correctness on hardware**: on the iGPU, resumable with
  `zpd=137` (12 batches) is **bitwise identical** (max rel diff 0.0) to
  single-batch resumable AND to Tier 1 global — the state-residency logic is
  exact.
- **Budget-bound regime** (`nit_max = nit_fixed`, every path runs to the cap —
  adaptive's worst case; only a tail of 13–127 points is unconverged):
  - iGPU m=256, 80×80: resumable 0.98× · compact 0.83× · global 0.39×
  - iGPU m=128, 128×128 (16k pts): resumable 0.90× · compact 0.88× · global 0.38×
  - iGPU m=512, 48×48 (interleaved repeats): resumable 0.99×
  - CPU m=256, 48×48: resumable 1.00× · compact 1.01× · global 0.40×
  Resumable's overhead is ≤2% at scale even when adaptivity buys nothing.
- **Convergence-bound regime** (generous budget = driver default territory):
  iGPU m=256, 80×80, budget 64: resumable converges at nit=48 →
  **1.27× faster than fixed(64)** (mechanistic: 25% less kernel work);
  compact 0.97×.
- **Measurement hygiene**: single-shot iGPU timings swing up to ±35% (clock
  ramp/thermal) — an initial m=512 run showed a spurious 1.35× for resumable
  at the same nit as fixed (no mechanism: single batch, same kernel work);
  interleaved repeated timing corrected it to 0.99×. GPU bench claims need
  interleaved repeats.
- The bench's `nit_max = 4·ceil(log2 m)` is too tight for Grcar at F32
  rtol=1e-4 (hard-point tail needs more); the driver's default
  `8·ceil(log2 m)` is the right ceiling.

## Hybrid (compact + resumable) — DONE — the headline speedup

Diagnosis that motivated it (user pushback: "no speedup — what are we doing?"):
- **Convergence census** (chunk=1 sweep recording each point's first-converged
  iteration; Grcar 80×80): at m=128/F32, 57% of points converge in ≤3
  iterations, 70% ≤5, 85% ≤8, tail to ~16–24; m=256 similar (60% ≤3, tail to
  32). Average needed work ≈ 5–6 iterations vs the 28–32 every lockstep path
  spends. The design doc's premise confirmed empirically.
- Why Tiers 1–3 showed no speedup: (a) retirement requires two consecutive
  chunk checkpoints and `nit_chunk` defaulted to log2(m)≈8–9, flooring every
  point at ≥16 iterations; (b) resumable is lockstep — scattered slow points
  keep whole batches alive; (c) compact retires points but restarts from
  iteration 1 each chunk. The promised win needs per-point retirement AND
  continuation AND small chunks simultaneously.

Implementation:
- `compact=true, resumable=true` now selects the hybrid (the prior
  ArgumentError exclusivity is removed — the rationale dissolved once state is
  gathered along with the active set). Default `nit_chunk` for the hybrid is 2
  (retirement floor = 2·nit_chunk); other modes keep log2(m).
- `_sdihlpsa_adaptive_hybrid`: per batch, after each chunk the converged
  points retire and survivors' state — workspace rows (Qv batch-major, v/x₀
  batch-minor, zv) plus α/β columns — is gathered to a packed prefix with
  plain array indexing (GPUArrays fancy indexing; **no custom kernels**).
  Gather traffic O(m·survivors) per chunk vs O(nit_chunk·m²·survivors) solve
  work avoided. `idx_glob` maps packed position → original flat grid index.
- The Tier-3 multi-device dispatcher was generalized to
  `_ihlpsa_adaptive_resident(sdworker, label, ...)` shared by resumable and
  hybrid; per-device independent stopping as before.

Verification:
- CPU suite all pass: new hybrid testset 10/10 (vs SVD oracle, vs Tier 1,
  B≠I, forced multi-batch `zpd=37`); Tier 3 testset updated (exclusivity test
  removed). iGPU suite all pass: cross-backend 6/6 incl. the on-device gather.
- Hybrid matches the SVD oracle to ~3e-11 (F64 CPU smoke).

Benchmarks (Grcar F32, generous budget = realistic "don't know nit" usage,
interleaved repeats, min-of-2; "oracle" = fixed at the converged nit, i.e. a
user with perfect foresight):
- iGPU m=128, 128×128, budget 56: fixed(budget) 4.57s · oracle(28) 2.37s ·
  resumable 1.48× · compact 1.58× · **hybrid 2.47s = 1.85×, matching the
  oracle without knowing nit** (launch-overhead-bound at this m).
- iGPU m=256, 100×100, budget 64: fixed(budget) 13.59s · oracle(32) 7.90s ·
  resumable 1.21× · **hybrid 4.31s = 3.15× vs budget, 1.83× vs the oracle**.
- CPU m=128, 128×128, budget 56: fixed(budget) 12.95s · oracle(28) 6.54s ·
  **hybrid 1.72s = 7.51× vs budget, 3.8× vs the oracle** — the design doc's
  ~8× estimate realized.
The hybrid is now the recommended adaptive mode; resumable remains the
zero-risk auto-nit knob, global/compact are kept as simpler references.

## Accuracy across the grid (hybrid vs dense SVD oracle, Grcar m=128 80×80)

- The adaptive paths stop at the STOPPING tolerance, not machine precision —
  by design. F64 default rtol=1e-6: max rel error over all contour-relevant
  points (σ > 1e-8) is **1.3e-7** (≈10× under rtol; successive-difference
  stopping over-converges), median 7.7e-13. Tightening to rtol=1e-10 pushes
  the median to 6.0e-15 ≈ the deep fixed-nit=96 floor (5.4e-15).
- The algorithm floor (fixed nit=96, no adaptivity): 2.8e-14 at σ>1e-2,
  degrading to ~5e-9 at σ>1e-8 — relative accuracy of σ_min inherently decays
  as σ→0 (absolute accuracy ~eps·‖A‖, standard perturbation behavior). F32
  floor: ~1e-5..3e-4 (F32 Lanczos roundoff; matches test tolerances).
- Raw max/p99 rel errors over the WHOLE grid look alarming (O(1) F64, O(1e6)
  F32) but are identical for deep fixed Lanczos — they live entirely at
  near-eigenvalue points (σ < 1e-10) where relative error is meaningless and
  log-contour plots are unaffected. Not an adaptive artifact.
- Rule of thumb: error ≈ rtol/10 on contour-relevant points; rtol is the
  accuracy knob, floor is the fixed-nit Lanczos floor.

### `nconfirm` confirmation streak (premature-retirement fix)

The successive-difference criterion can retire a point on a slow-convergence
plateau (|σ_k − σ_{k−1}| small while |σ_k − σ_∞| is not). Fix: a point (or the
grid/batch, per mode) must pass the criterion at `nconfirm` consecutive
checkpoints before stopping. New `nconfirm::Integer=2` kwarg on
`ihlpsa_adaptive`, implemented in all four modes (Tier 1 global streak;
compact per-point streak array; resumable per-batch streak; hybrid per-packed-
point streak, gathered with survivors). `nconfirm=1` restores the old
single-check behavior.

Measured effect (Grcar m=128, 80×80, vs ℂsvdpsa):
- **F64: max rel error on contour-relevant points 1.32e-7 → 9.4e-9 (14×
  better, ≈ rtol/100)**, meeting the deep fixed-nit=96 floor at the σ>1e-8
  stratum (4.6e-9). Cost: deepest nit 28 → 30.
- F32: 8.8e-4 → 8.4e-4 — unchanged, because the F32 max is NOT premature
  stopping: F32 kernel roundoff makes σ itself wobble at ~rtol=1e-4 scale, so
  the stopping diff cannot resolve below that noise (fixed-96 floor is 3e-4
  at σ>1e-4). F32 accuracy is method-floor-limited; only F64 kernels would
  improve it.
- Speedups retained at slightly reduced levels (≈2 extra iters/point):
  CPU m=128 128×128: 4.70× vs fixed(budget) / 2.85× vs oracle (was 7.51/3.8);
  iGPU m=256 100×100: 2.29× vs budget / 1.17× vs oracle (was 3.15/1.83).
  The accuracy/speed trade is now an explicit knob.

## Deferred / future
- Re-bench the hybrid at larger m / denser grids on discrete GPUs
  (CUDA/AMDGPU) — the multi-GPU box (Todoist, June 10) is the opportunity.
  Single-shot iGPU timings swing ±35% (clock ramp); use interleaved repeats.
- Possible hybrid refinement: stop gathering below a small active-set
  threshold (launch-overhead-bound tail) and just run the stragglers to
  convergence in one final chunk of size nit_max − nit_done.

## Consolidation — hybrid-only, merged into `ihlpsa` — DONE

Benchmarking (incl. a 3× K40m box, 500×500 grids) confirmed only the **hybrid**
(per-point retirement + resident state) is worth keeping; Tiers 1–3 never beat a
hand-picked fixed `nit` and only added flag-combinatorics. So the public surface
was collapsed to a single function:

- **Removed:** `ihlpsa_adaptive` (export + function), the `compact`/`resumable`
  kwargs, and the now-dead workers `_ihlpsa_adaptive_compact` (Tier 2),
  `_sdihlpsa_adaptive_resumable` (Tier 3), and the Tier-1 whole-grid loop.
- **Kept, renamed:** the hybrid worker `_sdihlpsa_adaptive_hybrid` →
  `_sdihlpsa_adaptive`; the multi-device dispatcher `_ihlpsa_adaptive_resident`
  → `_ihlpsa_adaptive` (specialized to the one worker — no `sdworker`/`label`).
- **`ihlpsa` is now two methods** sharing the engine: fixed
  `ihlpsa(b,zg,P,nit::Integer,γ,δ;…)` (forwards to the extracted `_ihlpsa_fixed`,
  returns `Matrix`) and adaptive `ihlpsa(b,zg,P; …)` (omit `nit` ⇒ per-point
  hybrid, returns `(σ, nit_used)`; `γ`,`δ` keyword). Arity dispatch: the fixed
  method dropped its default `nit`, so arity-3 unambiguously selects adaptive.
- **Tests/bench/precompile** updated: tier-specific testsets in
  `test_consistency.jl` deleted (the bare adaptive call now *is* the hybrid);
  `bench_adaptive.jl` reduced to fixed-vs-adaptive; GPU ext `@compile_workload`s
  now also precompile the adaptive path (first adaptive GPU call no longer cold);
  a real docstring added to `ihlpsa`, README gained a Usage section.
