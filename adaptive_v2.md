# Adaptive `nit` — v2 thoughts (post-PR)

Follow-on ideas for the adaptive `ihlpsa` driver, captured while the v1 PR is in
flight. Neither is blocking; both are deliberately deferred until after merge.
For the shipped design see `DESIGN.md`; for the engine see `src/ihlpsa.jl`
(`_ihlpsa_adaptive`, `_sdihlpsa_adaptive`, `_ihlpsa_fanout`,
`_device_column_partition`).

## 1. Progress reporting for the adaptive form (aesthetic)

The fixed-`nit` path takes `progress=true` and shows a ProgressBars.jl bar whose
total is `nit · length(zg)` — i.e. total Lanczos-iteration work, which is known up
front. The adaptive form can't reuse that: per-point depth isn't known ahead of
time, so there's no fixed denominator. Today it offers only `verbose=true`, which
`@info`-logs the deepest depth *after* the run.

The natural live quantity for adaptive is **retired (converged) grid points vs.
total** — a bar that fills as points retire, draining toward zero survivors. The
hook already exists: `_sdihlpsa_adaptive` knows `length(keep)` (survivors) after
every chunk, so after each chunk it could push `gtotal - survivors` to a progress
channel, mirroring the fixed engine's `pchnl`/`pbar` plumbing in `_ihlpsa_fixed`.
Two wrinkles: (a) under the multi-device fan-out each device runs its own loop, so
the bar would need to aggregate retirements across the `@sync`'d workers (one
shared channel, as the fixed engine already does); (b) the count isn't monotone in
"work" — late survivors are the expensive ones, so a points-retired bar races to
~90% quickly then crawls. That's arguably *informative* (it visualizes the hard
tail), but it's not a linear ETA. A secondary "deepest live depth" readout could
sit alongside it.

Purely cosmetic — no correctness or perf implication — so not before the PR.

## 2. Static round-robin partition vs. dynamic re-fan of survivors

**Current state.** Multi-device work is split once by `_device_column_partition`:
round-robin (strided) column→device assignment by default (`KAPSEUDO_STRIDED=0`
for legacy contiguous bands). Round-robin already decorrelates *spatial* clustering
— a hard region that would otherwise pile onto one device's contiguous band is
sprayed across all devices — so the naive "left half of the grid finishes before
the right half" imbalance from a block partition is largely handled. Each device
then runs its **own** adaptive loop and stops at its **own** deepest survivor;
there is no mid-run rebalancing.

**Where it can still be sub-optimal.** The partition is *static*. It balances
initial column counts and decorrelates difficulty, but:

- Survivor counts drift as the easy majority retires — after a few chunks two
  devices can hold very different numbers of unconverged points, and each device's
  wall-clock is bound by its own deepest survivor. The example in
  `examples/ihlpsa_adaptive.jl` (tight `rtol`, parter) is a good stress case: the
  hard region is concentrated, and even round-robin leaves per-device survivor
  sets uneven once most points retire. Devices that drew an easier mix go idle
  while one device grinds its stragglers.
- The decorrelation is statistical, not guaranteed — at low `ndev` or with a very
  localized hard region the spread can be lumpy.

**Proposed comparison (v2).** A *dynamic* scheme: every chunk (or every k chunks),
gather all survivors to the host, identify the still-unconverged set globally, and
re-fan **just those points** evenly across all devices. This *guarantees* balanced
survivor counts at each rebalance, so no device idles on another's tail.

**The catch — and why it needs measuring, not just adopting.** Re-fanning fights
the resident-state continuation that DESIGN.md calls a core pillar: the single-
device gather is cheap precisely because each survivor's Lanczos state (workspace
rows + α/β columns) stays put on its device. Redistributing survivors across
devices means **migrating that resident state across device memories** every
rebalance — exactly the traffic the in-place gather avoids. So v2 trades idle time
for migration + host-gather cost, and the balance is regime-dependent:

- **Integrated GPU (shared system RAM, e.g. this Intel iGPU):** migration is
  nearly free — likely a clear win, and the cleanest place to prototype/measure.
- **Discrete multi-GPU (PCIe / NVLink):** cross-device state copies are expensive;
  the rebalance must save more idle time than it spends moving state. Plausibly a
  loss at small `m` or frequent rebalancing; possibly a win at large `m` with a
  long hard tail and infrequent (every-k-chunk) rebalancing.

It also only matters for `ndev ≥ 2` — single-device runs are unaffected.

**Suggested experiment.** Add a `bench/` harness that runs the same grid under
(a) static round-robin and (b) dynamic re-fan at a few rebalance cadences, on both
the iGPU and a discrete multi-GPU box, reporting per-device wall-clock spread
(idle time) alongside total time and bytes migrated. The parter tight-`rtol`
example is a ready-made adversarial input. Decide the default from that curve, and
keep the mode behind an env flag like the existing `KAPSEUDO_STRIDED`.

## 3. Open PR-review items carried forward

Items from the PR #2 self-review that are genuine future work (the rest of the
review notes are already resolved in the branch — `γ`/`δ` keywords, the diagnostic
return now wrapped so `ihlpsa` yields only `srg`, the test reusing the package's
`_adaptive_x₀`, the device interface kept because KA's device addressing is ordinal-only
(`device`/`device!`/`ndevices`) and — through 0.10.0-dev / `main` — still lacks
device-memory queries, a backend-wide reclaim, the `adapt` array type, and
handle-based iteration; `supports_fp64` already delegates to KA's
`supports_float64`; `bench/Project.toml` tracked with all four backends, etc.):

- **Principled `nit_chunk` default (vs hardcoded 2).** A real auto-tune would
  balance a chunk's compute (∝ `nit_chunk · m² · survivors`) against the fixed
  host↔device round-trip, raising `nit_chunk` exactly when the round-trip
  dominates — ideally from device properties (compute-unit count, memory
  bandwidth) and `m`. `bench/nit_chunk_sweep.jl` already measures the curve;
  `DESIGN.md` reports the iGPU result (optimum drifts 2→1 as `m` grows). Blocked
  on a *discrete*-GPU sweep to confirm the opposite (latency-bound) regime before
  committing a model. Until then the default stays 2, kwarg-overridable.

- **Higher-precision lower-level pts in `KATRSM`.** Worth examining whether the
  pivoted triangular-solve inner loop runs reductions/accumulations at working
  precision on non-CPU backends, and whether that costs accuracy or perf on other
  backends (the review flagged it as likely sub-optimal beyond the path tested).
  Needs per-backend profiling + an accuracy comparison against the dense oracle.

- **Dynamic re-fan of survivors** — see §2 above.
