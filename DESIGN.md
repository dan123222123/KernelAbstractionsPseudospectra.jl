# Adaptive `nit` — design notes

Why `ihlpsa`'s adaptive iteration depth works the way it does. For usage see the
`ihlpsa` docstring and the README; for the API change see the CHANGELOG. The
feature was explored as four tiers (global / compact / resumable / hybrid); only
the per-point hybrid survived consolidation — `git log` and
`git show e763b8a:ADAPTIVE_NIT_PLAN.md` / `git show <sha>:PROGRESS.md` recover the
full exploration and its intermediate benchmarks.

## Why adaptive iteration depth

`ihlpsa` finds σ_min(zB − A) at each grid point via inverse Lanczos on the
largest eigenvalue of `[(zB−A)(zB−A)ᴴ]⁻¹` (= 1/σ_min²). Its convergence rate is
set by the **relative gap between σ_min and σ_2nd**:

- Near an eigenvalue, σ_min is well-isolated → 1–2 iterations suffice.
- Far from the spectrum, `zB−A ≈ zB`, so the singular values cluster near `|z|`
  and σ_min ≈ σ_2nd → tens of iterations.

A fixed `nit` is therefore dictated by the slowest grid point, massively
over-iterating the easy majority. A convergence census (Grcar 80×80, m=128/F32)
finds ~57% of points converge in ≤3 iterations, ~85% in ≤8, with a tail to
~16–24 — average needed work ≈ 5–6 vs the ~28–32 a lockstep run spends. Adaptive
depth retires each point at its own converged depth.

## Why the per-point hybrid (not the simpler variants)

Capturing that win requires three things **simultaneously**; any one or two alone
is roughly wall-clock-neutral:

- **per-point retirement** — stop iterating points that have converged (a
  whole-grid re-run keeps paying the slowest point);
- **resident state / continuation** — keep each point's Lanczos state and continue
  (restarting from iteration 1 each chunk wastes the early iterations);
- **small chunks** — retirement can't happen before ≈ `nconfirm · nit_chunk`
  iterations, so large chunks floor every point (default `nit_chunk = 2`).

The shipped driver does all three: after each chunk the converged points retire
and the survivors' resident state (workspace rows + α/β columns) is gathered to a
packed prefix with plain array indexing (GPUArrays fancy indexing — no custom
kernels), so subsequent chunks touch live points only.

## Convergence criterion & accuracy

A point retires when its σ stops changing between chunks —
`|Δσ| ≤ atol + rtol·|σ|`, confirmed over `nconfirm` consecutive checkpoints (a
single small step can be a slow-convergence plateau, not convergence). An
eps-sentinel short-circuits points pinned to `eps` at true eigenvalues. A fixed
deterministic `x₀` is reused across chunks, so successive σ are one Lanczos run
sampled at increasing depth (a fresh random `x₀` per chunk would compare
independent runs and make the criterion meaningless).

- Adaptive stops at the **stopping tolerance, not machine precision** — by design.
  Rule of thumb: error ≈ `rtol/10` on contour-relevant points (`σ ≳ 1e-8`).
  Tighten `rtol` (with `nit_max` headroom) for more accuracy.
- F64 default `rtol = 1e-6` lands ~`1e-8`…`1e-7` vs a dense-SVD oracle on contour
  points. F32 default `rtol = 1e-4` is method-floor-limited: F32 Lanczos roundoff
  makes σ itself wobble at ~`1e-4`, so the stopping difference can't resolve below
  that — only F64 kernels would improve F32 accuracy.
- Relative accuracy of σ_min inherently decays as σ → 0 (absolute accuracy
  ~`eps·‖A‖`, standard perturbation behavior). This is identical for deep
  fixed-`nit` Lanczos and irrelevant to log-contour plots, where those points sit
  off the bottom of the colour scale.
