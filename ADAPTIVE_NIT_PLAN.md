# Adaptive `nit` for `ihlpsa` — design plan

## Problem

`ihlpsa` runs lockstep inverse Lanczos: every grid point in a batch executes
the same `nit` iterations regardless of how fast σ_min has converged at that
point. In practice convergence rate varies enormously across a pseudospectra
grid:

- Near eigenvalues, σ_min is well-isolated from σ_2nd → 1–2 iterations suffice.
- Mid-pseudospectrum, σ_min ≈ σ_2nd within ~1% → tens of iterations.
- The slowest point dictates `nit` for the entire grid.

For a typical Grcar at m=128 on a 16×16 grid, the worst point needs ~28 iters
for 1e-10 precision while ~80% of the grid converges in ≤4. Fixed-`nit` calls
do roughly **5–10× more work than necessary** on the easy majority of points.

## Goal

Replace the user-supplied `nit` with an `ihlpsa_adaptive` driver that runs
lockstep Lanczos in chunks, checks convergence, and only continues iterating
on points that are still moving. No kernel changes required.

## Recommended architecture: two tiers

Both tiers live as host-side wrappers around the existing `ihlpsa` /
`sdihlpsa` / `lockstep_ihl!` stack. No GPU kernel modifications.

---

### Tier 1 — Global adaptive `nit` (easy)

Run `ihlpsa` on the whole grid in chunks of `nit_chunk`. After each chunk,
compare the result to the previous chunk; stop when the **maximum** per-point
relative change drops below `rtol`.

```julia
function ihlpsa_adaptive(backend, zg, P;
                        nit_chunk=8, nit_max=128, rtol=1e-6,
                        x₀=missing, kwargs...)
    σ_prev = nothing
    nit_total = 0
    while nit_total < nit_max
        nit_new = nit_total + nit_chunk
        σ_new = ihlpsa(backend, zg, P, nit_new; x₀, kwargs...)
        if σ_prev !== nothing &&
           maximum(abs.(σ_new .- σ_prev) ./ abs.(σ_new)) < rtol
            return σ_new, nit_new
        end
        σ_prev = σ_new
        nit_total = nit_new
    end
    @warn "ihlpsa_adaptive did not reach rtol=$rtol within nit_max=$nit_max"
    return σ_prev, nit_max
end
```

**Cost:** approximately 2× the work of a single fixed-`nit` call to converge
(one redundant chunk for the convergence check). Each chunk re-runs Lanczos
from iter 1 — wasteful, but the total `nit` is bounded by the slowest point.

**Pros:** ~30 lines, no workspace surgery, identical results to fixed-`nit` at
the converged value, ships today.

**Cons:** still pays the slowest-point cost on the whole grid. The fast 95%
of points get re-iterated for the slow 5%.

**Use case:** "I don't know what `nit` to pick" — an `auto-nit` knob.

---

### Tier 2 — Per-point adaptive with host-side compaction (medium)

Same idea, but after each chunk, **prune** the converged points from the
active set and re-launch only on the remainder. This is where the real
speedup lives — the active set shrinks geometrically as easy points retire.

```julia
function ihlpsa_adaptive_compact(backend, zg, P;
                                 nit_chunk=8, nit_max=128, rtol=1e-6,
                                 x₀=missing, kwargs...)
    n = length(zg)
    σ_out = zeros(real(eltype(zg)), n)
    σ_prev = zeros(real(eltype(zg)), n)
    active = trues(n)
    zg_flat = vec(zg)
    nit_used = 0
    while any(active) && nit_used < nit_max
        idx = findall(active)
        zg_sub = reshape(zg_flat[idx], :, 1)
        nit_new = nit_used + nit_chunk
        σ_sub = vec(ihlpsa(backend, zg_sub, P, nit_new; x₀, kwargs...))
        # convergence check, per-point
        rel = abs.(σ_sub .- σ_prev[idx]) ./ abs.(σ_sub)
        for (k, ai) in enumerate(idx)
            σ_prev[ai] = σ_sub[k]
            if rel[k] < rtol
                σ_out[ai] = σ_sub[k]
                active[ai] = false
            end
        end
        nit_used = nit_new
    end
    σ_out[active] .= σ_prev[active]   # cap reached, accept current value
    return reshape(σ_out, size(zg))
end
```

**Cost model.** If `f(k)` is the fraction of points still active after `k`
chunks, total work ≈ `nit_chunk * Σ f(k) * gtotal`. For Grcar on a typical
square grid the geometric series is ≈ `1 + 0.5 + 0.2 + 0.05 + … ≈ 2`, so total
work ≈ `2 * nit_chunk * gtotal`. Versus the fixed-`nit_max * gtotal` cost,
that's a **~`nit_max / (2 * nit_chunk)`× speedup** (≈ 8× for
`nit_max=128, nit_chunk=8`).

**Bonus:** as the active set shrinks, `findmaxbatchihl` will pick a larger
zpd → more parallelism per device per kernel launch, partially compensating
for the smaller workload per chunk.

**Pros:** real wall-clock speedup proportional to grid heterogeneity. Drop-in
replacement for `ihlpsa` from the user perspective.

**Cons:**
- Each chunk re-runs Lanczos from iter 1 on the active subset (the
  simplification that lets us avoid workspace state surgery). Net redundant
  work per surviving point: roughly `nit_used * (k_chunks + 1) / 2`. Bounded.
- `idxb` slicing into `vec(zg)` and the back-mapping into `σ_out` add per-call
  overhead — small relative to the kernel cost above ~m=64.

---

### Tier 3 (optional / later) — resumable lockstep

Modify `lockstep_ihl!` to take `start_iter` and continue from saved
`Qv[1], Qv[2], v, β_prev`. Eliminates the per-chunk re-run cost. Adds ~50
lines of kernel-arg plumbing and a stash struct. Worth doing only after
Tier 2 lands and we measure the chunk-restart overhead in production.

---

## API

Surface both tiers as exported functions, plus keep `ihlpsa(..., nit)` for
backward compat:

```julia
ihlpsa(backend, zg, P, nit::Integer, ...)             # existing fixed-nit
ihlpsa_adaptive(backend, zg, P; ...)                  # Tier 1
ihlpsa_adaptive(backend, zg, P; compact=true, ...)    # Tier 2 (single entry)
```

Default kwargs: `nit_chunk=8, nit_max=2 * ceil(Int, log2(m)), rtol=1e-6`.
The `nit_max` default is conservative — the reasoning: most points converge
in O(log m) iterations even on hard problems; budget 2× that for safety.

---

## Implementation order

1. **Tier 1** in `src/ihlpsa.jl` (≈ 30 lines, one new function, no struct
   changes). Add export in `src/KAPseudospectra.jl`.
2. **Tests** in `test/test_consistency.jl`:
   - `ihlpsa_adaptive(P; rtol=1e-6) ≈ ihlpsa(P; nit=64)` to within `rtol`.
   - `nit_returned ≤ ihlpsa(P; nit=64)`'s required nit.
   - F32 + F64, CPU + AMDGPU.
3. **Tier 2** (≈ 100 lines, in same file). Add `compact=true` kwarg to
   `ihlpsa_adaptive`.
4. **Bench** in `bench/bench_adaptive.jl`: compare fixed `nit=64` vs Tier 1
   vs Tier 2 across m ∈ {64, 128, 256, 512} on Grcar; report wall-time
   speedup and total Lanczos iterations performed.
5. **Documentation**: docstring on `ihlpsa_adaptive` explaining the
   convergence criterion and when it's the wrong choice (e.g. when the user
   really does want a fixed iteration budget for benchmarking).

---

## Open questions

- **Convergence criterion.** Relative change of σ between chunks works for
  pseudospectra plotting (we care about contour values, ε ≥ 1e-5 typically).
  An alternative is to use the off-diagonal β at iter k to bound the eigmax
  error: a converged tridiagonal has β_k → 0. Cheaper to compute than re-doing
  eigmax. Worth comparing; the current proposal uses re-running eigmax which
  is what `ihlpsa` already does.
- **F32 numerics.** `rtol=1e-6` is right at the F32 precision floor. For F32
  the criterion should probably loosen to `rtol=1e-4` by default. Plumb the
  default off `eltype(zg)`.
- **Pencil with B ≠ I.** Adaptive logic is identical (the σ values returned
  are γ + δ|z| times σ_min). Just verify the test uses a non-trivial pencil.
- **Multi-device.** The current `ihlpsa` already partitions zg across devices.
  The adaptive driver runs on top of `ihlpsa` so it inherits multi-device
  routing for free; the active-subset slicing happens before the device fan-out.

---

## Status

- Textbook Lanczos parity (the prerequisite that revealed the adaptive `nit`
  motivation): **done**, in `src/core.jl` (added `Z` field to
  `SchurMatrixPencil`) and `src/ihlpsa.jl` (`x₀ ← P.Z' * x₀` in the
  `IHLworkspace` constructor when user supplies `x₀`).
- Adaptive `nit`: **planned**, this document.
