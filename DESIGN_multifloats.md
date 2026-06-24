# Extended-precision pseudospectra with MultiFloats.jl — design & exploration notes

These notes record an exploration of running KAPseudospectra's inverse-Lanczos engine
(`ihlpsa`) in **extended precision** (double-double / quad-double) via
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl), including on the GPU. See
the runnable examples `examples/ihlpsa_multifloats.jl` (demo) and
`examples/ihlpsa_multifloats_accuracy.jl` (oracle check).

## TL;DR

- `ihlpsa` is generic over the complex element type, and `MultiFloat{Float64,N}` is an
  isbits struct, so **extended-precision pseudospectra run on the GPU with no changes to the
  core/kernels** — only a generic Schur factorization (GenericSchur.jl) and a generic
  tridiagonal `eigen` (GenericLinearAlgebra.jl) are needed.
- One package change: the final σ extraction (`ihlsrg!`) now **follows the input precision**.
  It computes the Lanczos-tridiagonal `eigmax` with LAPACK for Float64 and with
  GenericLinearAlgebra's `eigen` for extended-precision types, so the output precision tracks
  the input type (~27 digits for `Float64x2`, ~59 for `Float64x4`). Float64/Float32 keep the
  exact LAPACK path.
- Validated against an **independent** BigFloat dense-SVD oracle (the package's own
  `ℂsvdpsa` at ~77 digits): double-double matches the oracle to its own precision, where
  plain Float64 is off by up to ~84% in the ill-conditioned band near the spectrum.
- Performance: double-double on the GPU costs ~2× (small/overhead-bound) up to ~8× (large,
  compute-bound) over Float64 — cheap for ~16 extra digits.
- The Float64-based types (`Float64x2`/`Float64x4`) are the right choice. Float32-based
  MultiFloats (`Float32x2`/`Float32x4`) are slower at matched precision and range-limited
  (~1e38), so they can't reach the small-σ regime — see the performance section.

## Motivation

For a strongly **non-normal** matrix the resolvent `(zB − A)⁻¹` is highly ill-conditioned
near the spectrum. A Float64 inverse-Lanczos solve then loses accuracy and reports the wrong
`σ_min(zB − A)` there — exactly where the interesting pseudospectral structure lives. Running
the solve in double-double keeps the Lanczos recurrence accurate and recovers the correct
levels.

## Why it works with no core changes

- The inverse-Lanczos kernels (`src/ihlpsa.jl`) and the KATRSM triangular-solve kernels
  (`src/KATRSM.jl/`) use only `+ - * / conj sqrt real abs2` — all provided by MultiFloats.
- `MultiFloat{Float64,N}` is an immutable isbits struct, so it passes into KA/CUDA kernels
  by value; arrays of `Complex{Float64xN}` adapt to `CuArray` element-wise.
- `KATRSM._pdiv` uses full-precision generic complex division for `Complex{Float64xN}` (it
  only special-cases `Complex{Float16/Float32}`). The pivoting/division strategy is also
  selectable at runtime via the exported `set_pdiv_accurate`.

## Getting a Schur form in extended precision

`MatrixPencil(A)` calls `schur(A)`. Loading `GenericSchur` (or `GenericLinearAlgebra`) makes
`schur(::Matrix{Complex{Float64x2}})` dispatch to a pure-Julia factorization returning a
`LinearAlgebra.Schur`, which flows straight into the existing `MatrixPencil(F::Schur)`
method. Any dense matrix then works in extended precision through the normal API. (For an
already-triangular matrix, skip schur and build the pencil directly with
`SchurMatrixPencil{T}(A, A', I, I)`.)

## Precision-following σ extraction — the `ihlsrg!` eigmax

`ihlsrg!` returns the `(γ,δ)`-pseudospectral value `σ = σ_min(zB − A) / (γ + δ|z|)`, with
`σ_min = 1 / √(eigmax(T_k))` from the small Lanczos tridiagonal `T_k`. Previously `T_k` was
promoted to **Float64** before the `eigmax`, so the returned σ was capped at Float64 accuracy
even when the solves ran in double-double.

It now selects a work type `R = promote_type(Float64, real(eltype(α)))` — a Float64 *floor*
that is a no-op for Float64 and every extended type (so those follow the input precision) and
only lifts the sub-Float64 types (an F32 tridiagonal near an eigenvalue has a λmax past F32's
~1e38 range). `_eigmax_tridiag` then dispatches:

- **Float64** → LAPACK `eigmax` (fast, unchanged, bit-identical results for existing users);
- **extended precision** → `maximum(eigen(SymTridiagonal(d, e)).values)` from
  GenericLinearAlgebra.

`eigen` rather than `eigmax`/`eigvals` for the generic path: near a true eigenvalue
`σ_min → 0`, so `λmax = 1/σ_min²` is large and the tridiagonal spans a wide dynamic range.
GenericLinearAlgebra's `eigen` (plain QL) resolves the extreme eigenvalue to ~machine-eps on
such matrices — e.g. on a synthetic Float64x2 tridiagonal spanning ~1e0…1e33, `eigen`'s λmax
has relative error ~7e-33. Its square-root-free `eigvals` (which `eigmax` calls) is less
reliable in that regime, so the generic path goes through `eigen`. The tridiagonal is tiny
(nit entries per grid point), so the eigenvectors `eigen` also returns cost nothing here.

## Validation against an independent oracle

Oracle: the package's own dense-SVD pseudospectra `ℂsvdpsa` (`σ_min = svdvals(zI − A)[end]`)
evaluated in **BigFloat / MPFR at 256 bits (~77 digits)** — a different algorithm *and* a
different precision library from the inverse-Lanczos method under test, reached through the
same public API. Matrix: `chebspec` (the Chebyshev spectral differentiation operator, a
standard strongly non-normal example), m=20, 21×21 grid, nit=12, errors measured in BigFloat:

| precision | median rel err | worst-case rel err |
| --- | --- | --- |
| Float64 (~16 dig) | 1.4e-3 | **8.4e-1** (84%; 182/441 points >1% off) |
| Float64x2 (~32 dig) | 0 (exact to Float64 rounding) | 2.1e-16 |

At the worst Float64 point (z ≈ 1.4 + 2.64i, true σ = 4.38e-15) plain Float64 reports
8.06e-15 — **1.8× too large** — while Float64x2 returns the oracle value exactly. Switching
the demo to `Float64x4` pushes the worst case down further. The GPU result is validated
transitively: GPU-MF ≡ CPU-MF after Float64 rounding (the demo's correctness gate), and
CPU-MF ≡ oracle (above).

## Performance (6× NVIDIA GTX 1080 Ti, Pascal; 1/32 FP64)

### Double-double on the GPU

Representative throughput (m=24, 60×60 grid, nit=32):

| backend | Float64 | Float64x2 | Float64x4 |
| --- | --- | --- | --- |
| GPU (gridpts/s) | ~30k (1.0×) | ~16k (1.9×) | ~2.9k (10×) |
| CPU 1-thread | 6.6k (1.0×) | 0.49k (14×) | 0.06k (110×) |

Double-double's GPU penalty grows with problem size as the kernel becomes compute-bound:
~2.3× at m=24 → ~2.9× (m=64) → ~5.8× (m=128) → ~8.3× (m=256). Small grids are
launch/latency/fan-out bound, so the extra arithmetic partly hides; the GPU makes extended
precision far cheaper *relative to Float64* than the CPU does.

Note: the host-side σ extraction now runs the tridiagonal `eigen` in extended precision; for
small grids (fast GPU solve) this host cost is a non-trivial fraction of end-to-end time, so
very-small-grid ratios understate the GPU solve's own efficiency. The size-sweep ratios above
(GPU-solve-dominated) are the cleaner guide.

### Float32-based MultiFloats vs Float64-based

Tested on the hypothesis that the 1080 Ti's full-speed FP32 (vs 1/32 FP64) would make
`Float32x2`/`Float32x4` win. The Float64-based types are still the right choice:

| eltype | digits | range | gridpts/s | vs F64 |
| --- | --- | --- | --- | --- |
| ComplexF64 | 16 | 1e308 | 31,099 | 1.0× |
| Float32x2 | 14 | **1e38** | 12,191 | 2.55× slower |
| Float64x2 | 32 | 1e308 | 4,244 | 7.33× slower |
| Float32x4 | 28 | **1e38** | 740 | 42× slower |

- `Float32x2` (14 dig) is slower *and* less precise than native Float64 (16 dig).
- `Float32x4` (28 dig) is 5.7× slower than `Float64x2` (740 vs 4,244), less precise (28 < 32),
  and range-limited.

Why the FP32 throughput edge doesn't materialize:
1. **The triangular solves are latency/instruction-bound, not throughput-bound.** Each pencil
   solve is a serial dependency chain; the number of instructions in the chain dominates, not
   FP32-vs-FP64 peak throughput. Double-single is ~10 FP32 instructions where native Float64
   is 1, so even throttled FP64 wins. The 1/32 FP64 penalty isn't the binding constraint.
2. **Matching precision costs 2× the limbs** (`Float32x4` vs `Float64x2`), and the extra limbs
   swamp FP32's per-op speed.
3. **Range.** `Float32xN` shares Float32's ~1e38 exponent range, so `λmax = 1/σ_min²` overflows
   once `σ_min ≲ 1e-19` — it can't reach the deep pseudospectra this package targets (the
   Float64-based types, range ~1e308, do).

(FP32 emulation *is* faster than FP64 emulation at equal limb count, but you can't compare at
equal limb count because precision differs; at equal precision/range the Float64-based type
wins.)

## Recommendations

- Use `Complex{Float64x2}` (double-double, ~32 digits) or `Complex{Float64x4}` (~64) for
  extended-precision pseudospectra; they run on CPU and GPU unchanged.
- Add `using GenericSchur` and `using GenericLinearAlgebra` — the first for
  `MatrixPencil(schur(A))` on dense matrices, the second for the extended-precision σ
  extraction (and the SVD oracle in the accuracy example).
- **Don't over-specify `nit`.** Inverse-Lanczos targets the extreme eigenvalue
  (`λmax = 1/σ_min²`), which it captures in a handful of steps — well below `m`, and *faster*
  near the spectrum where that eigenvalue is huge and isolated. On the chebspec(20) example
  the σ are bit-identical from nit=4 through nit=40, and the adaptive driver picks nit=6 at
  every grid point. (More iterations can't rescue Float64 either — its in-band error is
  precision-limited, flat in `nit`.) Prefer the adaptive driver, or a fixed `nit` near the
  package default `~log2(m)` with a little margin.
- The **adaptive driver works in extended precision** too. Its default tolerances target
  Float64-level convergence (`rtol = 1e-6`); tighten them (e.g. `rtol = 1e-25`,
  `atol = eps(real(T))`) to converge to full double-double accuracy. The demo includes an
  adaptive run that matches the fixed-`nit` result.
- Prefer the Float64-based MultiFloats over the Float32-based ones for this solver.

## Known limitations & future work

- On CUDA, MultiFloats can be degraded by NVPTX FMA-contraction of its error-free transforms
  (MultiFloats #23). Current releases emit FMA-safe operators by default; the demo includes a
  GPU-vs-CPU correctness gate that checks this empirically. Don't `@fastmath` the kernels.
  A built-in package self-test that verifies error-free-transform integrity on the active
  device (rather than the example-level gate) would be a nice addition.
- The precompile workload pins `ComplexF32`/`ComplexF64`, so extended types compile on first
  use at runtime — a startup cost, not a correctness issue, and only at the example level
  for now.

## Reproducing

```
cd examples
julia --project=. ihlpsa_multifloats.jl            # F64 vs Float64x2 demo (+ adaptive run; optional figure)
julia --project=. ihlpsa_multifloats_accuracy.jl   # vs BigFloat ℂsvdpsa oracle; set HP=Float64x4
```
The examples depend on `MultiFloats`, `GenericSchur`, `GenericLinearAlgebra`, `MatrixDepot`
and `Random`. The package itself gains no new required dependencies — the extended-precision
σ extraction calls `eigen`, whose method for extended types is supplied by the user's already-
loaded `GenericLinearAlgebra`.
