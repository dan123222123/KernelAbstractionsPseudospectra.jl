# Changelog

All notable changes to KAPseudospectra.jl are documented here.

## [Unreleased]

### Added
- **`psaplot` / `psaplot!` plot recipe.** Contour plots of a pseudospectra field
  (`log₁₀ σ` colorbar, optional eigenvalue overlay) via a `RecipesBase` package
  extension: `using Plots; psaplot(gx, gy, σ, eigvals(A); levels=-6:0)`.
- **`MatrixPencil(A, cI)` with a scaled `UniformScaling` now keeps the scale**,
  materializing `cI` and building a generalized pencil (it used to silently drop
  `c` and build a `B = I` pencil, disagreeing with `ℂsvdpsa`'s handling of the
  same input).
- Aqua.jl quality testset (ambiguities, unbound params, compat coverage, piracy)
  plus `@test_throws` coverage for the input-validation error paths.
- `[compat]` bounds for every dependency and weak dependency, and a `julia = "1.10"`
  floor matching CI.
- **Adaptive `nit` in `ihlpsa`.** Calling `ihlpsa(backend, zg, P)` **without** a
  `nit` argument now runs the adaptive (per-point hybrid) inverse-Lanczos driver:
  each grid point retires at its own converged depth (relative `rtol` / absolute
  `atol`, confirmed over `nconfirm` chunks), with survivors' Lanczos state kept
  resident across chunks. Returns the grid-shaped `Matrix` of σ, exactly like the
  fixed form; the convergence depth reached is a diagnostic, logged when
  `verbose=true`. New keywords: `rtol`, `atol`, `nconfirm`, `nit_chunk`, `nit_max`,
  `seed`, `verbose`. Runs across multiple GPUs via the same device fan-out as the
  fixed path.
- Docstring for `ihlpsa` (both fixed and adaptive forms) and a Usage section in
  the README.
- Adaptive path added to every backend's precompile workload (CPU + CUDA/AMDGPU/
  oneAPI/Metal extensions), so the first adaptive GPU call no longer pays cold
  compilation latency.
- **Precision toggle for the triangular solves.** `set_pdiv_accurate(flag)` selects how the
  Float16/Float32 solves divide: the default (`true`) uses Base's widening division (most
  accurate); `false` keeps the divide in the input precision — slightly less accurate, but the
  only form FP64-less GPUs (Intel iGPUs, Apple Metal) can compile. It is a `Preferences.jl`
  preference, so it takes effect after a Julia restart. Float64 is unaffected.

### Changed
- **Breaking:** `ihlpsa` no longer has a default `nit`. Omitting `nit` selects the
  adaptive driver instead of fixed-`nit` with `nit = ceil(log2 m)`. Pass an explicit
  `nit::Integer` for the fixed-depth behaviour. Both forms return the grid-shaped
  `Matrix` of σ; the adaptive convergence depth is a diagnostic, surfaced via
  `verbose=true` logging (or from the un-exported `KAPseudospectra._ihlpsa_adaptive`
  driver, which returns `(σ, nit_grid)` — a same-shape map of each grid point's
  retirement depth; `maximum(nit_grid)` is the deepest).
- **Breaking:** perturbation scaling `γ`,`δ` are now **keyword** arguments in both
  forms (previously positional in the fixed form): `ihlpsa(b, zg, P, nit; γ, δ)`.
- **Breaking:** `set_pdiv_accurate` is renamed `set_pdiv_accurate!`, following the
  mutating-`!` convention of the other preference setters (`ihlpsa`'s internal solve
  router `trsmIHL` is likewise now `trsmIHL!`).
- `ℂsvdpsa`/`ℝsvdpsa` return a concrete `Matrix` (via `permutedims`) instead of a
  lazy `Adjoint` wrapper, matching `ihlpsa`.
- Invalid user input (empty grid, mismatched pencil sizes, bad `(γ, δ)` weights)
  now throws `ArgumentError`/`DimensionMismatch` instead of `AssertionError`.
- `set_trsm_strategy!` takes effect immediately in the running session (the
  persisted preference is now cached and updated in place; `KAPSEUDO_TRSM` still
  overrides), and the per-iteration preference re-read is gone from the solve loop.
- Internal reorganization: `src/KAPseudospectra.jl` is now a thin module index
  (preferences API → `src/preferences.jl`, precompile workloads →
  `src/precompile.jl`, `qgrid`/`psaplot` stubs → `src/grid.jl`); the KATRSM
  submodule directory is `src/KATRSM/` (was `src/KATRSM.jl/`); struct fields of
  `MatrixPencil`/`SchurMatrixPencil`/`IHLworkspace` are concretely parametrized.

### Fixed
- **Breaking (δ≠0 only):** the `(γ,δ)` pseudospectral value is now
  `σ_min(zB − A)/(γ + δ|z|)`, matching the Frayssé–Gueury–Nicoud–Toumazou
  definition (`z ∈ σ_ε^{(γ,δ)}` iff the value is `< ε`). Previously both `ihlpsa`
  and `ℂsvdpsa` returned the reciprocal weighting `(γ + δ|z|)·σ_min`, which does
  not correspond to any `σ_ε^{(γ,δ)}`. The default `γ=1, δ=0` (standard
  pseudospectrum, `= σ_min`) is unchanged; only `δ≠0` results differ.
- The `γ`,`δ` weights are no longer required to sum to 1. Any `γ, δ ≥ 0` (not both
  zero) is accepted — including the literature's `(1,0)` and `(1,1)` — since the
  set is invariant to a common scaling of `(γ,δ)`. The check is now applied
  uniformly across `ℂsvdpsa`, `ihlpsa`, and the adaptive driver (previously only
  `ℂsvdpsa` validated, and it rejected `(1,1)`).
- **MultiFloats on FP64-less GPUs**: `examples/ihlpsa_multifloats.jl` restored to
  its portable CPU default and its documented Float32/double-single precision pair.
- `src/KATRSM/trsm_tiled_kernels.jl` now declares its own `using KernelAbstractions`
  (previously load-order dependent through a sibling include).
- Metal `device_bytes_available` is floored at 0 so batch sizing never sees a
  negative budget under memory pressure.
