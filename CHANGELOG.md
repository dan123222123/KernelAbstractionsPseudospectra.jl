# Changelog

All notable changes to KAPseudospectra.jl are documented here.

## [Unreleased]

### Added
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

### Removed
- **Breaking:** `ihlpsa_adaptive` and its tier flags (`compact`, `resumable`).
  Benchmarking showed only the per-point hybrid (formerly `compact=true,
  resumable=true`) was worth keeping; the global / compact-only / resumable-only
  tiers never beat a well-chosen fixed `nit`. The hybrid is now the single
  adaptive path reached by omitting `nit`. Migration: replace
  `ihlpsa_adaptive(b, zg, P; …)` with `ihlpsa(b, zg, P; …)`, and any
  `compact=`/`resumable=` flags can be dropped.
