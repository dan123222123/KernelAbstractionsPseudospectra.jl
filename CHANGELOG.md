# Changelog

All notable changes to KAPseudospectra.jl are documented here.

## [Unreleased]

### Added
- **Adaptive `nit` in `ihlpsa`.** Calling `ihlpsa(backend, zg, P)` **without** a
  `nit` argument now runs the adaptive (per-point hybrid) inverse-Lanczos driver:
  each grid point retires at its own converged depth (relative `rtol` / absolute
  `atol`, confirmed over `nconfirm` chunks), with survivors' Lanczos state kept
  resident across chunks. Returns `(σ, nit_used)`. New keywords: `rtol`, `atol`,
  `nconfirm`, `nit_chunk`, `nit_max`, `seed`, `verbose`. Runs across multiple GPUs
  via the same device fan-out as the fixed path.
- Docstring for `ihlpsa` (both fixed and adaptive forms) and a Usage section in
  the README.
- Adaptive path added to every backend's precompile workload (CPU + CUDA/AMDGPU/
  oneAPI/Metal extensions), so the first adaptive GPU call no longer pays cold
  compilation latency.

### Changed
- **Breaking:** `ihlpsa` no longer has a default `nit`. Omitting `nit` selects the
  adaptive driver (returns `(σ, nit_used)`) instead of fixed-`nit` with
  `nit = ceil(log2 m)` (returned a `Matrix`). Pass an explicit `nit::Integer` for
  the fixed-depth behaviour. Perturbation scaling `γ`,`δ` are positional in the
  fixed form and keyword in the adaptive form.

### Removed
- **Breaking:** `ihlpsa_adaptive` and its tier flags (`compact`, `resumable`).
  Benchmarking showed only the per-point hybrid (formerly `compact=true,
  resumable=true`) was worth keeping; the global / compact-only / resumable-only
  tiers never beat a well-chosen fixed `nit`. The hybrid is now the single
  adaptive path reached by omitting `nit`. Migration: replace
  `ihlpsa_adaptive(b, zg, P; …)` with `ihlpsa(b, zg, P; …)`, and any
  `compact=`/`resumable=` flags can be dropped.
