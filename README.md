# KAPseudospectra.jl
Accelerated pseudospectral calculations using KernelAbstractions.jl

# Installation
1) Clone this repo
2) Inside of the Julia REPL (preferably within a top-level environment e.g. @v1.10, etc.) run `]dev path/to/KAPseudospectra.jl`
3) You may need to run `]instantiate` to resolve package dependencies of `KAPseudospectra.jl`

# Usage
`ihlpsa` computes pseudospectra (resolvent-norm contours) of a matrix pencil over
a grid of complex shifts via batched inverse Lanczos, fanned out across all
devices of the chosen KernelAbstractions backend.

```julia
using KAPseudospectra, KernelAbstractions   # + `using CUDA`/`AMDGPU`/... for a GPU backend
A = my_matrix
P = MatrixPencil(A)                          # or MatrixPencil(A, B) for a pencil zB − A
_, _, zg = qgrid(ComplexF64, (-2, 5), (-4.5, 4.5), (300, 300))   # grid of shifts

# Fixed depth: every grid point runs exactly `nit` Lanczos iterations.
srg = ihlpsa(CPU(), zg, P, 16)               # returns the grid Matrix of σ values

# Adaptive depth (omit `nit`): each grid point retires at its own converged
# depth (per-point hybrid). Returns the grid Matrix, just like the fixed form.
srg = ihlpsa(CPU(), zg, P)
srg = ihlpsa(CUDABackend(), zg, P)           # same call, multi-GPU
srg = ihlpsa(CPU(), zg, P; verbose=true)     # also logs the convergence depth reached
```

Adaptive convergence is tunable with keyword arguments (`rtol`, `atol`, `nit_chunk`,
`nit_max`, …) and `γ`,`δ` for perturbation scaling — see the `ihlpsa` docstring for
the full list. Pass `devs` to restrict which GPUs are used.

On FP64-less GPUs (Intel iGPUs, Apple Metal) call `set_pdiv_accurate(false)` so the
Float32 solves compile — the default uses Base's more accurate division, which needs FP64.

# Examples
Check out `examples/` for scripts that showcase usage of this package. See
`examples/README.md` for the full rundown; in brief:

- `ihlpsa_adaptive.jl` -- adaptive iteration-depth `ihlpsa`: σ contours alongside the per-grid-point Lanczos-depth maps, at two tolerances
- `ihlpsa_adaptive_advanced.jl` -- the intricate adaptive machinery: generalized pencils (`B ≠ I`, γ/δ weights), the convergence knobs, reproducible seeds, the `nit_max` cap, `zpd` multi-batch, and multi-device fan-out
- `loewner_pseudospectra.jl` -- reproduces Example 1 of Embree & Ioniţă, *Pseudospectra of Loewner Matrix Pencils* (2022): how interpolation-point placement controls the sensitivity of poles recovered by Loewner realization
- `ihlpsa_backends.jl` -- a good starting point, showing how to switch between device-specific backends (`CUDA` and `AMDGPU` have been tested thusfar)
  Note, `AMDGPU` currently requires running Julia with a single thread (there is a bug in Julia when running with multiple threads, likely related to premature garbage collection)
- `test_real_structured_psa.jl` -- compute structured/unstructured pseudospectra for a matrix using `CPU()` and plot them together
- `test_ihlpsa_large.jl` -- timing sweep over increasingly large matrices (CPU by default; uncomment a backend for an accelerator), writing timing information and plots to `examples/test_large_results/`

The `examples/` directory has its own `Project.toml`. It is **CPU-only and
plot-ready** out of the box: plotting uses `Plots.jl`'s default GR backend (no
Python/matplotlib), and a `[sources]` entry resolves `KAPseudospectra` from the
checkout, so a single instantiate suffices (no `dev` step). Add a GPU backend
yourself to run on a device -- see `examples/README.md`.

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

then run any example as:

```sh
julia --project=examples examples/ihlpsa_backends.jl
```
