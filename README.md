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

Adaptive knobs are keywords on the no-`nit` form: `rtol`/`atol` (convergence
tolerance), `nconfirm` (confirmation chunks), `nit_chunk`, `nit_max`, and `γ`,`δ`
for perturbation scaling (`γ`,`δ` are keywords in both forms). Pass `devs` to
restrict which GPUs are used. Adaptive runs across multiple GPUs the same way the
fixed path does. The convergence depth reached is a diagnostic — pass
`verbose=true` to log it, or call `KAPseudospectra._ihlpsa_adaptive(...)` for the
`(σ, nit_used)` tuple.

# Examples
Check out `examples/` for three scripts that showcase usage of this package.

They are:
- `ihlpsa_backends.jl` -- a good starting point, showing how to switch between device-specific backends (`CUDA` and `AMDGPU` have been tested thusfar)
  Note, `AMDGPU` currently requires running Julia with a single thread (there is a bug in Julia when running with multiple threads, likely related to premature garbage collection)
- `test_real_structured_psa.jl` -- compute structured/unstructured pseudospectra for a matrix using `CPU()` and plot them together
- `test_ihlpsa_large.jl` -- compute pseudospectra for increasingly large matrices using `CUDABackend()`, writing timing information and plots to `examples/test_large_results/`

The `examples/` directory has its own `Project.toml` listing every dependency the
scripts use (including PyPlot for nice colorbar formatting via `Plots.jl`'s
PyPlot backend, plus optional `CUDA`/`AMDGPU`/`Metal`). Instantiate it once:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

then run any example as:

```sh
julia --project=examples examples/ihlpsa_backends.jl
```
