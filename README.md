# KAPseudospectra.jl
Accelerated pseudospectral calculations using KernelAbstractions.jl

# Installation
1) Clone this repo
2) Inside of the Julia REPL (preferably within a top-level environment e.g. @v1.10, etc.) run `]dev path/to/KAPseudospectra.jl`
3) You may need to run `]instantiate` to resolve package dependencies of `KAPseudospectra.jl`

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
