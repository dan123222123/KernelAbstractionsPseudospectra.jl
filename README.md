# KAPseudospectra.jl
Accelerated pseudospectral calculations using KernelAbstractions.jl

# Installation
1) Clone this repo
2) Inside of the Julia REPL (preferably within a top-level environment e.g. @v1.10, etc.) run `]dev path/to/KAPseudospectra.jl`
3) You may need to run `]instantiate` to resolve package dependencies of `KAPseudospectra.jl`
4) Run `]add KernelAbstractions` — the backend types (`CPU()`, and the GPU backends) come from
   it, so it has to be in *your* environment, not just this package's. Add the vendor package
   for your device (`]add CUDA`/`AMDGPU`/`Metal`/`oneAPI`) to run on a GPU.

# Usage
`ihlpsa` computes pseudospectra (resolvent-norm contours) of a matrix pencil over
a grid of complex shifts via batched inverse Lanczos, fanned out across all
devices of the chosen KernelAbstractions backend.

```julia
using KAPseudospectra, KernelAbstractions   # + `using CUDA`/`AMDGPU`/... for a GPU backend
A = ComplexF64.(my_matrix)                   # a complex element type is required
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
the full list. Pass `devs` to restrict which GPUs are used, and `on_batch` to receive
each group of grid points as it retires, rather than waiting for the whole field:

```julia
ihlpsa(CPU(), zg, P; on_batch = (idx, σ, nit) -> checkpoint(idx, σ))
```

`idx` indexes the returned matrix, every grid point arrives exactly once, and the values
match the final return — useful for checkpointing a long solve. Deliveries are serialized,
so the callback need not be thread-safe.

On FP64-less GPUs (Intel iGPUs, Apple Metal) call `set_pdiv_accurate!(false)` so the
Float32 solves compile — the default uses Base's more accurate division, which needs FP64.
It writes the setting to `LocalPreferences.toml`, so it is a one-time call, but Julia has to
be restarted before it takes effect.

# Extended precision

`ihlpsa` and the KATRSM triangular solves are generic over the complex element type, so
MultiFloats.jl's isbits extended floats run *inside* the kernels — GPU included — with no
package changes. Load `GenericSchur` alongside `MultiFloats` and `MatrixPencil` reduces the
pencil at `BigFloat`, rounding the triangular factor back to the working type:

```julia
using KAPseudospectra, KernelAbstractions, MatrixDepot
using MultiFloats, GenericSchur, GenericLinearAlgebra   # generic Schur + tridiagonal eigen

A = MatrixDepot.grcar(Float64x4, 64)                    # ~64 digits
P = MatrixPencil(A)                                     # reduced at BigFloat, rounded to Float64x4
_, _, zg = qgrid(Complex{Float64x4}, (-1, 3), (-3, 3), (60, 60))
srg = ihlpsa(CPU(), zg, P)
```

The reduction is what makes this worth doing: a Float64 LAPACK Schur caps the end-to-end
result near 1e-15 however wide the solve arithmetic is, so widening the solve alone buys
nothing. Without `GenericSchur` loaded, a MultiFloat pencil fails with a clean `MethodError`.
See `examples/ihlpsa_multifloats.jl` and `examples/ihlpsa_chebspec_oracle.jl`.

# Device tuning

The triangular solve has per-device knobs — trailing-tile width, warps per block, grid points
per warp, and the column solve's workgroup size — which default to heuristics.
`KAPseudospectra.tune_trsm!` probes a device and can persist what it finds as a profile:

```julia
KAPseudospectra.tune_trsm!(CUDABackend(); profile = "mybox.toml")
```

Point `KAPSEUDO_TUNE_PROFILE` at that file to use it. `tune_profile_path()`, `tune_profile()`
and `reload_tuning!()` inspect and re-read the active profile. Per knob the resolution order is
`KAPSEUDO_TRSM_*` environment variable > profile > `LocalPreferences.toml` > heuristic. Tuned
profiles for the CI machines are tracked in `bench/tuning/`.

# Examples
Check out `examples/` for scripts that showcase usage of this package. See
`examples/README.md` for the full rundown; in brief:

- `ihlpsa_adaptive.jl` -- adaptive iteration-depth `ihlpsa`: σ contours alongside the per-grid-point Lanczos-depth maps, at two tolerances
- `loewner_pseudospectra.jl` -- reproduces Example 1 of Embree & Ioniţă, *Pseudospectra of Loewner Matrix Pencils* (2022): how interpolation-point placement controls the sensitivity of poles recovered by Loewner realization
- `ihlpsa_backends.jl` -- a good starting point, showing how to switch between device-specific backends (`CUDA` and `AMDGPU` have been tested thusfar)
  Note, `AMDGPU` currently requires running Julia with a single thread (there is a bug in Julia when running with multiple threads, likely related to premature garbage collection)
- `ihlpsa_multifloats.jl` -- extended precision in practice: where a `ComplexF32` solve reports the wrong σ near a strongly non-normal spectrum, and double-single recovers it
- `ihlpsa_chebspec_oracle.jl` -- the same pseudospectra three ways on chebspec (`ComplexF64`, `Complex{Float64x2}`, and a 256-bit BigFloat dense-SVD oracle), reporting median and worst relative error
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

# Benchmarking

The benchmark experiments (per-device parameter sweeps, roofline, driver comparisons)
live in `bench/`. A single command runs the applicable sweeps for a backend:

```sh
julia --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA")'   # backend per device
julia --project=bench bench/gpu.jl cuda                                    # or amdgpu / oneapi / cpu
```

Sizes are controlled with `BENCH_MS` / `BENCH_GRIDN` / `BENCH_REPS` (or `KAPSEUDO_BENCH_FULL=1`).
GPU benchmarks also run on Buildkite (`.buildkite/pipeline.yml`), with experiment selection
and sizes carried in as build environment variables. See **`bench/README.md`** for the full
rundown.
