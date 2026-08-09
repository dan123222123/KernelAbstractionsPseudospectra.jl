# KAPseudospectra examples

Runnable example scripts for `KAPseudospectra`. The environment here is
**CPU-only and plot-ready**: it instantiates on any machine without GPU drivers
or vendor toolkits, and ships the plotting stack (`Plots` with its default GR
backend, `LaTeXStrings`) so every script produces figures out of the box on
`CPU()` — GR needs no Python/matplotlib, so there's nothing extra to install.

## Quick start

```console
$ julia --project=examples -e 'using Pkg; Pkg.instantiate()'
$ julia --project=examples examples/ihlpsa_adaptive.jl
```

Or interactively, running the `##` cells block by block (VS Code, Pluto-style):

```julia
julia> # started with: julia --project=examples
julia> include("examples/ihlpsa_adaptive.jl")
```

`KAPseudospectra` itself resolves from the repo checkout one directory up (via
the `[sources]` entry in `Project.toml`), so there's no separate `dev` step —
`Pkg.instantiate()` is enough.

The scripts:

- `ihlpsa_adaptive.jl` — adaptive iteration-depth `ihlpsa`: σ contours alongside
  the per-grid-point Lanczos-depth maps, at two tolerances. Also shows the
  convergence knobs and reading the per-point depth via `_ihlpsa_adaptive`.
- `loewner_pseudospectra.jl` — reproduces Example 1 of Embree & Ioniţă, *Pseudospectra
  of Loewner Matrix Pencils* (2022): four Loewner realizations of a tiny SISO system,
  showing how interpolation-point placement controls the sensitivity of the recovered
  poles. Uses the dense `ℂsvdpsa` (as the paper does for small examples).
- `ihlpsa_backends.jl` — the same pseudospectra computed on a selectable backend.
- `ihlpsa_multifloats.jl` — extended-precision pseudospectra with MultiFloats.jl (double-single
  `Complex{Float32x2}` by default, so it also runs on FP64-less GPUs); the isbits extended floats
  run inside the GPU kernels unchanged.
- `ihlpsa_chebspec_oracle.jl` — the accuracy-validation MVE: Float64 vs Float64x2
  inverse-Lanczos against an independent BigFloat dense-SVD oracle on chebspec (CPU-only).
- `test_ihlpsa_large.jl` — large-`n` timing sweep (wants an accelerator).
- `test_real_structured_psa.jl` — real/structured pseudospectra.

## Running on a GPU

The package's GPU support is delivered through **weakdep extensions** (one per
backend: CUDA, AMDGPU, Metal, oneAPI). None of those backend packages are
dependencies of this environment — that's what keeps it CPU-only and portable.
To use a GPU you add the one backend you have, and the matching extension loads
automatically the moment you `using` it.

**Option A — add the backend to this project (quick):**

```julia
julia> # started with: julia --project=examples
julia> using Pkg
julia> Pkg.add("CUDA")        # or "AMDGPU" / "Metal" / "oneAPI"
```

then uncomment the matching backend line near the top of the script, e.g.

```julia
#backend = CPU()
using CUDA; backend = CUDABackend()
```

This edits `examples/Project.toml` (adds the backend dep). If you'd rather not
touch the tracked file, use Option B.

**Option B — your own examples project (keeps the repo clean):** copy the scripts
somewhere and point a private environment at the package. A named shared
environment outside the repo is convenient and reusable:

```julia
julia> # started with: julia --project=@kaps-examples
julia> using Pkg
julia> Pkg.develop(path="/path/to/KAPseudospectra.jl")   # the repo root
julia> Pkg.add(["KernelAbstractions", "MatrixDepot", "Plots", "LaTeXStrings"])
julia> Pkg.add("oneAPI")        # whichever backend you have
```

Backend ↔ hardware ↔ caveats:

| `using` …  | `backend = …`     | hardware           | notes                                  |
|------------|-------------------|--------------------|----------------------------------------|
| `CUDA`     | `CUDABackend()`   | NVIDIA             | FP32 + FP64; `wgs = 256`               |
| `AMDGPU`   | `ROCBackend()`    | AMD                | `wgs = 16`                             |
| `Metal`    | `MetalBackend()`  | Apple              | **Float32 only** (Metal has no FP64)   |
| `oneAPI`   | `oneAPIBackend()` | Intel              | **Float32 only** on FP64-less iGPUs; `wgs = 32` |

On FP64-less GPUs (Metal, Intel iGPUs) keep matrices in `ComplexF32` — the
example scripts already default to `ComplexF32` for exactly this reason.
