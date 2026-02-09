# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

KAPseudospectra.jl is a Julia package for accelerated pseudospectral calculations using KernelAbstractions.jl. It computes pseudospectra (resolvent norms) for matrices and matrix pencils using both SVD-based methods and Inverse Lanczos methods, with support for multiple compute backends (CPU, CUDA, AMDGPU, Metal).

## Installation & Development Setup

```julia
# From Julia REPL (in a top-level environment like @v1.10)
]dev path/to/KAPseudospectra.jl
]instantiate
```

For running examples, additional dependencies are needed:
```julia
]add LinearAlgebra MatrixDepot Plots LaTeXStrings PyPlot KernelAbstractions
]add CUDA  # Optional, for GPU acceleration
]add AMDGPU  # Optional, for AMD GPU acceleration
```

## Running Tests

```bash
# Run all tests
julia --project=test test/runtests.jl

# Run specific backend tests
julia --project=test test/runtests.jl cpu
julia --project=test test/runtests.jl cuda
julia --project=test test/runtests.jl amdgpu
```

Note: Tests compare against reference data from EigTool (see test/eigtool_core.jl).

## Architecture

### Core Abstractions

**Matrix Pencils** (src/core.jl): The package works with matrix pencils (A, B) representing eigenvalue problems of the form `(zB - A)x = 0`.

- `MatrixPencil{T}`: Standard matrix pencil with matrices A and B
- `SchurMatrixPencil{T}`: Optimized pencil storing both Schur forms and their conjugate transposes (A, A', B, B') for efficient triangular solves
- Both types support KernelAbstractions' `adapt` for multi-backend execution

### Pseudospectra Methods

**SVD-based methods** (src/svdpsa.jl): CPU-only, multi-threaded computations using LAPACK's GESDD
- `ℂsvdpsa`: Complex (unstructured) pseudospectra - computes `σₘᵢₙ(zB - A)` at each grid point
- `ℝsvdpsa`: Real (structured) pseudospectra - finds minimum real perturbation making `zB - A` singular, uses optimization (Optim.jl) over the algorithm from §50 of SAP2005

**Inverse Hermitian Lanczos** (src/ihlpsa.jl): GPU-accelerated method for large matrices
- Uses Lanczos iteration to approximate smallest singular value without full SVD
- The `ihlpsa` function is the main entry point supporting multi-device execution
- Key components:
  - Custom kernels for Lanczos iteration (`_qₙnext`, `_ihl_ttr_qₙnext`)
  - Batched triangular solves via KATRSM submodule
  - Device abstraction layer for backend-specific operations
  - Host-side eigenvalue computation of tridiagonal matrices (`ihlsrg!`)

### KATRSM Submodule

Located in src/KATRSM.jl/, this submodule provides batched triangular system solvers optimized for different backends:
- Forward/backward solves for both standard matrices and matrix pencils
- Column-oriented kernels for GPU backends (better memory coalescing)
- Row-oriented wrappers for CPU backend
- Files organized by operation type: `trsm_*` for standard matrices, `trsm_pencil_*` for matrix pencils

### Backend Extensions

The package uses Julia's package extension system (Project.toml `[extensions]`) to provide device-specific implementations:

- **ext/CUDAPseudospectra.jl**: CUDA backend support with precompilation
- **ext/AMDGPUPseudospectra.jl**: AMDGPU/ROCm backend support
- **ext/MetalPseudospectra.jl**: Apple Metal backend support

Each extension implements the device abstraction interface:
- `device()`, `device!()`, `devices()`: Device management
- `get_bgarray()`: Backend-specific array type (CuArray, ROCArray, etc.)
- `device_bytes_available()`, `device_reclaim()`: Memory management

### Multi-Device Execution

The `ihlpsa` function automatically distributes work across multiple devices:
1. Grid points (columns of `zg`) are partitioned across available devices
2. Each device spawns an independent task processing its partition
3. Within each device, grid points are further batched based on available memory
4. Results are collected and concatenated

## Key Implementation Details

**Grid Generation**: Use `qgrid(T, (xmin, xmax), (ymin, ymax), (nx, ny))` to create complex grids for pseudospectra computation.

**Memory Management**: The `findmaxbatchihl` function automatically determines optimal batch size based on device memory with a configurable margin of error (default 10%).

**Workgroup Size (`wgs`)**: The triangular solve kernels use a workgroup size parameter that should be tuned per backend:
- CPU/CUDA: 256 (default)
- AMDGPU: 16 recommended

**AMDGPU Threading Limitation**: AMDGPU backend currently requires Julia to run with a single thread due to a garbage collection bug in multi-threaded mode.

**Progress Tracking**: The `ihlpsa` function supports a `progress=true` flag that displays a progress bar tracking completion of Lanczos iterations across all grid points.

## Example Usage Patterns

See examples/ directory:
- `ihlpsa_backends.jl`: Demonstrates backend switching between CPU, CUDA, AMDGPU, Metal
- `test_real_structured_psa.jl`: Compares structured vs unstructured pseudospectra
- `test_ihlpsa_large.jl`: Performance testing with increasingly large matrices
