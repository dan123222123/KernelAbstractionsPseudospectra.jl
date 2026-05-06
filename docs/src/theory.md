# Pseudospectra Theory

```@meta
CurrentModule = KAPseudospectra
```

## Introduction

!!! todo "TODO: Add introduction"
    - Explain what pseudospectra are and why they matter
    - Contrast with standard eigenvalue analysis
    - Motivate with examples of nonnormal matrices

## Mathematical Definition

### ε-Pseudospectrum

!!! todo "TODO: Define ε-pseudospectrum"
    - Standard definition: Λ_ε(A) = {z ∈ ℂ : ||(zI - A)^{-1}|| ≥ ε^{-1}}
    - Equivalent definitions (eigenvalues of perturbed matrices, resolvent norm)
    - Connection to singular values

### Matrix Pencils

!!! todo "TODO: Generalized eigenvalue problems"
    - Definition of matrix pencil (A, B)
    - Pseudospectra for pencils: Λ_ε(A, B) involving (zB - A)
    - Normalization factors γ and δ

## Structured vs Unstructured Pseudospectra

### Unstructured (Complex) Pseudospectra

The unstructured pseudospectra computed by [`ℂsvdpsa`](@ref) allows arbitrary complex perturbations:
```math
\sigma_{min}(zB - A)
```

!!! todo "TODO: Expand on unstructured case"
    - Physical interpretation
    - When to use unstructured analysis

### Structured (Real) Pseudospectra

The structured pseudospectra computed by [`ℝsvdpsa`](@ref) restricts to real perturbations.

!!! todo "TODO: Expand on structured case"
    - Real stability radius
    - Applications with real-valued systems
    - Comparison with unstructured case (always ℝsvdpsa ≥ ℂsvdpsa)

## Computational Methods

### SVD-Based Methods

Direct computation using singular value decomposition.

!!! todo "TODO: SVD method details"
    - LAPACK GESDD algorithm
    - Computational complexity: O(m³) per grid point
    - Multi-threading on CPU
    - When to use: Small to medium matrices (m < 500)

### Inverse Hermitian Lanczos

GPU-accelerated iterative approximation method implemented in [`ihlpsa`](@ref).

!!! todo "TODO: IHL method details"
    - Lanczos iteration on (zB - A)^{-1}(z̄B - A)^{-H}
    - Tridiagonal reduction and eigenvalue extraction
    - Computational complexity: O(m² × nit) per grid point with nit ≈ log(m)
    - Advantages: GPU acceleration, batched solves, multi-device support
    - When to use: Large matrices (m ≥ 500), GPU available

## Applications

!!! todo "TODO: Add applications section"
    - Hydrodynamic stability
    - Control theory and robust stability
    - Markov chains and matrix iterations
    - Discretized differential operators

## References

!!! todo "TODO: Add key references"
    - Trefethen & Embree (2005). "Spectra and Pseudospectra"
    - Wright & Trefethen (2001). EigTool
    - Lui (1997). Inverse Lanczos method
    - Qiu et al. (1995). Real stability radius
