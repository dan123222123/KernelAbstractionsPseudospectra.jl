# API Reference

```@meta
CurrentModule = KAPseudospectra
```

## Grid Generation

```@docs
qgrid
```

## Matrix Pencils

### Types

```@docs
AbstractMatrixPencil
MatrixPencil
SchurMatrixPencil
```

### Constructors

```@docs
MatrixPencil(::Schur)
MatrixPencil(::GeneralizedSchur)
```

### Validation

```@docs
validate
```

## SVD-Based Pseudospectra

### Unstructured (Complex) Pseudospectra

```@docs
ℂsvdpsa
ℂsvdpsa!
```

### Structured (Real) Pseudospectra

```@docs
ℝsvdpsa
ℝsvdpsa!
distzeigAB
```

## Inverse Hermitian Lanczos

### Main Functions

```@docs
ihlpsa
```

### Helper Functions

```@docs
IHLworkspace
findmaxbatchihl
lockstep_ihl!
ihlsrg!
```

## Index

```@index
```
