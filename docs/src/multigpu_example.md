# Multi-GPU Computation

```@meta
CurrentModule = KAPseudospectra
```

This page demonstrates the multi-device capabilities of [`ihlpsa`](@ref) for large-scale pseudospectra computations across multiple GPUs.

## Overview

The Inverse Hermitian Lanczos method in KAPseudospectra.jl supports automatic work distribution across multiple GPUs:

1. **Automatic grid partitioning**: The complex grid is divided column-wise across available devices
2. **Independent device tasks**: Each device processes its partition asynchronously
3. **Memory-aware batching**: Within each device, grid points are further batched based on available memory
4. **Result aggregation**: Device results are automatically collected and concatenated

## Backend Support

The package supports multiple GPU backends via KernelAbstractions.jl:

- **CUDA**: NVIDIA GPUs (via CUDA.jl)
- **AMDGPU**: AMD GPUs (via AMDGPU.jl / ROCm)
- **Metal**: Apple Silicon GPUs (via Metal.jl)

## Example 1: Single GPU with CUDA

Basic GPU usage with NVIDIA CUDA:

```julia
using KAPseudospectra
using LinearAlgebra
using CUDA
using KernelAbstractions

# Check GPU availability
@assert CUDA.functional()

# Create large matrix
m = 2000
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))

# Create fine grid
gx, gy, zg = qgrid(ComplexF64, (-5, 5), (-5, 5), (300, 300))

# Compute on GPU with progress tracking
backend = CUDABackend()
psa = ihlpsa(backend, zg, P, progress=true)

println("Computed pseudospectra on CUDA device")
```

### Memory Management

The function automatically determines optimal batch size:

```julia
# Automatic batching (recommended)
psa = ihlpsa(CUDABackend(), zg, P)

# Manual batch size override
zpd = 5000  # Process 5000 grid points per batch
psa = ihlpsa(CUDABackend(), zg, P, zpd=zpd)

# Query what batch size will be used
m = size(P, 1)
max_batch = findmaxbatchihl(CUDABackend(), ComplexF64, m, moe=0.1)
println("Maximum batch size: $max_batch grid points")
```

## Example 2: Multi-GPU CUDA

Distributing work across multiple NVIDIA GPUs:

```julia
using CUDA

# Check available devices
n_devices = length(CUDA.devices())
println("Found $n_devices CUDA devices")

# Create large problem
m = 3000
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))

# High-resolution grid
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (500, 500))

# Automatic multi-GPU: uses all available devices
backend = CUDABackend()
psa = ihlpsa(backend, zg, P, progress=true)

# Manual device selection: use specific GPUs
selected_devices = CUDA.devices()[1:2]  # Use first 2 GPUs
psa = ihlpsa(backend, zg, P, devs=selected_devices, progress=true)

println("Computation distributed across $(length(selected_devices)) GPUs")
```

### Load Balancing

The grid is partitioned column-wise for balanced workload:

```julia
# For a 500×500 grid across 4 GPUs:
# - GPU 0: columns 1:125    (125 columns × 500 rows = 62,500 points)
# - GPU 1: columns 126:250  (125 columns × 500 rows = 62,500 points)
# - GPU 2: columns 251:375  (125 columns × 500 rows = 62,500 points)
# - GPU 3: columns 376:500  (125 columns × 500 rows = 62,500 points)

n_devices = length(CUDA.devices())
nx, ny = size(zg)
points_per_device = nx ÷ n_devices * ny

println("Grid size: $nx × $ny = $(nx*ny) total points")
println("Points per device: ≈ $points_per_device")
```

## Example 3: AMDGPU Multi-Device

Using AMD GPUs with ROCm:

```julia
using AMDGPU
using KernelAbstractions

# Check AMDGPU availability
@assert AMDGPU.functional()

# List available ROCm devices
devices_list = AMDGPU.devices()
println("Available AMD GPUs: ", length(devices_list))

# Create problem
m = 2500
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))
gx, gy, zg = qgrid(ComplexF64, (-3, 3), (-3, 3), (400, 400))

# Compute on AMDGPU with optimized workgroup size
backend = ROCBackend()
wgs = 16  # AMDGPU performs better with smaller workgroup size
psa = ihlpsa(backend, zg, P, wgs=wgs, progress=true)

# Multi-GPU AMD
psa = ihlpsa(backend, zg, P, devs=devices_list, wgs=wgs, progress=true)
```

!!! warning "AMDGPU Threading Limitation"
    Due to a known garbage collection bug in AMDGPU.jl, Julia must run with a single thread when using AMDGPU backend. Start Julia with:
    ```bash
    julia --threads=1
    ```

## Example 4: Apple Metal

Using Apple Silicon GPUs:

```julia
using Metal
using KernelAbstractions

# Check Metal availability
@assert Metal.functional()

# Create problem
m = 1500
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (300, 300))

# Compute on Metal
backend = MetalBackend()
psa = ihlpsa(backend, zg, P, progress=true)
```

## Example 5: Comparative Benchmark

Comparing performance across backends:

```julia
using BenchmarkTools

m = 1000
A = randn(ComplexF64, m, m)
P = MatrixPencil(schur(A))
gx, gy, zg = qgrid(ComplexF64, (-3, 3), (-3, 3), (200, 200))

# CPU baseline
println("CPU computation:")
@time psa_cpu = ihlpsa(CPU(), zg, P)

# Single GPU
println("\nSingle CUDA GPU:")
@time psa_gpu1 = ihlpsa(CUDABackend(), zg, P)

# Multi-GPU (if available)
if length(CUDA.devices()) > 1
    println("\nMulti-GPU ($(length(CUDA.devices())) devices):")
    @time psa_multigpu = ihlpsa(CUDABackend(), zg, P, devs=CUDA.devices())
end

# Verify consistency
@assert maximum(abs.(psa_cpu - psa_gpu1)) < 1e-10
```

## Example 6: Large-Scale Production Run

Complete workflow for a publication-quality computation:

```julia
using KAPseudospectra
using LinearAlgebra
using CUDA
using Plots
using LaTeXStrings

# Problem setup: Large non-normal matrix
m = 5000
A = randn(ComplexF64, m, m)
A += 5I  # Shift for numerical stability

# Convert to Schur form (expensive but done once)
println("Computing Schur decomposition...")
@time P = MatrixPencil(schur(A))

# High-resolution grid
println("Creating grid...")
gx, gy, zg = qgrid(ComplexF64, (-2, 12), (-5, 5), (800, 800))
println("Grid: $(size(zg,1))×$(size(zg,2)) = $(prod(size(zg))) points")

# Query available GPUs
backend = CUDABackend()
devices_list = CUDA.devices()
println("Using $(length(devices_list)) GPUs")

# Estimate memory requirements
max_batch = findmaxbatchihl(backend, ComplexF64, m, moe=0.15)
println("Batch size per device: $max_batch grid points")

# Compute with all available GPUs
println("\nComputing pseudospectra...")
@time psa = ihlpsa(backend, zg, P, 15;
                   devs=devices_list,
                   progress=true)  # 15 Lanczos iterations for accuracy

# Visualize results
println("\nGenerating visualization...")
contour(gx, gy, psa,
    levels = 10 .^ (-8:0.5:0),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)",
    title = "Large-Scale Pseudospectra (m=$m, Multi-GPU)",
    color = :viridis,
    size = (800, 700),
    dpi = 300
)

# Sample eigenvalues for visualization (plotting all 5000 is slow)
evals = eigvals(A)
sample_idx = rand(1:length(evals), min(500, length(evals)))
scatter!(real(evals[sample_idx]), imag(evals[sample_idx]),
    marker = :cross, markersize = 2, color = :red,
    label = "Eigenvalues (sample)"
)

savefig("large_scale_pseudospectra.png")
println("Done!")
```

## Performance Optimization Tips

### 1. Workgroup Size Tuning

Different backends have optimal workgroup sizes:

```julia
# CUDA: 256 (default)
psa = ihlpsa(CUDABackend(), zg, P, wgs=256)

# AMDGPU: 16 recommended
psa = ihlpsa(ROCBackend(), zg, P, wgs=16)

# Metal: 256 (default)
psa = ihlpsa(MetalBackend(), zg, P, wgs=256)
```

### 2. Iteration Count

Trade accuracy for speed by adjusting Lanczos iterations:

```julia
m = size(P, 1)

# Fast (lower accuracy)
psa = ihlpsa(backend, zg, P, ceil(Int, log2(m)/2))

# Standard (default)
psa = ihlpsa(backend, zg, P, ceil(Int, log2(m)))

# High accuracy
psa = ihlpsa(backend, zg, P, 2*ceil(Int, log2(m)))
```

### 3. Memory Margin

Adjust safety margin for batch size calculation:

```julia
# Conservative (10% margin, default)
max_batch = findmaxbatchihl(backend, ComplexF64, m, moe=0.10)

# Aggressive (5% margin, use more memory)
max_batch = findmaxbatchihl(backend, ComplexF64, m, moe=0.05)

# Very conservative (20% margin)
max_batch = findmaxbatchihl(backend, ComplexF64, m, moe=0.20)
```

### 4. Device Selection Strategy

```julia
# Strategy 1: All devices (default)
psa = ihlpsa(backend, zg, P)

# Strategy 2: Select fastest devices
using CUDA
devices_sorted = sort(CUDA.devices(),
                     by = d -> CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE),
                     rev = true)
psa = ihlpsa(backend, zg, P, devs=devices_sorted[1:2])

# Strategy 3: Exclude busy devices
available_devices = filter(d -> CUDA.memory_status(d).free > 4e9, CUDA.devices())
psa = ihlpsa(backend, zg, P, devs=available_devices)
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

```julia
# 1. Reduce batch size manually
zpd = 1000  # Small batch
psa = ihlpsa(backend, zg, P, zpd=zpd)

# 2. Increase memory margin
max_batch = findmaxbatchihl(backend, ComplexF64, m, moe=0.25)
psa = ihlpsa(backend, zg, P, zpd=max_batch)

# 3. Use fewer devices (more memory per device)
psa = ihlpsa(backend, zg, P, devs=CUDA.devices()[1:1])

# 4. Reduce grid resolution
gx, gy, zg = qgrid(ComplexF64, (-4, 4), (-4, 4), (200, 200))  # Smaller grid
```

### Device Synchronization

The function automatically synchronizes devices, but for debugging:

```julia
# Enable CUDA error checking
CUDA.allowscalar(false)

# Synchronize manually between operations
CUDA.synchronize()

# Check for kernel errors
CUDA.device_synchronize()
```

## Next Steps

- For standard CPU usage, see [Standard Usage Example](@ref)
- For complete function reference, see [API Reference](@ref)
- For theory and algorithms, see [Pseudospectra Theory](@ref)
