# Edge cases that catch off-by-ones in the IHL pipeline:
#  - m=1: degenerate Lanczos, trivially-sized SymTridiagonal in ihlsrg
#  - m=2: smallest non-trivial pencil
#  - nit=1: single Lanczos step, β has 2 rows but β[2:end-1] is empty
#  - long-aspect grid: nx ≠ ny, exercises the transpose/permutedims at return
#  - findmaxbatchihl returns sane positive values on each backend

using Test, KAPseudospectra, KernelAbstractions, LinearAlgebra
using KAPseudospectra: findmaxbatchihl, _device_column_partition

@testset "edge cases" begin
    @testset "m=1" begin
        A = reshape([2.0 + 0.0im], 1, 1)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        # Default nit was clamped to ≥ 1 to handle this.
        s = ihlpsa(CPU(), zg, P)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "m=2" begin
        A = ComplexF64[1.0 0.5; 0.0 2.0]
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        s = ihlpsa(CPU(), zg, P)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "nit=1" begin
        A = randn(ComplexF64, 16, 16)
        P = MatrixPencil(A)
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 4))
        s = ihlpsa(CPU(), zg, P, 1)
        @test all(isfinite, s)
        @test size(s) == (4, 4)
    end

    @testset "long-aspect grid" begin
        A = randn(ComplexF64, 8, 8)
        P = MatrixPencil(A)
        # nx=4, ny=32. ihlpsa returns permutedims(result), so output shape is
        # (ny, nx) = (32, 4).
        gx, gy, zg = qgrid(ComplexF64, (-1.0, 1.0), (-1.0, 1.0), (4, 32))
        s = ihlpsa(CPU(), zg, P)
        @test size(s) == (32, 4)
        @test all(isfinite, s)
    end
end

@testset "findmaxbatchihl sanity" begin
    # CPU path uses Sys.free_memory(); should be >> any reasonable single-pencil
    # workspace, so the returned batch should be huge but finite.
    n_cpu = findmaxbatchihl(CPU(), ComplexF64, 32, 5)
    @test n_cpu > 0
    @test isfinite(n_cpu)

    if isdefined(Main, :AMDGPU) && AMDGPU.functional()
        n_gpu = findmaxbatchihl(ROCBackend(), ComplexF64, 32, 5)
        @test n_gpu > 0
        @test isfinite(n_gpu)
    end
    if isdefined(Main, :CUDA) && CUDA.functional()
        n_gpu = findmaxbatchihl(CUDABackend(), ComplexF64, 32, 5)
        @test n_gpu > 0
        @test isfinite(n_gpu)
    end
end

@testset "_device_column_partition invariants" begin
    # The multi-device dispatch in ihlpsa indexes zgidxbatches[1:nblocks] and
    # zips blocks against devices, so the partition must always yield exactly
    # min(ndev, ncols) contiguous, ordered, balanced blocks. The old ceil-based
    # Iterators.partition could yield fewer blocks than devices (e.g. 9 cols /
    # 4 devs → 3 blocks of 3) and BoundsError the device loops — these are the
    # host-side regression tests for that fix (there is no GPU CI).
    for ncols in 1:20, ndev in 1:6
        blocks = _device_column_partition(ncols, ndev)
        @test blocks isa Vector{UnitRange{Int}}
        @test length(blocks) == min(ndev, ncols)
        # coverage + ordering + disjointness in one shot
        @test reduce(vcat, collect.(blocks)) == collect(1:ncols)
        # balance: block sizes differ by at most 1
        @test maximum(length.(blocks)) - minimum(length.(blocks)) <= 1
    end

    # Pinned regression cases (verified failures of the old partition):
    @test _device_column_partition(1, 4) == [1:1]              # was 1 block, loop hit [2]
    @test _device_column_partition(9, 4) == [1:3, 4:5, 6:7, 8:9]  # was 3,3,3 (3 blocks)
    @test _device_column_partition(2, 4) == [1:1, 2:2]
    @test isempty(_device_column_partition(0, 4))
    @test_throws ArgumentError _device_column_partition(4, 0)
end
