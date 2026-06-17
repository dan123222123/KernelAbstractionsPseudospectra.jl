# Correctness of the tiled + warp KA/KI solves vs the column-oriented baseline, end-to-end
# through ihlpsa, across element types and matrix sizes incl. non-multiples of 32 (partial
# last panel). Single device, small grid (fast).
#   unset LD_LIBRARY_PATH; julia --project=bench bench/tiled_check.jl
using KAPseudospectra, LinearAlgebra, Random, Printf, CUDA

backend = CUDABackend()
dev = collect(CUDA.devices())[1]
run(flag, zg, P, nit) = withenv("KAPSEUDO_TRSM" => flag) do
    ihlpsa(backend, zg, P, nit; devs=[dev])
end

@printf("%-12s %-6s %-14s %-14s\n", "T", "m", "tiled-vs-col", "warp-vs-col")
for T in (ComplexF32, ComplexF64), m in (32, 64, 100, 128, 256, 300, 512)
    rng = Random.seed!(2024)
    P = MatrixPencil(schur(randn(rng, T, m, m)))
    _, _, zg = KAPseudospectra.qgrid(T, (-3, 3), (-3, 3), (50, 50))
    nit = 12
    σc = run("column", zg, P, nit)
    σt = run("tiled", zg, P, nit)
    σw = run("warp", zg, P, nit)
    dt = maximum(abs.(σt .- σc)) / max(maximum(abs.(σc)), eps())
    dw = maximum(abs.(σw .- σc)) / max(maximum(abs.(σc)), eps())
    tol = T === ComplexF32 ? 1e-4 : 1e-10
    @printf("%-12s %-6d %-14s %-14s\n", T, m,
            @sprintf("%.2e %s", dt, dt < tol ? "ok" : "FAIL"),
            @sprintf("%.2e %s", dw, dw < tol ? "ok" : "FAIL"))
end
