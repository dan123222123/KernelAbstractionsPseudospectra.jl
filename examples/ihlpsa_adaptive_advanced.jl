## Adaptive-nit ihlpsa — advanced features
#
# Companion to `ihlpsa_adaptive.jl`. That script shows the drop-in adaptive call;
# this one exercises the intricate machinery added alongside it on this branch:
#
#   1. generalized matrix pencil  B ≠ I  with perturbation weights γ, δ
#   2. convergence-control knobs: rtol, atol, nconfirm, nit_chunk, nit_max
#   3. reproducibility via the deterministic `seed` (fixed x₀ reused across chunks)
#   4. the nit_max cap path (graceful @warn + fallback, not an error)
#   5. zpd memory-capping → the multi-batch resident-state gather (runs on CPU!)
#   6. multi-device fan-out via `devs` + the round-robin column partition
#   7. the per-point nit_grid as a convergence census
#
# Everything here runs on CPU out of the box. For GPU, see README.md and the
# commented backend lines below. Run the `##` cells block by block.
using LinearAlgebra, MatrixDepot
using KernelAbstractions
using KAPseudospectra
using Random
##

## backend — CPU default (see README.md to add a GPU backend)
backend = CPU()
#using CUDA;   backend = CUDABackend()
#using AMDGPU; backend = ROCBackend()
#using Metal;  backend = MetalBackend()    # Float32 only — no FP64
#using oneAPI; backend = oneAPIBackend()   # Float32 only on FP64-less iGPUs
wgs = 256                                  # 16 AMDGPU · 32 Intel · 256 CPU/CUDA
##

## ── 1. Generalized pencil B ≠ I with weights γ, δ ──────────────────────────────
# A standard pencil is B = I (zB − A = zI − A). `MatrixPencil(A, B)` builds the
# generalized pencil zB − A; the adaptive driver carries γ, δ through unchanged,
# with each point's value (γ + δ|z|)·σ_min(zB − A). γ+δ ≈ 1 keeps it a structured
# pseudospectrum.
T = ComplexF32
m = 24
Random.seed!(0xBEEF)
A = randn(T, m, m)
B = randn(T, m, m) + T(5)I                 # well-conditioned B ≠ I
_, _, zg = qgrid(T, (-1.5, 1.5), (-1.5, 1.5), (60, 60))

P_std = MatrixPencil(A)                     # B = I
P_gen = MatrixPencil(A, B)                  # B ≠ I
srg_std = ihlpsa(backend, zg, P_std; wgs)
srg_gen = ihlpsa(backend, zg, P_gen; wgs, γ=0.5, δ=0.5)
@info "standard vs generalized pencil" std_extrema = extrema(srg_std) gen_extrema = extrema(srg_gen)
##

## ── 2. Convergence-control knobs: rtol, nconfirm, nit_chunk ────────────────────
# Each knob trades depth (cost) against how strictly convergence is declared.
# Tabulate the deepest point and mean depth for a few settings on the same grid.
P = P_std
println("  rtol     nconfirm  nit_chunk |  max_depth  mean_depth")
for (rtol, nconfirm, nit_chunk) in [
    (1e-3, 2, 2),       # loose tol
    (1e-5, 2, 2),       # tight tol → more depth
    (1e-5, 3, 2),       # stricter confirmation → a touch more
    (1e-5, 2, 4),       # coarser chunks → retires later, over-iterates a bit
]
    _, ng = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, rtol, nconfirm, nit_chunk)
    println("  $(rpad(rtol,8)) $(rpad(nconfirm,9)) $(rpad(nit_chunk,9)) |  $(rpad(maximum(ng),9)) $(sum(ng)/length(ng))")
end
##

## ── 3. Reproducibility: the deterministic seed ────────────────────────────────
# Adaptive reuses ONE fixed start vector x₀ across all chunks (so successive σ are
# one Lanczos run sampled at increasing depth — that's what makes the convergence
# test meaningful). The vector is derived from `seed`, so runs are bit-reproducible;
# change `seed` and the path — hence the per-point depths — can shift slightly.
s1, n1 = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, seed=0x1234)
s2, n2 = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, seed=0x1234)
s3, n3 = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, seed=0x9999)
@info "seed reproducibility" same_seed_identical = (s1 == s2 && n1 == n2) diff_seed_maxdepth = (maximum(n1), maximum(n3))
# You can also pass your own start vector directly via x₀ (overrides seed).
##

## ── 4. The nit_max cap (graceful degradation, not an error) ────────────────────
# With nconfirm=2 a point needs ≥2 confirming checkpoints; nit_max=3 can't supply
# them, so every point hits the cap. The driver @warns, fills nit_grid with the cap,
# and falls back to the deepest-chunk σ — it does NOT error or return garbage.
s_cap, n_cap = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs, nit_max=3, rtol=1e-12)
@info "nit_max cap" all_at_cap = all(==(3), n_cap) finite = all(isfinite, s_cap)
##

## ── 5. zpd memory-capping → the multi-batch resident-state gather ──────────────
# `zpd` (z-points per device) caps how many grid points are resident at once. A
# small zpd forces several batches that each run the per-point gather machinery;
# the result must match the single-batch default exactly (batching changes only
# the stopping granularity). This exercises the multi-batch path WITHOUT a GPU.
s_one = ihlpsa(backend, zg, P; wgs)                    # single batch (default)
s_many = ihlpsa(backend, zg, P; wgs, zpd=400)          # 3600 pts → 9 batches
@info "zpd multi-batch" batches = cld(length(zg), 400) matches_single = isapprox(s_one, s_many; rtol=1e-4)
##

## ── 6. Multi-device fan-out via `devs` + round-robin partition ─────────────────
# Grid columns fan out across all devices of `backend` by default; pass `devs` to
# restrict to a subset. Columns are assigned round-robin (strided) so spatially
# clustered hard regions spread across devices instead of piling onto one — see
# DESIGN.md and adaptive_v2.md. CPU reports a single device, so this is a no-op
# here, but the API is identical on GPU:
@info "devices visible to this backend" devs = KAPseudospectra.devices(backend)
#   srg = ihlpsa(CUDABackend(), zg, P; devs=KAPseudospectra.devices(CUDABackend())[1:2])  # first 2 GPUs
#   # KAPSEUDO_STRIDED=0 julia …   # opt back into legacy contiguous bands
##

## ── 7. Per-point nit_grid as a convergence census ─────────────────────────────
# The depth map is the empirical convergence distribution (cf. DESIGN.md: most
# points converge shallow, a hard tail runs deep). Summarize it as a histogram.
_, ng = KAPseudospectra._ihlpsa_adaptive(backend, zg, P; wgs)
depths = sort(unique(ng))
println("depth : fraction of grid points retired at that depth")
for d in depths
    frac = count(==(d), ng) / length(ng)
    println("  $(lpad(d,3))  : $(round(100*frac, digits=1))%  ", "█"^round(Int, 50*frac))
end
println("mean depth = ", sum(ng)/length(ng), "   (a fixed-nit run would pay max = ", maximum(ng), " everywhere)")
##

## ── 8. (optional) plot the census + the depth map ─────────────────────────────
# using Plots
# h = histogram(vec(ng); bins=minimum(ng):maximum(ng)+1, legend=false,
#               xlabel="Lanczos depth at retirement", ylabel="grid points",
#               title="adaptive convergence census")
# hm = heatmap(ng; color=:viridis, title="depth per grid point", aspect_ratio=1)
# plot(h, hm; layout=(1, 2), size=(1400, 600)) |> display
##
