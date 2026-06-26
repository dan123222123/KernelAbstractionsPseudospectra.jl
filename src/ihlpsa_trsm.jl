# Trsm-specific device hooks (warp width, shared-memory budget, warp-shuffle safety), the
# `trsm_strategy` routing (`trsmIHL`) + the three solve drivers, and the `lockstep_ihl!` inner
# Lanczos loop. The general per-backend device interface lives in src/backend.jl. Split out of
# ihlpsa.jl; included after ihlpsa_workspace.jl.

## DEVICE FUNCTIONS ##

# Warp/subgroup width used by the register-warp / tiled solves (one shuffle domain). The default
# is 32, but this is a per-backend QUERY hook, not a baked constant: each GPU extension overrides
# it with the actual hardware value — `CUDA.warpsize` (32), `AMDGPU.wavefront_size()` (32 on RDNA,
# 64 on CDNA), 32 for Metal SIMD-groups, and 32 for Intel under the SIMD32 pin
# (`set_intel_force_simd32!`). KernelAbstractions has no portable pre-launch query, so the value
# comes from each backend's own API in its extension.
warp_width(backend) = 32
# Workgroup size for the trsm kernels: the warp width (capped at m) on GPU, 1 on CPU. Override via
# the `wgs` kwarg to ihlpsa.
default_wgs(backend, m) = KernelAbstractions.isgpu(backend) ? min(m, warp_width(backend)) : 1

# Shared-memory bytes per workgroup available to the tiled solve's `@localmem` tiles. Default a
# conservative 48 KB (the static-shared-memory limit on Volta/Turing/early-Ampere); each GPU
# extension overrides it with the real device query so the tiled-vs-column routing uses the
# actual per-device budget instead of an assumed constant.
device_smem_bytes(backend) = 48 * 1024

# Whether the tiled solve's `@localmem` tiles fit this device's shared memory for pencil `P`.
# Each trailing tile is 32×32×sizeof(ET); the generic (B≠I) kernels use two tiles (sA+sB), the
# B=I kernels one. This replaces the former `!wide || b_is_identity` type-proxy with the actual
# computed footprint vs the queried per-device limit — so a wide B≠I pencil can still tile on a
# large-shared-memory device, and a normally-fitting type is correctly rejected on a tiny one.
function tiled_tiles_fit(backend, P)
    tile = sizeof(eltype(P.A)) * 32 * 32
    ntiles = b_is_identity(P) ? 1 : 2
    return ntiles * tile <= device_smem_bytes(backend)
end

# Whether the register-warp / tiled solves (which broadcast pivots with warp shuffles) are
# correct under the "auto" strategy on this backend. True for CUDA / AMDGPU / Metal: fixed
# warp/wavefront/SIMD width + a hardware shuffle. On oneAPI it requires BOTH the
# KernelIntrinsics oneAPI shuffle backend AND a pinned SIMD width, so the oneAPI extension
# overrides this; without them, `auto` stays on the shuffle-free `column` solve — correct,
# just not the fast path. (Explicit KAPSEUDO_TRSM=warp/tiled is opt-in and not gated.)
# `wide` is true for non-IEEE element types (MultiFloats / BigFloat); the oneAPI override
# keeps those on `column` regardless (their warp/tiled SPIR-V codegen is not yet functional).
warp_trsm_safe(backend, wide) = true

# non-cpu solve step in lockstep_ihl!
#
# Two GPU solve kernels are available:
#   * warp-register (default): one warp/grid-point with the RHS held in registers and
#     pivots broadcast by warp shuffle — no block barriers, no global round-trips on b.
#     Specialized on R = cld(m, ws) (a Val), so one compile per matrix size m (m is fixed
#     per ihlpsa run; the survivor count g stays the dynamic ndrange).
#   * column-oriented (fallback): the original `@synchronize()`-per-column kernel. Select
#     it with KAPSEUDO_WARP_TRSM=0.
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=missing)
    wgs = ismissing(wgs) ? default_wgs(backend, size(P, 1)) : wgs
    strat = trsm_strategy()
    # The default `column` (and explicit `warp`) need no routing info, so return early — this keeps
    # the per-iteration device-property queries (`device_smem_bytes`, etc.) off the default path.
    if strat == "column"
        return _column_trsm!(backend, bV, zv, P, wgs)
    elseif strat == "warp"
        return _warp_trsm!(backend, bV, zv, P, wgs)
    end
    # `tiled` / `auto`: element-type + size routing for the GPU inner solve.
    #  * The warp solve is `@generated` on R = ⌈m/32⌉: excellent at small m, but its compile
    #    time and register pressure blow up with R (≈26 s compile at R=16, register-bound
    #    occupancy collapse for wide elements). So large problems avoid warp.
    #  * The tiled solve's trailing-update `@localmem` tiles must fit this device's shared memory
    #    (`tiled_tiles_fit` computes 32²·sizeof(ET)·{1 if B=I else 2} vs the queried
    #    `device_smem_bytes`). IEEE floats fit on any GPU; a wide (non-IEEE: MultiFloats/BigFloat)
    #    generalized pencil needs two tiles and typically overflows, falling back to the shuffle-
    #    free column solve, while a wide B=I pencil uses the single-tile sB-free kernels and fits.
    #    Wide warp/tiled use the per-limb `_trsm_shfl` override (MultiFloatsPseudospectra).
    #    Size threshold for warp→tiled = `trsm_crossover()`.
    wide = !(real(eltype(P.A)) <: Base.IEEEFloat)
    big = size(P, 1) >= trsm_crossover()
    tiled_ok = tiled_tiles_fit(backend, P)         # tiled's @localmem tiles fit device shared memory
    if strat == "tiled"
        tiled_ok ? _tiled_trsm!(backend, bV, zv, P, wgs) :
                   (big ? _column_trsm!(backend, bV, zv, P, wgs) : _warp_trsm!(backend, bV, zv, P, wgs))
    elseif !warp_trsm_safe(backend, wide)           # "auto" where shuffle isn't safe (stock oneAPI) / unsupported (oneAPI MultiFloats) → column
        _column_trsm!(backend, bV, zv, P, wgs)
    elseif big && tiled_ok                          # "auto", large m, tiles fit (IEEE, or wide B=I)
        _tiled_trsm!(backend, bV, zv, P, wgs)
    elseif big                                      # "auto", large m, wide B≠I → no tiled
        _column_trsm!(backend, bV, zv, P, wgs)
    else                                            # "auto", small m
        _warp_trsm!(backend, bV, zv, P, wgs)
    end
end

# Warp-register solve. Uses the portable KA + KernelIntrinsics kernels on every backend.
# `_warp_trsm!` stays a separate dispatch hook (rather than calling `_warp_trsm_ka!` directly at
# the call site) so a backend extension could specialize it later, but no extension overrides it
# today — the earlier opt-in CUDA-native (`@cuda` + `CUDA.shfl_sync`) override was removed.
_warp_trsm!(backend, bV, zv, P, wgs) = _warp_trsm_ka!(backend, bV, zv, P, wgs)

# Tiled / blocked solve: right-looking panel sweep with shared-memory A,B-tile reuse across
# the grid-point batch (see src/KATRSM.jl/trsm_tiled_kernels.jl). `z` is conjugated once for
# the forward (lower-tri Ac,Bc) sweep; the backward sweep uses (A,B,z) directly. `gt` (grid
# points per trailing tile = the A,B reuse factor) is tunable via KAPSEUDO_TRSM_GT.
function _tiled_trsm!(backend, bV, zv, P, wgs)
    m = size(P, 1)
    g = length(zv)
    gt = parse(Int, get(ENV, "KAPSEUDO_TRSM_GT", "32"))
    nblk = cld(m, 32)
    zc = conj(zv)
    eye = b_is_identity(P)   # B = I ⇒ sB-free trailing kernels (half the shared memory)
    # forward (lower-triangular), panels ascending
    for k in 1:nblk
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        @views _tiled_panel_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen; ndrange=(32, g))
        rbase = koff + plen
        ntrail = m - rbase
        if ntrail > 0
            rtiles = cld(ntrail, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_forward_eye(backend, 32)(bV, P.Ac, koff, plen, rbase, m, gt, rtiles; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen, rbase, m, gt, rtiles; ndrange=32 * rtiles * ggrid)
            end
        end
    end
    # backward (upper-triangular), panels descending
    for k in nblk:-1:1
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        @views _tiled_panel_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen; ndrange=(32, g))
        if koff > 0
            rtiles = cld(koff, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_backward_eye(backend, 32)(bV, P.A, koff, plen, gt, rtiles; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen, gt, rtiles; ndrange=32 * rtiles * ggrid)
            end
        end
    end
end
function _warp_trsm_ka!(backend, bV, zv, P, wgs)
    g = length(zv)
    R = cld(size(P, 1), wgs)
    @views _batched_warp_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc, Val(R); ndrange=(wgs, g))
    @views _batched_warp_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B, Val(R); ndrange=(wgs, g))
end
function _column_trsm!(backend, bV, zv, P, wgs)
    g = length(zv)
    @views _batched_column_oriented_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc; ndrange=(wgs, g))
    @views _batched_column_oriented_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B; ndrange=(wgs, g))
end

# cpu solve step in lockstep_ihl!. `wgs` is accepted and ignored so this CPU method
# isn't shadowed by the generic (column-oriented) trsmIHL when called with `; wgs`.
function trsmIHL(backend::CPU, bV, zv, P::SchurMatrixPencil; wgs=missing)
    g = length(zv)
    _batched_forward_solve_pencil(backend)(bV, conj(zv), P.A', P.B', ndrange=g)
    _batched_backward_solve_pencil(backend)(bV, zv, P.A, P.B, ndrange=g)
end

# Runs Lanczos iterations `start:nit` on the first g grid points of the batch.
# `start == 1` (the default) seeds q₁ from x₀ first; `start > 1` RESUMES a prior
# call: the loop body at iteration n only reads β[n] plus the Qv[1]/Qv[2]/v state
# left in the workspace, so continuing is just running more iterations. Resume
# contract: same `ihl` instance (Qv/v/zv untouched since the previous call), the
# same `α`/`β` arrays (sized for the deepest nit up front), the same g, and the
# previous call ended at iteration start-1.
function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs=missing, start::Integer=1)
    backend = get_backend(ihl)
    if start == 1
        # `_qₙnext` writes Qv[2] (= q₁); `_v2v` then copies all m elements of
        # each q₁ into v[1:g] before the trsm loop runs. v[1:g] is fully
        # overwritten there, so no separate seed copy is needed.
        _qₙnext(backend)(view(ihl.x₀, 1:g), view(β, 2, 1:g), view(ihl.Qv[2], 1:g, :), ndrange=g)
        _v2v(backend)(view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), ndrange=g)
    end
    for n = start:nit
        trsmIHL(backend, view(ihl.v, 1:g), view(ihl.zv, 1:g), ihl.P; wgs)
        _ihl_ttr_qₙnext(backend)(view(β, n, 1:g), view(ihl.Qv[1], 1:g, :), view(α, n, 1:g), view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), view(β, n + 1, 1:g), ndrange=g)
    end
    synchronize(backend)
end

# The general per-backend device interface (get_bgarray / device / devices / device! /
# device_bytes_available / device_reclaim / supports_fp64) lives at module level in
# `src/backend.jl` — it's the broad backend abstraction, not trsm-specific. Only the
# trsm-specific device hooks above (warp_width, device_smem_bytes, warp_trsm_safe) stay here.

## END DEVICE FUNCTIONS ##
