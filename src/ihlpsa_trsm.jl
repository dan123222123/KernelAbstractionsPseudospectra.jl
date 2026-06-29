# Trsm strategy routing (`trsmIHL`) + the three solve drivers + the `lockstep_ihl!` inner Lanczos
# loop. The per-backend device hooks these consume — the device/array/memory interface plus
# `warp_width` / `device_smem_bytes` / `warp_trsm_safe` — all live in src/backend.jl (overridden by
# the GPU extensions). Split out of ihlpsa.jl; included after ihlpsa_workspace.jl.

# Workgroup size for the column trsm kernel: the warp width (capped at m) on GPU, 1 on CPU. Override
# via the `wgs` kwarg to ihlpsa. (`warp_width` is a per-backend hook in src/backend.jl.) NOTE: the
# tiled SHUFFLE solve does NOT use this — its kernels are intrinsically 32-wide.
default_wgs(backend, m) = KernelAbstractions.isgpu(backend) ? min(m, warp_width(backend)) : 1

# Whether the tiled solve can run for pencil `P` on this device. Each trailing tile is
# 32 × TC × sizeof(ET); the generic (B≠I) kernels use two tiles (sA+sB), the B=I kernels one. The tile
# WIDTH TC is a tunable knob (`tiled_tc`), so this checks the NARROWEST candidate — i.e. tiled is
# usable as long as SOME TC fits `device_smem_bytes`. That lets a wide non-IEEE pencil (MultiFloats:
# Float64xN etc.) whose 32×32 tile would overflow still tile at a narrow TC instead of dropping to the
# `column` solve; only a type too wide for even the narrowest TC (or a tiny-shared-memory device) is
# rejected. `tiled_tc` then picks the actual width — always one that fits, by construction.
function tiled_tiles_fit(backend, P)
    ntiles = b_is_identity(P) ? 1 : 2
    tile = sizeof(eltype(P.A)) * 32 * minimum(_TC_CANDIDATES)
    return ntiles * tile <= device_smem_bytes(backend)
end

# Trailing-tile COLUMN width — the `Val{TC}` of the trailing kernels (src/KATRSM.jl/trsm_tiled_kernels.jl).
# A shared-memory occupancy knob DECOUPLED from the 32-wide panel solve: a narrower TC shrinks the
# 32×TC @localmem tile, raising resident blocks/SM (≈ occupancy) on a shared-memory-bound device and,
# measured on a 1080 Ti, the trailing update ~1.3–1.7× faster — at the cost of ⌈plen/TC⌉ tile reloads
# per panel (A,B DRAM traffic is unchanged). TC=32 reproduces the original single-tile sweep.
#
# Resolution order: `KAPSEUDO_TRSM_TC` env var > a value persisted by the `tune_trsm_tc!` probe
# (src/tune.jl), keyed per element type + eye/generic > the analytic estimate `_auto_tiled_tc`. The
# probe is the accurate per-device path (it MEASURES the compute+bandwidth tradeoff directly, so it
# needs no occupancy model); the analytic estimate is the zero-setup default when no probe has run.
# Only called for pencils that passed `tiled_tiles_fit`, so the 32×32 tile fits and TC=32 is valid.
const _TC_CANDIDATES = (32, 16, 8)         # largest → smallest; 32 reproduces the pre-optimization sweep
const _MAX_BLOCKS_PER_SM = 32              # NVIDIA/AMD per-SM resident-block cap (Pascal…Hopper, CDNA/RDNA)
const _TARGET_BLOCKS = 3 * _MAX_BLOCKS_PER_SM ÷ 4   # ~¾ of the cap = the measured occupancy sweet spot
_tc_pref_key(T, eye::Bool) = "trsm_tc_$(T)_$(eye ? "eye" : "gen")"

# The pref/analytic resolution does a `@load_preference` (a TOML read) — too slow to repeat on every
# `_tiled_trsm!` call (it dominates tiny solves). Resolve once per (element type, eye) and cache.
# The env override is never cached (so `tune_trsm_tc!` / experiments can force a value per call); the
# probe empties this after persisting so a same-session tune takes effect. Lock-guarded since the
# multi-GPU driver may resolve concurrently (all devices compute the same value — idempotent — the
# lock just protects the Dict). Keyed by type+eye, not device: a session is assumed to use one GPU
# model (true for the homogeneous multi-GPU box and typical single-GPU setups).
const _TC_CACHE = Dict{Tuple{DataType,Bool},Int}()
const _TC_CACHE_LOCK = ReentrantLock()
_clear_tc_cache!() = (@lock _TC_CACHE_LOCK empty!(_TC_CACHE); nothing)

function tiled_tc(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_TC")
        tc = parse(Int, ENV["KAPSEUDO_TRSM_TC"])
        tc in _TC_CANDIDATES || error("KAPSEUDO_TRSM_TC must be one of $(_TC_CANDIDATES) (got $tc)")
        return tc
    end
    @lock _TC_CACHE_LOCK get!(_TC_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = @load_preference(_tc_pref_key(eltype(P.A), b_is_identity(P)), nothing)
        tc = tuned === nothing ? nothing : tryparse(Int, tuned)
        (tc !== nothing && tc in _TC_CANDIDATES) ? tc : _auto_tiled_tc(backend, P)
    end
end

# Analytic per-device+type estimate — NO benchmark run. The end-to-end optimum is NOT the narrowest
# tile: more sub-tiles mean more @localmem reloads / panel passes, whose overhead eventually outweighs
# the occupancy gain (measured 1080 Ti: the 1-tile eye solve peaks at TC=16, the 2-tile generic at
# TC=8 — both ≈24 resident blocks/SM). So target the occupancy SWEET SPOT, ~¾ of the per-SM block cap
# resident, and use the LARGEST TC that reaches it (least narrowing → least overhead); fall back to
# the narrowest that fits a block if none do. `device_smem_per_sm` gives the per-SM budget for the
# blocks-per-SM estimate (= smem_sm ÷ tile, capped by the hardware block limit). NOTE the target is a
# fixed ¾-of-cap heuristic — the true sweet spot also depends on the type's arithmetic intensity and
# the device's compute:bandwidth ratio, which is exactly what `tune_trsm_tc!` measures directly.
function _auto_tiled_tc(backend, P)
    ntiles = b_is_identity(P) ? 1 : 2
    elt = sizeof(eltype(P.A))
    smem_block = device_smem_bytes(backend)        # per-block shared-memory limit
    smem_sm = device_smem_per_sm(backend)          # per-SM shared memory (occupancy estimate)
    best = 0
    for tc in _TC_CANDIDATES                        # largest first: take the least narrowing that suffices
        tile = ntiles * 32 * tc * elt
        tile <= smem_block || continue              # must fit a single block
        best = tc
        min(smem_sm ÷ tile, _MAX_BLOCKS_PER_SM) >= _TARGET_BLOCKS && return tc   # hit the occupancy target
    end
    return best == 0 ? last(_TC_CANDIDATES) : best                               # nothing fit (shouldn't happen post tiled_tiles_fit) → narrowest
end

# non-cpu solve step in lockstep_ihl!
#
# Two GPU solve kernels are available:
#   * column-oriented (default): a `@synchronize()`-per-column workgroup kernel — barrier-based,
#     shuffle-free, correct for every element type and backend. The portable default.
#   * tiled (opt-in `tiled`): a right-looking blocked solve with shared-memory A,B-tile reuse across
#     the grid-point batch (panel solves broadcast pivots by `@shfl`). Bandwidth-optimal and the
#     performance path on a shuffle-capable backend; it SELF-GATES to the column solve wherever the
#     shuffle / shared memory isn't usable, so `tiled` is safe to request on any backend.
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=missing)
    wgs = ismissing(wgs) ? default_wgs(backend, size(P, 1)) : wgs
    # The default `column` needs no routing info, so return early — this keeps the per-iteration
    # device-property queries (`device_smem_bytes`, etc.) off the default path.
    if trsm_strategy() == "column"
        return _column_trsm!(backend, bV, zv, P, wgs)
    end
    # `tiled`: run the tiled solve where it's usable for this backend+type, else fall back to the
    # shuffle-free column solve (so requesting `tiled` is always correct). "Usable" needs BOTH: SOME
    # trailing-tile width fits this device's shared memory (`tiled_tiles_fit` checks the narrowest
    # 32×TC×{1 if B=I else 2}·sizeof(ET) vs `device_smem_bytes`; `tiled_tc` then picks the actual TC)
    # AND a hardware warp shuffle usable here (`warp_trsm_safe`; false on stock oneAPI / Metal-without-
    # opt-in, and for wide non-IEEE types lacking the per-limb `_trsm_shfl` override —
    # MultiFloatsPseudospectra). Because the tile width adapts, a wide non-IEEE pencil (MultiFloats:
    # Float64xN etc.) tiles at a narrow TC rather than dropping to `column`; only a type too wide for
    # even the narrowest TC falls back.
    wide       = !(real(eltype(P.A)) <: Base.IEEEFloat)  # non-IEEE (MultiFloats/BigFloat)
    tiled_ok   = tiled_tiles_fit(backend, P)             # tiled's @localmem tiles fit device shared memory
    shuffle_ok = warp_trsm_safe(backend, wide)           # tiled shuffle usable for this backend+type
    if shuffle_ok && tiled_ok
        _tiled_trsm!(backend, bV, zv, P, wgs)
    else
        _column_trsm!(backend, bV, zv, P, wgs)
    end
end

# Tiled / blocked solve: right-looking panel sweep with shared-memory A,B-tile reuse across
# the grid-point batch (see src/KATRSM.jl/trsm_tiled_kernels.jl). `z` is conjugated once for
# the forward (lower-tri Ac,Bc) sweep; the backward sweep uses (A,B,z) directly. `gt` (grid
# points per trailing tile = the A,B reuse factor) is tunable via KAPSEUDO_TRSM_GT.
function _tiled_trsm!(backend, bV, zv, P, wgs)
    m = size(P, 1)
    g = length(zv)
    gt = parse(Int, get(ENV, "KAPSEUDO_TRSM_GT", "32"))
    gt >= 1 || error("KAPSEUDO_TRSM_GT must be a positive integer (got $gt)")
    vtc = Val(tiled_tc(backend, P))   # trailing-tile column width (shared-mem occupancy knob, per device+type)
    nblk = cld(m, 32)
    zc = conj(zv)
    eye = b_is_identity(P)   # B = I ⇒ sB-free trailing kernels (half the shared memory)
    # forward (lower-triangular), panels ascending
    for k in 1:nblk
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        if eye
            @views _tiled_panel_forward_eye(backend, 32)(bV, zc, P.Ac, koff, plen; ndrange=(32, g))
        else
            @views _tiled_panel_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen; ndrange=(32, g))
        end
        rbase = koff + plen
        ntrail = m - rbase
        if ntrail > 0
            rtiles = cld(ntrail, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_forward_eye(backend, 32)(bV, P.Ac, koff, plen, rbase, m, gt, rtiles, vtc; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_forward(backend, 32)(bV, zc, P.Ac, P.Bc, koff, plen, rbase, m, gt, rtiles, vtc; ndrange=32 * rtiles * ggrid)
            end
        end
    end
    # backward (upper-triangular), panels descending
    for k in nblk:-1:1
        koff = (k - 1) * 32
        plen = min(32, m - koff)
        if eye
            @views _tiled_panel_backward_eye(backend, 32)(bV, zv, P.A, koff, plen; ndrange=(32, g))
        else
            @views _tiled_panel_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen; ndrange=(32, g))
        end
        if koff > 0
            rtiles = cld(koff, 32)
            ggrid = cld(g, gt)
            if eye
                @views _tiled_trailing_backward_eye(backend, 32)(bV, P.A, koff, plen, gt, rtiles, vtc; ndrange=32 * rtiles * ggrid)
            else
                @views _tiled_trailing_backward(backend, 32)(bV, zv, P.A, P.B, koff, plen, gt, rtiles, vtc; ndrange=32 * rtiles * ggrid)
            end
        end
    end
end
function _column_trsm!(backend, bV, zv, P, wgs)
    g = length(zv)
    if b_is_identity(P)   # B = I ⇒ skip the identity-B reads (the eye column kernels)
        @views _batched_column_oriented_forward_solve_eye(backend, wgs)(bV, conj(zv), P.Ac; ndrange=(wgs, g))
        @views _batched_column_oriented_backward_solve_eye(backend, wgs)(bV, zv, P.A; ndrange=(wgs, g))
    else
        @views _batched_column_oriented_forward_solve_pencil(backend, wgs)(bV, conj(zv), P.Ac, P.Bc; ndrange=(wgs, g))
        @views _batched_column_oriented_backward_solve_pencil(backend, wgs)(bV, zv, P.A, P.B; ndrange=(wgs, g))
    end
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

# (All per-backend device hooks — the device/array/memory interface, supports_fp64, warp_width,
# device_smem_bytes, warp_trsm_safe — live in src/backend.jl, overridden by the GPU extensions.)
