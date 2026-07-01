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
# 32×TC @localmem tile, raising resident blocks/SM at the cost of ⌈plen/TC⌉ tile reloads per panel
# (A,B DRAM traffic is unchanged). TC=32 reproduces the original single-tile sweep. See DESIGN_TRSM.md
# "Trailing-tile width and occupancy" for the measured tradeoff.
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

# Warps per trailing-update block — the `Val{W}` of the trailing kernels. The W warps SHARE one
# 32×TC @localmem tile (shared mem/block UNCHANGED) and split the `gt` grid points across warps, so
# the shared-mem-capped resident blocks/SM each carry W warps → occupancy rises ~W× off the W=1
# ceiling until registers bind. W=1 reproduces the original single-warp launch exactly; the result is
# bit-identical across W (per-(row,grid-point) FMA order is unchanged). Resolution order mirrors
# `tiled_tc`: `KAPSEUDO_TRSM_W` env > a value persisted by `tune_trsm_tc!` (per type+eye) >
# `_DEFAULT_W`. W is independent of `tiled_tiles_fit` (it does not change shared-mem/block); the only
# device limit is registers, which the tuner discovers by skipping any (TC,W) whose launch fails.
const _W_CANDIDATES = (1, 2, 4, 8)
const _DEFAULT_W = 4                        # A100 ComplexF64 sweet spot; the tuner refines per device+type
_w_pref_key(T, eye::Bool) = "trsm_w_$(T)_$(eye ? "eye" : "gen")"
const _W_CACHE = Dict{Tuple{DataType,Bool},Int}()
const _W_CACHE_LOCK = ReentrantLock()
_clear_w_cache!() = (@lock _W_CACHE_LOCK empty!(_W_CACHE); nothing)

function tiled_w(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_W")
        w = parse(Int, ENV["KAPSEUDO_TRSM_W"])
        w >= 1 || error("KAPSEUDO_TRSM_W must be a positive integer (got $w)")
        return w
    end
    @lock _W_CACHE_LOCK get!(_W_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = @load_preference(_w_pref_key(eltype(P.A), b_is_identity(P)), nothing)
        w = tuned === nothing ? nothing : tryparse(Int, tuned)
        (w !== nothing && w >= 1) ? w : _DEFAULT_W
    end
end

# Analytic per-device+type estimate — NO benchmark run. The end-to-end optimum is NOT the narrowest
# tile: more sub-tiles mean more @localmem reloads / panel passes, whose overhead eventually outweighs
# the occupancy gain. So target the occupancy SWEET SPOT, ~¾ of the per-SM block cap resident, and use
# the LARGEST TC that reaches it (least narrowing → least overhead); fall back to the narrowest that
# fits a block if none do. `device_smem_per_sm` gives the per-SM budget for the blocks-per-SM estimate
# (= smem_sm ÷ tile, capped by the hardware block limit). NOTE the target is a fixed ¾-of-cap
# heuristic — the true sweet spot also depends on the type's arithmetic intensity and the device's
# compute:bandwidth ratio, which is exactly what `tune_trsm_tc!` measures directly.
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

# Non-CPU solve step in lockstep_ihl! — routes to column / tiled / tiled-gemm per `trsm_strategy()`.
# See DESIGN_TRSM.md "Choosing a solve" for the full routing rationale.
function trsmIHL(backend, bV, zv, P::SchurMatrixPencil; wgs=missing)
    wgs = ismissing(wgs) ? default_wgs(backend, size(P, 1)) : wgs
    # The default `column` needs no routing info, so return early — this keeps the per-iteration
    # device-property queries (`device_smem_bytes`, etc.) off the default path.
    strat = trsm_strategy()
    if strat == "column"
        return _column_trsm!(backend, bV, zv, P, wgs)
    end
    # `tiled`/`tiled-gemm` need BOTH some trailing-tile width to fit shared memory (`tiled_tiles_fit`)
    # AND a usable warp shuffle (`warp_trsm_safe`); otherwise self-gate to `column`.
    wide       = !(real(eltype(P.A)) <: Base.IEEEFloat)  # non-IEEE (MultiFloats/BigFloat)
    tiled_ok   = tiled_tiles_fit(backend, P)             # tiled's @localmem tiles fit device shared memory
    shuffle_ok = warp_trsm_safe(backend, wide)           # tiled shuffle usable for this backend+type
    if shuffle_ok && tiled_ok
        # `tiled-gemm` replaces the trailing kernel with a vendor-BLAS `mul!` where `tiled_gemm_safe`;
        # otherwise gemm=false runs the regular `tiled` trailing kernel.
        gemm = strat == "tiled-gemm" && tiled_gemm_safe(backend, eltype(P.A))
        _tiled_trsm!(backend, bV, zv, P, wgs; gemm)
    else
        _column_trsm!(backend, bV, zv, P, wgs)
    end
end

# Tiled / blocked solve: right-looking panel sweep with shared-memory A,B-tile reuse across
# the grid-point batch (see src/KATRSM.jl/trsm_tiled_kernels.jl). `z` is conjugated once for
# the forward (lower-tri Ac,Bc) sweep; the backward sweep uses (A,B,z) directly. `gt` (grid
# points per trailing tile = the A,B reuse factor) is tunable via KAPSEUDO_TRSM_GT.
function _tiled_trsm!(backend, bV, zv, P, wgs; gemm::Bool=false)
    m = size(P, 1)
    g = length(zv)
    ET = eltype(P.A)
    Xc = flatview(bV)   # the m×g RHS as a matrix (for the `gemm` trailing `mul!`); shares storage with bV
    gt = parse(Int, get(ENV, "KAPSEUDO_TRSM_GT", "32"))
    gt >= 1 || error("KAPSEUDO_TRSM_GT must be a positive integer (got $gt)")
    # W = warps per trailing-update block (occupancy knob; see `tiled_w`). Each block then covers
    # W*gt grid points (warp w handles the w-th gt-chunk). Resolution: KAPSEUDO_TRSM_W env >
    # tuned preference > default.
    W = tiled_w(backend, P)
    vw = Val(W)
    vtc = Val(tiled_tc(backend, P))   # trailing-tile column width (shared-mem occupancy knob, per device+type)
    nblk = cld(m, 32)
    zc = conj(zv)
    eye = b_is_identity(P)   # B = I ⇒ sB-free trailing kernels (half the shared memory)
    # B≠I gemm scratch: the z-scaled panel (≤32 rows). Scaling x[panel] columns by z before the B GEMM
    # gives B·(z⊙x) = z⊙(B·x), avoiding a full m×g temp. Only allocated for the generalized gemm path.
    Xz = gemm && !eye ? similar(Xc, 32, g) : Xc
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
            ggrid = cld(g, W * gt)
            if eye && gemm
                # trailing update as a GEMM (grid points = wide dim): b[rbase+1:m] += Ac[rbase+1:m, panel]·b[panel].
                # Matches the eye kernel formula (z-independent off-diagonal for B=I); α=β=+1.
                @views mul!(Xc[rbase+1:m, :], P.Ac[rbase+1:m, koff+1:koff+plen], Xc[koff+1:koff+plen, :], one(ET), one(ET))
            elseif gemm
                # B≠I: b[rbase+1:m] += Ac·x[panel] − zc⊙(Bc·x[panel]). Scale the panel by zc first so the
                # Bc GEMM yields zc⊙(Bc·x) directly (matches the generic forward kernel's z·Bc − Ac formula).
                pan = (koff + 1):(koff + plen)
                @views mul!(Xc[rbase+1:m, :], P.Ac[rbase+1:m, pan], Xc[pan, :], one(ET), one(ET))
                @views Xz[1:plen, :] .= Xc[pan, :] .* transpose(zc)
                @views mul!(Xc[rbase+1:m, :], P.Bc[rbase+1:m, pan], Xz[1:plen, :], -one(ET), one(ET))
            elseif eye
                # tiled kernel trailing update (multi-warp): W warps/block share one tile
                @views _tiled_trailing_forward_eye(backend, 32 * W)(bV, P.Ac, koff, plen, rbase, m, gt, rtiles, vtc, vw; ndrange=32 * W * rtiles * ggrid)
            else
                @views _tiled_trailing_forward(backend, 32 * W)(bV, zc, P.Ac, P.Bc, koff, plen, rbase, m, gt, rtiles, vtc, vw; ndrange=32 * W * rtiles * ggrid)
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
            ggrid = cld(g, W * gt)
            if eye && gemm
                # b[1:koff] += A[1:koff, panel]·b[panel], all grid points at once.
                @views mul!(Xc[1:koff, :], P.A[1:koff, koff+1:koff+plen], Xc[koff+1:koff+plen, :], one(ET), one(ET))
            elseif gemm
                # B≠I: b[1:koff] += A·x[panel] − z⊙(B·x[panel]).
                pan = (koff + 1):(koff + plen)
                @views mul!(Xc[1:koff, :], P.A[1:koff, pan], Xc[pan, :], one(ET), one(ET))
                @views Xz[1:plen, :] .= Xc[pan, :] .* transpose(zv)
                @views mul!(Xc[1:koff, :], P.B[1:koff, pan], Xz[1:plen, :], -one(ET), one(ET))
            elseif eye
                @views _tiled_trailing_backward_eye(backend, 32 * W)(bV, P.A, koff, plen, gt, rtiles, vtc, vw; ndrange=32 * W * rtiles * ggrid)
            else
                @views _tiled_trailing_backward(backend, 32 * W)(bV, zv, P.A, P.B, koff, plen, gt, rtiles, vtc, vw; ndrange=32 * W * rtiles * ggrid)
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
