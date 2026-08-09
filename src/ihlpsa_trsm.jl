# Trsm strategy routing (`trsmIHL!`) + the three solve drivers + the `lockstep_ihl!` inner Lanczos

# Workgroup size for the column trsm kernel (1 on CPU). Override via the `wgs` kwarg to ihlpsa.
# Resolution order: `KAPSEUDO_TRSM_WGS` env > schedule persisted by `tune_trsm_wgs!` (per type+eye,
# via `tuned_knob`) > `_auto_wgs`.
const _WGS_CANDIDATES = (32, 64, 128, 256)
_wgs_pref_key(T, eye::Bool) = "trsm_wgs_$(T)_$(eye ? "eye" : "gen")"
const _WGS_CACHE = Dict{Tuple{DataType, Bool}, Vector{Tuple{Int, Int}}}()
const _WGS_CACHE_LOCK = ReentrantLock()
_clear_wgs_cache!() = (@lock _WGS_CACHE_LOCK empty!(_WGS_CACHE); nothing)

# Zero-setup estimate via the warp width
function _auto_wgs(backend, m)
    KernelAbstractions.isgpu(backend) || return 1
    w = warp_width(backend)
    m >= 1024 && return min(m, 4w)
    m >= 512 && return min(m, 2w)
    return min(m, w)
end

# A tuned `wgs` is a piecewise-constant schedule over `m`: "m₀:wgs₀,m₁:wgs₁,…" ascending;
# resolves to the wgs of the last threshold ≤ m. Malformed input returns `nothing`.
function _parse_wgs_schedule(s::AbstractString)
    sched = Tuple{Int, Int}[]
    for entry in split(s, ',')
        isempty(strip(entry)) && continue
        kv = split(entry, ':')
        length(kv) == 2 || return nothing
        mth, w = tryparse(Int, strip(kv[1])), tryparse(Int, strip(kv[2]))
        (mth === nothing || w === nothing || mth < 1 || w < 1) && return nothing
        push!(sched, (mth, w))
    end
    return isempty(sched) ? nothing : sort!(sched; by = first)
end

_format_wgs_schedule(sched) = join(("$mth:$w" for (mth, w) in sched), ",")

function _wgs_from_schedule(sched, m)
    wgs = last(first(sched))               # below the first threshold: the schedule's floor
    for (mth, w) in sched
        mth <= m || break
        wgs = w
    end
    return wgs
end

function column_wgs(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_WGS")
        wgs = parse(Int, ENV["KAPSEUDO_TRSM_WGS"])
        wgs >= 1 || error("KAPSEUDO_TRSM_WGS must be a positive integer (got $wgs)")
        return wgs
    end
    m = size(P, 1)
    sched = @lock _WGS_CACHE_LOCK get!(_WGS_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = tuned_knob(_wgs_pref_key(eltype(P.A), b_is_identity(P)))
        s = tuned === nothing ? nothing : _parse_wgs_schedule(tuned)
        s === nothing ? Tuple{Int, Int}[] : s
    end
    return isempty(sched) ? _auto_wgs(backend, m) : min(m, _wgs_from_schedule(sched, m))
end

# Whether the tiled solve can run for pencil `P` on this device: true iff the narrowest `TC`
# candidate's tile(s) fit `device_smem_bytes` (two tiles for B≠I, one for B=I). Lets wide
# non-IEEE types fall back to a narrow TC instead of `column`; `tile_cols` picks the actual width.
function tiled_tiles_fit(backend, P)
    ntiles = b_is_identity(P) ? 1 : 2
    tile = sizeof(eltype(P.A)) * tile_warp_rows(backend) * minimum(_TILECOLS_CANDIDATES)
    return ntiles * tile <= device_smem_bytes(backend)
end

# Trailing tile row count — one warp's worth, so consecutive lanes take consecutive rows and the
# column-major A/b loads coalesce. Same hook as the panel width, but a distinct role: constrained by
# memory layout here, not by `@shfl` reach (no shuffle in the trailing kernel).
tile_warp_rows(backend) = warp_width(backend)

const _TILECOLS_CANDIDATES = (32, 16, 8)         # largest → smallest
const _MAX_BLOCKS_PER_SM = 32              # per-SM resident-block cap
const _TARGET_BLOCKS = 3 * _MAX_BLOCKS_PER_SM ÷ 4   # occupancy target for `_auto_tile_cols`
_tilecols_pref_key(T, eye::Bool) = "trsm_tilecols_$(T)_$(eye ? "eye" : "gen")"

# Cached per (element type, eye): `@load_preference` is a TOML read, too slow per call. Env override
# is never cached; the tuning probe clears this cache after persisting a new value. Lock-guarded for
# the concurrent multi-GPU driver. Keyed by type+eye only — assumes one GPU model per session.
const _TILECOLS_CACHE = Dict{Tuple{DataType, Bool}, Int}()
const _TILECOLS_CACHE_LOCK = ReentrantLock()
_clear_tilecols_cache!() = (@lock _TILECOLS_CACHE_LOCK empty!(_TILECOLS_CACHE); nothing)

# Trailing-tile column width — the `Val{NumTileCols}` of the trailing kernels
# (src/KATRSM/trsm_tiled_kernels.jl). Resolution order: `KAPSEUDO_TRSM_TILECOLS` env >
# `tune_trsm_tiled!`-persisted value (via `tuned_knob`, keyed by type+eye) > `_auto_tile_cols`.
# `tiled_tiles_fit` only guarantees the NARROWEST candidate fits shared memory, so `_auto_tile_cols`
# re-checks the width it picks; the env override is validated for membership only and bypasses both.
function tile_cols(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_TILECOLS")
        tc = parse(Int, ENV["KAPSEUDO_TRSM_TILECOLS"])
        tc in _TILECOLS_CANDIDATES ||
            error("KAPSEUDO_TRSM_TILECOLS must be one of $(_TILECOLS_CANDIDATES) (got $tc)")
        return tc
    end
    @lock _TILECOLS_CACHE_LOCK get!(_TILECOLS_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = tuned_knob(_tilecols_pref_key(eltype(P.A), b_is_identity(P)))
        tc = tuned === nothing ? nothing : tryparse(Int, tuned)
        (tc !== nothing && tc in _TILECOLS_CANDIDATES) ? tc : _auto_tile_cols(backend, P)
    end
end

# Warps per trailing-update block. Not a kernel argument: the trailing kernels recover it from the
# launch as `@groupsize()[1] ÷ NumTileWarpRows`, so it cannot disagree with the geometry. The warps
# share one `warp_width × TC` @localmem tile and split the `gt` grid points between them.
# Bit-identical across the count (FMA order per row/grid-point is unchanged). Resolution:
# `KAPSEUDO_TRSM_BLOCKWARPS` env > `tuned_knob` (per type+eye, persisted by `tune_trsm_tiled!`) >
# `_DEFAULT_BLOCKWARPS`.
const _BLOCKWARPS_CANDIDATES = (1, 2, 4, 8)
const _DEFAULT_BLOCKWARPS = 4                        # zero-setup default; the tuner refines per device+type
_blockwarps_pref_key(T, eye::Bool) = "trsm_blockwarps_$(T)_$(eye ? "eye" : "gen")"
const _BLOCKWARPS_CACHE = Dict{Tuple{DataType, Bool}, Int}()
const _BLOCKWARPS_CACHE_LOCK = ReentrantLock()
_clear_blockwarps_cache!() = (@lock _BLOCKWARPS_CACHE_LOCK empty!(_BLOCKWARPS_CACHE); nothing)

function block_warps(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_BLOCKWARPS")
        w = parse(Int, ENV["KAPSEUDO_TRSM_BLOCKWARPS"])
        w >= 1 || error("KAPSEUDO_TRSM_BLOCKWARPS must be a positive integer (got $w)")
        return w
    end
    @lock _BLOCKWARPS_CACHE_LOCK get!(_BLOCKWARPS_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = tuned_knob(_blockwarps_pref_key(eltype(P.A), b_is_identity(P)))
        w = tuned === nothing ? nothing : tryparse(Int, tuned)
        (w !== nothing && w >= 1) ? w : _DEFAULT_BLOCKWARPS
    end
end

# Grid points per warp per trailing tile — the tile's reuse factor, and (with W) the block count
# g/(W·gt) of the trailing launch. Resolution: `KAPSEUDO_TRSM_WARPGRIDPTS` env > `tuned_knob`
# (per type+eye) > `_DEFAULT_WARPGRIDPTS`.
const _WARPGRIDPTS_CANDIDATES = (32, 16, 8, 4)
const _DEFAULT_WARPGRIDPTS = 32                      # zero-setup default; the tuner refines per device+type
_warpgridpts_pref_key(T, eye::Bool) = "trsm_warpgridpts_$(T)_$(eye ? "eye" : "gen")"
const _WARPGRIDPTS_CACHE = Dict{Tuple{DataType, Bool}, Int}()
const _WARPGRIDPTS_CACHE_LOCK = ReentrantLock()
_clear_warpgridpts_cache!() = (@lock _WARPGRIDPTS_CACHE_LOCK empty!(_WARPGRIDPTS_CACHE); nothing)

function warp_gridpts(backend, P)
    if haskey(ENV, "KAPSEUDO_TRSM_WARPGRIDPTS")
        n = parse(Int, ENV["KAPSEUDO_TRSM_WARPGRIDPTS"])
        n >= 1 || error("KAPSEUDO_TRSM_WARPGRIDPTS must be a positive integer (got $n)")
        return n
    end
    @lock _WARPGRIDPTS_CACHE_LOCK get!(_WARPGRIDPTS_CACHE, (eltype(P.A), b_is_identity(P))) do
        tuned = tuned_knob(_warpgridpts_pref_key(eltype(P.A), b_is_identity(P)))
        n = tuned === nothing ? nothing : tryparse(Int, tuned)
        (n !== nothing && n >= 1) ? n : _DEFAULT_WARPGRIDPTS
    end
end

# Analytic per-device+type estimate (no benchmark run): targets `_TARGET_BLOCKS` resident blocks/SM,
# picking the largest TC that reaches it, or the narrowest that fits a block if none do.
function _auto_tile_cols(backend, P)
    ntiles = b_is_identity(P) ? 1 : 2
    elt = sizeof(eltype(P.A))
    smem_block = device_smem_bytes(backend)        # per-block shared-memory limit
    smem_sm = device_smem_per_sm(backend)          # per-SM shared memory (occupancy estimate)
    best = 0
    for tc in _TILECOLS_CANDIDATES                        # largest first: take the least narrowing that suffices
        tile = ntiles * tile_warp_rows(backend) * tc * elt
        tile <= smem_block || continue              # must fit a single block
        best = tc
        min(smem_sm ÷ tile, _MAX_BLOCKS_PER_SM) >= _TARGET_BLOCKS && return tc   # hit the occupancy target
    end
    return best == 0 ? last(_TILECOLS_CANDIDATES) : best                               # nothing fit (shouldn't happen post tiled_tiles_fit) → narrowest
end

# Non-CPU solve step in lockstep_ihl! — solves in place into bV, routing to column /
# tiled / tiled-gemm per `trsm_strategy()`.
function trsmIHL!(backend, bV, zv, P::SchurMatrixPencil; wgs = missing)
    wgs = ismissing(wgs) ? column_wgs(backend, P) : wgs
    # `column` needs no routing info; return early to keep per-iteration device-property
    # queries (`device_smem_bytes`, etc.) off the default path.
    strat = trsm_strategy()
    if strat == "column"
        return _column_trsm!(backend, bV, zv, P, wgs)
    end
    # `tiled`/`tiled-gemm` need both a trailing-tile width that fits shared memory (`tiled_tiles_fit`)
    # and a usable warp shuffle (`warp_trsm_safe`); otherwise self-gate to `column`.
    wide = !(real(eltype(P.A)) <: Base.IEEEFloat)  # non-IEEE (MultiFloats/BigFloat)
    tiled_ok = tiled_tiles_fit(backend, P)             # tiled's @localmem tiles fit device shared memory
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
# the grid-point batch (see src/KATRSM/trsm_tiled_kernels.jl). `z` is conjugated once for
# the forward (lower-tri Ac,Bc) sweep; the backward sweep uses (A,B,z) directly. `gt` (grid
# points per trailing tile = the A,B reuse factor) is tunable via KAPSEUDO_TRSM_WARPGRIDPTS.
function _tiled_trsm!(backend, bV, zv, P, wgs; gemm::Bool = false)
    m = size(P, 1)
    zpd = length(zv)          # grid points in this batch — the driver's zpd
    ET = eltype(P.A)
    Xc = flatview(bV)   # the m×g RHS as a matrix (for the `gemm` trailing `mul!`); shares storage with bV
    gridpts = warp_gridpts(backend, P)   # env > tuned preference > default
    # Warps per trailing-update block; each covers nwarps*gridpts grid points, warp w the w-th chunk.
    nwarps = block_warps(backend, P)
    vcols = Val(tile_cols(backend, P))   # trailing-tile column width (shared-mem knob)
    # Panel width = the hardware warp width: the panel solve broadcasts each pivot via `@shfl`, so
    # hardcoding 32 would idle half the lanes on wave64 (CDNA). Trailing tile rows use the same hook
    # for an unrelated reason (lane→row coalescing, no shuffle) — via `tile_warp_rows`.
    wp = warp_width(backend)
    vrows = Val(tile_warp_rows(backend))
    nblk = cld(m, wp)
    zc = conj(zv)
    eye = b_is_identity(P)   # B = I ⇒ sB-free trailing kernels (half the shared memory)
    # B≠I gemm scratch (≤ wp rows): scaling x[panel] by z before the B GEMM gives B·(z⊙x) = z⊙(B·x),
    # avoiding a full m×g temp. Only allocated for the generalized gemm path.
    Xz = gemm && !eye ? similar(Xc, wp, zpd) : Xc
    # forward (lower-triangular), panels ascending
    for k in 1:nblk
        poff = (k - 1) * wp
        psize = min(wp, m - poff)
        if eye
            @views _tiled_panel_forward_eye(backend, wp)(
                bV, zc, P.Ac, poff, psize; ndrange = (wp, zpd))
        else
            @views _tiled_panel_forward(backend, wp)(
                bV, zc, P.Ac, P.Bc, poff, psize; ndrange = (wp, zpd))
        end
        rows = (poff + psize + 1):m
        if !isempty(rows)
            rtiles = cld(length(rows), wp)
            ggrid = cld(zpd, nwarps * gridpts)
            if eye && gemm
                # trailing update as a GEMM (grid points = wide dim): b[rows] += Ac[rows, panel]·b[panel].
                # Matches the eye kernel formula (z-independent off-diagonal for B=I); α=β=+1.
                @views mul!(
                    Xc[rows, :], P.Ac[rows, (poff + 1):(poff + psize)],
                    Xc[(poff + 1):(poff + psize), :], one(ET), one(ET))
            elseif gemm
                # B≠I: b[rows] += Ac·x[panel] − zc⊙(Bc·x[panel]). Scale the panel by zc first so the
                # Bc GEMM yields zc⊙(Bc·x) directly (matches the generic forward kernel's z·Bc − Ac formula).
                pan = (poff + 1):(poff + psize)
                @views mul!(Xc[rows, :], P.Ac[rows, pan],
                    Xc[pan, :], one(ET), one(ET))
                @views Xz[1:psize, :] .= Xc[pan, :] .* transpose(zc)
                @views mul!(Xc[rows, :], P.Bc[rows, pan],
                    Xz[1:psize, :], -one(ET), one(ET))
            elseif eye
                # tiled kernel trailing update: the block's warps share one tile, and their count
                # comes from the launch's workgroupsize rather than an argument (see `_tiled_trailing`).
                @views _tiled_trailing_eye(backend, (wp * nwarps, 1))(
                    bV, P.Ac, poff, psize, rows, gridpts,
                    vrows, vcols; ndrange = (wp * nwarps * rtiles, ggrid))
            else
                @views _tiled_trailing(backend, (wp * nwarps, 1))(
                    bV, zc, P.Ac, P.Bc, poff, psize, rows, gridpts,
                    vrows, vcols; ndrange = (wp * nwarps * rtiles, ggrid))
            end
        end
    end
    # backward (upper-triangular), panels descending
    for k in nblk:-1:1
        poff = (k - 1) * wp
        psize = min(wp, m - poff)
        if eye
            @views _tiled_panel_backward_eye(backend, wp)(
                bV, zv, P.A, poff, psize; ndrange = (wp, zpd))
        else
            @views _tiled_panel_backward(backend, wp)(
                bV, zv, P.A, P.B, poff, psize; ndrange = (wp, zpd))
        end
        rows = 1:poff
        if !isempty(rows)
            rtiles = cld(length(rows), wp)
            ggrid = cld(zpd, nwarps * gridpts)
            if eye && gemm
                # b[1:poff] += A[1:poff, panel]·b[panel], all grid points at once.
                @views mul!(Xc[1:poff, :], P.A[1:poff, (poff + 1):(poff + psize)],
                    Xc[(poff + 1):(poff + psize), :], one(ET), one(ET))
            elseif gemm
                # B≠I: b[1:poff] += A·x[panel] − z⊙(B·x[panel]).
                pan = (poff + 1):(poff + psize)
                @views mul!(Xc[1:poff, :], P.A[1:poff, pan], Xc[pan, :], one(ET), one(ET))
                @views Xz[1:psize, :] .= Xc[pan, :] .* transpose(zv)
                @views mul!(
                    Xc[1:poff, :], P.B[1:poff, pan], Xz[1:psize, :], -one(ET), one(ET))
            elseif eye
                @views _tiled_trailing_eye(backend, (wp * nwarps, 1))(
                    bV, P.A, poff, psize, rows, gridpts,
                    vrows, vcols; ndrange = (wp * nwarps * rtiles, ggrid))
            else
                @views _tiled_trailing(backend, (wp * nwarps, 1))(
                    bV, zv, P.A, P.B, poff, psize, rows, gridpts,
                    vrows, vcols; ndrange = (wp * nwarps * rtiles, ggrid))
            end
        end
    end
end
function _column_trsm!(backend, bV, zv, P, wgs)
    g = length(zv)
    if b_is_identity(P)   # B = I ⇒ skip the identity-B reads (the eye column kernels)
        @views _batched_column_oriented_forward_solve_eye(backend, wgs)(bV, conj(zv), P.Ac; ndrange = (
            wgs, g))
        @views _batched_column_oriented_backward_solve_eye(backend, wgs)(bV, zv, P.A; ndrange = (
            wgs, g))
    else
        @views _batched_column_oriented_forward_solve_pencil(backend, wgs)(
            bV, conj(zv), P.Ac, P.Bc; ndrange = (wgs, g))
        @views _batched_column_oriented_backward_solve_pencil(backend, wgs)(
            bV, zv, P.A, P.B; ndrange = (wgs, g))
    end
end

# cpu solve step in lockstep_ihl!. `wgs` is accepted and ignored so this CPU method
# isn't shadowed by the generic (column-oriented) trsmIHL! when called with `; wgs`.
function trsmIHL!(backend::CPU, bV, zv, P::SchurMatrixPencil; wgs = missing)
    g = length(zv)
    _batched_forward_solve_pencil(backend)(bV, conj(zv), P.A', P.B', ndrange = g)
    _batched_backward_solve_pencil(backend)(bV, zv, P.A, P.B, ndrange = g)
end

# Runs Lanczos iterations `start:nit` on the first g grid points of the batch. `start == 1`
# (default) seeds q₁ from x₀; `start > 1` resumes a prior call — only valid with the same `ihl`
# instance (Qv/v/zv untouched since), the same sized `α`/`β` arrays, the same g, and the previous
# call having ended at iteration start-1.
function lockstep_ihl!(α, β, ihl::IHLworkspace, nit, g; wgs = missing, start::Integer = 1)
    backend = get_backend(ihl)
    if start == 1
        # Qv[1] (= q₀) is read at n=1 scaled by β[1] ≡ 0, canceling any finite residue from a
        # reused workspace — but not a Lanczos-breakdown NaN (0·NaN = NaN), so q₀ must be reset.
        fill!(view(ihl.Qv[1], 1:g, :), zero(eltype(ihl.zv)))
        # `_qₙnext!` writes Qv[2] (= q₁); `_v2v!` copies it into v[1:g], fully overwriting it.
        _qₙnext!(backend)(view(ihl.Qv[2], 1:g, :), view(β, 2, 1:g), view(ihl.x₀, 1:g), ndrange = g)
        _v2v!(backend)(view(ihl.v, 1:g), view(ihl.Qv[2], 1:g, :), ndrange = g)
    end
    for n in start:nit
        trsmIHL!(backend, view(ihl.v, 1:g), view(ihl.zv, 1:g), ihl.P; wgs)
        _ihl_ttr_qₙnext!(backend)(
            view(β, n, 1:g), view(ihl.Qv[1], 1:g, :), view(α, n, 1:g),
            view(ihl.Qv[2], 1:g, :), view(ihl.v, 1:g), view(β, n + 1, 1:g), ndrange = g)
    end
    synchronize(backend)
end

# (All per-backend device hooks — the device/array/memory interface, supports_fp64, warp_width,
# device_smem_bytes, warp_trsm_safe — live in src/backend.jl, overridden by the GPU extensions.)
