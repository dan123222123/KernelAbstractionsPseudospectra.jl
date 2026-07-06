# GPU triangular solves — design notes

Why `ihlpsa`'s batched pencil triangular solve has two implementations and a
strategy preference to pick between them. For the adaptive-depth driver that
calls these solves see [`DESIGN.md`](DESIGN.md); for usage see the `ihlpsa`
docstring and README. The exploration that led here — including the now-removed
register-warp solve (see ["Why the register-warp solve was removed"](#why-the-register-warp-solve-was-removed)) —
and its intermediate benchmarks live in `git log` and the `bench/` timing script
`trsm_bench.jl` (the earlier correctness scripts were folded into the test
suite — see `test/test_katrsm.jl`).

## Why the solve is the thing to optimize

Each Lanczos step in `ihlpsa` applies `(zB − A)⁻¹(zB − A)⁻ᴴ` to a vector — a
forward and a backward triangular solve of the Schur pencil `M = zB − A`, batched
over every grid point (the pencil `A,B` is shared across grid points; only the
shift `z` varies). Profiling on the 6× GTX 1080 Ti node shows these two solves
are **~76% of GPU compute at m=512 and ~85% at m=1024** — the dominant cost.

The pencil is built on the fly (`zBAij`, never materialised) to save memory, and
divided with `_pdiv` (a precision-preserving complex divide that keeps F32/F16 in
their own precision instead of widening to F64 — see `src/KATRSM/KATRSM.jl`).

## The two solves

Both produce numerically equivalent results (same `_pdiv`, same `zBAij`); they
differ in how the work is mapped onto the warp and the memory hierarchy. They
live in `src/KATRSM/` and are selected at runtime by `trsmIHL!`
(`src/ihlpsa_trsm.jl`).

### 1. Column-oriented (`column`) — the baseline

`_batched_column_oriented_{forward,backward}_solve_pencil`
(`trsm_pencil_kernels.jl`). One warp per grid point; the RHS `b` lives in global
memory; for each of the `m` columns, lane 0 computes the pivot into `@localmem`
and a `@synchronize()` block barrier broadcasts it before the strided column
update — i.e. **~2m block barriers and m global round-trips on `b`** per solve.
It uses no warp shuffles, so it works for any element type. This is the safe
fallback (see the routing section).

### 2. Tiled (`tiled`)

`_tiled_panel_{forward,backward}` + `_tiled_trailing_{forward,backward}`
(`trsm_tiled_kernels.jl`), driven by `_tiled_trsm!` (`src/ihlpsa_trsm.jl`).

For large `m` the per-grid-point solves become **bandwidth-bound**: every warp
re-streams the whole `m×m` `A,B` pencil from DRAM with zero reuse across grid
points. While `A,B` fit in L2 (small m) this is free; once they don't (m ≳ 512),
each grid point pays full DRAM bandwidth for `A,B`. This is why large `A,B` are
the worst case.

The tiled solve is a right-looking blocked algorithm, panel width = warp size
(kernel-level implementation: `src/KATRSM/trsm_tiled_kernels.jl`, which has an
ASCII block-layout diagram in its header):

1. **Panel solve** — solve the ≤32×32 *triangular* diagonal tile for every grid
   point (lower-triangular for the forward sweep, upper-triangular for the
   backward; one warp per grid point, pivots by `@shfl`).
2. **Trailing update** — a tiled GEMM that subtracts the panel's contribution
   from the trailing rows. Each workgroup loads the `A,B[row-tile, panel]` tile
   into `@localmem` **once** and reuses it across `gt` grid points (tunable via
   `KAPSEUDO_TRSM_GT`, default 32). The pencil `M = zB − A` is kept as separate,
   z-independent `A,B` tiles plus a per-grid-point `z` combine, so the shared
   tiles are reused across the whole batch — turning the streaming into
   compute-bound work.

Its per-panel launch overhead makes it lose to `column` at small m and win
increasingly at large m. The panel solve broadcasts pivots with `_trsm_shfl` (a
warp shuffle, `src/KATRSM/trsm_tiled_kernels.jl`); the MultiFloats per-limb
shuffle override (`MultiFloatsPseudospectra`) applies here so wide IEEE-free
types still tile on a shuffle-capable backend.

**The "dead" zero triangle of `A,B` is never touched.** The right-looking
blocking keeps every trailing-tile index in the *filled* part of the triangle:
the forward sweep loads rows `i > koff+plen` against panel columns `j ≤ koff+plen`
(so `i > j`, the filled subdiagonal), and the backward sweep loads rows
`i ≤ koff` against columns `j > koff` (so `i < j`, the filled superdiagonal). The
structurally-zero entries exist as device storage but are never loaded into the
shared tiles or multiplied — dead storage, not dead computation (and no wasted
no-op flops).

#### Trailing-tile width (`TC`) and occupancy

The trailing tile is **32 rows × `TC` columns**: the row count is a fixed full
warp (a half-warp row tile wastes lanes and measured slower), and `TC` is a
compile-time `Val` **decoupled from the 32-wide panel solve** — a wider panel is
subtracted in `⌈plen/TC⌉` column sub-tiles, leaving the `A,B` DRAM traffic
unchanged. `TC` is the key occupancy knob, because the trailing kernel is
**shared-memory-bound** on Pascal: a 32×32 ComplexF32 *generic* tile is
`2·32·32·4 = 16 KB`, so only **6 blocks/SM** are resident (≈9% occupancy).
Halving `TC` halves the tile and roughly doubles resident blocks. End-to-end on a
1080 Ti, versus the original `TC=32`:

| element / `B` | optimal `TC` | speedup at m = 512 / 1024 |
|---|---|---|
| ComplexF32, `B=I` (1 tile)  | 16 | 13% / 18% |
| ComplexF32, `B≠I` (2 tiles) | 8  | 33% / 41% |

The optimum is **not** the narrowest tile: more sub-tiles mean more `@localmem`
reloads, whose overhead eventually outweighs the occupancy gain — both cases peak
at ≈24 resident blocks/SM (the 1-tile eye reaches that at `TC=16`, the 2-tile
generic at `TC=8`). `tiled_tc` (`src/ihlpsa_trsm.jl`) resolves `TC` per device +
element type in three tiers:

1. `KAPSEUDO_TRSM_TC=8|16|32` env override (tuning / measurement; `TC=32`
   reproduces the pre-optimization single-tile sweep exactly);
2. a value persisted by the **`tune_trsm_tc!` probe** (`src/tune.jl`) — it times
   the tiled solve at each `TC` per `(type, eye/generic)` and stores the fastest in
   `LocalPreferences.toml`. A timed probe needs *no* occupancy model, so it captures
   the type's arithmetic intensity and the device's compute:bandwidth ratio
   directly — the accurate per-device path, run once on the target hardware;
3. otherwise a zero-setup **analytic estimate**: the *largest* `TC` whose `32×TC`
   tile reaches ≈¾ of the device's blocks/SM cap (from `device_smem_per_sm`), else
   the narrowest that fits a block. The ¾-of-cap target is a fixed heuristic
   calibrated on Pascal/ComplexF32 — good enough as a default, but exactly the
   quantity the probe measures rather than assumes.

`device_smem_per_sm` is a per-backend hook (CUDA/AMDGPU query the real per-SM/CU
figure; oneAPI/Metal fall back to per-workgroup proxies since their execution
models don't expose a clean per-SM shared-memory budget — there the probe is the
intended path).

### 3. Tiled-GEMM (`tiled-gemm`)

Same driver as `tiled` (`_tiled_trsm!`, with a `gemm` flag) and the **same warp
panel solve** — the *only* difference is the trailing update. `tiled`'s custom
shared-memory kernel reuses the A-tile across the grid-point batch (a *bandwidth*
win); `tiled-gemm` instead does the trailing update as a vendor-BLAS **`mul!`**
with the grid points as the GEMM's wide dimension. Because each panel's trailing
matrix block is **z-independent for B=I** (the off-diagonal of `zI−A` is just
`−A`), the update is a single `mul!(X[trailing,:], Mat[trailing,panel], X[panel,:],
1, 1)` over the whole `flatview(bV)` (`Mat = Ac` forward / `A` backward) — and the
existing grid-point-major RHS layout needs no transpose (cuBLAS handles the
strided `lda=m` sub-matrices at full speed).

This trades `tiled`'s bandwidth win for a **compute-bound** trailing GEMM, which
helps where the solve is bandwidth/latency-bound — i.e. **large-m ComplexF32/F64**
(where `tiled` already wins, and the GEMM extends it). It is gated by
`tiled_gemm_safe(backend, T)`: true for ComplexF32/F64 on CUDA/AMDGPU (fast complex
GEMM), false for MultiFloats (no vendor GEMM — `mul!` would drop to a slow generic
matmul) and on oneAPI/Metal (patchy complex GEMM). When false, `tiled-gemm`
self-gates to the regular `tiled` trailing kernel. Covers both `B=I` (single
`mul!`) and `B≠I` (two `mul!`s: `A·x − z⊙(B·x)`, scaling the panel by `z` before
the `B` GEMM so it needs no full `m×g` temp). MultiFloats are *compute-bound by
construction* (the limb arithmetic), so a memory-side strategy buys them nothing
anyway, which is why routing them to `tiled` loses nothing.

## Choosing a solve: the `trsm_strategy` local preference

The choice is a **local preference** (Preferences.jl → `LocalPreferences.toml`)
so it can be set per checkout, with `ENV["KAPSEUDO_TRSM"]` as a no-recompile
runtime override. `set_trsm_strategy!(s)` persists it; `trsm_strategy()` reads it.
Values:

| value    | behaviour |
|----------|-----------|
| `column` (**default**) | the column-oriented solve — **shuffle-free**, correct for every element type and backend |
| `tiled`  | the tiled solve where it is usable (tiles fit **and** the shuffle is safe), else an automatic fall back to `column` |
| `tiled-gemm` | `tiled`'s panel solve + a vendor-BLAS `mul!` trailing update, where `tiled_gemm_safe` (ComplexF32/F64 on CUDA/AMDGPU); else an automatic fall back to the regular `tiled` trailing kernel |

**`column` is the shipped default; `tiled` is the opt-in performance mode.**
There is **no size crossover** — `tiled` routes on capability, not `m`, checking
**both** of:

- `tiled_tiles_fit(backend, P)` — whether *some* trailing-tile width fits
  `device_smem_bytes`. Because the width `TC` is tunable (see below), this checks
  the **narrowest** `32×TC` tile (one if `B = I`, two if `B ≠ I`), so a wide
  non-IEEE `B ≠ I` pencil (MultiFloats: `Float64xN`) whose `32×32` tile would
  overflow still tiles at a narrow `TC` rather than dropping to `column`; only a
  type too wide for even the narrowest `TC` falls back.
- `warp_trsm_safe(backend, wide)` — the panel-solve shuffle is usable for this
  backend+type. It is `false` on stock oneAPI (no shuffle backend / no SIMD32 pin)
  and on Metal unless opted in, and for wide non-IEEE types (MultiFloats /
  BigFloat) lacking the per-limb `_trsm_shfl` override
  (`MultiFloatsPseudospectra`). On CUDA / AMDGPU it is `true` regardless of element
  type, so a wide pencil tiles through the per-limb override rather than falling
  back to `column`.

Because `tiled` **self-gates** to `column` wherever either check fails, it never
runs a broken kernel and is safe to request on any backend. (An earlier ungated
`tiled` that bypassed the shuffle check — and a separate `auto` that did the gating
— were collapsed into this single gated `tiled`.) `column` nonetheless stays the
default: it's the fully-validated, bitwise-stable baseline, and a user opts into the
fast path only once they've confirmed it on their hardware. `ComplexF32` and
`ComplexF64` are the tested fast-path types (`test_katrsm.jl`'s `test_katrsm_kernels`
and `test_trsm_strategies`, the latter over both standard and generalized `B≠I`
pencils); the genuinely untested/risky case is MultiFloats. Switching to the fast
mode is one `set_trsm_strategy!("tiled")` / `KAPSEUDO_TRSM=tiled` away.

## Why the register-warp solve was removed

An earlier `warp` strategy kept the RHS in registers (lane ℓ owning rows ℓ, ℓ+32,
…, `R = ⌈m/32⌉` slots) and broadcast each pivot lane-to-lane with a warp shuffle —
barrier-free and the fastest path at small `m`. It was **removed**. Benchmarked
end-to-end `ihlpsa` on CUDA (GTX 1080 Ti, ComplexF32, standard `B = I` pencil),
warm best-of-N, as a runtime ratio over the `column` baseline (lower = faster):

| m (R) | `warp` | `tiled` |
|-------|--------|---------|
| 128 (R=4)  | **0.71×** | 0.83× |
| 256 (R=8)  | 0.97×     | 1.03× |
| 512 (R=16) | 0.96×     | **0.78×** |

`warp` only won at small `m`, where the absolute saving was tiny (~0.2 s). The
disqualifier was **compile cost**: the warp kernel was `@generated` on
`R = ⌈m/32⌉`, so every new matrix-size bucket triggered a fresh codegen + `ptxas`
pass, growing superlinearly — measured pure per-size compile (at `m ≥ 256`, where
one-time costs are already paid): 3.5 s (R=8), 8.4 s (R=12), **17.5 s (R=16)**, and
minutes at R=32 (`ptxas` still running after 6+ min). `column` and `tiled` compile
their kernels **once** and pay ~0 marginal cost as `m` changes (they are not
specialised on `R`). For a tool where users sweep sizes / run adaptive grids, the
per-`R` recompile dwarfed warp's small-`m` runtime edge. `tiled` is the
better-engineered fast path (bandwidth-optimal, compiles once, wins at large `m`);
`column` stays the portable default.

## Benchmarks (6× GTX 1080 Ti, ComplexF32)

End-to-end `ihlpsa` (300² grid, all 6 GPUs, nit=20, best-of-3), `tiled` speedup
over `column`:

| m    | `tiled` |
|------|---------|
| 128  | 1.10×   |
| 256  | 1.19×   |
| 512  | **1.46×** |
| 1024 | **2.77×** |

The tiled win grows with `m` as the solve dominates more and the column baseline's
`m` barriers scale badly (m=1024: 43.9 s → 15.8 s). σ grids match `column` to F32
round-off (~3e-7); the tiny end-to-end delta is benign FMA-contraction difference
in the tiled trailing-update GEMM, accumulated over the iterations.

## Implementation map

- `src/KATRSM/trsm_pencil_kernels.jl` — column-oriented baseline kernels.
- `src/KATRSM/trsm_tiled_kernels.jl` — KA/KI tiled panel + trailing kernels;
  also home to `_trsm_shfl` (the panel-solve warp shuffle).
- `src/ihlpsa_trsm.jl` — `trsmIHL!` strategy dispatch; `_tiled_trsm!`,
  `_column_trsm!` drivers; `default_wgs`, `tiled_tiles_fit`, `tiled_tc`; `lockstep_ihl!`.
- `src/tune.jl` — `tune_trsm_tc!`, the per-device trailing-tile-width probe.
- `src/backend.jl` — per-backend device interface (CPU defaults; GPU extensions override):
  `warp_width`, `device_smem_bytes`, `device_smem_per_sm`, `warp_trsm_safe`, the device/array/memory ops, `supports_fp64`.
- `src/KAPseudospectra.jl` — `trsm_strategy()`, `set_trsm_strategy!()`.
- `ext/CUDAPseudospectra.jl` — CUDA device-interface overrides + GPU precompile
  workload (runs via `column` so the headless precompile worker never executes the
  shuffle kernels — they JIT at runtime, and CUDA PTX would not survive the
  precompile→runtime boundary anyway).
- `test/test_katrsm.jl` — per-kernel correctness (tiled vs LAPACK and vs `column`)
  and `test_trsm_strategies` (end-to-end `tiled` vs `column`).

## Dependency note

`KernelIntrinsics.jl` (the portable warp shuffle) caps CUDA at `< 6.0` through its own
`[compat]` (`CUDA = "5.9"`, i.e. `[5.9, 6.0)` — 5.9–5.11 in practice). That ceiling is KI's,
not ours: we pin only `KernelIntrinsics = "0.1.7"` (`[0.1.7, 0.2)` — the 0.1.7 floor is the
first release with a working `@shfl`/`Idx`; 0.1.8 is the current newest and that range admits
it).

KI 0.1.8 does *not* force the backends apart: its `[compat]` is `CUDA = "5.9"`,
`AMDGPU = "1"`, `Metal = "1"`, so as far as KI is concerned CUDA 5.9–5.11, AMDGPU 1 and Metal 1
are mutually compatible in one environment. (oneAPI is the exception: KI ships its oneAPI
`@shfl` as a stub, so only the shuffle-free `column` path runs on Intel — see
`ext/oneAPIPseudospectra.jl`.) The one real ceiling is CUDA `< 6.0`: if you need CUDA ≥ 6.0,
KI's cap is the blocker, and the fix is a newer KI (none is published beyond 0.1.8) or
hand-rolling the shuffle per backend before widening the support matrix.

## Open items

- Tune `tiled` further: `gt` sweep, fuse the per-panel diagonal+trailing launches,
  double-buffer the `A,B` tiles. (The trailing-tile width `TC` is already chosen
  analytically per device+type — see "Trailing-tile width and occupancy".)
- Subgroup-width-adaptive panel solve for Intel without the SIMD32 pin. The panel
  solve's `@shfl` assumes a 32-lane shuffle domain, so on oneAPI we currently force
  IGC to SIMD32 (`set_intel_force_simd32!`) rather than adapt. A more general path
  would query the kernel's actual dispatched subgroup size `W` (Level-Zero kernel
  properties / SYCL `get_sub_group_size`; Metal's `threadExecutionWidth`) and
  specialise the panel solve on `Val{W}` (a W-lane register shuffle), with the
  trailing-tile height a multiple of `W`. The query is easy; the `Val{W}` shuffle
  solve is the work, and the only payoff is running on Intel *without* the pin (and
  losing the `reqd_sub_group_size` guarantee) — so the pin stays the default. (Note
  the trailing kernel does NOT need this: it has no shuffle, and 32 is already a
  multiple of any SIMD width; its height is also occupancy-neutral.) We pin SIMD32
  via the `IGC_ForceOCLSIMDWidth` env var rather than oneAPI's `reqd_subgroup_size!`
  (the per-kernel `intel_reqd_sub_group_size` mode): applied globally the latter
  crashes the SPIR-V translator on the MultiFloat kernels, whereas the env var (a
  driver-level flag with no per-kernel metadata) pins both the IEEE tiled kernels
  and the MultiFloat column kernels fine.
- Validate / extend the tiled solve for higher-precision element types, or
  keep them on `column`.
