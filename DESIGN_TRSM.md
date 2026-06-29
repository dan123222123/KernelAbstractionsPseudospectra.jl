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
their own precision instead of widening to F64 — see `src/KATRSM.jl/KATRSM.jl`).

## The two solves

Both produce numerically equivalent results (same `_pdiv`, same `zBAij`); they
differ in how the work is mapped onto the warp and the memory hierarchy. They
live in `src/KATRSM.jl/` and are selected at runtime by `trsmIHL`
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
(kernel-level implementation: `src/KATRSM.jl/trsm_tiled_kernels.jl`, which has an
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
warp shuffle, `src/KATRSM.jl/trsm_tiled_kernels.jl`); the MultiFloats per-limb
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

## Choosing a solve: the `trsm_strategy` local preference

The choice is a **local preference** (Preferences.jl → `LocalPreferences.toml`)
so it can be set per checkout, with `ENV["KAPSEUDO_TRSM"]` as a no-recompile
runtime override. `set_trsm_strategy!(s)` persists it; `trsm_strategy()` reads it.
Values:

| value    | behaviour |
|----------|-----------|
| `column` (**default**) | the column-oriented solve — **shuffle-free**, correct for every element type and backend |
| `auto`   | the tiled solve where it is usable (tiles fit **and** the shuffle is safe), else `column` |
| `tiled`  | the tiled solve where its tiles fit (shuffle gate bypassed — user opt-in), else `column` |

**`column` is the shipped default; `auto`/`tiled` are opt-in performance modes.**
There is **no size crossover** — both opt-in modes route on capability, not `m`:

- `tiled_tiles_fit(backend, P)` — the 32×32 trailing-update `@localmem` tiles (one
  tile if `B = I`, two if `B ≠ I`) fit `device_smem_bytes`. A wide `B = I` pencil
  uses the single-tile kernels and fits; a wide `B ≠ I` pencil needs two tiles and
  typically overflows → `column`.
- `warp_trsm_safe(backend, wide)` — the panel-solve shuffle is usable for this
  backend+type. It is `false` on stock oneAPI (no shuffle backend / no SIMD32 pin)
  and on Metal unless opted in, and for wide non-IEEE types (MultiFloats /
  BigFloat) lacking the per-limb `_trsm_shfl` override
  (`MultiFloatsPseudospectra`). On CUDA / AMDGPU it is `true` regardless of element
  type, so a wide pencil tiles through the per-limb override rather than falling
  back to `column`.

`auto` checks **both** gates; an explicit `tiled` checks only the tiles (the user
opted in to the shuffle). Making `column` the default means a user can't silently
get garbage by setting the strategy without knowing their setup is safe.
`ComplexF32` and `ComplexF64` are the tested fast-path types (`test_katrsm.jl`'s
`test_katrsm_kernels` and `test_trsm_strategies`, the latter over both standard and
generalized `B≠I` pencils); the genuinely untested/risky case is MultiFloats, which
is what the conservative default protects against. Switching to a fast mode is one
`set_trsm_strategy!("auto")` / `KAPSEUDO_TRSM=auto` away.

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

- `src/KATRSM.jl/trsm_pencil_kernels.jl` — column-oriented baseline kernels.
- `src/KATRSM.jl/trsm_tiled_kernels.jl` — KA/KI tiled panel + trailing kernels;
  also home to `_trsm_shfl` (the panel-solve warp shuffle).
- `src/ihlpsa_trsm.jl` — `trsmIHL` strategy dispatch; `_tiled_trsm!`,
  `_column_trsm!` drivers; `default_wgs`, `tiled_tiles_fit`; `lockstep_ihl!`.
- `src/backend.jl` — per-backend device interface (CPU defaults; GPU extensions override):
  `warp_width`, `device_smem_bytes`, `warp_trsm_safe`, the device/array/memory ops, `supports_fp64`.
- `src/KAPseudospectra.jl` — `trsm_strategy()`, `set_trsm_strategy!()`.
- `ext/CUDAPseudospectra.jl` — CUDA device-interface overrides + GPU precompile
  workload (runs via `column` so the headless precompile worker never executes the
  shuffle kernels — they JIT at runtime, and CUDA PTX would not survive the
  precompile→runtime boundary anyway).
- `test/test_katrsm.jl` — per-kernel correctness (tiled vs LAPACK and vs `column`)
  and `test_trsm_strategies` (end-to-end `tiled`/`auto` vs `column`).

## Dependency note

`KernelIntrinsics.jl` (the portable warp shuffle) pins CUDA to 5.9–5.11, which can
make a single environment that loads *all* GPU backends (CUDA + AMDGPU + Metal +
oneAPI together, as the bench Project does) unsatisfiable. Resolve per-backend, or
revisit whether the shuffle should be hand-rolled per backend, before widening the
support matrix.

## Open items

- Tune `tiled`: `gt` sweep, fuse the per-panel diagonal+trailing launches,
  double-buffer the `A,B` tiles.
- Validate / extend the tiled solve for higher-precision element types, or
  keep them on `column`.
