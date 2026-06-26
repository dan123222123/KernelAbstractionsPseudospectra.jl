# GPU triangular solves — design notes

Why `ihlpsa`'s batched pencil triangular solve has three implementations and a
strategy preference to pick between them. For the adaptive-depth driver that
calls these solves see [`DESIGN.md`](DESIGN.md); for usage see the `ihlpsa`
docstring and README. The exploration (warp-register → CUDA-native diagnosis →
tiled) and its intermediate benchmarks live in `git log` and the `bench/`
timing script `warp_trsm_bench.jl` (the earlier `warp_smoke.jl` / `tiled_check.jl`
correctness scripts were folded into the test suite — see `test/test_katrsm.jl`).

## Why the solve is the thing to optimize

Each Lanczos step in `ihlpsa` applies `(zB − A)⁻¹(zB − A)⁻ᴴ` to a vector — a
forward and a backward triangular solve of the Schur pencil `M = zB − A`, batched
over every grid point (the pencil `A,B` is shared across grid points; only the
shift `z` varies). Profiling on the 6× GTX 1080 Ti node shows these two solves
are **~76% of GPU compute at m=512 and ~85% at m=1024** — the dominant cost.

The pencil is built on the fly (`zBAij`, never materialised) to save memory, and
divided with `_pdiv` (a precision-preserving complex divide that keeps F32/F16 in
their own precision instead of widening to F64 — see `src/KATRSM.jl/KATRSM.jl`).

## The three solves

All three produce numerically equivalent results (same `_pdiv`, same `zBAij`,
same per-column/per-row update order); they differ only in how the warp is mapped
onto the work. They live in `src/KATRSM.jl/` and are selected at runtime by
`trsmIHL` (`src/ihlpsa_trsm.jl`).

### 1. Column-oriented (`column`) — the baseline

`_batched_column_oriented_{forward,backward}_solve_pencil`
(`trsm_pencil_kernels.jl`). One warp per grid point; the RHS `b` lives in global
memory; for each of the `m` columns, lane 0 computes the pivot into `@localmem`
and a `@synchronize()` block barrier broadcasts it before the strided column
update — i.e. **~2m block barriers and m global round-trips on `b`** per solve.
It uses no warp shuffles, so it works for any element type. This is the safe
fallback (see the preference section).

### 2. Warp-register (`warp`)

`_batched_warp_{forward,backward}_solve_pencil` (`trsm_warp_kernels.jl`). The
workgroup is already one warp (`default_wgs = min(m,32) = 32`), so the block
barriers are wildly overpriced. This version keeps the RHS **in registers** —
lane ℓ owns rows ℓ, ℓ+32, … → `R = cld(m,32)` register slots — and broadcasts each
pivot lane-to-lane with a warp shuffle (`KernelIntrinsics.@shfl`), which *also*
synchronises the warp. Result: **no block barriers, no global round-trips on `b`**.

Key points:
- **Why registers, not shared memory.** A shared-`b` version would still need a
  per-column barrier for the write→next-pivot dependency. Register-resident `b`
  carries that dependency lane-to-lane through `@shfl` alone — that is what makes
  it barrier-free.
- **`R` is a compile-time `Val`.** The solve is a `@generated` function that emits
  the unrolled panel loops with register variables named by literal slot. `m` is
  fixed for an entire run, so this specialises once per problem size — it does
  **not** recompile on the per-iteration survivor count `g` (that stays the
  dynamic ndrange; cf. the dynamic-ndrange fix in `git log`).
- **Warp-safety.** Every `@shfl` is guarded by a warp-uniform condition
  (`j = (p-1)·32 + jj` depends only on the panel/column, never the lane), so a
  partial last panel never diverges the warp on a shuffle.

Bitwise-identical to `column` for the standalone solve; ~1.4–1.6× faster at the
kernel level for small/medium m.

### 3. Tiled (`tiled`)

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

This loses to `warp` at small m (per-panel launch overhead) and wins
increasingly at large m.

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
| `column` (**default**) | the column-oriented solve — **shuffle-free, no per-warp register semantics**, correct for every element type and backend |
| `auto`   | register-`warp` for `m < trsm_crossover()` (default 512), `tiled` for `m ≥` it |
| `warp`   | always the register-warp solve |
| `tiled`  | always the tiled solve |

**`column` is the shipped default; `auto`/`warp`/`tiled` are opt-in performance
modes.** The fast solves rely on warp shuffles and per-lane register residency,
which are only correct on a backend with a fixed, hardware-shuffled 32-lane warp
(CUDA / AMDGPU / Metal, and Intel only with the SIMD32 pin). Stock oneAPI and
non-IEEE element types (MultiFloats / BigFloat) are routed back to `column`
automatically *inside* the `auto` branch — but an explicit `warp`/`tiled` bypasses
that gate, so making `column` the default means a user can't silently get garbage
by setting the strategy without knowing their setup is safe. `ComplexF32` and
`ComplexF64` are the tested fast-path types (`test_katrsm.jl`'s
`test_katrsm_kernels` and `test_trsm_strategies` both default to both); the
genuinely untested/risky cases are MultiFloats and very large `m` (register
budget), which is what the conservative default protects against. Switching to a
fast mode is one `set_trsm_strategy!("auto")` / `KAPSEUDO_TRSM=auto` away.

The `auto` crossover at 512 also keeps that opt-in path entirely within
KernelAbstractions + KernelIntrinsics: the KA+KI register-warp kernel has a
codegen regression at `R = 16` (m ≈ 512, see below), and routing `m ≥ 512` to the
tiled solve sidesteps it.

## The R=16 codegen regression (and the retired CUDA-native diagnosis)

While benchmarking, the KA+KI `warp` solve was found to run *slower than the
baseline* at exactly m=512 (R=16) — non-monotonic (fine again at R=32). Compiling
the identical algorithm straight through `@cuda` + `CUDA.shfl_sync` showed smooth
register use (71→224 for R=4→32, only 32 B spill at every R) and no regression, so
the cliff is a **KA+KI lowering artifact, not inherent register pressure**.

This diagnosis was prototyped as a CUDA-native warp solve (`_warp_cuda_{fwd,bwd}!`
through `@cuda` + `CUDA.shfl_sync`), kept for a while as an opt-in override. It has
since been **removed** (see `git log` for the implementation): the whole point of
the package is the minimal portable KA+KI interface, and `auto` already sidesteps
the R=16 cliff by routing `m ≥ 512` to the `tiled` solve — so the native path was
only reachable under the unusual combination of `KAPSEUDO_TRSM=warp` *and* a forced
large `m`, which no default user hits. If the cliff ever needs to be addressed
head-on, the right fix is upstream in KA/KI lowering, not a hand-rolled per-backend
kernel here.

## Benchmarks (6× GTX 1080 Ti, ComplexF32)

Solve-only, single GPU, full 90k-point grid, forward solve (solve-only
microbench): `warp` is 1.39–1.56× the column kernel for m≤256; the R=16 cliff
shows at m=512 (KA+KI 0.87×, CUDA-native 1.24×); 1.37× at m=1024.

End-to-end `ihlpsa` (300² grid, all 6 GPUs, nit=20, best-of-3), speedup over
`column`:

| m    | `warp` | `tiled` |
|------|--------|---------|
| 128  | 1.13×  | 1.10×   |
| 256  | 1.18×  | 1.19×   |
| 512  | 1.05×  | **1.46×** |
| 1024 | 1.83×  | **2.77×** |

`auto` picks the winner at each size (1.13 / 1.18 / 1.46 / 2.77×). The tiled win
grows with `m` as the solve dominates more and the column baseline's `m` barriers
scale badly (m=1024: 43.9 s → 15.8 s). σ grids match `column` to F32 round-off
(~3e-7) — the standalone solves are bitwise-identical; the tiny end-to-end delta
is benign FMA-contraction difference accumulated over the iterations.

## Implementation map

- `src/KATRSM.jl/trsm_pencil_kernels.jl` — column-oriented baseline kernels.
- `src/KATRSM.jl/trsm_warp_kernels.jl` — `@generated` register-warp kernels.
- `src/KATRSM.jl/trsm_tiled_kernels.jl` — KA/KI tiled panel + trailing kernels.
- `src/ihlpsa_trsm.jl` — `trsmIHL` strategy dispatch; `_warp_trsm_ka!`, `_tiled_trsm!`,
  `_column_trsm!` drivers; `default_wgs`, `tiled_tiles_fit`; `lockstep_ihl!`.
- `src/backend.jl` — per-backend device interface (CPU defaults; GPU extensions override):
  `warp_width`, `device_smem_bytes`, `warp_trsm_safe`, the device/array/memory ops, `supports_fp64`.
- `src/tune.jl` — `tune_trsm_crossover!(backend, dev)`: per-device warp↔tiled crossover probe.
- `src/KAPseudospectra.jl` — `trsm_strategy()`, `trsm_crossover()`,
  `set_trsm_strategy!()`.
- `ext/CUDAPseudospectra.jl` — CUDA device-interface overrides + GPU precompile
  workload (runs via `column` so the headless precompile worker never executes the
  shuffle kernels — they JIT at runtime, and CUDA PTX would not survive the
  precompile→runtime boundary anyway).
- `test/test_katrsm.jl` — per-kernel correctness (warp vs LAPACK and bitwise vs
  column) and `test_trsm_strategies` (end-to-end `warp`/`tiled`/`auto` vs `column`).

## Dependency note

`KernelIntrinsics.jl` (the portable warp shuffle) pins CUDA to 5.9–5.11, which can
make a single environment that loads *all* GPU backends (CUDA + AMDGPU + Metal +
oneAPI together, as the bench Project does) unsatisfiable. Resolve per-backend, or
revisit whether the shuffle should be hand-rolled per backend, before widening the
support matrix.

## Open items

- Tune `tiled`: `gt` sweep, fuse the per-panel diagonal+trailing launches,
  double-buffer the `A,B` tiles, lower the `auto` crossover if a `gt`-tuned tiled
  wins at m=256.
- Validate / extend the per-warp solves for higher-precision element types, or
  keep them on `column`.
