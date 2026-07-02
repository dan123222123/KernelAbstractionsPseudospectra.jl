# Perf / backend-default notes stripped from inline comments

Running collection of the performance numbers, backend-default tables, and tuning
rationale that were removed from source comments during the code-comment cleanup,
so they can be re-homed on the package documentation website. Nothing here is lost
— it is verbatim (or lightly edited) from the comments it was pulled out of.

Organized by source file. Each entry: what the number/table was attached to.

---

## src/backend.jl

**`warp_width` (default 32) — per-vendor warp/subgroup widths**
- CUDA: 32 (`CUDA.warpsize`)
- AMDGPU: 32 on RDNA, 64 on CDNA (`warpSize`)
- Metal: 32 (SIMD-groups)
- Intel oneAPI: 32, under the SIMD32 pin (`set_intel_force_simd32!`)
- KernelAbstractions has no portable pre-launch query, so the value comes from each backend's API.
- The warp width sets the column solve's workgroup size (`default_wgs`). The tiled solve's panel
  width is a fixed 32 regardless.

**`device_smem_bytes` (default 48 KB) — shared memory per workgroup / per block**
- 48 KB is the static-shared-memory limit on Volta / Turing / early-Ampere.
- Feeds the tiled solve's `@localmem` tiles and the tiled-vs-column routing (per-device budget).

**`device_smem_per_sm` (default 64 KB) — shared memory per SM (multiprocessor)**
- Used only to estimate the tiled trailing kernel's resident-blocks-per-SM when picking the
  trailing-tile width `TC` (`tiled_tc`).
- Distinct from `device_smem_bytes`, which is the per-BLOCK limit.

**`warp_trsm_safe` (default true) — warp-shuffle pivot broadcast correctness**
- Defaults true on CUDA / AMDGPU / Metal; oneAPI overrides to false (see `ext/oneAPIPseudospectra.jl`).

**`tiled_gemm_safe` (default: any IEEE float) — fast vendor complex GEMM availability**
- Defaults to "any IEEE float type"; oneAPI and Metal override to false (see their extension files).

---

## src/core.jl

**`adapt_structure` (Schur pencil → device) — coalescing + memory layout**
- Ac/Bc are lazy conjugate-transpose views (`F.T'`). The GPU forward solve reads them transposed
  (column-strided) → uncoalesced, **~1.4–1.8× slower**. Materializing them as dense contiguous
  arrays (`Matrix(P.Ac)`) makes the forward read coalesced; values unchanged, ~free in device memory
  (`adapt` already copies each field; this just lays the bytes out for coalescing).
- A standard pencil (B = Bc = I) ships a `Diagonal` identity (m elements, shared between B and Bc)
  instead of a dense m×m, since no kernel reads B/Bc after the B=I "eye" kernels. This cuts the
  standard pencil from **5·m² to ~3·m² per device**. Generalized pencils keep B/Bc dense.

---

## src/KATRSM.jl/trsm_tiled_kernels.jl

**`_tiled_trailing_forward` / `_tiled_trailing_backward` — trailing-tile width `TC` and occupancy**
- A half-warp row tile wastes lanes and is slower (row tile is fixed at a full warp = 32 rows).
- Narrowing `TC` raises resident blocks/SM; see DESIGN_TRSM.md "Trailing-tile width and occupancy"
  for the measured tradeoff.
- `TC = 32` reproduces the single-tile (pre-optimization) sweep exactly.

**B = I trailing-update kernel — shared-memory / occupancy**
- Dropping the B tile halves shared memory (one 32×TC tile instead of two) → **~2× the resident
  blocks** of the generic kernel.

---

## src/ihlpsa_trsm.jl

**`tiled_tc` — trailing-tile width selection**
- A timed probe measures the compute+bandwidth tradeoff directly, so it needs no occupancy model.
- `TC = 32` reproduces the original single-tile sweep.
- The TOML-read cost dominates tiny solves → cache it rather than re-read per call.

**`_MAX_BLOCKS_PER_SM = 32`** — NVIDIA/AMD per-SM resident-block cap (Pascal…Hopper, CDNA/RDNA).

**`_TARGET_BLOCKS = 3 * _MAX_BLOCKS_PER_SM ÷ 4`** — ~¾ of the cap is the measured occupancy sweet spot.

**`tiled_w` — warp-multiplicity `W`**
- Occupancy rises ~W× off the W=1 ceiling until registers bind.

**`_DEFAULT_W = 4`** — the A100 ComplexF64 sweet spot (zero-setup default).

**`_auto_tiled_tc` — TC auto-selection tradeoff**
- More sub-tiles ⇒ more `@localmem` reloads / panel passes, whose overhead eventually outweighs the
  occupancy gain. The true sweet spot also depends on the type's arithmetic intensity and the
  device's compute:bandwidth ratio.

Note: the concrete per-type / eye-vs-generic `TC` optima, the ~1.1–1.4× end-to-end speedup, the
`TC = 8/16/32` shared-mem sizing, and resident-blocks/SM counts already live in DESIGN_TRSM.md
("Trailing-tile width (`TC`) and occupancy").

---

## src/KATRSM.jl/trsm_pencil_kernels.jl

**Column-oriented kernel — accuracy & bandwidth**
- The eye (B = I) result matches the generic kernel to round-off (**~1 ULP**, via FMA order).
- Skipping the identity-B DRAM read saves **~half the matrix bandwidth for B = I**.

---

## src/ihlpsa.jl

- `_eigmax_tridiag`: the tridiagonal is tiny (nit per grid point), so the `eigen`-vs-`eigmax` cost
  difference is negligible on extended-precision types.
- `ihlsrg!`: the `R` work-type promotion is a no-op for Float64 / extended-precision inputs.

---

## src/tune.jl

No numeric values stripped — the removed header text was qualitative architecture rationale (how
`TC`/`W` interact via shared-mem vs. register-limited occupancy; that a timed probe beats an
occupancy model on any shuffle-capable backend). The concrete numbers (optimal `TC` per type,
~1.1–1.4× end-to-end speedup, `TC = 8/16/32` sizing, resident-blocks/SM) already live in
DESIGN_TRSM.md § "Trailing-tile width (`TC`) and occupancy".
