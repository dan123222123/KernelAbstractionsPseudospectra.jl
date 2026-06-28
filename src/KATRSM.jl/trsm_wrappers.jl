# Brings in the non-pencil single-matrix KA kernels (trsm_kernels.jl) — kept only as LAPACK-checked
# references in test/test_katrsm.jl; ihlpsa's production path uses the *_pencil / warp / tiled solves.
# (The non-pencil host wrappers that used to live here — forward_solve, batched_back_solve,
# blkco_*_solve! — were dead and broken, referencing undefined kernels, and were removed.)
include("trsm_kernels.jl")
