#!/usr/bin/env bash
# bench_correctness: end-to-end σ vs a 256-bit BigFloat dense-SVD oracle — pure CPU (the caller parks it
# on idle cores; no CUDA). The arithmetic floor is m-insensitive, so a modest m suffices;
# cap sizes at build time via BENCH_CORR_MS / BENCH_CORR_GRIDN / BENCH_CORR_ELTYPES.
# Untagged output: the check is the same on any box.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :abacus: correctness (BigFloat oracle, CPU)"
export JULIA_NUM_THREADS=auto # the BigFloat SVD threads over the grid
export BENCH_MS="${BENCH_CORR_MS:-64,128}"
export BENCH_CORR_GRIDN="${BENCH_CORR_GRIDN:-12}"
export BENCH_ELTYPES="${BENCH_CORR_ELTYPES:-f32,f32x2,f64,f32x4,f64x2,f64x4}"
julia --project=bench bench/bench_correctness.jl || echo "correctness failed (soft_fail)"
publish bench_correctness.csv
