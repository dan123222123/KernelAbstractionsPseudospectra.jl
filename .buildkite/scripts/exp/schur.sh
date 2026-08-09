#!/usr/bin/env bash
# bench_schur: Schur/QZ break-even. The driver places it after the faster sections — its m=4096 QZ
# factorization alone is ~40 min.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :scales: schur"
julia --project=bench bench/bench_schur.jl cuda || echo "schur failed (soft_fail)"
publish bench_schur.csv
