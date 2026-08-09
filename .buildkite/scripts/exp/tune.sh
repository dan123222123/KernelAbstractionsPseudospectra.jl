#!/usr/bin/env bash
# device tuning probe, writes to a git tracked device profile at bench/tuning/$KAPSEUDO_TUNE_KEY.toml
# runs on BENCH_EXP=tune -- see bench/tuning/README.md.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/tag.sh"

echo "--- :wrench: device tuning probe (tiled tile_cols/block_warps/warp_gridpts + column wgs schedule)"
# Runs unprofiled (tune_device.jl unsets it), then rewrites bench/tuning/$KAPSEUDO_TUNE_KEY.toml.
julia --project=bench bench/tuning/tune_device.jl cuda 2>&1 | tee bench/results/tune_log.txt ||
  echo "tune failed (soft_fail) — sections below fall back to the committed profile"
publish tune_log.txt

# make sure the tune exists
if [ -f "bench/tuning/$KAPSEUDO_TUNE_KEY.toml" ]; then
  buildkite-agent artifact upload "bench/tuning/$KAPSEUDO_TUNE_KEY.toml" || true
  echo "^^^ download bench/tuning/$KAPSEUDO_TUNE_KEY.toml and COMMIT it — an uncommitted re-tune does not exist"
else
  echo "^^^ TUNE PRODUCED NO PROFILE: bench/tuning/$KAPSEUDO_TUNE_KEY.toml was not written. The probe"
  echo "    above may have completed and still failed on the write — read tune_log.txt."
fi
