#!/usr/bin/env bash
# 1080 Ti box (cuda-6xgtx1080ti): multi-GPU scaling + the f32-limb kernel ladder + nvprof
# counters + the nsys timeline (the only profiler for pre-Volta). Gates and order only —
# each section body lives in exp/<name>.sh and runs as a child process; selection is the
# BENCH_EXP build env var (bench/README.md § Running on Buildkite).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$HERE/exp"
. "$HERE/lib/prelude.sh"

bench_hygiene
bench_instantiate
export BENCH_FAST_SCHUR=1 # kernel-perf box: skip the BigFloat Schur
# f32 by default on this box: FP64 here is reduced-rate; run the f64 ladder on a
# full-rate-FP64 box. Captured before the default is applied, so the kernels ladder below can
# tell "the build asked for these rungs" from "nobody asked, use the box default".
BENCH_ELTYPES_BUILD="${BENCH_ELTYPES:-}"
export BENCH_ELTYPES="${BENCH_ELTYPES:-f32}"
bench_exps "multigpu,kernels" # what "all" means on this box

# ---- get the tuned profile; the probe targets the FIRST device (see exp/tune.sh) ----------
bench_box 1080ti
if has_exp tune; then "$EXP/tune.sh"; fi
bench_tune_profile

if has_exp multigpu; then "$EXP/multigpu.sh"; fi
if has_exp golub; then "$EXP/golub.sh"; fi # opt-in only (hours-scale); not under "all"
if has_exp nsys; then "$EXP/nsys.sh"; fi
if has_exp kernels; then
  # BENCH_ELTYPES_1080TI (this box only) > a build-level BENCH_ELTYPES > the f32-limb ladder.
  BENCH_ELTYPES="${BENCH_ELTYPES_1080TI:-${BENCH_ELTYPES_BUILD:-f32,f32x2,f32x4}}" \
    "$EXP/kernels.sh" "" "kernels (f32 limbs)"
fi

# counters hard-fail; deferred past bench_bundle so earlier sections' artifacts still upload.
FAIL=0
if has_exp counters; then "$EXP/counters_nvprof.sh" || FAIL=1; fi

bench_bundle
exit "$FAIL"
