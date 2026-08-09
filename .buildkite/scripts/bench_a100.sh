#!/usr/bin/env bash
# A100 mega-step (cuda-a100): every section in ONE job — one Pkg resolve + CUDA init +
# MatrixDepot load for all of them. Gates and order only — each section body lives in
# exp/<name>.sh and runs as a child process (so a section cannot leak an `export` into the
# next); selection is the BENCH_EXP build env var (bench/README.md § Running on Buildkite).
# Each exp script is independently runnable by hand from the repo root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$HERE/exp"
. "$HERE/lib/prelude.sh"

bench_hygiene
bench_instantiate
export BENCH_FAST_SCHUR=1 # kernel-perf: reduction precision ⊥ trsm timing (drivers/cpu_gpu/schur are IEEE ⇒ unaffected)
bench_exps "kernels,drivers,cpu_gpu,batching,baseline,schur" # what "all" means on this box

# ---- device tuning: a PREREQUISITE for every section below, not an experiment ------------
bench_box a100
if has_exp tune; then "$EXP/tune.sh"; fi
bench_tune_profile

if has_exp kernels; then "$EXP/kernels.sh" "" "kernels (full ladder)"; fi
if has_exp drivers; then "$EXP/drivers.sh"; fi
if has_exp cpu_gpu; then "$EXP/cpu_gpu.sh"; fi
if has_exp stopping; then "$EXP/stopping.sh"; fi # opt-in; not under "all"
if has_exp batching; then "$EXP/batching.sh"; fi
if has_exp baseline; then "$EXP/baseline.sh"; fi
# schur after the faster sections: its m=4096 QZ factorization alone is ~40 min.
if has_exp schur; then "$EXP/schur.sh"; fi
if has_exp golub; then "$EXP/golub.sh"; fi # opt-in only (hours-scale); not under "all"

# counters hard-fail; deferred past bench_bundle so earlier sections' artifacts still upload.
FAIL=0
if has_exp counters; then "$EXP/counters_ncu.sh" || FAIL=1; fi

# ---- re-race against the freshly probed knobs, in a FRESH process so it reads the
# just-written profile from disk. IEEE rungs pinned: the tuned-vs-untuned A/B is an
# IEEE-knob comparison; run the full ladder separately if the MultiFloat knobs are at issue.
if has_exp tune; then
  BENCH_ELTYPES="f32,f64" KAPSEUDO_TUNE_PROFILE="$PWD/bench/tuning/$TUNE_KEY.toml" \
    "$EXP/kernels.sh" _tuned "kernels RE-RACE with tuned knobs"
fi

# ---- drivers --trace runs LAST: a fatal CUPTI crash kills the agent, and by now every other
# section has uploaded its artifacts.
if has_exp drivers; then "$EXP/drivers_trace.sh"; fi

bench_bundle
exit "$FAIL"
