# shellcheck shell=bash
# Shared driver scaffolding, SOURCED by bench_a100.sh / bench_1080ti.sh.

# ---- agent hygiene + a clean tuning slate ----------------------------------------------------
bench_hygiene() {
  export BENCH_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  unset LD_LIBRARY_PATH
  mkdir -p bench/results
  # remove any local prefs/device tuning
  rm -f bench/LocalPreferences.toml bench/tuning/device.toml LocalPreferences.toml
}

bench_instantiate() {
  julia --project=bench -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA")'
  export JULIA_NUM_THREADS=auto
}

# Experiment selection: BENCH_EXP is a comma list of section names set on the build
# (New Build → Environment Variables); the token "all" (alone or in the list) expands to
# the umbrella list the box script passes in. Unset selects nothing. Gate each section
# with `has_exp <name>`.
bench_exps() {
  BENCH_EXPS="${BENCH_EXP:-}"
  case ",$BENCH_EXPS," in *",all,"*) BENCH_EXPS="$BENCH_EXPS,$1" ;; esac
}

has_exp() { case ",$BENCH_EXPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac }

# export box identity + tune
bench_box() {
  export BENCH_BOX="${BENCH_BOX:-$1}"
  export KAPSEUDO_TUNE_KEY="${KAPSEUDO_TUNE_KEY:-$BENCH_BOX}"
  TUNE_KEY="$KAPSEUDO_TUNE_KEY"
}

# grab the box tune, if it exists
bench_tune_profile() {
  if [ -f "bench/tuning/$TUNE_KEY.toml" ]; then
    export KAPSEUDO_TUNE_PROFILE="$PWD/bench/tuning/$TUNE_KEY.toml"
    echo "--- :wrench: tuning profile $KAPSEUDO_TUNE_PROFILE"
    cat "bench/tuning/$TUNE_KEY.toml"
  else
    echo "^^^ NO tuning profile at bench/tuning/$TUNE_KEY.toml — sections below run on heuristic"
    echo "    defaults and every stamp will read tuning=UNTUNED. Start a build with BENCH_EXP=tune"
    echo "    to generate one."
  fi
}

# bundle bench artifacts at the end
bench_bundle() {
  local bundle="bench/results/bundle_${BENCH_BOX}_build${BUILDKITE_BUILD_NUMBER:-0}.tar.gz"
  if [ "$(ls -A bench/results 2>/dev/null || true)" != "" ]; then
    echo "--- :package: bundle run artifacts"
    # provenance: which commit/build produced these CSVs
    printf 'commit=%s\nbuild=%s\nbox=%s\n' "${BUILDKITE_COMMIT:-unknown}" \
      "${BUILDKITE_BUILD_NUMBER:-0}" "${BENCH_BOX:-}" >bench/results/meta.txt
    # don't need nsys_tmp
    tar czf "$bundle" --exclude='nsys_tmp' --exclude='*.tar.gz' -C bench results &&
      ls -lh "$bundle" &&
      buildkite-agent artifact upload "$bundle" ||
      echo "bundle failed (soft_fail) — per-section artifacts above are unaffected."
  else
    echo "bench/results is empty — nothing to bundle."
  fi
}
