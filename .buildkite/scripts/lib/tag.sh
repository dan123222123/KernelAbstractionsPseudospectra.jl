# shellcheck shell=bash
# artifact naming by box, SOURCED by every exp/*.sh.
#
# The bench_*.jl scripts name their own outputs and need to have associated machine information
# `publish` makes it uniform: the box is the LAST component before the extension
#
# With BENCH_BOX unset (running an exp script by hand) nothing is added

# Box-tagged form of a path, idempotent
boxed() {
  local f="$1" dir base ext
  [ "${BENCH_BOX:-}" != "" ] || {
    printf '%s' "$f"
    return
  }
  dir="$(dirname "$f")"
  base="$(basename "$f")"
  case "$base" in
  *_"$BENCH_BOX" | *_"$BENCH_BOX".*)
    printf '%s' "$f"
    return
    ;;
  esac
  case "$base" in
  *.*)
    ext=".${base##*.}"
    base="${base%.*}"
    ;;
  *) ext="" ;;
  esac
  printf '%s/%s_%s%s' "$dir" "$base" "$BENCH_BOX" "$ext"
}

# publish <glob> [<glob>…] — patterns are relative to bench/results.
# Renames each match to its box-tagged form, then uploads it
publish() {
  local pat f out
  for pat in "$@"; do
    # shellcheck disable=SC2231 — $pat is UNQUOTED: quoting suppresses glob
    # expansion; a non-matching pattern stays literal and fails the -e check below.
    for f in bench/results/$pat; do
      [ -e "$f" ] || continue
      out="$(boxed "$f")"
      [ "$out" = "$f" ] || mv -f "$f" "$out"
      command -v buildkite-agent >/dev/null 2>&1 &&
        { buildkite-agent artifact upload "$out" || true; }
    done
  done
  return 0
}
