#!/usr/bin/env bash
# Assert the Flash Next observability goal's commits stay inside the declared
# source scope.
#
# Usage: scripts/check_flash_next_commit_scope.sh <baseline-sha>
#
# Asserts:
#   * the baseline is an ancestor of HEAD;
#   * the baseline..HEAD range is nonempty;
#   * every commit subject is nonempty;
#   * every changed path of every commit (deletions, both rename sides, merges
#     resolved against the first parent) is inside the declared scope.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE="${1:-}"
if [ -z "$BASELINE" ]; then
  echo "usage: $0 <baseline-sha>" >&2
  exit 2
fi

# Declared scope (goal plan). Entries ending in '/' are directory prefixes.
# The three disclosed companions are marked below; they are required by the
# frozen `server-metrics-name` check and the per-reason /metrics exposure, and
# the goal report names them explicitly.
SCOPE_PATHS=(
  "crates/ax-engine-mlx/src/runner/"
  "crates/ax-engine-mlx/src/model/qwen4_exp_mtp.rs"
  "crates/ax-engine-server/src/metrics.rs"
  "crates/ax-engine-server/src/tests/metrics.rs"
  "scripts/"
  ".internal/reports/flash-next-"
  "docs/model-certifications/qwen3.8-flash-next.md"
)
# Disclosed scope expansion (see the goal report's "scope" section).
SCOPE_COMPANIONS=(
  "crates/ax-engine-server/src/app_state.rs"
  "crates/ax-engine-server/src/flash_next_fallback_keys.rs"
  "crates/ax-engine-server/tests/metrics.rs"
)

in_scope() {
  local path="$1" entry
  for entry in "${SCOPE_PATHS[@]}"; do
    case "$entry" in
      */) case "$path" in "$entry"*) return 0 ;; esac ;;
      *) [ "$path" = "$entry" ] && return 0 ;;
    esac
  done
  for entry in "${SCOPE_COMPANIONS[@]}"; do
    [ "$path" = "$entry" ] && return 0
  done
  return 1
}

fail=0

if ! git merge-base --is-ancestor "$BASELINE" HEAD; then
  echo "FAIL: baseline $BASELINE is not an ancestor of HEAD" >&2
  exit 1
fi

count="$(git rev-list --count "$BASELINE..HEAD")"
if [ "$count" -eq 0 ]; then
  echo "FAIL: empty commit range $BASELINE..HEAD" >&2
  exit 1
fi

echo "range $BASELINE..HEAD: $count commit(s)"
echo "declared scope + ${#SCOPE_COMPANIONS[@]} disclosed companion(s)"

for commit in $(git rev-list "$BASELINE..HEAD"); do
  subject="$(git show -s --format=%s "$commit")"
  if [ -z "$subject" ]; then
    echo "FAIL: commit $commit has an empty subject" >&2
    fail=1
  fi
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    if ! in_scope "$path"; then
      echo "FAIL: $commit changed out-of-scope path: $path" >&2
      fail=1
    fi
  done < <(git log -1 --format= --name-only --first-parent -m "$commit")
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "OK: every changed path of $count commit(s) is inside the declared scope"
