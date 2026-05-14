#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
ARTIFACT_ROOT="${AX_OFFLINE_POLICY_SEARCH_ARTIFACT_ROOT:-benchmarks/results/offline-policy-search}"

cd "$ROOT_DIR"

artifacts=()
while IFS= read -r artifact; do
    artifacts+=("$artifact")
done < <(
    find "$ARTIFACT_ROOT" \
        -type f \
        -name '*.json' \
        -print 2>/dev/null | sort
)

if [[ "${#artifacts[@]}" -eq 0 ]]; then
    echo "offline policy search artifact check failed: no artifacts found under $ARTIFACT_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/check_offline_policy_search_artifact.py "${artifacts[@]}"
