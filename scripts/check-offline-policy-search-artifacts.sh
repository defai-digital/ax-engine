#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
ARTIFACT_ROOT="${AX_OFFLINE_POLICY_SEARCH_ARTIFACT_ROOT:-benchmarks/results/profiling/offline-policy-search}"
PUBLICATION_CUTOFF_DATE="2026-05-31"

cd "$ROOT_DIR"

artifacts=()
while IFS= read -r artifact; do
    if [[ -z "${AX_OFFLINE_POLICY_SEARCH_ARTIFACT_ROOT:-}" && "$artifact" =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then
        if [[ "${BASH_REMATCH[1]}" < "$PUBLICATION_CUTOFF_DATE" ]]; then
            continue
        fi
    fi
    artifacts+=("$artifact")
done < <(
    find "$ARTIFACT_ROOT" \
        -type f \
        -name '*.json' \
        -print 2>/dev/null | sort
)

if [[ "${#artifacts[@]}" -eq 0 ]]; then
    if [[ -z "${AX_OFFLINE_POLICY_SEARCH_ARTIFACT_ROOT:-}" ]]; then
        echo "[skip] no current offline policy search artifacts under $ARTIFACT_ROOT"
        exit 0
    fi
    echo "offline policy search artifact check failed: no artifacts found under $ARTIFACT_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/check_offline_policy_search_artifact.py "${artifacts[@]}"
