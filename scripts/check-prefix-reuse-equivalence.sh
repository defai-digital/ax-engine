#!/usr/bin/env bash
#
# Prefix-reuse equivalence regression gate.
#
# Runs `scripts/verify_prefix_reuse_equivalence.py --pad-to-block-size 16`
# against the mounted MLX model and asserts 5/5 PASS. Any token drift
# between cold and warm_repeat produces a non-zero exit code (3 per the
# harness's contract); ADR 0018 §Strategy 1/2/3 enabled architecture-
# specific prefix cache paths that must remain bit-exact.
#
# Required env:
#   AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR  Directory containing the MLX model
#                                     artifacts (config.json, tokenizer.json,
#                                     and safetensors). The CI workflow gates
#                                     this script on the directory being
#                                     available.
#
# Optional env:
#   AX_PREFIX_REUSE_MODEL_ID    Model id label passed to Session(...). The
#                               harness does not require this to match any
#                               specific name; it's just a label. Defaults
#                               to a value derived from the artifacts dir.
#   AX_PREFIX_REUSE_OUTPUT_DIR  Where to write the artifact. Defaults to a
#                               freshly-allocated temp directory that is
#                               removed on exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"

: "${AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR:?AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR is required}"
if [[ ! -d "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" ]]; then
    echo "error: AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR='$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR' is not a directory" >&2
    exit 2
fi

MODEL_ID="${AX_PREFIX_REUSE_MODEL_ID:-$(basename "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" | tr -c 'A-Za-z0-9._-' '_')}"

VENV_DIR="$(ax_tmp_dir ax-prefix-reuse-equivalence-venv)"
OUTPUT_DIR_SOURCE="external"
if [[ -z "${AX_PREFIX_REUSE_OUTPUT_DIR:-}" ]]; then
    AX_PREFIX_REUSE_OUTPUT_DIR="$(ax_tmp_dir ax-prefix-reuse-equivalence)"
    OUTPUT_DIR_SOURCE="ephemeral"
fi

cleanup() {
    ax_rm_rf "$VENV_DIR"
    if [[ "$OUTPUT_DIR_SOURCE" == "ephemeral" ]]; then
        ax_rm_rf "$AX_PREFIX_REUSE_OUTPUT_DIR"
    fi
}
trap cleanup EXIT

# Use a dedicated venv so the system Python isn't polluted and the maturin
# build picks up a clean ABI3 wheel for this run.
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --quiet --upgrade pip
python -m pip install --quiet "maturin>=1.7,<2" tokenizers

cd "$ROOT_DIR"

python -m maturin develop --quiet

# Run both modes; both must PASS for the gate to exit 0. Output artifacts
# go to separate files inside the output dir (the harness names them by
# model_id + UTC date, so we pass distinct output dirs per mode).
#   warm_repeat: same-prompt full-prefix snapshot path. Verifies Strategies
#       1/2/3 do bit-exact restore (Decode-mode item, snapshot HIT).
#   warm_extend: cold(P+suffix) vs warm(P then P+suffix). Verifies both the
#       cee4227e full-recompute fallback path AND, for non-MLA architectures,
#       the snapshot+chunked_prefill extension path. MLA's Prefill-mode
#       snapshot path is intentionally gated to fall through to full
#       recompute (the snapshot+extend math drifts on MLA layers).
WARM_REPEAT_DIR="$AX_PREFIX_REUSE_OUTPUT_DIR/warm_repeat"
WARM_EXTEND_DIR="$AX_PREFIX_REUSE_OUTPUT_DIR/warm_extend"
mkdir -p "$WARM_REPEAT_DIR" "$WARM_EXTEND_DIR"

echo "--- warm_repeat ---"
python scripts/verify_prefix_reuse_equivalence.py \
    --model-id "$MODEL_ID" \
    --mlx-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
    --mode warm_repeat \
    --pad-to-block-size 16 \
    --output "$WARM_REPEAT_DIR"

echo "--- warm_extend ---"
python scripts/verify_prefix_reuse_equivalence.py \
    --model-id "$MODEL_ID" \
    --mlx-artifacts-dir "$AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR" \
    --mode warm_extend \
    --pad-to-block-size 16 \
    --output "$WARM_EXTEND_DIR"

echo "prefix-reuse equivalence regression gate: PASS (warm_repeat + warm_extend)"
