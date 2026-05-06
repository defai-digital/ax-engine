#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
ROOT_DIR="$AX_REPO_ROOT"
PYTHON_BIN="$AX_PYTHON_BIN"
TMP_DIR="$(ax_tmp_dir ax-turboquant-microbench-gate)"

cleanup() {
    ax_rm_rf "$TMP_DIR" "$ROOT_DIR/scripts/__pycache__"
}

trap cleanup EXIT

cd "$ROOT_DIR"

cat > "$TMP_DIR/microbench.json" <<'JSON'
{
  "schema_version": "ax.turboquant_fused_decode_microbench.v1",
  "decode_path": "fused_compressed_decode",
  "kernel": "turboquant_fused_cold_decode_k8v4",
  "preset": "k8v4",
  "key_bits": 8,
  "value_bits": 4,
  "config": {
    "cold_tokens": [8192],
    "hot_tokens": 128,
    "variants": ["dim_parallel", "two_stage_scores"]
  },
  "rows": [
    {
      "cold_tokens": 8192,
      "n_kv_heads": 1,
      "head_dim": 128,
      "cpu_reference_wall_us": 11252,
      "full_precision_cold_kv_bytes": 8388608,
      "compressed_buffer_bytes": 1966080,
      "estimated_cold_saved_bytes": 6520832,
      "hot_tail_merge": {
        "hot_tokens": 128,
        "contract": "shared_logsumexp_partition_merge",
        "quality": {
          "max_abs_diff": 0.00000015,
          "mean_abs_diff": 0.00000003,
          "min_cosine_similarity": 0.99999994
        }
      },
      "kernel_variants": [
        {
          "name": "dim_parallel",
          "metal_wall_us": {
            "median": 78363,
            "min": 78133,
            "max": 78370,
            "samples": [78370, 78363, 78133]
          },
          "quality": {
            "max_abs_diff": 0.00000015,
            "mean_abs_diff": 0.00000003,
            "min_cosine_similarity": 0.99999994
          }
        },
        {
          "name": "two_stage_scores",
          "metal_wall_us": {
            "median": 1041,
            "min": 1013,
            "max": 1074,
            "samples": [1074, 1013, 1041]
          },
          "quality": {
            "max_abs_diff": 0.00000015,
            "mean_abs_diff": 0.00000003,
            "min_cosine_similarity": 0.99999994
          }
        }
      ]
    }
  ]
}
JSON

"$PYTHON_BIN" scripts/check_turboquant_microbench_artifact.py "$TMP_DIR/microbench.json"

cat > "$TMP_DIR/regressed-microbench.json" <<'JSON'
{
  "schema_version": "ax.turboquant_fused_decode_microbench.v1",
  "decode_path": "fused_compressed_decode",
  "kernel": "turboquant_fused_cold_decode_k8v4",
  "preset": "k8v4",
  "key_bits": 8,
  "value_bits": 4,
  "config": {
    "cold_tokens": [8192],
    "hot_tokens": 128,
    "variants": ["two_stage_scores"]
  },
  "rows": [
    {
      "cold_tokens": 8192,
      "n_kv_heads": 1,
      "head_dim": 128,
      "cpu_reference_wall_us": 11252,
      "full_precision_cold_kv_bytes": 8388608,
      "compressed_buffer_bytes": 1966080,
      "estimated_cold_saved_bytes": 6520832,
      "hot_tail_merge": {
        "hot_tokens": 128,
        "contract": "shared_logsumexp_partition_merge",
        "quality": {
          "max_abs_diff": 0.01,
          "mean_abs_diff": 0.00000003,
          "min_cosine_similarity": 0.99999994
        }
      },
      "kernel_variants": [
        {
          "name": "two_stage_scores",
          "metal_wall_us": {
            "median": 12000,
            "min": 12000,
            "max": 12000,
            "samples": [12000]
          },
          "quality": {
            "max_abs_diff": 0.00000015,
            "mean_abs_diff": 0.00000003,
            "min_cosine_similarity": 0.99999994
          }
        }
      ]
    }
  ]
}
JSON

if "$PYTHON_BIN" scripts/check_turboquant_microbench_artifact.py \
  "$TMP_DIR/regressed-microbench.json"; then
  echo "error: regressed TurboQuant microbench artifact unexpectedly passed gate" >&2
  exit 1
fi

echo "ok: TurboQuant microbench gate CLI pipeline"
