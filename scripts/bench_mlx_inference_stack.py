#!/usr/bin/env python3
"""Benchmark AX Engine MLX inference against MLX reference runtimes.

The primary reference is upstream `mlx_lm.benchmark`.

Every comparison run includes `mlx_lm.benchmark`. If that baseline fails, the
run fails instead of emitting AX-only numbers.

Examples:
  cargo build -p ax-engine-server --release

  python3 scripts/bench_mlx_inference_stack.py \
    --model-repo-id mlx-community/Qwen3.5-9B-MLX-4bit \
    --prompt-tokens 128,512,2048 \
    --generation-tokens 128 \
    --repetitions 5 \
    --cooldown 15

Optional llama.cpp Metal baseline:
  python3 scripts/bench_mlx_inference_stack.py \
    --llama-cpp-bench .internal/reference/llama.cpp/build/bin/llama-bench \
    --llama-cpp-gguf /path/to/model.gguf

The llama.cpp row is a shape-compatible external GGUF baseline. `llama-bench`
does not consume the MLX random-token prompt JSON, so it is not a prompt-hash
parity baseline and must not be used as repo-owned MLX throughput evidence.
Metal backend is selected at llama.cpp build time and verified at parse time
via the row's `backends` field; `--llama-cpp-extra-args` is for flags like
`-fa 1` (flash attention), not for selecting the backend.

Optional llama.cpp decode-at-depth evidence:
  add --llama-cpp-decode-at-depth to run a second llama-bench pass with
  `-p 0 -n {generation_tokens} -d {prompt_tokens}`. This records depth-aware
  decode metrics without replacing the regular shape-compatible `pp`/`tg` row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_ENGINE_SERVER = REPO_ROOT / "target/release/ax-engine-server"
DEFAULT_MODEL_REPO_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_MODEL_ID = DEFAULT_MODEL_REPO_ID
DEFAULT_PROMPT_TOKENS = "128,512,2048"
DEFAULT_GENERATION_TOKENS = 128
DEFAULT_REPETITIONS = 5
DEFAULT_COOLDOWN = 15.0
AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE = 0.97
AXENGINE_PORT = 0
MLX_LM_RANDOM_SEED = 0
GATEDDELTA_PREFILL_PROFILE_PROMPT_TOKENS = [512, 2048, 8192, 32768]

AX_ENGINE_DIRECT_KEY = "ax_engine_mlx"
AX_ENGINE_NGRAM_ACCEL_KEY = "ax_engine_mlx_ngram_accel"
AX_ENGINE_PURE_MTP_KEY = "ax_engine_mlx_pure_mtp"
AX_ENGINE_LINEAR_ATTENTION_PACK_KEY = "ax_engine_mlx_linear_pack"
AX_ENGINE_DENSE_FFN_PACK_KEY = "ax_engine_mlx_dense_ffn_pack"
AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY = "ax_engine_gemma4_assistant_mtp"
AX_ENGINE_GEMMA4_ASSISTANT_MTP_NGRAM_KEY = "ax_engine_gemma4_assistant_mtp_ngram"
AX_ENGINE_DIRECT_LINEAR_ATTENTION_INPUTS_KEY = (
    "ax_engine_mlx_direct_linear_attention_inputs"
)
AX_ENGINE_DIRECT_LINEAR_ATTENTION_POST_INPUT_KEY = (
    "ax_engine_mlx_direct_linear_attention_post_input"
)
PHASE0_CLAIM_GATE_SCHEMA_VERSION = "ax.phase0_claim_gate.v1"

AX_MLX_RUNTIME_IDENTITY = {
    "selected_backend": "mlx",
    "route_identity": "repo_owned_mlx",
    "resolution_policy": "mlx_only",
    "benchmark_surface": "mlx_inference_stack",
}

AX_PREFIX_CACHE_DISABLED_ENV = {
    "AX_MLX_PREFIX_CACHE_MAX_BYTES": "0",
    "AX_MLX_PREFIX_CACHE_MAX_ENTRIES": "0",
    "AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1",
}
AX_PREFIX_CACHE_DISABLED_MODE = "disabled_for_cold_prefill_benchmark"
AX_PREFIX_CACHE_ENABLED_MODE = "enabled_by_cli_for_prefix_cache_experiment"

LLAMA_CPP_METAL_RUNTIME_IDENTITY = {
    "selected_backend": "llama_cpp",
    "route_identity": "external_llama_cpp_metal",
    "resolution_policy": "external_gguf_baseline",
    "benchmark_surface": "llama_cpp_bench",
}


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def hf_cache_roots(explicit_root: Path | None = None) -> list[Path]:
    """Return Hugging Face Hub cache roots in the same order users expect."""
    if explicit_root is not None:
        return [explicit_root.expanduser()]

    roots: list[Path] = []
    if hf_hub_cache := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(hf_hub_cache).expanduser())
    if hf_home := os.environ.get("HF_HOME"):
        roots.append(Path(hf_home).expanduser() / "hub")
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(xdg_cache_home).expanduser() / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def latest_hf_cache_snapshot(repo_id: str, roots: list[Path]) -> Path | None:
    """Resolve a repo id to the current local Hugging Face cache snapshot."""
    repo_cache_name = f"models--{_slug_repo_id(repo_id)}"
    candidates: list[Path] = []

    for root in roots:
        repo_cache = root / repo_cache_name
        refs_main = repo_cache / "refs" / "main"
        if refs_main.is_file():
            revision = refs_main.read_text().strip()
            if revision:
                snapshot = repo_cache / "snapshots" / revision
                if snapshot.is_dir():
                    return snapshot

        snapshots = repo_cache / "snapshots"
        if snapshots.is_dir():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())

    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def missing_ax_model_artifacts(model_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not (model_dir / "model-manifest.json").is_file():
        missing.append("model-manifest.json")
    if not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    return missing


def resolve_model_dir(
    model_dir: Path | None, repo_id: str, hf_cache_root: Path | None
) -> Path:
    if model_dir is not None:
        return model_dir

    roots = hf_cache_roots(hf_cache_root)
    snapshot = latest_hf_cache_snapshot(repo_id, roots)
    if snapshot is None:
        roots_text = ", ".join(str(root) for root in roots)
        raise RuntimeError(
            f"no Hugging Face cache snapshot found for {repo_id}; searched: {roots_text}\n"
            "Download and prepare it first, for example:\n"
            f"  python3 scripts/download_model.py {repo_id}\n"
            "or pass --model-dir /path/to/AX-ready/model-artifacts"
        )
    missing = missing_ax_model_artifacts(snapshot)
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"Hugging Face cache snapshot for {repo_id} is not AX-ready: {snapshot}\n"
            f"Missing: {missing_text}\n"
            "Prepare the snapshot first, for example:\n"
            f"  ax-engine-bench generate-manifest {snapshot}\n"
            "or pass --model-dir /path/to/AX-ready/model-artifacts"
        )
    return snapshot


def arg_was_provided(argv: list[str], flag: str) -> bool:
    return any(value == flag or value.startswith(f"{flag}=") for value in argv)


def looks_like_hf_repo_id(value: str) -> bool:
    if "://" in value:
        return False
    if value.startswith(("/", "./", "../", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def normalize_model_repo_id_for_cache(
    *,
    model: str,
    model_repo_id: str,
    model_arg_explicit: bool,
    model_repo_id_arg_explicit: bool,
    model_dir_explicit: bool,
) -> str:
    if (
        model_arg_explicit
        and not model_repo_id_arg_explicit
        and not model_dir_explicit
        and model != DEFAULT_MODEL_ID
    ):
        if looks_like_hf_repo_id(model):
            return model
        raise ValueError(
            "--model was provided without --model-repo-id or --model-dir. "
            "Pass --model-repo-id for Hugging Face cache resolution, or "
            "--model-dir for an AX-ready local artifact directory."
        )
    return model_repo_id


def infer_hf_repo_id_from_path(path: Path) -> str | None:
    """Infer a HuggingFace repo_id from an HF cache snapshot path.

    HF cache layout: .../hub/models--<org>--<repo>/snapshots/<hash>
    Returns 'org/repo' or None when the path does not follow that structure.
    """
    for parent in (path, *path.parents):
        name = parent.name
        if name.startswith("models--"):
            slug = name[len("models--") :]
            if "--" in slug:
                org, rest = slug.split("--", 1)
                if org and rest:
                    return f"{org}/{rest}"
    return None


def ensure_ax_engine_server_binary(*, build: bool = True) -> None:
    if build:
        cmd = ["cargo", "build", "-p", "ax-engine-server", "--release"]
        print(f"  [build] {' '.join(cmd)}", file=sys.stderr)
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                "cargo was not found; cannot build ax-engine-server"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"cargo build -p ax-engine-server --release failed with exit={error.returncode}"
            ) from error

    if not AX_ENGINE_SERVER.exists():
        raise RuntimeError(
            f"ax-engine-server not found at {AX_ENGINE_SERVER}. "
            "Run: cargo build -p ax-engine-server --release"
        )


CLAIMS_REQUIRING_ARTIFACT_EVIDENCE = [
    "continuous_batching",
    "prefix_reuse",
    "long_context_prefill_improvement",
]

# Prefix-based telemetry collection: any key the Rust side emits under these
# prefixes is automatically included in benchmark artifacts without requiring
# a matching entry in a static allowlist.  Adding a new ax_ngram_* or ax_mtp_*
# key in runner.rs is sufficient — no Python change needed.
AX_NGRAM_TELEMETRY_PREFIXES: tuple[str, ...] = (
    "ax_ngram_",
    "ax_mtp_",
    "ax_prompt_class_",
)

# Keys whose values are categorical/enum (not additive counters) and must be
# aggregated with max() across repetitions instead of sum().  Naming convention:
# any key whose suffix matches one of these strings is treated as max-merge.
_AX_NGRAM_MAX_MERGE_SUFFIXES: tuple[str, ...] = ("_code", "_variant")

# PRD §8 Phase 6 helper: stable ordered list of the accept-at-depth keys so
# downstream aggregation can iterate without re-deriving the count.
AX_NGRAM_ACCEPT_AT_DEPTH_KEYS = [
    "ax_ngram_accept_at_depth_0",
    "ax_ngram_accept_at_depth_1",
    "ax_ngram_accept_at_depth_2",
    "ax_ngram_accept_at_depth_3",
    "ax_ngram_accept_at_depth_4",
    "ax_ngram_accept_at_depth_5",
    "ax_ngram_accept_at_depth_6",
    "ax_ngram_accept_at_depth_7",
]

AX_NGRAM_ACCEPT_RATE_KEY = "ax_ngram_accept_rate_micros"

AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUT_KEYS = [
    "ax_mlx_direct_cpp_linear_attention_inputs_attempts",
    "ax_mlx_direct_cpp_linear_attention_inputs_hits",
    "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
    "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked",
]

AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT_KEYS = [
    "ax_mlx_direct_cpp_linear_attention_post_input_attempts",
    "ax_mlx_direct_cpp_linear_attention_post_input_hits",
    "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks",
    "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked",
]

AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_KEYS = [
    "ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts",
    "ax_mlx_qwen_linear_attention_decode_post_input_metal_hits",
    "ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks",
    "ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked",
]

AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_KEYS = [
    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts",
    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits",
    "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks",
]

AX_MLX_TELEMETRY_KEYS = [
    # Resolved speculation profile (ADR-022): 0=auto, 1=coding, 2=agentic, 3=chatbot.
    "ax_mlx_speculation_profile",
    "ax_mlx_prefill_steps",
    "ax_mlx_prefill_wall_us",
    "ax_mlx_prefill_forward_wall_us",
    "ax_mlx_prefill_prefix_cache_wall_us",
    "ax_mlx_prefill_generation_state_wall_us",
    "ax_mlx_prefill_eval_barriers",
    "ax_mlx_prefill_drain_async_evals",
    "ax_mlx_decode_steps",
    "ax_mlx_decode_wall_us",
    "ax_mlx_direct_bootstrap_steps",
    "ax_mlx_direct_bootstrap_wall_us",
    "ax_mlx_direct_pipeline_steps",
    "ax_mlx_direct_pipeline_wall_us",
    "ax_mlx_direct_pipeline_forward_wall_us",
    "ax_mlx_direct_pipeline_forward_layer_loop_wall_us",
    "ax_mlx_direct_pipeline_forward_head_wall_us",
    "ax_mlx_direct_pipeline_argmax_wall_us",
    "ax_mlx_direct_pipeline_async_eval_wall_us",
    "ax_mlx_direct_pipeline_next_complete_wall_us",
    "ax_mlx_direct_pipeline_pending_eval_wall_us",
    "ax_mlx_direct_pipeline_pending_read_wall_us",
    "ax_mlx_direct_pipeline_op_count",
    "ax_mlx_direct_pipeline_linear_attention_layer_ops",
    "ax_mlx_direct_pipeline_linear_attention_layer_count",
    "ax_mlx_direct_pipeline_full_attention_layer_ops",
    "ax_mlx_direct_pipeline_full_attention_layer_count",
    "ax_mlx_single_decode_steps",
    "ax_mlx_single_decode_wall_us",
    "ax_mlx_ngram_decode_steps",
    "ax_mlx_ngram_decode_wall_us",
    "ax_mlx_diffusion_blocks",
    "ax_mlx_diffusion_denoise_steps",
    "ax_mlx_diffusion_converged_blocks",
    "ax_mlx_diffusion_denoise_wall_us",
    "ax_mlx_diffusion_commit_wall_us",
    "ax_mlx_diffusion_block_wall_us",
    "ax_mlx_bonus_tokens",
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_blocked",
    "ax_mlx_prefix_cache_blocked_policy_disabled",
    "ax_mlx_prefix_cache_blocked_unsupported_layout",
    "ax_mlx_prefix_cache_blocked_trim_failure",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_evictions",
    "ax_mlx_prefix_cache_reused_tokens",
    "ax_mlx_prefix_cache_warmup_tokens",
    "ax_mlx_prefix_cache_entries",
    "ax_mlx_prefix_cache_bytes_kib",
    "ax_mlx_dense_ffn_gate_up_packed_layers",
    "ax_mlx_dense_ffn_split_gate_up_layers",
    "ax_mlx_dense_attention_qkv_packed_layers",
    "ax_mlx_dense_attention_split_qkv_layers",
    "ax_mlx_linear_attention_qkvz_ba_packed_layers",
    "ax_mlx_linear_attention_split_qkvba_layers",
    *AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUT_KEYS,
    *AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT_KEYS,
    *AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL_KEYS,
    *AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL_KEYS,
    # Affine quantization bit summary — constant per model load, max-merged across trials.
    "ax_mlx_affine_tensor_count",
    "ax_mlx_affine_min_bits",
    "ax_mlx_affine_max_bits",
    "ax_mlx_affine_2bit_count",
    "ax_mlx_affine_3bit_count",
    "ax_mlx_affine_4bit_count",
    "ax_mlx_affine_5bit_count",
    "ax_mlx_affine_6bit_count",
    "ax_mlx_affine_8bit_count",
    "ax_mlx_experimental_3bit_gate",
]

# Affine quantization bit keys are constant per model load (set at startup, not
# accumulated per step). They must be max-merged across trials, not summed.
AX_MLX_AFFINE_MAX_KEYS: frozenset[str] = frozenset(
    key
    for key in AX_MLX_TELEMETRY_KEYS
    if key.startswith("ax_mlx_affine_") or key == "ax_mlx_experimental_3bit_gate"
)

AX_MLX_PREFIX_CACHE_MAX_KEYS = {
    "ax_mlx_prefix_cache_entries",
    "ax_mlx_prefix_cache_bytes_kib",
    "ax_mlx_dense_ffn_gate_up_packed_layers",
    "ax_mlx_dense_ffn_split_gate_up_layers",
    "ax_mlx_dense_attention_qkv_packed_layers",
    "ax_mlx_dense_attention_split_qkv_layers",
}
AX_MLX_PREFIX_CACHE_SUM_KEYS = {
    key
    for key in AX_MLX_TELEMETRY_KEYS
    if key.startswith("ax_mlx_prefix_cache_")
    and key not in AX_MLX_PREFIX_CACHE_MAX_KEYS
}

AX_SCHEDULER_TELEMETRY_KEYS = [
    "ax_scheduler_scheduled_prefill_tokens",
    "ax_scheduler_scheduled_decode_tokens",
    "ax_scheduler_skipped_prefill_tokens",
    "ax_scheduler_skipped_decode_tokens",
    "ax_scheduler_mixed_prefill_decode_batches",
]

AX_MLX_GEMMA4_MOE_PROFILE_KEYS = [
    "ax_mlx_gemma4_moe_profile_enabled",
    "ax_mlx_gemma4_moe_profile_decode_layers",
    "ax_mlx_gemma4_moe_profile_topk_selections",
    "ax_mlx_gemma4_moe_profile_sorted_gather_layers",
    "ax_mlx_gemma4_moe_profile_unsorted_gather_layers",
    "ax_mlx_gemma4_moe_profile_attention_wall_us",
    "ax_mlx_gemma4_moe_profile_dense_wall_us",
    "ax_mlx_gemma4_moe_profile_router_wall_us",
    "ax_mlx_gemma4_moe_profile_expert_wall_us",
    "ax_mlx_gemma4_moe_profile_post_wall_us",
]

AX_MLX_GEMMA4_ASSISTANT_MTP_KEYS = [
    "ax_mlx_gemma4_assistant_mtp_configured",
    "ax_mlx_gemma4_assistant_mtp_validated",
    "ax_mlx_gemma4_assistant_mtp_enabled",
    "ax_mlx_gemma4_assistant_mtp_attach_failed",
    "ax_mlx_gemma4_assistant_mtp_disable_reason",
    "ax_mlx_gemma4_assistant_mtp_depth",
    "ax_mlx_gemma4_assistant_mtp_confidence_mode",
    "ax_mlx_gemma4_assistant_mtp_draft_tokens",
    "ax_mlx_gemma4_assistant_mtp_accepted_tokens",
    "ax_mlx_gemma4_assistant_mtp_rejected_tokens",
    "ax_mlx_gemma4_assistant_mtp_corrections",
    "ax_mlx_gemma4_assistant_mtp_accept_rate_x1000",
    "ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us",
    "ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us",
    "ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us",
]

AX_MLX_LINEAR_ATTENTION_PROFILE_KEYS = [
    "ax_mlx_linear_attention_profile_enabled",
    "ax_mlx_linear_attention_profile_layers",
    "ax_mlx_linear_attention_profile_tokens",
    "ax_mlx_linear_attention_profile_projection_wall_us",
    "ax_mlx_linear_attention_profile_projection_qkvz_wall_us",
    "ax_mlx_linear_attention_profile_projection_ba_wall_us",
    "ax_mlx_linear_attention_profile_projection_qkv_wall_us",
    "ax_mlx_linear_attention_profile_projection_z_wall_us",
    "ax_mlx_linear_attention_profile_projection_a_wall_us",
    "ax_mlx_linear_attention_profile_projection_b_wall_us",
    "ax_mlx_linear_attention_profile_conv_wall_us",
    "ax_mlx_linear_attention_profile_qk_norm_wall_us",
    "ax_mlx_linear_attention_profile_recurrent_wall_us",
    "ax_mlx_linear_attention_profile_output_wall_us",
]

AX_MLX_PREFILL_PROFILE_KEYS = [
    "ax_mlx_prefill_profile_enabled",
    "ax_mlx_prefill_profile_prefill_steps",
    "ax_mlx_prefill_profile_layers",
    "ax_mlx_prefill_profile_tokens",
    "ax_mlx_prefill_profile_per_layer_input_wall_us",
    "ax_mlx_prefill_profile_pre_sdpa_wall_us",
    "ax_mlx_prefill_profile_pre_sdpa_qkv_proj_wall_us",
    "ax_mlx_prefill_profile_pre_sdpa_qk_norm_wall_us",
    "ax_mlx_prefill_profile_pre_sdpa_rope_kv_wall_us",
    "ax_mlx_prefill_profile_sdpa_wall_us",
    "ax_mlx_prefill_profile_post_attn_wall_us",
    "ax_mlx_prefill_profile_post_attn_ffn_wall_us",
    "ax_mlx_prefill_profile_post_attn_ffn_gate_up_wall_us",
    "ax_mlx_prefill_profile_post_attn_ffn_activation_wall_us",
    "ax_mlx_prefill_profile_post_attn_ffn_down_wall_us",
    "ax_mlx_prefill_profile_post_attn_output_proj_wall_us",
    "ax_mlx_prefill_profile_post_attn_residual_norm_wall_us",
    "ax_mlx_prefill_profile_post_attn_residual_gate_wall_us",
    "ax_mlx_prefill_profile_lm_head_wall_us",
]

AX_MLX_DECODE_PROFILE_KEYS = [
    "ax_mlx_decode_profile_enabled",
    "ax_mlx_decode_profile_decode_steps",
    "ax_mlx_decode_profile_layers",
    "ax_mlx_decode_profile_per_layer_input_wall_us",
    "ax_mlx_decode_profile_pre_sdpa_wall_us",
    "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us",
    "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us",
    "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us",
    "ax_mlx_decode_profile_sdpa_wall_us",
    "ax_mlx_decode_profile_post_attn_wall_us",
    "ax_mlx_decode_profile_post_attn_ffn_wall_us",
    "ax_mlx_decode_profile_post_attn_ffn_gate_up_wall_us",
    "ax_mlx_decode_profile_post_attn_ffn_activation_wall_us",
    "ax_mlx_decode_profile_post_attn_ffn_down_wall_us",
    "ax_mlx_decode_profile_post_attn_output_proj_wall_us",
    "ax_mlx_decode_profile_post_attn_residual_norm_wall_us",
    "ax_mlx_decode_profile_post_attn_residual_gate_wall_us",
    "ax_mlx_decode_profile_lm_head_wall_us",
]

AX_MLX_KV_COMPRESSION_TELEMETRY_KEYS = [
    "ax_mlx_kv_compression_request_snapshots",
    "ax_mlx_kv_compression_status",
    "ax_mlx_kv_compression_preset",
    "ax_mlx_kv_compression_key_bits",
    "ax_mlx_kv_compression_value_bits",
    "ax_mlx_kv_compression_eligible_layers",
    "ax_mlx_kv_compression_candidate_token_layers",
    "ax_mlx_kv_compression_hot_token_layers",
    "ax_mlx_kv_compression_full_precision_kib",
    "ax_mlx_kv_compression_estimated_compressed_kib",
    "ax_mlx_kv_compression_estimated_saved_kib",
    "ax_mlx_kv_compression_ratio_milli",
    "ax_mlx_kv_compression_route_metadata_schema",
    "ax_mlx_kv_compression_production_ready",
    "ax_mlx_kv_compression_production_blockers",
    "ax_mlx_kv_compression_runtime_storage_layers",
    "ax_mlx_kv_compression_runtime_storage_token_layers",
    "ax_mlx_kv_compression_runtime_storage_kib",
    "ax_mlx_kv_compression_runtime_storage_written_slots",
    "ax_mlx_kv_compression_shadow_sync_calls",
    "ax_mlx_kv_compression_shadow_sync_wall_us",
    "ax_mlx_kv_compression_decode_path",
    "ax_mlx_kv_compression_fused_decode_candidates",
    "ax_mlx_kv_compression_fused_decode_attempts",
    "ax_mlx_kv_compression_fused_decode_successes",
    "ax_mlx_kv_compression_fused_decode_metal_successes",
    "ax_mlx_kv_compression_fused_decode_fallbacks",
    "ax_mlx_kv_compression_fused_decode_fallback_reason",
    "ax_mlx_kv_compression_fused_decode_ready_candidates",
    "ax_mlx_kv_compression_fused_decode_blocked_prefill_only",
    "ax_mlx_kv_compression_fused_decode_blocked_attention_kind",
    "ax_mlx_kv_compression_fused_decode_blocked_linear_attention",
    "ax_mlx_kv_compression_fused_decode_blocked_sliding_window",
    "ax_mlx_kv_compression_fused_decode_blocked_kv_shared",
    "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer",
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset",
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim",
    "ax_mlx_kv_compression_fused_decode_blocked_gqa",
    "ax_mlx_kv_compression_fused_decode_blocked_missing_storage",
    "ax_mlx_kv_compression_fused_decode_query_readback_wall_us",
    "ax_mlx_kv_compression_fused_decode_cold_metal_wall_us",
    "ax_mlx_kv_compression_fused_decode_hot_tail_merge_wall_us",
    "ax_mlx_kv_compression_fused_decode_output_staging_wall_us",
]

KV_COMPRESSION_FUSED_DECODE_BLOCKED_COUNTERS = {
    "prefill_only": "ax_mlx_kv_compression_fused_decode_blocked_prefill_only",
    "attention_kind": "ax_mlx_kv_compression_fused_decode_blocked_attention_kind",
    "ineligible_layer": "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer",
    "unsupported_preset": "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset",
    "unsupported_head_dim": "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim",
    "gqa": "ax_mlx_kv_compression_fused_decode_blocked_gqa",
    "missing_storage": "ax_mlx_kv_compression_fused_decode_blocked_missing_storage",
}

KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND_COUNTERS = {
    "linear_attention": "ax_mlx_kv_compression_fused_decode_blocked_linear_attention",
    "sliding_window": "ax_mlx_kv_compression_fused_decode_blocked_sliding_window",
    "kv_shared": "ax_mlx_kv_compression_fused_decode_blocked_kv_shared",
}


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def collect_host_metadata() -> dict[str, Any]:
    """Gather Apple Silicon host provenance for benchmark artifact labelling."""
    chip = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
    memory_bytes = _sysctl("hw.memsize")
    try:
        memory_gb = round(int(memory_bytes) / (1024**3))
    except ValueError:
        memory_gb = "unknown"

    os_version = "unknown"
    try:
        os_version = subprocess.check_output(
            ["sw_vers", "-productVersion"], text=True
        ).strip()
    except Exception:
        pass

    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "os_version": os_version,
        "platform": sys.platform,
    }


def collect_build_metadata() -> dict[str, Any]:
    """Collect git commit and build profile for artifact provenance."""
    commit = "unknown"
    tracked_status: list[str] = []
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        tracked_status = [line for line in status.splitlines() if line.strip()]
    except Exception:
        pass

    return {
        "commit": commit,
        "build_profile": "release",
        "server_binary": str(AX_ENGINE_SERVER),
        "git_tracked_dirty": bool(tracked_status),
        "git_tracked_status": tracked_status[:50],
    }


def collect_model_metadata(model_dir: Path) -> dict[str, Any]:
    """Read model config.json for provenance; returns empty dict if unavailable."""
    metadata: dict[str, Any] = {}
    config_path = model_dir / "config.json"
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        cfg = {}
    keys = ("model_type", "num_hidden_layers", "vocab_size", "quantization")
    metadata.update({k: cfg[k] for k in keys if k in cfg})

    manifest_path = model_dir / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        manifest = {}

    if manifest:
        linear_attention = manifest.get("linear_attention") or {}
        metadata["model_family"] = manifest.get("model_family")
        metadata["linear_attention_enabled"] = native_manifest_has_linear_attention(
            linear_attention
        )
        if isinstance(linear_attention, dict):
            metadata["linear_attention"] = {
                key: linear_attention.get(key)
                for key in (
                    "full_attention_interval",
                    "num_value_heads",
                    "num_key_heads",
                    "key_head_dim",
                    "value_head_dim",
                    "conv_kernel_dim",
                )
                if key in linear_attention
            }
        projection_layout = linear_attention_projection_layout(manifest)
        if projection_layout:
            metadata["linear_attention_projection_layout"] = projection_layout

    return metadata


def linear_attention_projection_layout(manifest: dict[str, Any]) -> dict[str, Any]:
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        return {}

    role_counts: dict[str, int] = {}
    layer_roles: dict[int, set[str]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        role = tensor.get("role")
        layer_index = tensor.get("layer_index")
        if not isinstance(role, str) or not role.startswith(
            "linear_attention_in_proj_"
        ):
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
        if isinstance(layer_index, int):
            layer_roles.setdefault(layer_index, set()).add(role)

    if not role_counts:
        return {}

    packed_layers = sum(
        1
        for roles in layer_roles.values()
        if {
            "linear_attention_in_proj_qkvz",
            "linear_attention_in_proj_ba",
        }.issubset(roles)
    )
    split_layers = sum(
        1
        for roles in layer_roles.values()
        if {
            "linear_attention_in_proj_qkv",
            "linear_attention_in_proj_z",
            "linear_attention_in_proj_a",
            "linear_attention_in_proj_b",
        }.issubset(roles)
    )
    if packed_layers and split_layers:
        layout = "mixed"
    elif packed_layers:
        layout = "packed_qkvz_ba"
    elif split_layers:
        layout = "split_qkv_z_a_b"
    else:
        layout = "incomplete"

    return {
        "schema_version": "ax.linear_attention_projection_layout.v1",
        "layout": layout,
        "linear_layers": len(layer_roles),
        "packed_layers": packed_layers,
        "split_layers": split_layers,
        "role_counts": dict(sorted(role_counts.items())),
        "offline_pack_candidate": split_layers > 0 and packed_layers == 0,
    }


def native_manifest_has_linear_attention(linear_attention: Any) -> bool:
    if not isinstance(linear_attention, dict):
        return False
    return any(value is not None for value in linear_attention.values())


def parse_prompt_lengths(value: str) -> list[int]:
    prompt_lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not prompt_lengths:
        raise ValueError("--prompt-tokens must include at least one prompt length")
    if any(prompt_tokens <= 0 for prompt_tokens in prompt_lengths):
        raise ValueError("--prompt-tokens values must be positive integers")
    return prompt_lengths


def normalize_gateddelta_prefill_profile_prompt_lengths(value: str) -> list[int]:
    if value == DEFAULT_PROMPT_TOKENS:
        return list(GATEDDELTA_PREFILL_PROFILE_PROMPT_TOKENS)
    prompt_lengths = parse_prompt_lengths(value)
    if prompt_lengths != GATEDDELTA_PREFILL_PROFILE_PROMPT_TOKENS:
        expected = ",".join(
            str(item) for item in GATEDDELTA_PREFILL_PROFILE_PROMPT_TOKENS
        )
        raise ValueError(
            "--gateddelta-prefill-profile requires --prompt-tokens "
            f"{expected} so profile rows are comparable across runs"
        )
    return prompt_lengths


def build_gateddelta_prefill_profile_contract(
    model_metadata: dict[str, Any],
    prompt_lengths: list[int],
) -> dict[str, Any]:
    if not model_metadata.get("linear_attention_enabled"):
        raise RuntimeError(
            "--gateddelta-prefill-profile requires a Qwen3.5/Qwen3-Next-style "
            "linear-attention MLX manifest"
        )
    return {
        "schema_version": "ax.gateddelta_prefill_profile.v1",
        "evidence_contract": "gateddelta_prefill_profile",
        "purpose": "evidence_first_gateddelta_long_prompt_prefill_profile",
        "model_family": model_metadata.get("model_family")
        or model_metadata.get("model_type")
        or "unknown",
        "linear_attention_required": True,
        "direct_ax_row_required": True,
        "ngram_policy_allowed": False,
        "kv_compression_allowed": False,
        "prompt_tokens": prompt_lengths,
        "required_prompt_tokens": GATEDDELTA_PREFILL_PROFILE_PROMPT_TOKENS,
        "model_preflight": {
            "schema_version": model_metadata.get(
                "model_preflight_schema_version",
                "ax.gateddelta_prefill_model_preflight.v1",
            ),
            "status": "passed",
            "checker": "scripts/check_gateddelta_prefill_model.py",
            "model_family": model_metadata.get("model_family")
            or model_metadata.get("model_type")
            or "unknown",
            "model_type": model_metadata.get("model_type"),
            "linear_attention": model_metadata.get("linear_attention", {}),
        },
        "primary_metrics": [
            "prefill_tok_s",
            "ax_mlx_prefill_steps",
            "ax_mlx_prefill_wall_us",
            "ax_mlx_linear_attention_profile_layers",
            "ax_mlx_linear_attention_profile_tokens",
            "ax_mlx_linear_attention_profile_recurrent_wall_us",
            "baseline.prefill_ratio_to_mlx_lm",
        ],
        "runtime_profile_env": "AX_MLX_LINEAR_ATTENTION_PROFILE=1",
        "interpretation": (
            "This profile establishes prompt-length prefill slope evidence before "
            "GatedDelta scan or fusion kernel changes. Runtime profile counters "
            "are diagnostic timing-barrier counters, not headline throughput. "
            "Decode/n-gram improvements must not be inferred from these rows."
        ),
    }


def validate_gateddelta_prefill_profile_model(model_dir: Path) -> dict[str, Any]:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from check_gateddelta_prefill_model import (
        GatedDeltaPrefillModelError,
        validate_gateddelta_prefill_model,
    )

    try:
        return validate_gateddelta_prefill_model(model_dir)
    except GatedDeltaPrefillModelError as exc:
        raise RuntimeError(str(exc)) from exc


def ax_decode_policy(
    model_metadata: dict[str, Any],
    *,
    direct_mode: bool,
    mtp_disable_ngram_stacking: bool = False,
    gemma4_assistant_mtp: bool = False,
) -> str:
    if direct_mode:
        return "direct_no_ngram_acceleration"
    if gemma4_assistant_mtp:
        if mtp_disable_ngram_stacking:
            return "gemma4_assistant_mtp_no_ngram_stacking"
        return "gemma4_assistant_mtp_ngram_stacking"
    if mtp_disable_ngram_stacking:
        return "mtp_head_only_no_ngram_stacking"
    if model_metadata.get("linear_attention_enabled"):
        return "ngram_acceleration_linear_attention_branch_recompute"
    return "ngram_acceleration_kv_trim"


def ax_decode_claim_status(
    direct_mode: bool,
    telemetry: dict[str, int],
    *,
    mtp_disable_ngram_stacking: bool = False,
) -> str:
    """Return the throughput-and-fallback verdict for an AX decode row."""
    if direct_mode:
        return "direct_same_policy_baseline"
    if mtp_disable_ngram_stacking:
        if int(telemetry.get("ax_mtp_ngram_hit_steps", 0)) > 0:
            return "mtp_head_only_contract_violation"
        if int(telemetry.get("ax_mtp_draft_tokens", 0)) > 0:
            return "mtp_head_only_effective"
        return "mtp_head_only_no_observed_draft_path"
    if not telemetry or (
        int(telemetry.get("ax_ngram_draft_attempts", 0)) == 0
        and int(telemetry.get("ax_ngram_no_draft_steps", 0)) == 0
        and int(telemetry.get("ax_ngram_request_disabled_steps", 0)) == 0
    ):
        return "ngram_no_observed_draft_path"
    if int(telemetry.get("ax_ngram_draft_attempts", 0)) == 0 and (
        int(telemetry.get("ax_ngram_no_draft_steps", 0)) > 0
        or int(telemetry.get("ax_ngram_request_disabled_steps", 0)) > 0
    ):
        return "ngram_no_draft_direct_fallback"
    if int(telemetry.get("ax_ngram_accepted_tokens", 0)) == 0:
        return "ngram_no_accept_fallback"
    return "ngram_acceleration_effective_throughput"


def ax_ngram_outcome_tier(
    *,
    direct_mode: bool,
    status: str,
    route: str,
    mtp_disable_ngram_stacking: bool = False,
) -> str:
    """Map n-gram counters to a promotion-tier label.

    Categories (PRD Slice-5 implementation notes):
      direct_baseline              -- direct-mode row, not n-gram
      effective_throughput         -- accepted drafts produced a real speedup
      no_draft_fallback            -- no drafts attempted; ran as direct mode
      zero_accept_fallback         -- drafts attempted but none accepted
      direct_fallback_cost_reduction -- fallback mode but throughput beats mlx_lm
                                       baseline (determined by the outperformance
                                       checker, not by counter values alone)
      regression_or_neutral        -- n-gram at or below mlx_lm baseline

    The two comparison-dependent tiers (direct_fallback_cost_reduction and
    regression_or_neutral) cannot be determined from counters alone.  When this
    function is called without mlx_lm baseline data, fallback rows are labeled
    "no_draft_fallback" or "zero_accept_fallback".  The outperformance checker
    (check_direct_ngram_outperformance.py) assigns the comparison tiers when
    processing completed artifact directories.
    """
    if direct_mode:
        return "direct_baseline"
    if mtp_disable_ngram_stacking:
        if status == "mtp_head_only_effective":
            return "pure_mtp"
        if status == "mtp_head_only_contract_violation":
            return "contract_violation"
        return "no_draft_fallback"
    if (
        status == "ngram_acceleration_effective_throughput"
        and route == "ngram_verified_bonus_tokens"
    ):
        return "effective_throughput"
    if status == "ngram_no_draft_direct_fallback":
        return "no_draft_fallback"
    if status == "ngram_no_accept_fallback":
        return "zero_accept_fallback"
    # Unknown or unobserved path — treat conservatively as no acceleration.
    return "no_draft_fallback"


def ax_decode_effective_route(
    *,
    direct_mode: bool,
    model_metadata: dict[str, Any],
    telemetry: dict[str, int],
    ax_mlx_telemetry: dict[str, int],
    mtp_disable_ngram_stacking: bool = False,
) -> str:
    """Classify the route actually observed after fallback decisions.

    ``ax_decode_policy`` records the requested policy. This field records the
    effective route shown by runtime counters so linear-attention no-draft rows
    cannot be mistaken for a working n-gram acceleration path.
    """
    direct_steps = int(ax_mlx_telemetry.get("ax_mlx_direct_pipeline_steps", 0))
    single_steps = int(ax_mlx_telemetry.get("ax_mlx_single_decode_steps", 0))
    ngram_decode_steps = int(ax_mlx_telemetry.get("ax_mlx_ngram_decode_steps", 0))

    if direct_mode:
        return (
            "direct_pipeline_baseline"
            if direct_steps > 0
            else "direct_single_decode_baseline"
        )

    if mtp_disable_ngram_stacking:
        if int(telemetry.get("ax_mtp_ngram_hit_steps", 0)) > 0:
            return "mtp_head_only_contract_violation"
        if int(telemetry.get("ax_mtp_draft_tokens", 0)) > 0:
            return "mtp_head_only_verify_loop"
        return "mtp_head_only_not_observed"

    draft_attempts = int(telemetry.get("ax_ngram_draft_attempts", 0))
    accepted_tokens = int(telemetry.get("ax_ngram_accepted_tokens", 0))
    no_draft_steps = int(telemetry.get("ax_ngram_no_draft_steps", 0))
    request_disabled_steps = int(telemetry.get("ax_ngram_request_disabled_steps", 0))
    linear_no_draft_steps = int(
        telemetry.get("ax_ngram_fallback_linear_no_draft_steps", 0)
    )
    has_linear_attention = bool(model_metadata.get("linear_attention_enabled"))

    if draft_attempts == 0:
        if no_draft_steps > 0 or request_disabled_steps > 0:
            if has_linear_attention and linear_no_draft_steps > 0:
                if direct_steps > 0 and single_steps == 0:
                    return "linear_no_draft_direct_pipeline_fallback"
                if direct_steps > 0 and single_steps > 0:
                    return "linear_no_draft_mixed_fallback"
                return "linear_no_draft_single_decode_fallback"
            return "no_draft_fallback"
        return "ngram_route_not_observed"

    if accepted_tokens == 0:
        return "ngram_attempted_no_accept_fallback"
    if ngram_decode_steps > 0:
        return "ngram_verified_bonus_tokens"
    return "ngram_accepted_without_decode_route"


def canonical_prompt_hash(tokens: list[int]) -> str:
    """Stable SHA256-hex(prefix) hash of a token sequence.

    Used by the same-policy promotion gate (Rust harness +
    ``ax_decode_same_policy_baseline_identity``) so a direct-vs-n-gram
    pairing can verify both rows ran on the exact same prompt without
    embedding the full token vector in every artifact.

    Encoding: little-endian u32 per token, fed into SHA256. The 16-byte
    prefix of the hex digest is what gets recorded — sufficient to make
    accidental collisions effectively impossible while keeping artifact
    sizes manageable. Two callers MUST agree on the encoding; this is
    the single source of truth.
    """
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(4, byteorder="little", signed=False))
    return h.hexdigest()[:16]


def canonical_sampler_signature(sampler: dict[str, Any] | None) -> str:
    """Stable string encoding of sampler settings (PRD §7.1).

    Greedy-equivalent values collapse to ``"greedy"`` so a sampler dict
    of ``{}`` and ``{"temperature": 0.0, "top_p": 1.0}`` produce the
    same signature — the same-policy gate would otherwise reject a
    pairing that is actually identical.
    """
    if not sampler or not _sampler_breaks_greedy_exactness(sampler):
        return "greedy"
    parts = []
    if (t := sampler.get("temperature")) is not None:
        parts.append(f"temperature={float(t)}")
    if (p := sampler.get("top_p")) is not None:
        parts.append(f"top_p={float(p)}")
    if (k := sampler.get("top_k")) is not None:
        parts.append(f"top_k={int(k)}")
    if (rp := sampler.get("repetition_penalty")) is not None:
        parts.append(f"repetition_penalty={float(rp)}")
    return "sampling[" + ",".join(parts) + "]"


def build_row_identity(
    *,
    model_id: str,
    tokens: list[int],
    seed: int,
    max_output_tokens: int,
    sampler: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct the identity block embedded in every emitted bench row.

    Mirrors the Rust ``crates/ax-engine-bench/src/harness/ngram_claim_gate.rs::RowIdentity``
    so a Python aggregator and the Rust promotion gate can decide on the
    same five fields.
    """
    return {
        "schema": "ax.row_identity.v1",
        "model_id": model_id,
        "prompt_hash": canonical_prompt_hash(tokens),
        "seed": int(seed),
        "max_output_tokens": int(max_output_tokens),
        "sampler_signature": canonical_sampler_signature(sampler),
    }


def ax_decode_claim_mode(
    direct_mode: bool,
    sampler: dict[str, Any] | None = None,
    *,
    mtp_disable_ngram_stacking: bool = False,
) -> str:
    """Return the *correctness mode* an n-gram or direct row is claiming.

    Distinct from `ax_decode_claim_status`, which reports fallback / promotion
    state. `claim_mode` records what kind of claim the row can support at all:

    - ``direct_greedy_exact_baseline``: direct decode with greedy sampling.
      A row in this mode is the canonical same-policy baseline against which
      `ngram_greedy_exact_candidate` rows are compared (PRD §7.1).
    - ``ngram_greedy_exact_candidate``: n-gram-accelerated decode with greedy
      sampling. May be promoted to a same-policy baseline-equivalent claim if
      the direct baseline row has identical model identity, prompt hash, seed,
      token budget, and sampler config, *and* the generated token IDs match.
    - ``direct_sampling_not_distribution_exact``: direct decode with
      ``temperature > 0`` or top-p / top-k / repetition penalty active.
    - ``ngram_sampling_not_distribution_exact``: n-gram-accelerated decode
      with any of the above sampling knobs active. PRD §7.1 forbids promoting
      these rows as distribution-exact under the current n-gram verifier;
      such a claim would require probability-ratio acceptance plus residual
      correction (out of scope for the current PRD).

    `sampler` is the row's sampler config dict (`{"temperature": ..., ...}`).
    A missing or empty dict is interpreted as greedy.
    """
    sampling_active = _sampler_breaks_greedy_exactness(sampler)
    if direct_mode and sampling_active:
        return "direct_sampling_not_distribution_exact"
    if direct_mode:
        return "direct_greedy_exact_baseline"
    if mtp_disable_ngram_stacking and sampling_active:
        return "mtp_sampling_distribution_corrected"
    if mtp_disable_ngram_stacking:
        return "mtp_greedy_exact_candidate"
    if sampling_active:
        return "ngram_sampling_not_distribution_exact"
    return "ngram_greedy_exact_candidate"


def _sampler_breaks_greedy_exactness(sampler: dict[str, Any] | None) -> bool:
    """Return True when any sampler knob breaks deterministic argmax decode.

    The set of knobs is intentionally narrow: each is a documented PRD §7.2
    "sampling-mode warning" trigger. New knobs added later should be appended
    here so future rows produced under them are not silently promoted.
    """
    if not sampler:
        return False
    if float(sampler.get("temperature", 0.0)) > 0.0:
        return True
    top_p = sampler.get("top_p")
    if top_p is not None and float(top_p) < 1.0:
        return True
    top_k = sampler.get("top_k")
    if top_k is not None and int(top_k) > 0:
        return True
    rep_pen = sampler.get("repetition_penalty")
    if rep_pen is not None and float(rep_pen) != 1.0:
        return True
    return False


def assert_no_distribution_exact_promotion_under_sampling(row: dict[str, Any]) -> None:
    """Promotion gate: refuse to label a sampling-mode row as distribution-exact.

    PRD §7.1: sampling-mode n-gram rows may report throughput and acceptance
    telemetry, but must not be labeled distribution-exact. This helper is the
    single chokepoint test harnesses and aggregation jobs use to enforce
    that rule; it raises `ValueError` rather than returning so a misused
    pipeline halts before publishing a misleading claim.
    """
    mode = row.get("ax_decode_claim_mode")
    status = row.get("ax_decode_claim_status")
    if (
        mode
        in (
            "direct_sampling_not_distribution_exact",
            "ngram_sampling_not_distribution_exact",
        )
        and status == "ngram_acceleration_effective_throughput"
    ):
        # The status field is allowed to record effective throughput, but the
        # *promotion* — labeling the row as distribution-exact — is forbidden.
        # Currently nothing in this script emits a distribution_exact label
        # for sampling rows; this guard exists so a future patch cannot do so
        # without intentionally removing the check.
        promoted = bool(row.get("ax_decode_distribution_exact_claim"))
        if promoted:
            raise ValueError(
                "sampling-mode n-gram row cannot be promoted as distribution-exact "
                "without a probability-ratio acceptance + residual-correction implementation; "
                f"got mode={mode!r}, status={status!r}"
            )


MLX_AVERAGES_RE = re.compile(
    r"Averages:\s+prompt_tps=(?P<prompt>[0-9.]+),\s+"
    r"generation_tps=(?P<generation>[0-9.]+),\s+"
    r"peak_memory=(?P<memory>[0-9.]+)"
)
MLX_TRIAL_RE = re.compile(
    r"Trial\s+(?P<trial>\d+):\s+"
    r"prompt_tps=(?P<prompt>[0-9.]+),\s+"
    r"generation_tps=(?P<generation>[0-9.]+),\s+"
    r"peak_memory=(?P<memory>[0-9.]+),\s+"
    r"total_time=(?P<total>[0-9.]+)"
)


def wait_for_server(
    url: str,
    timeout: float = 600.0,
    proc: subprocess.Popen[Any] | None = None,
) -> bool:
    """Poll `url` (`/health`) until it returns < 500 or the process dies.

    Requires *two* consecutive 200s ~200 ms apart before declaring ready
    (one-shot 200 from a server about to crash or lose its port to
    TIME_WAIT does not count). Also probes once for "the spawned `proc`
    is the actual listener" by checking the response includes our
    server's signature and `proc.poll()` is still None — if the bench
    `pkill`ed a previous server but left an orphan listener, the new
    spawn would silently bind nothing while the orphan still owns the
    port. In that case `proc.poll()` returns non-None (bind failure
    exited the new process) and we bail out so the bench's retry path
    can clean up properly instead of recording the wrong model's
    numbers.
    """
    deadline = time.monotonic() + timeout
    consecutive_ok = 0
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace")
                if response.status < 500 and "ax-engine-server" in body:
                    consecutive_ok += 1
                    if consecutive_ok >= 2:
                        return True
                    time.sleep(0.2)
                    continue
                consecutive_ok = 0
        except Exception:
            consecutive_ok = 0
        time.sleep(0.5)
    return False


def ensure_port_available(port: int, host: str = "127.0.0.1") -> None:
    """Fail closed when an existing listener could contaminate benchmark rows."""
    if port == 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"AX Engine benchmark port {host}:{port} is already in use; "
                "stop the existing ax-engine-server process or pass "
                "--axengine-port with a free port before benchmarking."
            )


def allocate_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def process_stderr_snapshot(proc: subprocess.Popen[Any], limit: int = 2000) -> str:
    if proc.stderr is None:
        return ""
    if proc.poll() is None:
        return "<process still running; stderr is unavailable until process exit>"
    try:
        _stdout, stderr = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        return "<timed out while reading process stderr>"
    if not stderr:
        return ""
    return stderr.decode(errors="replace")[:limit]


def kill_proc(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def process_rss_gb(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    try:
        rss_kib = float(output.splitlines()[-1].strip())
    except ValueError:
        return None
    if rss_kib <= 0.0:
        return None
    return rss_kib / (1024.0 * 1024.0)


def model_vocab_size(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        vocab_size = config.get("vocab_size")
        if vocab_size is None:
            vocab_size = config.get("text_config", {}).get("vocab_size")
        if vocab_size is not None:
            return int(vocab_size)

    print(
        f"  [prompt] loading mlx_lm config because {config_path} did not expose vocab_size",
        file=sys.stderr,
    )
    from mlx_lm import load

    loaded = load(str(model_dir), return_config=True)
    config = loaded[2]
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    return int(vocab_size)


def mlx_lm_reference_prompt_tokens(vocab_size: int, target_tokens: int) -> list[int]:
    import mlx.core as mx

    mx.random.seed(MLX_LM_RANDOM_SEED)
    prompt = mx.random.randint(0, vocab_size, (1, target_tokens)).tolist()[0]
    tokens = [int(token) for token in prompt]
    if len(tokens) != target_tokens:
        raise RuntimeError(
            f"generated prompt length mismatch: target={target_tokens} actual={len(tokens)}"
        )
    return tokens


def token_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_prompt_tokens(
    artifact_root: Path,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    vocab_size: int,
    tokens: list[int],
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    token_hash = token_sha256(tokens)
    path = (
        artifact_root
        / f"prompt-{prompt_tokens}-gen-{generation_tokens}-{token_hash[:12]}.json"
    )
    payload = {
        "schema_version": "ax.mlx_reference_prompt.v1",
        "source": "mlx_lm.benchmark",
        "random_seed": MLX_LM_RANDOM_SEED,
        "prompt_distribution": "mx.random.randint(0, vocab_size, (1, prompt_tokens))",
        "vocab_size": vocab_size,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "sha256": token_hash,
        "token_ids": tokens,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"  [prompt] prompt_tokens={prompt_tokens} sha256={token_hash[:12]} path={path}",
        file=sys.stderr,
    )
    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "vocab_size": vocab_size,
        "random_seed": MLX_LM_RANDOM_SEED,
        "token_ids_path": str(path),
        "token_ids_sha256": token_hash,
        "token_count": len(tokens),
    }


def build_reference_prompts(
    prompt_lengths: list[int],
    generation_tokens: int,
    model_dir: Path,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    vocab_size = model_vocab_size(model_dir)
    prompts = []
    for prompt_tokens in prompt_lengths:
        tokens = mlx_lm_reference_prompt_tokens(vocab_size, prompt_tokens)
        prompt_doc = write_prompt_tokens(
            artifact_root,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            vocab_size=vocab_size,
            tokens=tokens,
        )
        prompt_doc["token_ids"] = tokens
        prompt_doc["prompt_source"] = "random"
        prompts.append(prompt_doc)
    return prompts


REAL_PROMPT_SCHEMA_VERSION = "ax.real_prompt.v1"


@dataclass(frozen=True)
class RealPromptCase:
    """A single real-text prompt case loaded from a JSONL suite.

    Schema mirrors MTPLX's `PromptCase` so suite files written against
    one project can be reused against the other without translation.
    `max_tokens` caps generation per-case; the bench harness still
    clamps to its own `--generation-tokens` budget at decode time.
    """

    id: str
    category: str
    prompt: str
    max_tokens: int = 128


def load_real_prompt_suite(path: Path) -> list[RealPromptCase]:
    """Load a JSONL real-prompt suite.

    Each non-empty, non-comment line must decode to a JSON object with
    `id`, `category`, and `prompt` keys. `max_tokens` defaults to 128
    when omitted. Duplicate ids and malformed JSON raise ValueError so
    misconfigured suites fail at load time, not silently mid-bench.
    """
    cases: list[RealPromptCase] = []
    seen_ids: set[str] = set()
    raw_text = Path(path).read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        for required in ("id", "category", "prompt"):
            if required not in payload:
                raise ValueError(
                    f"{path}:{line_no} is missing required key {required!r}"
                )
        case_id = str(payload["id"])
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_no} duplicate prompt id {case_id!r}")
        seen_ids.add(case_id)
        cases.append(
            RealPromptCase(
                id=case_id,
                category=str(payload["category"]),
                prompt=str(payload["prompt"]),
                max_tokens=int(payload.get("max_tokens", 128)),
            )
        )
    if not cases:
        raise ValueError(f"{path} contains no prompt cases")
    return cases


def load_model_tokenizer(model_dir: Path) -> Any:
    """Load only the tokenizer from an MLX model directory.

    Reuses `mlx_lm.load` so the tokenizer choice matches the reference
    path used by mlx_lm.benchmark. The full model load is wasted work,
    but it keeps the tokenizer choice byte-identical to the reference
    runtime for architectures mlx_lm supports.

    Some AX-native architectures (e.g. Gemma 4's unified ``gemma4_unified``
    12B) have no upstream mlx_lm graph, so the full `load` raises before it
    ever reaches the tokenizer. mlx_lm builds its tokenizer from the same
    `mlx_lm.tokenizer_utils.load` helper regardless of model class, so we
    fall back to that model-free loader to obtain the identical tokenizer
    without instantiating an unsupported model graph. This only affects
    `--prompt-source real` AX-only runs (mlx_lm rows are skipped for those
    architectures anyway).
    """
    from mlx_lm import load

    try:
        _model, tokenizer, _config = load(str(model_dir), return_config=True)
        return tokenizer
    except (ValueError, ModuleNotFoundError) as error:
        from mlx_lm.tokenizer_utils import load as load_tokenizer_only

        print(
            f"  [tokenizer] mlx_lm.load could not instantiate the model graph "
            f"({error}); falling back to model-free tokenizer load",
            file=sys.stderr,
        )
        return load_tokenizer_only(Path(model_dir))


def tokenize_real_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    chat_template: bool,
    enable_thinking: bool = True,
) -> list[int]:
    """Encode a real prompt to token IDs.

    `chat_template=True` applies the tokenizer's chat template with a
    single user turn and `add_generation_prompt=True`. This is the
    correct path for instruction-tuned models (Gemma 4 IT, Qwen
    chat-tuned, etc.): raw-text encoding on an IT model triggers
    immediate EOS at decode step 0 because the model never saw an
    end-of-turn marker. `chat_template=False` encodes the literal text
    and is the right choice for base (non-IT) models.

    `enable_thinking=False` passes `enable_thinking=False` to
    `apply_chat_template`, which for Qwen3-series models pre-fills the
    `<think></think>` block so the model skips reasoning and generates
    the answer directly.  Ignored when `chat_template=False`.
    """
    if chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "tokenizer does not expose apply_chat_template; "
                "rerun with --no-real-prompt-chat-template for raw encoding"
            )
        kwargs: dict[str, Any] = {}
        if not enable_thinking:
            kwargs["enable_thinking"] = False
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        return [int(token) for token in encoded]
    return [int(token) for token in tokenizer.encode(prompt)]


def write_real_prompt_tokens(
    artifact_root: Path,
    *,
    suite_id: str,
    case: RealPromptCase,
    tokens: list[int],
    generation_tokens: int,
    vocab_size: int,
    chat_template_applied: bool = True,
) -> dict[str, Any]:
    """Persist a real-prompt token artifact and return its prompt_doc.

    The on-disk filename includes the suite id, case id, generation
    budget, and the first 12 hex chars of the token sha256 so artifacts
    are stable and easy to grep across runs. The returned dict carries
    the full token list so the AX harness can feed it without re-reading
    the file.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    token_hash = token_sha256(tokens)
    prompt_text_hash = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
    path = (
        artifact_root
        / f"real-{suite_id}-{case.id}-gen-{generation_tokens}-{token_hash[:12]}.json"
    )
    payload = {
        "schema_version": REAL_PROMPT_SCHEMA_VERSION,
        "source": "ax_real_prompt_suite",
        "prompt_source": "real",
        "prompt_suite_id": suite_id,
        "prompt_case_id": case.id,
        "prompt_category": case.category,
        "prompt_text_sha256": prompt_text_hash,
        "prompt_tokens": len(tokens),
        "generation_tokens": generation_tokens,
        "vocab_size": vocab_size,
        "case_max_tokens": case.max_tokens,
        "chat_template_applied": bool(chat_template_applied),
        "sha256": token_hash,
        "token_ids": tokens,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"  [prompt] real suite={suite_id} case={case.id} "
        f"tokens={len(tokens)} text_sha256={prompt_text_hash[:12]} path={path}",
        file=sys.stderr,
    )
    return {
        "prompt_source": "real",
        "prompt_suite_id": suite_id,
        "prompt_case_id": case.id,
        "prompt_category": case.category,
        "prompt_text_sha256": prompt_text_hash,
        "prompt_tokens": len(tokens),
        "generation_tokens": generation_tokens,
        "vocab_size": vocab_size,
        "case_max_tokens": case.max_tokens,
        "chat_template_applied": bool(chat_template_applied),
        "token_ids_path": str(path),
        "token_ids_sha256": token_hash,
        "token_count": len(tokens),
        "token_ids": tokens,
    }


def build_real_prompts(
    suite_path: Path,
    generation_tokens: int,
    model_dir: Path,
    artifact_root: Path,
    *,
    chat_template: bool = True,
    enable_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Build prompt_doc list from a real-text JSONL suite.

    Parallel to `build_reference_prompts` but driven by a static suite
    rather than `mx.random.randint`. The suite filename stem becomes the
    `prompt_suite_id` carried on every row; the suite case ids become
    the `prompt_case_id`. `chat_template=True` (the default) is required
    for instruction-tuned models; pass `False` for base/raw models.
    `enable_thinking=False` disables Qwen3-series think mode.
    """
    cases = load_real_prompt_suite(suite_path)
    suite_id = Path(suite_path).stem
    tokenizer = load_model_tokenizer(model_dir)
    vocab_size = model_vocab_size(model_dir)
    prompts: list[dict[str, Any]] = []
    for case in cases:
        tokens = tokenize_real_prompt(
            tokenizer,
            case.prompt,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
        )
        prompts.append(
            write_real_prompt_tokens(
                artifact_root,
                suite_id=suite_id,
                case=case,
                tokens=tokens,
                generation_tokens=generation_tokens,
                vocab_size=vocab_size,
                chat_template_applied=chat_template,
            )
        )
    return prompts


def without_inline_tokens(prompt_doc: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in prompt_doc.items() if key != "token_ids"}


def validate_prompt_doc(
    prompt_doc: dict[str, Any],
    *,
    prompt_tokens: int,
    generation_tokens: int,
) -> None:
    if int(prompt_doc.get("prompt_tokens", -1)) != prompt_tokens:
        raise RuntimeError(
            "prompt artifact token count mismatch: "
            f"expected={prompt_tokens} actual={prompt_doc.get('prompt_tokens')}"
        )
    if int(prompt_doc.get("generation_tokens", -1)) != generation_tokens:
        raise RuntimeError(
            "prompt artifact generation count mismatch: "
            f"expected={generation_tokens} actual={prompt_doc.get('generation_tokens')}"
        )

    tokens = prompt_doc.get("token_ids")
    if tokens is not None:
        token_ids = [int(token) for token in tokens]
        if len(token_ids) != prompt_tokens:
            raise RuntimeError(
                "prompt artifact inline token length mismatch: "
                f"expected={prompt_tokens} actual={len(token_ids)}"
            )
        expected_hash = prompt_doc.get("token_ids_sha256")
        if expected_hash and token_sha256(token_ids) != expected_hash:
            raise RuntimeError("prompt artifact inline token hash mismatch")

    path = prompt_doc.get("token_ids_path")
    expected_hash = prompt_doc.get("token_ids_sha256")
    if path and expected_hash:
        payload = json.loads(Path(path).read_text())
        file_tokens = [int(token) for token in payload.get("token_ids", [])]
        if len(file_tokens) != prompt_tokens:
            raise RuntimeError(
                "prompt artifact file token length mismatch: "
                f"expected={prompt_tokens} actual={len(file_tokens)}"
            )
        if token_sha256(file_tokens) != expected_hash:
            raise RuntimeError("prompt artifact file token hash mismatch")


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_run_stability(
    runs: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    values = [float(run[metric]) for run in runs if run.get(metric) is not None]
    if len(values) < 2:
        return {
            "schema_version": "ax.benchmark_run_stability.v1",
            "metric": metric,
            "classification": "insufficient_repetitions",
            "trial_count": len(values),
        }

    first = values[0]
    last = values[-1]
    minimum = min(values)
    maximum = max(values)
    median = statistics.median(values)
    relative_spread_pct = (
        ((maximum - minimum) / median * 100.0) if median > 0.0 else 0.0
    )
    last_vs_first_pct = ((last / first - 1.0) * 100.0) if first > 0.0 else 0.0
    if last_vs_first_pct <= -5.0:
        classification = "tail_regression"
    elif relative_spread_pct >= 8.0:
        classification = "high_variance"
    else:
        classification = "stable_enough"

    return {
        "schema_version": "ax.benchmark_run_stability.v1",
        "metric": metric,
        "classification": classification,
        "trial_count": len(values),
        "first": first,
        "last": last,
        "min": minimum,
        "max": maximum,
        "median": median,
        "relative_spread_pct": relative_spread_pct,
        "last_vs_first_pct": last_vs_first_pct,
    }


def format_run_stability_label(cell: dict[str, Any]) -> str:
    stability = cell.get("run_stability")
    if not isinstance(stability, dict):
        return "n/a"

    classification = str(stability.get("classification", "unknown"))
    if classification == "stable_enough":
        return "stable"
    if classification == "insufficient_repetitions":
        return "insufficient"

    drift = stability.get("last_vs_first_pct")
    if isinstance(drift, (int, float)):
        if classification == "tail_regression":
            return f"tail {drift:+.1f}%"
        if classification == "high_variance":
            return f"var {drift:+.1f}%"
        return f"{classification} {drift:+.1f}%"
    return classification


def format_axengine_interim_summary(runs: list[dict[str, Any]]) -> str:
    decode = summarize_values([float(run["decode_tok_s"]) for run in runs])
    output = summarize_values([float(run["output_tokens"]) for run in runs])
    prefill_values = [
        float(run["prefill_tok_s"])
        for run in runs
        if run.get("prefill_tok_s") is not None
    ]
    prefill = summarize_values(prefill_values) if prefill_values else None
    prefill_label = (
        f"{prefill['median']:.1f} tok/s"
        if prefill is not None
        else "cache_warm"
    )
    stability = summarize_run_stability(runs, "decode_tok_s")
    stability_label = ""
    if stability["classification"] != "insufficient_repetitions":
        stability_label = (
            f" drift={stability['last_vs_first_pct']:+.1f}%"
            f"/{stability['classification']}"
        )
    return (
        f"median prefill={prefill_label} "
        f"decode={decode['median']:.1f} tok/s "
        f"range={decode['min']:.1f}-{decode['max']:.1f} "
        f"out_median={output['median']:.0f}"
        f"{stability_label}"
    )


def attach_derived_ttft_ms(
    cell: dict[str, Any],
    *,
    prompt_tokens: int,
    source: str,
) -> None:
    trials = cell.get("trials")
    if isinstance(trials, list) and trials:
        values = []
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            prefill_tok_s = float(trial.get("prefill_tok_s", 0.0))
            if prefill_tok_s <= 0.0:
                continue
            ttft_ms = prompt_tokens / prefill_tok_s * 1000.0
            trial["ttft_ms"] = ttft_ms
            values.append(ttft_ms)
        if values:
            cell["ttft_ms"] = summarize_values(values)
            cell["ttft_source"] = source
            return

    prefill_tok_s = metric_value(cell, "prefill_tok_s")
    if prefill_tok_s > 0.0:
        cell["ttft_ms"] = {"median": prompt_tokens / prefill_tok_s * 1000.0}
        cell["ttft_source"] = source


def parse_mlx_lm_benchmark_output(output: str) -> dict[str, Any]:
    trials = []
    for match in MLX_TRIAL_RE.finditer(output):
        trials.append(
            {
                "trial": int(match.group("trial")),
                "prefill_tok_s": float(match.group("prompt")),
                "decode_tok_s": float(match.group("generation")),
                "peak_memory_gb": float(match.group("memory")),
                "total_time_s": float(match.group("total")),
            }
        )

    averages = MLX_AVERAGES_RE.search(output)
    if not averages:
        raise RuntimeError("mlx_lm.benchmark output did not contain an Averages line")

    parsed = {
        "prefill_tok_s": {"mean": float(averages.group("prompt"))},
        "decode_tok_s": {"mean": float(averages.group("generation"))},
        "peak_memory_gb": {"mean": float(averages.group("memory"))},
        "trials": trials,
    }
    if trials:
        parsed["prefill_tok_s"] = summarize_values(
            [float(trial["prefill_tok_s"]) for trial in trials]
        )
        parsed["decode_tok_s"] = summarize_values(
            [float(trial["decode_tok_s"]) for trial in trials]
        )
        parsed["peak_memory_gb"] = summarize_values(
            [float(trial["peak_memory_gb"]) for trial in trials]
        )
        parsed["total_time_s"] = summarize_values(
            [float(trial["total_time_s"]) for trial in trials]
        )
        parsed["reported_averages"] = {
            "prefill_tok_s": float(averages.group("prompt")),
            "decode_tok_s": float(averages.group("generation")),
            "peak_memory_gb": float(averages.group("memory")),
        }
    return parsed


def run_mlx_lm_benchmark(
    model: str,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    prefill_step_size: int,
    prompt_doc: dict[str, Any],
) -> dict[str, Any]:
    validate_prompt_doc(
        prompt_doc,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )
    cmd = [
        "python3",
        "-m",
        "mlx_lm.benchmark",
        "--model",
        model,
        "--batch-size",
        "1",
        "--prompt-tokens",
        str(prompt_tokens),
        "--generation-tokens",
        str(generation_tokens),
        "--num-trials",
        str(repetitions),
        "--delay",
        str(int(cooldown)),
        "--prefill-step-size",
        str(prefill_step_size),
    ]
    print(f"  [mlx_lm] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"mlx_lm.benchmark failed with exit={result.returncode}:\n{combined}"
        )
    cell = parse_mlx_lm_benchmark_output(combined)
    cell.update(
        {
            "engine": "mlx_lm",
            "method": "mlx_lm.benchmark",
            "timing_scope": "upstream_mlx_lm_response_stats",
            "prompt_contract": "mlx_lm_random_tokens_seed_0",
            "random_seed": MLX_LM_RANDOM_SEED,
            "batch_size": 1,
            "prefill_step_size": prefill_step_size,
            "prompt_token_ids_origin": "reproduced_from_mlx_lm_benchmark_algorithm",
            "prompt_token_ids_path": prompt_doc["token_ids_path"],
            "prompt_token_ids_sha256": prompt_doc["token_ids_sha256"],
            "prompt_source": "random",
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
        }
    )
    attach_derived_ttft_ms(
        cell,
        prompt_tokens=prompt_tokens,
        source="derived_from_mlx_lm_prefill_tok_s",
    )
    return cell


def start_axengine(
    binary: Path,
    model_dir: Path,
    port: int,
    *,
    model_id: str,
    direct_mode: bool,
    kv_compression: str = "disabled",
    kv_compression_hot_window_tokens: int | None = None,
    kv_compression_min_context_tokens: int | None = None,
    gemma4_moe_profile: bool = False,
    linear_attention_profile: bool = False,
    prefill_profile: bool = False,
    decode_profile: bool = False,
    pack_linear_attention_projections: bool = False,
    pack_dense_ffn_gate_up: bool = False,
    qwen_dense_ffn_gate_up_matvec_metal: bool = False,
    direct_linear_attention_inputs_route: bool = False,
    direct_linear_attention_post_input_route: bool = False,
    gemma4_assistant_mtp: bool = False,
    mtp_max_depth: int | None = None,
    mtp_disable_ngram_stacking: bool = False,
    mtp_fast_tail_topk_sampling: bool = False,
    prefill_chunk: int | None = None,
    max_batch_tokens: int | None = None,
    prefix_cache_enabled: bool = False,
) -> subprocess.Popen[Any]:
    ensure_port_available(port)
    cmd = [
        str(binary),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--model-id",
        model_id,
        "--port",
        str(port),
    ]
    if direct_mode:
        cmd.append("--disable-ngram-acceleration")
    if mtp_disable_ngram_stacking:
        cmd.append("--mlx-mtp-disable-ngram-stacking")
    if prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(prefill_chunk)])
    if max_batch_tokens is not None:
        cmd.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if kv_compression != "disabled":
        cmd.extend(["--experimental-mlx-kv-compression", kv_compression])
        if kv_compression_hot_window_tokens is not None:
            cmd.extend(
                [
                    "--experimental-mlx-kv-compression-hot-window-tokens",
                    str(kv_compression_hot_window_tokens),
                ]
            )
        if kv_compression_min_context_tokens is not None:
            cmd.extend(
                [
                    "--experimental-mlx-kv-compression-min-context-tokens",
                    str(kv_compression_min_context_tokens),
                ]
            )
    env = {**os.environ, "AX_MLX_NATIVE_CONFIRM": "1"}
    if not prefix_cache_enabled:
        env.update(AX_PREFIX_CACHE_DISABLED_ENV)
    if gemma4_moe_profile:
        env["AX_MLX_GEMMA4_MOE_PROFILE"] = "1"
    if linear_attention_profile:
        env["AX_MLX_LINEAR_ATTENTION_PROFILE"] = "1"
    if prefill_profile:
        env["AX_MLX_PREFILL_PROFILE"] = "1"
    if decode_profile:
        env["AX_MLX_DECODE_PROFILE"] = "1"
    if pack_linear_attention_projections:
        env["AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS"] = "1"
    if pack_dense_ffn_gate_up:
        env["AX_MLX_PACK_DENSE_FFN_GATE_UP"] = "1"
    if qwen_dense_ffn_gate_up_matvec_metal:
        env["AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL"] = "1"
    if direct_linear_attention_inputs_route:
        env["AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS"] = "1"
    if direct_linear_attention_post_input_route:
        env["AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT"] = "1"
    if gemma4_assistant_mtp:
        env["AX_MLX_GEMMA4_ASSISTANT_MTP"] = "1"
    if mtp_max_depth is not None:
        env["AX_MLX_MTP_MAX_DEPTH"] = str(mtp_max_depth)
        if gemma4_assistant_mtp:
            env["AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH"] = str(mtp_max_depth)
    if mtp_fast_tail_topk_sampling:
        env["AX_MLX_MTP_FAST_TAIL_TOPK_SAMPLING"] = "1"
    print(f"  [ax-engine] {' '.join(cmd)}", file=sys.stderr)
    if mtp_max_depth is not None:
        print(f"  [ax-engine] AX_MLX_MTP_MAX_DEPTH={mtp_max_depth}", file=sys.stderr)
    if mtp_disable_ngram_stacking:
        print("  [ax-engine] MTP n-gram stacking disabled", file=sys.stderr)
    if mtp_fast_tail_topk_sampling:
        print("  [ax-engine] AX_MLX_MTP_FAST_TAIL_TOPK_SAMPLING=1", file=sys.stderr)
    if gemma4_assistant_mtp:
        print("  [ax-engine] AX_MLX_GEMMA4_ASSISTANT_MTP=1", file=sys.stderr)
    if not prefix_cache_enabled:
        print(
            "  [ax-engine] prefix cache disabled for cold prefill/TTFT measurement",
            file=sys.stderr,
        )
    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
    )


def ax_linear_attention_profile_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.gateddelta_prefill_profile
        or args.ax_linear_attention_profile
        or args.ax_compare_linear_attention_projection_pack
    )


def validate_direct_linear_attention_post_input_route_compare_args(
    args: argparse.Namespace,
) -> None:
    if not args.ax_compare_direct_linear_attention_post_input_route:
        return
    if args.skip_ax_engine:
        raise ValueError(
            "--ax-compare-direct-linear-attention-post-input-route requires AX rows"
        )
    if args.ax_ngram_accel or args.ax_compare_policies:
        raise ValueError(
            "--ax-compare-direct-linear-attention-post-input-route requires direct AX rows; "
            "do not combine it with --ax-ngram-accel or --ax-compare-policies"
        )
    if (
        args.ax_compare_linear_attention_projection_pack
        or args.ax_compare_dense_ffn_gate_up_pack
    ):
        raise ValueError(
            "--ax-compare-direct-linear-attention-post-input-route runs paired AX rows; "
            "run one comparison at a time"
        )
    if args.gateddelta_prefill_profile or args.ax_linear_attention_profile:
        raise ValueError(
            "--ax-compare-direct-linear-attention-post-input-route cannot be combined with "
            "linear-attention profiling because profiling blocks the route"
        )
    if args.ax_prefill_profile or args.ax_decode_profile:
        raise ValueError(
            "--ax-compare-direct-linear-attention-post-input-route cannot be combined with "
            "--ax-prefill-profile or --ax-decode-profile for throughput A/B artifacts"
        )


def extract_ax_ngram_telemetry(route: dict[str, Any] | None) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    telemetry = {
        k: int(v)
        for k, v in decisions.items()
        if any(k.startswith(p) for p in AX_NGRAM_TELEMETRY_PREFIXES)
    }
    if (
        telemetry.get("ax_ngram_draft_attempts", 0) > 0
        and AX_NGRAM_ACCEPT_RATE_KEY not in telemetry
    ):
        draft_tokens = telemetry.get("ax_ngram_draft_tokens", 0)
        accepted_tokens = telemetry.get("ax_ngram_accepted_tokens", 0)
        if draft_tokens > 0:
            telemetry[AX_NGRAM_ACCEPT_RATE_KEY] = int(
                round(accepted_tokens * 1_000_000 / draft_tokens)
            )
    return telemetry


def summarize_ngram_accept_at_depth(telemetry: dict[str, int]) -> dict[str, Any]:
    """Project the flat ``ax_ngram_accept_at_depth_*`` counters into a
    stable histogram artifact (PRD §8 Phase 6 / I-6).

    Schema ``ax.ngram_accept_at_depth.v1`` is used by downstream
    promotion aggregators and by ``profile_ngram_observability.py`` to
    surface a single object per row instead of eight flat keys.

    Returns ``{}`` when no depth telemetry is present so a row produced
    by an older runtime (or without n-gram) is not annotated with a
    spurious zero histogram.
    """
    buckets = []
    total_attempts = 0
    weighted_accepted = 0
    any_present = False
    for depth, key in enumerate(AX_NGRAM_ACCEPT_AT_DEPTH_KEYS):
        if key in telemetry:
            any_present = True
        attempts = int(telemetry.get(key, 0))
        buckets.append({"depth": depth, "attempts": attempts})
        total_attempts += attempts
        weighted_accepted += depth * attempts
    if not any_present:
        return {}
    return {
        "schema": "ax.ngram_accept_at_depth.v1",
        "bucket_count": len(AX_NGRAM_ACCEPT_AT_DEPTH_KEYS),
        "buckets": buckets,
        "total_attempts": total_attempts,
        # Lower bound on accepted tokens recoverable from the histogram.
        # Equal to `ax_ngram_accepted_tokens` exactly when no attempt
        # saturates into the last bucket (the saturating clamp can hide
        # accepts beyond depth 7); a sustained gap to
        # `ax_ngram_accepted_tokens` indicates draft lengths > 8 and
        # would justify raising NGRAM_ACCEPT_DEPTH_BUCKETS.
        "weighted_accepted_tokens_lower_bound": weighted_accepted,
    }


def extract_ax_mlx_telemetry(route: dict[str, Any] | None) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if not any(key in decisions for key in AX_MLX_TELEMETRY_KEYS):
        return {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_TELEMETRY_KEYS}


def extract_scheduler_telemetry(route: dict[str, Any] | None) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    return {key: int(decisions.get(key, 0)) for key in AX_SCHEDULER_TELEMETRY_KEYS}


def extract_ax_mlx_gemma4_moe_profile(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_gemma4_moe_profile_enabled" not in decisions:
        return {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_GEMMA4_MOE_PROFILE_KEYS}


def extract_ax_mlx_gemma4_assistant_mtp(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_gemma4_assistant_mtp_configured" not in decisions:
        return {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_GEMMA4_ASSISTANT_MTP_KEYS}


def extract_ax_mlx_linear_attention_profile(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_linear_attention_profile_enabled" not in decisions:
        return {}
    return {
        key: int(decisions.get(key, 0)) for key in AX_MLX_LINEAR_ATTENTION_PROFILE_KEYS
    }


def extract_ax_mlx_decode_profile(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_decode_profile_enabled" not in decisions:
        return {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_DECODE_PROFILE_KEYS}


def extract_ax_mlx_prefill_profile(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_prefill_profile_enabled" not in decisions:
        return {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_PREFILL_PROFILE_KEYS}


def extract_ax_mlx_kv_compression_telemetry(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    present = [key for key in AX_MLX_KV_COMPRESSION_TELEMETRY_KEYS if key in decisions]
    if not present:
        return {}
    return {
        key: int(decisions.get(key, 0)) for key in AX_MLX_KV_COMPRESSION_TELEMETRY_KEYS
    }


def route_with_more_decisions(
    candidate: dict[str, Any] | None, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not candidate:
        return current
    if not current:
        return candidate
    candidate_decisions = candidate.get("crossover_decisions") or {}
    current_decisions = current.get("crossover_decisions") or {}
    static_priority_keys = {
        *AX_MLX_TELEMETRY_KEYS,
        *AX_MLX_GEMMA4_ASSISTANT_MTP_KEYS,
        *AX_MLX_PREFILL_PROFILE_KEYS,
        *AX_MLX_DECODE_PROFILE_KEYS,
    }

    def priority_score(decisions: dict[str, Any]) -> tuple[int, int, int, int]:
        priority_values = [
            int(decisions.get(key, 0))
            for key in static_priority_keys
            if int(decisions.get(key, 0)) > 0
        ]
        priority_values += [
            int(v)
            for k, v in decisions.items()
            if any(k.startswith(p) for p in AX_NGRAM_TELEMETRY_PREFIXES) and int(v) > 0
        ]
        return (
            len(priority_values),
            sum(priority_values),
            len(decisions),
            sum(int(value) for value in decisions.values()),
        )

    if priority_score(candidate_decisions) > priority_score(current_decisions):
        return candidate
    return current


def merge_step_local_route_decisions(
    totals: dict[str, int],
    route: dict[str, Any] | None,
) -> None:
    if not route:
        return
    decisions = route.get("crossover_decisions") or {}
    for key in AX_MLX_PREFIX_CACHE_SUM_KEYS:
        totals[key] = totals.get(key, 0) + int(decisions.get(key, 0))
    for key in AX_SCHEDULER_TELEMETRY_KEYS:
        totals[key] = totals.get(key, 0) + int(decisions.get(key, 0))
    for key in AX_MLX_PREFIX_CACHE_MAX_KEYS:
        totals[key] = max(totals.get(key, 0), int(decisions.get(key, 0)))


def route_with_step_local_decisions(
    route: dict[str, Any] | None,
    step_local_decisions: dict[str, int],
) -> dict[str, Any] | None:
    if not step_local_decisions:
        return route
    merged = copy.deepcopy(route) if route else {}
    decisions = dict(merged.get("crossover_decisions") or {})
    decisions.update(step_local_decisions)
    merged["crossover_decisions"] = decisions
    return merged


def route_for_linear_attention_profile(
    prefill_route: dict[str, Any] | None,
    final_route: dict[str, Any] | None,
) -> dict[str, Any] | None:
    # The linear-attention profile is used for prefill/TTFT diagnosis. The final
    # response route usually points at the last decode step, which would report
    # seq=1 and hide prompt-length costs.
    return prefill_route or final_route


def summarize_telemetry(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ngram_acceleration_telemetry") or {}).items():
            if key == AX_NGRAM_ACCEPT_RATE_KEY:
                continue
            if any(key.endswith(s) for s in _AX_NGRAM_MAX_MERGE_SUFFIXES):
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    if totals.get("ax_ngram_draft_tokens", 0) > 0:
        totals[AX_NGRAM_ACCEPT_RATE_KEY] = int(
            round(
                totals.get("ax_ngram_accepted_tokens", 0)
                * 1_000_000
                / totals["ax_ngram_draft_tokens"]
            )
        )
    return totals


def summarize_ax_mlx_telemetry(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_telemetry") or {}).items():
            if key in AX_MLX_AFFINE_MAX_KEYS:
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_decode_route(telemetry: dict[str, int]) -> dict[str, Any]:
    decode_steps = int(telemetry.get("ax_mlx_decode_steps", 0))
    decode_wall_us = int(telemetry.get("ax_mlx_decode_wall_us", 0))
    direct_pipeline_steps = int(telemetry.get("ax_mlx_direct_pipeline_steps", 0))
    direct_pipeline_wall_us = int(telemetry.get("ax_mlx_direct_pipeline_wall_us", 0))
    direct_pipeline_forward_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_forward_wall_us", 0)
    )
    direct_pipeline_forward_layer_loop_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_forward_layer_loop_wall_us", 0)
    )
    direct_pipeline_forward_head_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_forward_head_wall_us", 0)
    )
    direct_pipeline_argmax_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_argmax_wall_us", 0)
    )
    direct_pipeline_async_eval_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_async_eval_wall_us", 0)
    )
    direct_pipeline_next_complete_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_next_complete_wall_us", 0)
    )
    direct_pipeline_pending_eval_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_pending_eval_wall_us", 0)
    )
    direct_pipeline_pending_read_wall_us = int(
        telemetry.get("ax_mlx_direct_pipeline_pending_read_wall_us", 0)
    )
    direct_pipeline_op_count = int(telemetry.get("ax_mlx_direct_pipeline_op_count", 0))
    direct_pipeline_linear_attention_layer_ops = int(
        telemetry.get("ax_mlx_direct_pipeline_linear_attention_layer_ops", 0)
    )
    direct_pipeline_linear_attention_layer_count = int(
        telemetry.get("ax_mlx_direct_pipeline_linear_attention_layer_count", 0)
    )
    direct_pipeline_full_attention_layer_ops = int(
        telemetry.get("ax_mlx_direct_pipeline_full_attention_layer_ops", 0)
    )
    direct_pipeline_full_attention_layer_count = int(
        telemetry.get("ax_mlx_direct_pipeline_full_attention_layer_count", 0)
    )
    single_decode_steps = int(telemetry.get("ax_mlx_single_decode_steps", 0))
    single_decode_wall_us = int(telemetry.get("ax_mlx_single_decode_wall_us", 0))
    ngram_decode_steps = int(telemetry.get("ax_mlx_ngram_decode_steps", 0))
    ngram_decode_wall_us = int(telemetry.get("ax_mlx_ngram_decode_wall_us", 0))
    diffusion_blocks = int(telemetry.get("ax_mlx_diffusion_blocks", 0))
    diffusion_denoise_steps = int(
        telemetry.get("ax_mlx_diffusion_denoise_steps", 0)
    )
    diffusion_converged_blocks = int(
        telemetry.get("ax_mlx_diffusion_converged_blocks", 0)
    )
    diffusion_denoise_wall_us = int(
        telemetry.get("ax_mlx_diffusion_denoise_wall_us", 0)
    )
    diffusion_commit_wall_us = int(
        telemetry.get("ax_mlx_diffusion_commit_wall_us", 0)
    )
    diffusion_block_wall_us = int(
        telemetry.get("ax_mlx_diffusion_block_wall_us", 0)
    )

    if decode_steps <= 0:
        classification = "no_decode_steps"
    elif (
        direct_pipeline_steps == decode_steps
        and single_decode_steps == 0
        and ngram_decode_steps == 0
    ):
        classification = "direct_pipeline"
    elif (
        single_decode_steps == decode_steps
        and direct_pipeline_steps == 0
        and ngram_decode_steps == 0
    ):
        classification = "single_decode"
    elif (
        ngram_decode_steps > 0
        and direct_pipeline_steps == 0
        and single_decode_steps == 0
    ):
        classification = "ngram"
    elif diffusion_blocks > 0:
        classification = "diffusion"
    else:
        classification = "mixed"

    def share_micros(part: int, total: int) -> int:
        if total <= 0:
            return 0
        return int(round(part * 1_000_000 / total))

    return {
        "classification": classification,
        "decode_steps": decode_steps,
        "decode_wall_us": decode_wall_us,
        "direct_pipeline_steps": direct_pipeline_steps,
        "direct_pipeline_step_share_micros": share_micros(
            direct_pipeline_steps,
            decode_steps,
        ),
        "direct_pipeline_wall_share_micros": share_micros(
            direct_pipeline_wall_us,
            decode_wall_us,
        ),
        "direct_pipeline_forward_wall_us": direct_pipeline_forward_wall_us,
        "direct_pipeline_forward_wall_share_micros": share_micros(
            direct_pipeline_forward_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_forward_layer_loop_wall_us": (
            direct_pipeline_forward_layer_loop_wall_us
        ),
        "direct_pipeline_forward_layer_loop_wall_share_micros": share_micros(
            direct_pipeline_forward_layer_loop_wall_us,
            direct_pipeline_forward_wall_us,
        ),
        "direct_pipeline_forward_head_wall_us": direct_pipeline_forward_head_wall_us,
        "direct_pipeline_forward_head_wall_share_micros": share_micros(
            direct_pipeline_forward_head_wall_us,
            direct_pipeline_forward_wall_us,
        ),
        "direct_pipeline_argmax_wall_us": direct_pipeline_argmax_wall_us,
        "direct_pipeline_argmax_wall_share_micros": share_micros(
            direct_pipeline_argmax_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_async_eval_wall_us": direct_pipeline_async_eval_wall_us,
        "direct_pipeline_async_eval_wall_share_micros": share_micros(
            direct_pipeline_async_eval_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_next_complete_wall_us": direct_pipeline_next_complete_wall_us,
        "direct_pipeline_next_complete_wall_share_micros": share_micros(
            direct_pipeline_next_complete_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_pending_eval_wall_us": direct_pipeline_pending_eval_wall_us,
        "direct_pipeline_pending_eval_wall_share_micros": share_micros(
            direct_pipeline_pending_eval_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_pending_read_wall_us": direct_pipeline_pending_read_wall_us,
        "direct_pipeline_pending_read_wall_share_micros": share_micros(
            direct_pipeline_pending_read_wall_us,
            direct_pipeline_wall_us,
        ),
        "direct_pipeline_op_count": direct_pipeline_op_count,
        "direct_pipeline_op_count_per_step": (
            direct_pipeline_op_count // direct_pipeline_steps
            if direct_pipeline_steps > 0
            else 0
        ),
        "direct_pipeline_linear_attention_layer_ops": (
            direct_pipeline_linear_attention_layer_ops
        ),
        "direct_pipeline_linear_attention_layer_count": (
            direct_pipeline_linear_attention_layer_count
        ),
        "direct_pipeline_linear_attention_ops_per_layer": (
            direct_pipeline_linear_attention_layer_ops
            // direct_pipeline_linear_attention_layer_count
            if direct_pipeline_linear_attention_layer_count > 0
            else 0
        ),
        "direct_pipeline_full_attention_layer_ops": direct_pipeline_full_attention_layer_ops,
        "direct_pipeline_full_attention_layer_count": (
            direct_pipeline_full_attention_layer_count
        ),
        "direct_pipeline_full_attention_ops_per_layer": (
            direct_pipeline_full_attention_layer_ops
            // direct_pipeline_full_attention_layer_count
            if direct_pipeline_full_attention_layer_count > 0
            else 0
        ),
        "single_decode_steps": single_decode_steps,
        "single_decode_step_share_micros": share_micros(
            single_decode_steps,
            decode_steps,
        ),
        "single_decode_wall_share_micros": share_micros(
            single_decode_wall_us,
            decode_wall_us,
        ),
        "ngram_decode_steps": ngram_decode_steps,
        "ngram_decode_step_share_micros": share_micros(
            ngram_decode_steps,
            decode_steps,
        ),
        "ngram_decode_wall_share_micros": share_micros(
            ngram_decode_wall_us,
            decode_wall_us,
        ),
        "bonus_tokens": int(telemetry.get("ax_mlx_bonus_tokens", 0)),
        "diffusion_blocks": diffusion_blocks,
        "diffusion_denoise_steps": diffusion_denoise_steps,
        "diffusion_converged_blocks": diffusion_converged_blocks,
        "diffusion_denoise_wall_us": diffusion_denoise_wall_us,
        "diffusion_commit_wall_us": diffusion_commit_wall_us,
        "diffusion_block_wall_us": diffusion_block_wall_us,
        "diffusion_denoise_step_share_micros": share_micros(
            diffusion_denoise_steps,
            decode_steps,
        ),
        "diffusion_block_wall_share_micros": share_micros(
            diffusion_block_wall_us,
            decode_wall_us,
        ),
    }


def summarize_attempted_fastpath(
    telemetry: dict[str, int],
    *,
    schema_version: str,
    attempts_key: str,
    hits_key: str,
    fallbacks_key: str,
    profile_blocked_key: str | None,
) -> dict[str, Any]:
    attempts = int(telemetry.get(attempts_key, 0))
    if attempts <= 0:
        return {}
    hits = int(telemetry.get(hits_key, 0))
    fallbacks = int(telemetry.get(fallbacks_key, 0))
    profile_blocked = (
        int(telemetry.get(profile_blocked_key, 0)) if profile_blocked_key else 0
    )
    accounted = hits + fallbacks + profile_blocked
    if accounted < attempts:
        classification = "incomplete_accounting"
    elif profile_blocked > 0 and hits > 0:
        classification = "mixed_hit_profile_blocked"
    elif profile_blocked > 0:
        classification = "profile_blocked_fallback"
    elif hits == attempts and fallbacks == 0:
        classification = "all_hits"
    elif hits > 0 and fallbacks > 0:
        classification = "mixed_hit_fallback"
    elif fallbacks >= attempts:
        classification = "all_fallback"
    else:
        classification = "incomplete_accounting"
    return {
        "schema_version": schema_version,
        "classification": classification,
        "attempts": attempts,
        "hits": hits,
        "fallbacks": fallbacks,
        "profile_blocked": profile_blocked,
        "hit_rate_micros": int(round(hits * 1_000_000 / attempts)),
    }


def summarize_ax_mlx_direct_cpp_linear_attention_inputs(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    return summarize_attempted_fastpath(
        telemetry,
        schema_version="ax.mlx_direct_cpp_linear_attention_inputs.v1",
        attempts_key="ax_mlx_direct_cpp_linear_attention_inputs_attempts",
        hits_key="ax_mlx_direct_cpp_linear_attention_inputs_hits",
        fallbacks_key="ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
        profile_blocked_key=(
            "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked"
        ),
    )


def summarize_ax_mlx_direct_cpp_linear_attention_post_input(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    return summarize_attempted_fastpath(
        telemetry,
        schema_version="ax.mlx_direct_cpp_linear_attention_post_input.v1",
        attempts_key="ax_mlx_direct_cpp_linear_attention_post_input_attempts",
        hits_key="ax_mlx_direct_cpp_linear_attention_post_input_hits",
        fallbacks_key="ax_mlx_direct_cpp_linear_attention_post_input_fallbacks",
        profile_blocked_key=(
            "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked"
        ),
    )


def summarize_ax_mlx_qwen_linear_attention_decode_post_input_metal(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    return summarize_attempted_fastpath(
        telemetry,
        schema_version="ax.mlx_qwen_linear_attention_decode_post_input_metal.v1",
        attempts_key=(
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts"
        ),
        hits_key="ax_mlx_qwen_linear_attention_decode_post_input_metal_hits",
        fallbacks_key=(
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks"
        ),
        profile_blocked_key=(
            "ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked"
        ),
    )


def summarize_ax_mlx_qwen_dense_ffn_gate_up_matvec_metal(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    return summarize_attempted_fastpath(
        telemetry,
        schema_version="ax.mlx_qwen_dense_ffn_gate_up_matvec_metal.v1",
        attempts_key="ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts",
        hits_key="ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits",
        fallbacks_key="ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks",
        profile_blocked_key=None,
    )


def summarize_counted_layout(
    *,
    packed_count: int,
    split_count: int,
) -> str:
    if packed_count > 0 and split_count == 0:
        return "packed"
    if split_count > 0 and packed_count == 0:
        return "split"
    if packed_count > 0 and split_count > 0:
        return "mixed"
    return "absent"


def summarize_ax_mlx_effective_routes(telemetry: dict[str, int]) -> dict[str, Any]:
    if not telemetry:
        return {}
    dense_attention_qkv_packed = int(
        telemetry.get("ax_mlx_dense_attention_qkv_packed_layers", 0)
    )
    dense_attention_qkv_split = int(
        telemetry.get("ax_mlx_dense_attention_split_qkv_layers", 0)
    )
    dense_ffn_gate_up_packed = int(
        telemetry.get("ax_mlx_dense_ffn_gate_up_packed_layers", 0)
    )
    dense_ffn_gate_up_split = int(
        telemetry.get("ax_mlx_dense_ffn_split_gate_up_layers", 0)
    )
    linear_attention_qkvz_ba_packed = int(
        telemetry.get("ax_mlx_linear_attention_qkvz_ba_packed_layers", 0)
    )
    linear_attention_qkvz_ba_split = int(
        telemetry.get("ax_mlx_linear_attention_split_qkvba_layers", 0)
    )
    return {
        "schema_version": "ax.mlx_effective_routes.v1",
        "dense_attention_qkv": {
            "status": summarize_counted_layout(
                packed_count=dense_attention_qkv_packed,
                split_count=dense_attention_qkv_split,
            ),
            "packed_layers": dense_attention_qkv_packed,
            "split_layers": dense_attention_qkv_split,
        },
        "dense_ffn_gate_up": {
            "status": summarize_counted_layout(
                packed_count=dense_ffn_gate_up_packed,
                split_count=dense_ffn_gate_up_split,
            ),
            "packed_layers": dense_ffn_gate_up_packed,
            "split_layers": dense_ffn_gate_up_split,
            "qwen_gate_up_matvec_metal": (
                summarize_ax_mlx_qwen_dense_ffn_gate_up_matvec_metal(telemetry)
                or {
                    "schema_version": (
                        "ax.mlx_qwen_dense_ffn_gate_up_matvec_metal.v1"
                    ),
                    "classification": "not_attempted",
                    "attempts": 0,
                    "hits": 0,
                    "fallbacks": 0,
                    "profile_blocked": 0,
                    "hit_rate_micros": 0,
                }
            ),
        },
        "linear_attention_qkvz_ba": {
            "status": summarize_counted_layout(
                packed_count=linear_attention_qkvz_ba_packed,
                split_count=linear_attention_qkvz_ba_split,
            ),
            "packed_layers": linear_attention_qkvz_ba_packed,
            "split_layers": linear_attention_qkvz_ba_split,
        },
        "linear_attention_direct_cpp_inputs": (
            summarize_ax_mlx_direct_cpp_linear_attention_inputs(telemetry)
            or {
                "schema_version": "ax.mlx_direct_cpp_linear_attention_inputs.v1",
                "classification": "not_attempted",
                "attempts": 0,
                "hits": 0,
                "fallbacks": 0,
                "profile_blocked": 0,
                "hit_rate_micros": 0,
            }
        ),
        "linear_attention_direct_cpp_post_input": (
            summarize_ax_mlx_direct_cpp_linear_attention_post_input(telemetry)
            or {
                "schema_version": "ax.mlx_direct_cpp_linear_attention_post_input.v1",
                "classification": "not_attempted",
                "attempts": 0,
                "hits": 0,
                "fallbacks": 0,
                "profile_blocked": 0,
                "hit_rate_micros": 0,
            }
        ),
        "qwen_linear_attention_decode_post_input_metal": (
            summarize_ax_mlx_qwen_linear_attention_decode_post_input_metal(telemetry)
            or {
                "schema_version": (
                    "ax.mlx_qwen_linear_attention_decode_post_input_metal.v1"
                ),
                "classification": "not_attempted",
                "attempts": 0,
                "hits": 0,
                "fallbacks": 0,
                "profile_blocked": 0,
                "hit_rate_micros": 0,
            }
        ),
    }


def summarize_scheduler_telemetry(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("scheduler_telemetry") or {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def classify_prefix_reuse_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    hit_observed = int(evidence.get("hit_count", 0)) > 0
    miss_warmup_observed = (
        int(evidence.get("miss_count", 0)) > 0
        and int(evidence.get("warmup_token_count", 0)) > 0
    )
    blocked_reason_count = (
        int(evidence.get("blocked_policy_disabled_count", 0))
        + int(evidence.get("blocked_unsupported_layout_count", 0))
        + int(evidence.get("blocked_trim_failure_count", 0))
    )
    blocked_count = int(evidence.get("blocked_count", 0))
    blocked_observed = blocked_count > 0

    if hit_observed and miss_warmup_observed:
        coverage = "hit_and_miss_warmup"
    elif miss_warmup_observed:
        coverage = "miss_warmup_only"
    elif hit_observed:
        coverage = "hit_only"
    elif blocked_observed:
        coverage = "blocked_only"
    else:
        coverage = "none_observed"

    return {
        "physical_snapshot_hit_observed": hit_observed,
        "physical_snapshot_miss_warmup_observed": miss_warmup_observed,
        "physical_snapshot_blocked_observed": blocked_observed,
        "physical_snapshot_coverage": coverage,
        "blocked_reason_count": blocked_reason_count,
        "blocked_reason_accounting_gap_count": max(
            0, blocked_count - blocked_reason_count
        ),
    }


def summarize_prefix_reuse_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = {
        "hit_count": 0,
        "miss_count": 0,
        "blocked_count": 0,
        "blocked_policy_disabled_count": 0,
        "blocked_unsupported_layout_count": 0,
        "blocked_trim_failure_count": 0,
        "stored_prefix_count": 0,
        "eviction_count": 0,
        "reused_token_count": 0,
        "warmup_token_count": 0,
        "cache_entry_count": 0,
        "cache_bytes_kib": 0,
    }
    for row in results:
        if not str(row.get("engine", "")).startswith("ax_engine"):
            continue
        telemetry = row.get("ax_mlx_telemetry") or {}
        evidence["hit_count"] += int(telemetry.get("ax_mlx_prefix_cache_hits", 0))
        evidence["miss_count"] += int(telemetry.get("ax_mlx_prefix_cache_misses", 0))
        evidence["blocked_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_blocked", 0)
        )
        evidence["blocked_policy_disabled_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_blocked_policy_disabled", 0)
        )
        evidence["blocked_unsupported_layout_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_blocked_unsupported_layout", 0)
        )
        evidence["blocked_trim_failure_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_blocked_trim_failure", 0)
        )
        evidence["stored_prefix_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_stores", 0)
        )
        evidence["eviction_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_evictions", 0)
        )
        evidence["reused_token_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_reused_tokens", 0)
        )
        evidence["warmup_token_count"] += int(
            telemetry.get("ax_mlx_prefix_cache_warmup_tokens", 0)
        )
        evidence["cache_entry_count"] = max(
            evidence["cache_entry_count"],
            int(telemetry.get("ax_mlx_prefix_cache_entries", 0)),
        )
        evidence["cache_bytes_kib"] = max(
            evidence["cache_bytes_kib"],
            int(telemetry.get("ax_mlx_prefix_cache_bytes_kib", 0)),
        )
    evidence.update(classify_prefix_reuse_evidence(evidence))
    return evidence


def summarize_artifact_run_stability(results: list[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": "ax.benchmark_run_stability_summary.v1",
        "scope": "ax_engine_rows",
        "row_count": 0,
        "stable_enough_count": 0,
        "unstable_count": 0,
        "missing_count": 0,
        "classification_counts": {},
        "unstable_rows": [],
        "publication_candidate": True,
    }
    for row in results:
        if not isinstance(row, dict):
            continue
        engine = str(row.get("engine", ""))
        if not engine.startswith("ax_engine"):
            continue
        summary["row_count"] += 1
        stability = row.get("run_stability")
        if not isinstance(stability, dict):
            summary["missing_count"] += 1
            summary["publication_candidate"] = False
            continue
        shape_problem = None
        if stability.get("schema_version") != "ax.benchmark_run_stability.v1":
            shape_problem = "invalid_run_stability_schema"
        elif stability.get("metric") != "decode_tok_s":
            shape_problem = "invalid_run_stability_metric"
        classification = shape_problem or str(
            stability.get("classification", "unknown")
        )
        counts = summary["classification_counts"]
        counts[classification] = int(counts.get(classification, 0)) + 1
        if shape_problem is None and classification == "stable_enough":
            summary["stable_enough_count"] += 1
            continue
        summary["unstable_count"] += 1
        summary["publication_candidate"] = False
        unstable_row = {
            "engine": engine,
            "prompt_tokens": row.get("prompt_tokens"),
            "generation_tokens": row.get("generation_tokens"),
            "classification": classification,
        }
        drift = stability.get("last_vs_first_pct")
        if isinstance(drift, (int, float)):
            unstable_row["last_vs_first_pct"] = float(drift)
        summary["unstable_rows"].append(unstable_row)
    if summary["row_count"] == 0:
        summary["publication_candidate"] = False
    return summary


def _sum_safetensor_bytes(model_dir: Path) -> int:
    total = 0
    for path in model_dir.rglob("*.safetensors"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _detect_moe_block(model_dir: Path) -> dict[str, Any] | None:
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return manifest.get("moe")


def _compute_active_expert_bytes(model_dir: Path) -> int | None:
    moe_block = _detect_moe_block(model_dir)
    if moe_block is None:
        return None
    expert_count = int(moe_block.get("expert_count", 0))
    experts_per_token = int(moe_block.get("experts_per_token", 0))
    if expert_count <= 0 or experts_per_token <= 0:
        return None
    manifest_path = model_dir / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    tensors = manifest.get("tensors", [])
    if not tensors:
        return None
    routed_bytes = 0
    other_bytes = 0
    for t in tensors:
        role = t.get("role", "")
        nbytes = t.get("length_bytes", 0)
        if "shared_expert" in role:
            other_bytes += nbytes
        elif any(tag in role for tag in ("_exps", "routed_expert")):
            routed_bytes += nbytes
        else:
            other_bytes += nbytes
    if routed_bytes == 0:
        return None
    active_ratio = experts_per_token / expert_count
    return other_bytes + int(routed_bytes * active_ratio)


def build_bandwidth_accounting(
    model_dir: Path,
    results: list[dict[str, Any]],
    peak_bandwidth_gb_s: float | None = None,
    peak_bandwidth_source: str | None = None,
) -> dict[str, Any]:
    safetensor_bytes = _sum_safetensor_bytes(model_dir)
    moe_block = _detect_moe_block(model_dir)
    active_bytes = _compute_active_expert_bytes(model_dir)

    if moe_block is not None:
        if active_bytes is not None:
            estimate_kind = "moe_active_estimate"
            bytes_for_estimate: int | None = active_bytes
        else:
            estimate_kind = "not_comparable"
            bytes_for_estimate = None
    else:
        estimate_kind = "dense_safetensor_total"
        bytes_for_estimate = safetensor_bytes

    per_row: list[dict[str, Any]] = []
    for cell in results:
        engine = str(cell.get("engine", ""))
        method = str(cell.get("method", ""))
        if engine != "ax_engine_mlx" or method != "server_sse_runner_time_us":
            continue
        decode_tok_s = metric_value(cell, "decode_tok_s")
        if decode_tok_s <= 0:
            continue
        if bytes_for_estimate is None or bytes_for_estimate <= 0:
            per_row.append(
                {
                    "engine": engine,
                    "prompt_tokens": cell.get("prompt_tokens"),
                    "generation_tokens": cell.get("generation_tokens"),
                    "decode_tok_s_median": decode_tok_s,
                    "ax_bandwidth_estimate_kind": estimate_kind,
                    "ax_bandwidth_estimate_blocked_reason": "moe_active_bytes_unavailable",
                }
            )
            continue
        bytes_per_token = bytes_for_estimate
        bandwidth_gb_s = (bytes_per_token * decode_tok_s) / 1e9
        row: dict[str, Any] = {
            "engine": engine,
            "prompt_tokens": cell.get("prompt_tokens"),
            "generation_tokens": cell.get("generation_tokens"),
            "decode_tok_s_median": decode_tok_s,
            "ax_effective_weight_bytes_per_token": bytes_per_token,
            "ax_effective_bandwidth_gb_s": round(bandwidth_gb_s, 3),
            "ax_bandwidth_estimate_kind": estimate_kind,
        }
        if peak_bandwidth_gb_s is not None and peak_bandwidth_gb_s > 0:
            row["ax_effective_bandwidth_percent_of_peak"] = round(
                bandwidth_gb_s / peak_bandwidth_gb_s * 100, 2
            )
            row["ax_bandwidth_peak_source"] = peak_bandwidth_source
        per_row.append(row)

    accounting: dict[str, Any] = {
        "safetensor_bytes": safetensor_bytes,
        "moe_block": moe_block,
        "moe_active_bytes": active_bytes,
        "bytes_used_for_estimate": bytes_for_estimate,
        "estimate_kind": estimate_kind,
        "peak_bandwidth_gb_s": peak_bandwidth_gb_s,
        "peak_bandwidth_source": peak_bandwidth_source,
        "ax_bandwidth_peak_source": peak_bandwidth_source,
        "per_row": per_row,
    }
    return accounting


def summarize_ax_mlx_gemma4_moe_profile(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_gemma4_moe_profile") or {}).items():
            if key == "ax_mlx_gemma4_moe_profile_enabled":
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_gemma4_assistant_mtp(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    max_keys = {
        "ax_mlx_gemma4_assistant_mtp_configured",
        "ax_mlx_gemma4_assistant_mtp_validated",
        "ax_mlx_gemma4_assistant_mtp_enabled",
        "ax_mlx_gemma4_assistant_mtp_attach_failed",
        "ax_mlx_gemma4_assistant_mtp_disable_reason",
        "ax_mlx_gemma4_assistant_mtp_depth",
        "ax_mlx_gemma4_assistant_mtp_confidence_mode",
    }
    for run in runs:
        for key, value in (run.get("ax_mlx_gemma4_assistant_mtp") or {}).items():
            if key == "ax_mlx_gemma4_assistant_mtp_accept_rate_x1000":
                continue
            if key in max_keys:
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    drafted = totals.get("ax_mlx_gemma4_assistant_mtp_draft_tokens", 0)
    if drafted > 0:
        totals["ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"] = int(
            totals.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens", 0)
            * 1000
            / drafted
        )
    return totals


def summarize_ax_mlx_linear_attention_profile(
    runs: list[dict[str, Any]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_linear_attention_profile") or {}).items():
            if key == "ax_mlx_linear_attention_profile_enabled":
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_decode_profile(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_decode_profile") or {}).items():
            if key == "ax_mlx_decode_profile_enabled":
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_prefill_profile(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_prefill_profile") or {}).items():
            if key == "ax_mlx_prefill_profile_enabled":
                totals[key] = max(totals.get(key, 0), int(value))
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_kv_compression_telemetry(
    runs: list[dict[str, Any]],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    latest_keys = {
        "ax_mlx_kv_compression_status",
        "ax_mlx_kv_compression_preset",
        "ax_mlx_kv_compression_key_bits",
        "ax_mlx_kv_compression_value_bits",
        "ax_mlx_kv_compression_ratio_milli",
        "ax_mlx_kv_compression_route_metadata_schema",
        "ax_mlx_kv_compression_production_ready",
        "ax_mlx_kv_compression_production_blockers",
        "ax_mlx_kv_compression_decode_path",
        "ax_mlx_kv_compression_fused_decode_fallback_reason",
    }
    for run in runs:
        telemetry = run.get("kv_compression_telemetry") or {}
        for key, value in telemetry.items():
            if key in latest_keys:
                totals[key] = int(value)
            else:
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def kv_compression_decode_path_label(telemetry: dict[str, int]) -> str:
    if telemetry.get("ax_mlx_kv_compression_decode_path") == 2:
        return "fused_compressed_decode"
    if telemetry.get("ax_mlx_kv_compression_decode_path") == 3:
        return "cpu_oracle_compressed_decode"
    return "full_precision_shadow"


def kv_compression_fused_decode_fallback_reason_label(
    telemetry: dict[str, int],
) -> str:
    reason = telemetry.get("ax_mlx_kv_compression_fused_decode_fallback_reason", 0)
    return {
        0: "none",
        1: "shadow_only",
        2: "missing_runtime_storage",
        3: "unsupported_preset",
        4: "runner_not_integrated",
        5: "cpu_oracle_unavailable",
    }.get(reason, f"unknown_{reason}")


def kv_compression_fused_decode_blocked_summary(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    counters = {
        label: int(telemetry.get(key, 0))
        for label, key in KV_COMPRESSION_FUSED_DECODE_BLOCKED_COUNTERS.items()
    }
    reasons = [label for label, value in counters.items() if value > 0]
    return {
        "total": sum(counters.values()),
        "reasons": reasons,
        "counters": counters,
    }


def kv_compression_fused_decode_blocked_attention_kind_summary(
    telemetry: dict[str, int],
) -> dict[str, Any]:
    counters = {
        label: int(telemetry.get(key, 0))
        for label, key in KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND_COUNTERS.items()
    }
    reasons = [label for label, value in counters.items() if value > 0]
    return {
        "total": sum(counters.values()),
        "reasons": reasons,
        "counters": counters,
    }


def is_ax_prefill_step(step: dict[str, Any], *, seen_prefill: bool) -> bool:
    route = step.get("route") or {}
    route_labels = [
        str(route.get("execution_plan") or "").lower(),
        str(route.get("attention_route") or "").lower(),
    ]
    if any("prefill" in label for label in route_labels):
        return True
    if any("decode" in label for label in route_labels):
        return False

    scheduled_tokens = int(step.get("scheduled_tokens") or 0)
    return not seen_prefill or scheduled_tokens > 1


def iter_sse_json_events_from_lines(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    current_event = ""
    data_parts: list[str] = []

    def flush_frame() -> tuple[str, Any] | None:
        nonlocal current_event, data_parts
        event_name = current_event
        data = "\n".join(data_parts)
        current_event = ""
        data_parts = []
        if not data:
            return None
        try:
            return event_name, json.loads(data)
        except json.JSONDecodeError:
            return None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if line == "":
            frame = flush_frame()
            if frame is not None:
                yield frame
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            data_parts.append(value)

    frame = flush_frame()
    if frame is not None:
        yield frame


def axengine_one_run(
    port: int,
    tokens: list[int],
    generation_tokens: int,
    *,
    capture_output_token_ids: bool = False,
    server_pid: int | None = None,
    sampler: dict[str, Any] | None = None,
    seed: int = MLX_LM_RANDOM_SEED,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    first_output_wall_s: float | None = None
    sampling_dict: dict[str, Any] = {"ignore_eos": True, "seed": seed}
    if sampler:
        sampling_dict.update(sampler)
    sampling_dict["seed"] = seed
    payload = json.dumps(
        {
            "input_tokens": tokens,
            "max_output_tokens": generation_tokens,
            "sampling": sampling_dict,
        }
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        conn.request(
            "POST",
            "/v1/generate/stream",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"ax-engine HTTP {response.status}: {response.read(300).decode(errors='replace')}"
            )

        prefill_us = 0
        seen_prefill = False
        decode_us = 0
        output_tokens = 0
        output_token_ids: list[int] = []
        final_route: dict[str, Any] | None = None
        prefill_route: dict[str, Any] | None = None
        step_local_decisions: dict[str, int] = {}

        decoded_lines = (raw.decode("utf-8", errors="replace") for raw in response)
        for current_event, obj in iter_sse_json_events_from_lines(decoded_lines):
            if current_event == "step":
                step = obj.get("step", {})
                runner_us = int(step.get("runner_time_us") or 0)
                output_len_raw = obj.get("request", {}).get("output_len")
                if output_len_raw is not None:
                    output_tokens = int(output_len_raw)
                    if output_tokens > 0 and first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - request_started
                step_route = step.get("route") or obj.get("request", {}).get("route")
                merge_step_local_route_decisions(step_local_decisions, step_route)
                final_route = route_with_more_decisions(
                    step_route,
                    final_route,
                )
                prefill_step = is_ax_prefill_step(step, seen_prefill=seen_prefill)
                if prefill_step:
                    prefill_route = route_with_more_decisions(step_route, prefill_route)
                    prefill_us += runner_us
                    seen_prefill = True
                else:
                    decode_us += runner_us
            elif current_event == "response":
                response_tokens = obj.get("response", {}).get("output_tokens")
                if isinstance(response_tokens, list):
                    output_tokens = len(response_tokens) or output_tokens
                    if response_tokens and first_output_wall_s is None:
                        first_output_wall_s = time.perf_counter() - request_started
                    if capture_output_token_ids:
                        output_token_ids = [int(token) for token in response_tokens]
                final_route = route_with_more_decisions(
                    obj.get("response", {}).get("route"),
                    final_route,
                )
        client_wall_total_ms = (time.perf_counter() - request_started) * 1000.0
    finally:
        conn.close()
    final_route = route_with_step_local_decisions(final_route, step_local_decisions)
    prompt_tokens = len(tokens)
    prefill_s = prefill_us / 1_000_000
    decode_s = decode_us / 1_000_000
    # Match `mlx_lm.benchmark`'s `generation_tps` contract.  Upstream
    # `stream_generate` resets its generation timer after the first token is
    # available, but the final response still reports `generation_tokens=N` and
    # computes throughput as N divided by that post-first-token interval.  AX's
    # streamed runner timing similarly keeps the prompt/prefill step separate
    # from decode step timing, so use the full generated-token count here
    # instead of subtracting the prefill-produced first token.
    measured_decode_tokens = max(output_tokens, 0)

    # The recent shared-prefix-cache change (commit 0887e8f) lets warmup
    # trials populate KV cache that subsequent measurement trials hit. On a
    # cache-hit trial the runner does ~0 forward work and reports a tiny
    # `runner_time_us`; dividing prompt_tokens by that yields nonsense rates
    # like 5M tok/s. Detect the hit via runner telemetry and emit None for
    # prefill-derived metrics so the artifact doesn't carry misleading
    # numbers. Decode metrics are unaffected (decode steps do not reuse a
    # cache shortcut) and stay valid.
    mlx_telemetry_probe = extract_ax_mlx_telemetry(final_route) or {}
    prefill_cache_warm = (
        int(mlx_telemetry_probe.get("ax_mlx_prefix_cache_hits", 0)) > 0
        and int(mlx_telemetry_probe.get("ax_mlx_prefix_cache_reused_tokens", 0)) > 0
    )
    if prefill_cache_warm:
        prefill_s_value: float | None = None
        prefill_tok_s_value: float | None = None
        ttft_ms_value: float | None = None
    else:
        prefill_s_value = prefill_s
        prefill_tok_s_value = prompt_tokens / prefill_s if prefill_s > 0 else 0.0
        ttft_ms_value = prefill_s * 1000.0

    run: dict[str, Any] = {
        "prefill_s": prefill_s_value,
        "decode_s": decode_s,
        "ttft_ms": ttft_ms_value,
        "prefill_tok_s": prefill_tok_s_value,
        "decode_tok_s": measured_decode_tokens / decode_s if decode_s > 0 else 0.0,
        "output_tokens": float(output_tokens),
        "client_wall_total_ms": client_wall_total_ms,
    }
    if first_output_wall_s is not None:
        run["client_wall_ttft_ms"] = first_output_wall_s * 1000.0
    if prefill_cache_warm:
        run["prefill_cache_warm"] = True
    rss_gb = process_rss_gb(server_pid)
    if rss_gb is not None:
        run["peak_memory_gb"] = rss_gb
    if capture_output_token_ids:
        run["output_token_ids"] = output_token_ids
    telemetry = extract_ax_ngram_telemetry(final_route)
    if telemetry:
        run["ngram_acceleration_telemetry"] = telemetry
    mlx_telemetry = extract_ax_mlx_telemetry(final_route)
    if mlx_telemetry:
        run["ax_mlx_telemetry"] = mlx_telemetry
    scheduler_telemetry = extract_scheduler_telemetry(final_route)
    if scheduler_telemetry:
        run["scheduler_telemetry"] = scheduler_telemetry
    gemma4_moe_profile = extract_ax_mlx_gemma4_moe_profile(final_route)
    if gemma4_moe_profile:
        run["ax_mlx_gemma4_moe_profile"] = gemma4_moe_profile
    gemma4_assistant_mtp = extract_ax_mlx_gemma4_assistant_mtp(final_route)
    if gemma4_assistant_mtp:
        run["ax_mlx_gemma4_assistant_mtp"] = gemma4_assistant_mtp
    linear_attention_profile = extract_ax_mlx_linear_attention_profile(
        route_for_linear_attention_profile(prefill_route, final_route)
    )
    if linear_attention_profile:
        run["ax_mlx_linear_attention_profile"] = linear_attention_profile
    prefill_profile = extract_ax_mlx_prefill_profile(prefill_route)
    if prefill_profile:
        run["ax_mlx_prefill_profile"] = prefill_profile
    decode_profile = extract_ax_mlx_decode_profile(final_route)
    if decode_profile:
        run["ax_mlx_decode_profile"] = decode_profile
    compression_telemetry = extract_ax_mlx_kv_compression_telemetry(final_route)
    if compression_telemetry:
        run["kv_compression_telemetry"] = compression_telemetry
    return run


def summarize_runs(runs: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [run[key] for run in runs if run.get(key) is not None]
    if not values:
        # Aggregation requested for a metric that no trial carries a valid
        # value for (e.g. prefill rows where every trial was a prefix-cache
        # hit and prefill_s was nulled out). Preserve the schema shape but
        # return None so downstream README updaters can detect and skip
        # rather than crashing on dict access.
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def ax_prefill_work_contract(
    prompt_tokens: int, *, sampler: dict[str, Any] | None
) -> str:
    if sampler is None and prompt_tokens > 512:
        return "mlx_lm_style_cache_only_prefix_plus_final_prompt_token"
    return "historical_full_logits_prefill_or_sampler_required"


def validate_axengine_policy_telemetry(
    *,
    direct_mode: bool,
    mtp_disable_ngram_stacking: bool,
    ngram_summary: dict[str, int],
    ax_mlx_telemetry: dict[str, int],
) -> None:
    observed: dict[str, int] = {}
    if direct_mode:
        observed = {
            key: int(ngram_summary.get(key, 0))
            for key in (
                "ax_ngram_draft_attempts",
                "ax_ngram_draft_tokens",
                "ax_ngram_accepted_tokens",
            )
            if int(ngram_summary.get(key, 0)) > 0
        }
        for key in ("ax_mlx_ngram_decode_steps", "ax_mlx_ngram_decode_wall_us"):
            value = int(ax_mlx_telemetry.get(key, 0))
            if value > 0:
                observed[key] = value
        if observed:
            raise RuntimeError(
                "direct AX benchmark row observed n-gram telemetry; "
                "the server may not be honoring --disable-ngram-acceleration, "
                f"or the harness may be connected to the wrong server: {observed}"
            )
    if mtp_disable_ngram_stacking:
        ngram_hit_steps = int(ngram_summary.get("ax_mtp_ngram_hit_steps", 0) or 0)
        if ngram_hit_steps > 0:
            raise RuntimeError(
                "pure-MTP AX benchmark row observed n-gram draft hits; "
                "the server may not be honoring --mlx-mtp-disable-ngram-stacking, "
                f"or the harness may be connected to the wrong server: {ngram_hit_steps}"
            )


def bench_axengine(
    port: int,
    tokens: list[int],
    generation_tokens: int,
    repetitions: int,
    warmup_repetitions: int | float = 1,
    cooldown: float | None = None,
    *,
    model_metadata: dict[str, Any],
    direct_mode: bool = False,
    engine_key_override: str | None = None,
    kv_compression: str = "disabled",
    capture_output_token_ids: bool = False,
    server_pid: int | None = None,
    prefix_cache_enabled: bool = False,
    sampler: dict[str, Any] | None = None,
    mtp_disable_ngram_stacking: bool = False,
    gemma4_assistant_mtp: bool = False,
    seed: int = MLX_LM_RANDOM_SEED,
    prompt_source: str = "random",
) -> dict[str, Any]:
    if cooldown is None:
        cooldown = float(warmup_repetitions)
        warmup_repetitions = 1
    warmup_repetitions = int(warmup_repetitions)
    engine_key = engine_key_override or (
        AX_ENGINE_DIRECT_KEY if direct_mode else AX_ENGINE_NGRAM_ACCEL_KEY
    )
    decode_policy = ax_decode_policy(
        model_metadata,
        direct_mode=direct_mode,
        mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
        gemma4_assistant_mtp=gemma4_assistant_mtp,
    )
    print(
        f"  [ax-engine/{engine_key}] prompt={len(tokens)} "
        f"generation={generation_tokens} policy={decode_policy} "
        f"kv_compression={kv_compression}",
        file=sys.stderr,
    )
    for warmup_index in range(warmup_repetitions):
        axengine_one_run(
            port,
            tokens,
            generation_tokens,
            server_pid=server_pid,
            sampler=sampler,
            seed=seed,
        )
        print(
            f"    warmup {warmup_index + 1}/{warmup_repetitions}",
            file=sys.stderr,
        )
        if cooldown > 0:
            time.sleep(cooldown)

    runs = []
    for index in range(repetitions):
        run = axengine_one_run(
            port,
            tokens,
            generation_tokens,
            capture_output_token_ids=capture_output_token_ids,
            server_pid=server_pid,
            sampler=sampler,
            seed=seed,
        )
        runs.append(run)
        run["random_seed"] = seed
        run["seed"] = seed
        prefill_label = (
            f"{run['prefill_tok_s']:.1f} tok/s"
            if run.get("prefill_tok_s") is not None
            else "cache_warm"
        )
        print(
            f"    rep {index + 1}: prefill={prefill_label} "
            f"decode={run['decode_tok_s']:.1f} tok/s out={run['output_tokens']:.0f}",
            file=sys.stderr,
        )
        if cooldown > 0 and index < repetitions - 1:
            print(
                f"    interim after {index + 1}/{repetitions}: "
                f"{format_axengine_interim_summary(runs)}; "
                f"cooldown {cooldown:.0f}s",
                file=sys.stderr,
            )
            time.sleep(cooldown)

    ngram_summary = summarize_telemetry(runs)
    ax_mlx_telemetry = summarize_ax_mlx_telemetry(runs)
    validate_axengine_policy_telemetry(
        direct_mode=direct_mode,
        mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
        ngram_summary=ngram_summary,
        ax_mlx_telemetry=ax_mlx_telemetry,
    )
    _claim_status = ax_decode_claim_status(
        direct_mode,
        ngram_summary,
        mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
    )
    _effective_route = ax_decode_effective_route(
        direct_mode=direct_mode,
        model_metadata=model_metadata,
        telemetry=ngram_summary,
        ax_mlx_telemetry=ax_mlx_telemetry,
        mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
    )
    row = {
        "engine": engine_key,
        "method": "server_sse_runner_time_us",
        "timing_scope": "ax_engine_runner_time_us",
        "runtime_identity": dict(AX_MLX_RUNTIME_IDENTITY),
        "ax_decode_policy": decode_policy,
        "ax_prefix_cache_mode": (
            AX_PREFIX_CACHE_ENABLED_MODE
            if prefix_cache_enabled
            else AX_PREFIX_CACHE_DISABLED_MODE
        ),
        "prefill_ttft_measurement_contract": (
            "cold_prefill_no_cross_request_prefix_cache"
            if not prefix_cache_enabled
            else "prefix_cache_enabled_prefill_metrics_invalidated_on_hit"
        ),
        "prefill_work_contract": ax_prefill_work_contract(len(tokens), sampler=None),
        "ax_decode_claim_status": _claim_status,
        "ax_ngram_outcome_tier": ax_ngram_outcome_tier(
            direct_mode=direct_mode,
            status=_claim_status,
            route=_effective_route,
            mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
        ),
        "ax_decode_effective_route": _effective_route,
        "ax_decode_claim_mode": ax_decode_claim_mode(
            direct_mode,
            sampler=sampler,
            mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
        ),
        "ax_mtp_draft_source": (
            "gemma4_assistant_head_only"
            if gemma4_assistant_mtp and mtp_disable_ngram_stacking
            else (
                "gemma4_assistant_head_or_ngram_stacked"
                if gemma4_assistant_mtp
                else (
                    "mtp_head_only"
                    if mtp_disable_ngram_stacking
                    else "mtp_head_or_ngram_stacked"
                )
            )
        ),
        "prompt_contract": (
            "real_prompt_tokenized"
            if prompt_source == "real"
            else "mlx_lm_random_tokens_seed_0"
        ),
        "random_seed": seed,
        "seed": seed,
        "batch_size": 1,
        "prompt_tokens": len(tokens),
        "generation_tokens": generation_tokens,
        "prefill_tok_s": summarize_runs(runs, "prefill_tok_s"),
        "decode_tok_s": summarize_runs(runs, "decode_tok_s"),
        "ttft_ms": summarize_runs(runs, "ttft_ms"),
        "ttft_source": "ax_engine_runner_prefill_time",
        "client_wall_ttft_ms": summarize_runs(runs, "client_wall_ttft_ms"),
        "client_wall_ttft_source": "http_sse_first_output_token_observed_by_client",
        "client_wall_total_ms": summarize_runs(runs, "client_wall_total_ms"),
        "prefill_s": summarize_runs(runs, "prefill_s"),
        "decode_s": summarize_runs(runs, "decode_s"),
        "run_stability": summarize_run_stability(runs, "decode_tok_s"),
        "ngram_acceleration_telemetry": ngram_summary,
        "ngram_accept_at_depth": summarize_ngram_accept_at_depth(ngram_summary),
        # Sampler config (PRD §7.1 release-claim artifact requirement).
        # Greedy by default; non-None when --ax-sampling is active.
        # canonical signature equals "greedy" when no knob is set.
        "sampler_settings": canonical_sampler_signature(sampler)
        if sampler
        else "greedy",
        # Row identity block + same-policy baseline pointer (PRD §8 Phase 6).
        "ax_decode_row_identity": build_row_identity(
            model_id=(
                model_metadata.get("model_family")
                or model_metadata.get("model_type")
                or "unknown"
            ),
            tokens=tokens,
            seed=seed,
            max_output_tokens=generation_tokens,
            sampler=sampler,
        ),
        "ax_decode_same_policy_baseline_identity": build_row_identity(
            model_id=(
                model_metadata.get("model_family")
                or model_metadata.get("model_type")
                or "unknown"
            ),
            tokens=tokens,
            seed=seed,
            max_output_tokens=generation_tokens,
            sampler=sampler,
        ),
        "ax_mlx_telemetry": ax_mlx_telemetry,
        "ax_mlx_decode_route": summarize_ax_mlx_decode_route(ax_mlx_telemetry),
        "scheduler_telemetry": summarize_scheduler_telemetry(runs),
        "ax_mlx_gemma4_moe_profile": summarize_ax_mlx_gemma4_moe_profile(runs),
        "ax_mlx_gemma4_assistant_mtp": summarize_ax_mlx_gemma4_assistant_mtp(runs),
        "ax_mlx_linear_attention_profile": summarize_ax_mlx_linear_attention_profile(
            runs
        ),
        "ax_mlx_prefill_profile": summarize_ax_mlx_prefill_profile(runs),
        "ax_mlx_decode_profile": summarize_ax_mlx_decode_profile(runs),
        "trials": runs,
    }
    direct_cpp_linear_inputs = summarize_ax_mlx_direct_cpp_linear_attention_inputs(
        ax_mlx_telemetry
    )
    if direct_cpp_linear_inputs:
        row["ax_mlx_direct_cpp_linear_attention_inputs"] = direct_cpp_linear_inputs
    direct_cpp_linear_post_input = (
        summarize_ax_mlx_direct_cpp_linear_attention_post_input(ax_mlx_telemetry)
    )
    if direct_cpp_linear_post_input:
        row["ax_mlx_direct_cpp_linear_attention_post_input"] = (
            direct_cpp_linear_post_input
        )
    effective_routes = summarize_ax_mlx_effective_routes(ax_mlx_telemetry)
    if effective_routes:
        row["ax_mlx_effective_routes"] = effective_routes
    cache_warm_trials = sum(1 for run in runs if run.get("prefill_cache_warm"))
    if cache_warm_trials > 0:
        # Document at row level so README updaters and aggregators can detect
        # without re-walking individual trials. Counted, not boolean, so a
        # mixed run (e.g. one cold + two warm trials) is still flagged.
        row["prefill_measurement_invalid_reason"] = (
            "prefix_cache_warm_on_all_trials"
            if cache_warm_trials == len(runs)
            else "prefix_cache_warm_on_some_trials"
        )
        row["prefill_cache_warm_trial_count"] = cache_warm_trials
    if all("peak_memory_gb" in run for run in runs):
        row["peak_memory_gb"] = summarize_runs(runs, "peak_memory_gb")
        row["memory_source"] = "server_process_rss_after_stream"
    compression_summary = summarize_ax_mlx_kv_compression_telemetry(runs)
    if kv_compression != "disabled":
        row["experimental_mlx_kv_compression"] = kv_compression
        decode_path = kv_compression_decode_path_label(compression_summary)
        row["kv_compression_decode_path"] = decode_path
        row["kv_compression_claim_status"] = (
            "integrated_fused_compressed_decode"
            if decode_path == "fused_compressed_decode"
            else "telemetry_only_full_precision_generation"
        )
        row["kv_compression_fused_decode_fallback_reason_label"] = (
            kv_compression_fused_decode_fallback_reason_label(compression_summary)
        )
        blocked_summary = kv_compression_fused_decode_blocked_summary(
            compression_summary
        )
        row["kv_compression_fused_decode_blocked_total"] = blocked_summary["total"]
        row["kv_compression_fused_decode_blocked_reasons"] = blocked_summary["reasons"]
        attention_kind_blocked_summary = (
            kv_compression_fused_decode_blocked_attention_kind_summary(
                compression_summary
            )
        )
        row["kv_compression_fused_decode_blocked_attention_kind_total"] = (
            attention_kind_blocked_summary["total"]
        )
        row["kv_compression_fused_decode_blocked_attention_kind_reasons"] = (
            attention_kind_blocked_summary["reasons"]
        )
    if compression_summary:
        row["kv_compression_telemetry"] = compression_summary
    return row


def _llama_cpp_metric_from_row(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples_ts")
    if isinstance(samples, list) and samples:
        values = [float(value) for value in samples]
        metric = summarize_values(values)
        stddev = row.get("stddev_ts")
        if stddev is not None:
            metric["stddev"] = float(stddev)
        avg = row.get("avg_ts")
        if avg is not None:
            metric["reported_mean"] = float(avg)
        return metric
    avg = row.get("avg_ts")
    if avg is None:
        raise RuntimeError("llama-bench JSON row missing avg_ts/samples_ts")
    return {"mean": float(avg), "median": float(avg)}


def _llama_cpp_trial_rows(
    row: dict[str, Any], metric_name: str
) -> list[dict[str, Any]]:
    samples = row.get("samples_ts")
    if not isinstance(samples, list):
        return []
    raw_ns = row.get("samples_ns")
    sample_ns = raw_ns if isinstance(raw_ns, list) else []
    trials = []
    for index, value in enumerate(samples):
        ns = (
            sample_ns[index]
            if index < len(sample_ns) and sample_ns[index] is not None
            else None
        )
        trials.append(
            {
                "trial": index + 1,
                metric_name: float(value),
                "sample_ns": int(ns) if ns is not None else None,
            }
        )
    return trials


def parse_llama_cpp_bench_json(
    stdout: str,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    require_metal: bool = True,
) -> dict[str, Any]:
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("llama-bench output was not valid JSON") from error
    if not isinstance(rows, list):
        raise RuntimeError("llama-bench JSON output must be a list")

    prefill_row = None
    decode_row = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        if n_prompt == prompt_tokens and n_gen == 0:
            prefill_row = row
        elif n_prompt == 0 and n_gen == generation_tokens:
            decode_row = row

    if prefill_row is None:
        raise RuntimeError(
            f"llama-bench JSON missing pp row for n_prompt={prompt_tokens}"
        )
    if decode_row is None:
        raise RuntimeError(
            f"llama-bench JSON missing tg row for n_gen={generation_tokens}"
        )

    backends = str(prefill_row.get("backends") or decode_row.get("backends") or "")
    if require_metal:
        tokens = {
            token.strip().lower() for token in backends.split(",") if token.strip()
        }
        if "metal" not in tokens and "mtl" not in tokens:
            raise RuntimeError(
                f"llama-bench row did not report Metal/MTL backend: {backends!r}"
            )

    prefill_trials = _llama_cpp_trial_rows(prefill_row, "prefill_tok_s")
    decode_trials = _llama_cpp_trial_rows(decode_row, "decode_tok_s")

    return {
        "prefill_tok_s": _llama_cpp_metric_from_row(prefill_row),
        "decode_tok_s": _llama_cpp_metric_from_row(decode_row),
        "prefill_trials": prefill_trials,
        "decode_trials": decode_trials,
        "trials_pairing_note": (
            "llama-bench runs pp (prefill) and tg (decode) as independent test "
            "invocations. prefill_trials[i] and decode_trials[i] are NOT from the "
            "same end-to-end run; do not compute per-trial joint statistics."
        ),
        "llama_cpp": {
            "build_commit": prefill_row.get("build_commit"),
            "build_number": prefill_row.get("build_number"),
            "backends": backends,
            "gpu_info": prefill_row.get("gpu_info"),
            "cpu_info": prefill_row.get("cpu_info"),
            "model_filename": prefill_row.get("model_filename"),
            "model_type": prefill_row.get("model_type"),
            "model_size": prefill_row.get("model_size"),
            "model_n_params": prefill_row.get("model_n_params"),
            "n_gpu_layers": prefill_row.get("n_gpu_layers"),
            "n_batch": prefill_row.get("n_batch"),
            "n_ubatch": prefill_row.get("n_ubatch"),
            "type_k": prefill_row.get("type_k"),
            "type_v": prefill_row.get("type_v"),
            "flash_attn": prefill_row.get("flash_attn"),
            "devices": prefill_row.get("devices"),
        },
        "raw_rows": [prefill_row, decode_row],
    }


def parse_llama_cpp_decode_depth_json(
    stdout: str,
    *,
    context_depth_tokens: int,
    generation_tokens: int,
    require_metal: bool = True,
) -> dict[str, Any]:
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("llama-bench depth output was not valid JSON") from error
    if not isinstance(rows, list):
        raise RuntimeError("llama-bench depth JSON output must be a list")

    decode_row = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        n_depth = int(row.get("n_depth", 0))
        if (
            n_prompt == 0
            and n_gen == generation_tokens
            and n_depth == context_depth_tokens
        ):
            decode_row = row
            break

    if decode_row is None:
        raise RuntimeError(
            "llama-bench JSON missing depth tg row for "
            f"n_depth={context_depth_tokens} n_gen={generation_tokens}"
        )

    backends = str(decode_row.get("backends") or "")
    if require_metal:
        tokens = {
            token.strip().lower() for token in backends.split(",") if token.strip()
        }
        if "metal" not in tokens and "mtl" not in tokens:
            raise RuntimeError(
                f"llama-bench depth row did not report Metal/MTL backend: {backends!r}"
            )

    return {
        "decode_at_depth_tok_s": _llama_cpp_metric_from_row(decode_row),
        "decode_at_depth_trials": _llama_cpp_trial_rows(
            decode_row,
            "decode_at_depth_tok_s",
        ),
        "llama_cpp_depth": {
            "build_commit": decode_row.get("build_commit"),
            "build_number": decode_row.get("build_number"),
            "backends": backends,
            "gpu_info": decode_row.get("gpu_info"),
            "cpu_info": decode_row.get("cpu_info"),
            "model_filename": decode_row.get("model_filename"),
            "model_type": decode_row.get("model_type"),
            "model_size": decode_row.get("model_size"),
            "model_n_params": decode_row.get("model_n_params"),
            "n_gpu_layers": decode_row.get("n_gpu_layers"),
            "n_batch": decode_row.get("n_batch"),
            "n_ubatch": decode_row.get("n_ubatch"),
            "n_depth": decode_row.get("n_depth"),
            "type_k": decode_row.get("type_k"),
            "type_v": decode_row.get("type_v"),
            "flash_attn": decode_row.get("flash_attn"),
            "devices": decode_row.get("devices"),
        },
        "raw_depth_rows": [decode_row],
    }


def _attach_llama_cpp_ttft(
    cell: dict[str, Any],
    *,
    prompt_tokens: int,
    source: str,
) -> None:
    values = []
    for trial in cell.get("prefill_trials", []):
        if not isinstance(trial, dict):
            continue
        prefill_tok_s = float(trial.get("prefill_tok_s", 0.0))
        if prefill_tok_s <= 0.0:
            continue
        ttft_ms = prompt_tokens / prefill_tok_s * 1000.0
        trial["ttft_ms"] = ttft_ms
        values.append(ttft_ms)
    if values:
        cell["ttft_ms"] = summarize_values(values)
        cell["ttft_source"] = source


def collect_llama_cpp_device_evidence(binary: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def run_llama_cpp_metal_benchmark(
    binary: Path,
    gguf: Path,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    n_gpu_layers: int,
    prompt_doc: dict[str, Any],
    extra_args: str | None = None,
) -> dict[str, Any]:
    validate_prompt_doc(
        prompt_doc,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )
    if not binary.exists():
        raise RuntimeError(f"llama.cpp benchmark binary not found: {binary}")
    if not gguf.exists():
        raise RuntimeError(f"llama.cpp GGUF model not found: {gguf}")

    delay_seconds = max(0, round(cooldown))
    if cooldown > 0 and float(delay_seconds) != float(cooldown):
        print(
            f"  [llama.cpp/metal] note: --delay rounds float cooldown={cooldown} "
            f"to integer seconds ({delay_seconds})",
            file=sys.stderr,
        )

    # llama-bench defaults to `-ub 512` (physical batch size) regardless of
    # `-p`, which artificially caps prefill throughput on prompts longer than
    # 512 tokens because the work is internally chunked four times for a
    # p=2048 prompt. Match the physical batch to the prompt length (capped at
    # 2048 to match mlx-lm's prefill_step_size default) so prefill is a single
    # full forward pass — the apples-to-apples comparison the bench claims.
    ubatch = min(max(prompt_tokens, 512), 2048)
    lbatch = max(ubatch, 2048)
    cmd = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "--delay",
        str(delay_seconds),
        "-ngl",
        str(n_gpu_layers),
        "-b",
        str(lbatch),
        "-ub",
        str(ubatch),
        "-o",
        "json",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    print(f"  [llama.cpp/metal] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed with exit={result.returncode}:\n"
            f"{result.stdout}{result.stderr}"
        )

    metrics = parse_llama_cpp_bench_json(
        result.stdout,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        require_metal=True,
    )
    cell: dict[str, Any] = {
        "engine": "llama_cpp_metal",
        "method": "llama-bench",
        "timing_scope": "external_llama_cpp_kernel_benchmark",
        "runtime_identity": dict(LLAMA_CPP_METAL_RUNTIME_IDENTITY),
        "prompt_contract": "shape_compatible_llama_bench_internal_tokens",
        "prompt_token_ids_origin": "not_applicable_llama_bench_internal_synthetic_tokens",
        "prompt_token_ids_path": prompt_doc["token_ids_path"],
        "prompt_token_ids_sha256": prompt_doc["token_ids_sha256"],
        "prompt_source": "llama_bench_internal_random",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "batch_size": 1,
        "gguf_model": str(gguf),
        "n_gpu_layers": n_gpu_layers,
        "prefill_tok_s": metrics["prefill_tok_s"],
        "decode_tok_s": metrics["decode_tok_s"],
        "prefill_trials": metrics["prefill_trials"],
        "decode_trials": metrics["decode_trials"],
        "trials_pairing_note": metrics["trials_pairing_note"],
        "llama_cpp": metrics["llama_cpp"],
        "external_baseline_role": "gguf_non_mlx_metal_reference",
        "claim_boundary": (
            "Shape-compatible external GGUF baseline. llama-bench does not consume "
            "the harness prompt-token JSON, so this row is not prompt-hash parity "
            "evidence for repo-owned MLX throughput."
        ),
    }
    device_evidence = collect_llama_cpp_device_evidence(binary)
    if device_evidence:
        cell["llama_cpp_device_evidence"] = device_evidence
    _attach_llama_cpp_ttft(
        cell,
        prompt_tokens=prompt_tokens,
        source="derived_from_llama_cpp_pp_tok_s",
    )
    return cell


def attach_llama_cpp_decode_at_depth_benchmark(
    cell: dict[str, Any],
    binary: Path,
    gguf: Path,
    *,
    context_depth_tokens: int,
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    n_gpu_layers: int,
    extra_args: str | None = None,
) -> None:
    if cell.get("engine") != "llama_cpp_metal":
        raise RuntimeError(
            "decode-at-depth evidence can only be attached to llama_cpp_metal rows"
        )
    if not binary.exists():
        raise RuntimeError(f"llama.cpp benchmark binary not found: {binary}")
    if not gguf.exists():
        raise RuntimeError(f"llama.cpp GGUF model not found: {gguf}")

    delay_seconds = max(0, round(cooldown))
    # See run_llama_cpp_metal_benchmark: ubatch defaults to 512 unless we set
    # it to match the context-depth that's being pre-filled. Without this,
    # decode-at-depth understates llama.cpp's throughput because the
    # depth-pre-fill itself runs in 512-token chunks.
    ubatch = min(max(context_depth_tokens, 512), 2048)
    lbatch = max(ubatch, 2048)
    cmd = [
        str(binary),
        "-m",
        str(gguf),
        "-p",
        "0",
        "-n",
        str(generation_tokens),
        "-d",
        str(context_depth_tokens),
        "-r",
        str(repetitions),
        "--delay",
        str(delay_seconds),
        "-ngl",
        str(n_gpu_layers),
        "-b",
        str(lbatch),
        "-ub",
        str(ubatch),
        "-o",
        "json",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    print(f"  [llama.cpp/metal/depth] {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-bench depth failed with exit={result.returncode}:\n"
            f"{result.stdout}{result.stderr}"
        )

    metrics = parse_llama_cpp_decode_depth_json(
        result.stdout,
        context_depth_tokens=context_depth_tokens,
        generation_tokens=generation_tokens,
        require_metal=True,
    )
    cell["context_depth_tokens"] = context_depth_tokens
    cell["decode_at_depth_contract"] = "llama_bench_n_depth"
    cell["decode_at_depth_tok_s"] = metrics["decode_at_depth_tok_s"]
    cell["decode_at_depth_trials"] = metrics["decode_at_depth_trials"]
    cell["llama_cpp_depth"] = metrics["llama_cpp_depth"]
    cell["raw_depth_rows"] = metrics["raw_depth_rows"]
    cell["decode_at_depth_claim_boundary"] = (
        "Shape-compatible external GGUF decode-depth baseline. llama-bench "
        "uses n_depth to prefill synthetic KV state before timed generation; "
        "this row is not prompt-hash parity evidence."
    )


def metric_value(cell: dict[str, Any], metric: str) -> float:
    data = cell.get(metric, {})
    if not isinstance(data, dict):
        return 0.0
    # `None` medians are emitted by `summarize_runs` when every trial for a
    # metric was invalid (e.g. prefill on all-cache-warm rows). Treat as
    # "no measurement" so downstream baseline attachment / aggregation does
    # not crash on `float(None)`.
    if data.get("median") is not None:
        return float(data["median"])
    if data.get("mean") is not None:
        return float(data["mean"])
    return 0.0


def attach_mlx_lm_baselines(results: list[dict[str, Any]]) -> None:
    baselines = {
        (int(cell["prompt_tokens"]), int(cell["generation_tokens"])): cell
        for cell in results
        if cell.get("engine") == "mlx_lm"
    }

    mlx_lm_present = bool(baselines)

    for cell in results:
        if cell.get("engine") == "mlx_lm":
            cell["baseline"] = {
                "engine": "mlx_lm",
                "method": "mlx_lm.benchmark",
                "role": "primary_reference",
                "prompt_contract": cell.get("prompt_contract"),
                "timing_scope": cell.get("timing_scope"),
            }
            continue

        if not mlx_lm_present:
            cell["baseline"] = {
                "engine": "mlx_lm",
                "method": "mlx_lm.benchmark",
                "role": "absent_skipped_via_cli",
                "note": "Row produced under --skip-mlx-lm; no in-run baseline attached.",
            }
            continue

        key = (int(cell["prompt_tokens"]), int(cell["generation_tokens"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise RuntimeError(
                "missing mlx_lm.benchmark baseline for "
                f"prompt_tokens={key[0]} generation_tokens={key[1]}"
            )

        baseline_prefill = metric_value(baseline, "prefill_tok_s")
        baseline_decode = metric_value(baseline, "decode_tok_s")
        cell_prefill = metric_value(cell, "prefill_tok_s")
        cell_decode = metric_value(cell, "decode_tok_s")
        cell["baseline"] = {
            "engine": "mlx_lm",
            "method": "mlx_lm.benchmark",
            "prompt_contract": baseline.get("prompt_contract"),
            "timing_scope": baseline.get("timing_scope"),
            "prompt_tokens": key[0],
            "generation_tokens": key[1],
            "prefill_tok_s": baseline_prefill,
            "decode_tok_s": baseline_decode,
            "prefill_ratio_to_mlx_lm": (
                cell_prefill / baseline_prefill if baseline_prefill > 0 else None
            ),
            "decode_ratio_to_mlx_lm": (
                cell_decode / baseline_decode if baseline_decode > 0 else None
            ),
        }


def load_reused_reference_rows(
    path: Path,
    *,
    prompt_lengths: list[int],
    generation_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = json.loads(path.read_text())
    wanted = {(prompt_tokens, generation_tokens) for prompt_tokens in prompt_lengths}
    rows: list[dict[str, Any]] = []
    seen_mlx_lm: set[tuple[int, int]] = set()

    for cell in doc.get("results", []):
        engine = cell.get("engine")
        if engine != "mlx_lm":
            continue
        key = (int(cell["prompt_tokens"]), int(cell["generation_tokens"]))
        if key not in wanted:
            continue
        if key in seen_mlx_lm:
            raise RuntimeError(
                f"{path} has duplicate mlx_lm reference rows for "
                f"prompt_tokens={key[0]} generation_tokens={key[1]}"
            )
        rows.append(copy.deepcopy(cell))
        seen_mlx_lm.add(key)

    missing = sorted(wanted - seen_mlx_lm)
    if missing:
        raise RuntimeError(
            f"{path} is missing mlx_lm reference rows for prompt/generation pairs: {missing}"
        )
    return rows, doc


def ax_row_key(cell: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(cell.get("engine", "")),
        int(cell.get("prompt_tokens", -1)),
        int(cell.get("generation_tokens", -1)),
    )


def summarize_ax_only_refresh_regression(
    *,
    results: list[dict[str, Any]],
    reference_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": "ax.ax_only_refresh_regression.v1",
        "scope": "matching_ax_rows_from_reused_reference_artifact",
        "metric": "decode_tok_s",
        "decode_min_ratio_to_reference": AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE,
        "row_count": 0,
        "matched_count": 0,
        "missing_reference_count": 0,
        "decode_regression_count": 0,
        "classification_counts": {},
        "missing_reference_rows": [],
        "rows": [],
        "publication_candidate": True,
    }
    reference_rows = {}
    if isinstance(reference_doc, dict):
        for row in reference_doc.get("results", []):
            if not isinstance(row, dict):
                continue
            if not str(row.get("engine", "")).startswith("ax_engine"):
                continue
            reference_rows[ax_row_key(row)] = row

    for row in results:
        if not str(row.get("engine", "")).startswith("ax_engine"):
            continue
        summary["row_count"] += 1
        key = ax_row_key(row)
        reference_row = reference_rows.get(key)
        if reference_row is None:
            summary["missing_reference_count"] += 1
            counts = summary["classification_counts"]
            counts["missing_reference"] = int(counts.get("missing_reference", 0)) + 1
            summary["missing_reference_rows"].append(
                {
                    "engine": key[0],
                    "prompt_tokens": key[1],
                    "generation_tokens": key[2],
                    "classification": "missing_reference",
                }
            )
            summary["publication_candidate"] = False
            continue
        reference_decode = metric_value(reference_row, "decode_tok_s")
        current_decode = metric_value(row, "decode_tok_s")
        if reference_decode <= 0.0 or current_decode <= 0.0:
            classification = "missing_decode_metric"
            decode_ratio = None
        else:
            decode_ratio = current_decode / reference_decode
            classification = (
                "decode_regression"
                if decode_ratio < AX_ONLY_REFRESH_DECODE_MIN_RATIO_TO_REFERENCE
                else "within_tolerance"
            )
        counts = summary["classification_counts"]
        counts[classification] = int(counts.get(classification, 0)) + 1
        if classification == "decode_regression":
            summary["decode_regression_count"] += 1
            summary["publication_candidate"] = False
        elif classification == "missing_decode_metric":
            summary["publication_candidate"] = False
        summary["matched_count"] += 1
        current_prefill = metric_value(row, "prefill_tok_s")
        reference_prefill = metric_value(reference_row, "prefill_tok_s")
        current_ttft = metric_value(row, "ttft_ms")
        reference_ttft = metric_value(reference_row, "ttft_ms")
        summary["rows"].append(
            {
                "engine": key[0],
                "prompt_tokens": key[1],
                "generation_tokens": key[2],
                "classification": classification,
                "current_decode_tok_s": current_decode,
                "reference_decode_tok_s": reference_decode,
                "decode_ratio_to_reference": decode_ratio,
                "current_prefill_tok_s": current_prefill,
                "reference_prefill_tok_s": reference_prefill,
                "prefill_ratio_to_reference": (
                    current_prefill / reference_prefill
                    if reference_prefill > 0.0 and current_prefill > 0.0
                    else None
                ),
                "current_ttft_ms": current_ttft,
                "reference_ttft_ms": reference_ttft,
                "ttft_ratio_to_reference": (
                    current_ttft / reference_ttft
                    if reference_ttft > 0.0 and current_ttft > 0.0
                    else None
                ),
            }
        )
    if summary["row_count"] == 0:
        summary["publication_candidate"] = False
    return summary


def validate_reused_reference_prompt_hashes(
    rows: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
) -> None:
    expected = {
        (int(prompt["prompt_tokens"]), int(prompt["generation_tokens"])): prompt[
            "token_ids_sha256"
        ]
        for prompt in prompts
    }
    for cell in rows:
        key = (int(cell["prompt_tokens"]), int(cell["generation_tokens"]))
        expected_hash = expected.get(key)
        actual_hash = cell.get("prompt_token_ids_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                "reused reference row prompt hash mismatch for "
                f"engine={cell.get('engine')} prompt_tokens={key[0]} "
                f"generation_tokens={key[1]} expected={expected_hash} actual={actual_hash}"
            )


def print_summary(doc: dict[str, Any]) -> None:
    print("\n" + "=" * 102)
    print("AX Engine MLX inference stack benchmark")
    print("=" * 102)
    print(
        f"{'Engine':<18} {'Prompt tok':>10} {'Prefill tok/s':>14} "
        f"{'Decode tok/s':>13} {'Decode vs mlx_lm':>16} {'Stability':>12}  Method"
    )
    print("-" * 102)
    for cell in doc["results"]:
        baseline = cell.get("baseline", {})
        decode_ratio = baseline.get("decode_ratio_to_mlx_lm")
        ratio_text = (
            "baseline"
            if cell["engine"] == "mlx_lm"
            else f"{decode_ratio:.3f}x"
            if isinstance(decode_ratio, (int, float))
            else "n/a"
        )
        print(
            f"{cell['engine']:<18} {cell['prompt_tokens']:>10} "
            f"{metric_value(cell, 'prefill_tok_s'):>14.1f} "
            f"{metric_value(cell, 'decode_tok_s'):>13.1f} "
            f"{ratio_text:>16} {format_run_stability_label(cell):>12}  "
            f"{cell['method']}"
        )
    print()


def validate_gateddelta_prefill_profile_output(path: Path) -> list[str]:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from check_gateddelta_prefill_profile_artifact import (
        validate_gateddelta_prefill_profile_artifact,
    )

    return validate_gateddelta_prefill_profile_artifact(path)


def render_gateddelta_prefill_profile_output(path: Path, output: Path) -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from render_gateddelta_prefill_profile_report import render_report

    report = render_report(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AX MLX against MLX reference runtimes"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-repo-id",
        default=DEFAULT_MODEL_REPO_ID,
        help=(
            "Hugging Face repo id used to resolve --model-dir from the local cache "
            "when --model-dir is omitted."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=(
            "AX-ready local MLX model artifact directory. When omitted, the "
            "benchmark resolves --model-repo-id from the Hugging Face cache."
        ),
    )
    parser.add_argument(
        "--hf-cache-root",
        type=Path,
        help="Optional Hugging Face Hub cache root used when resolving --model-repo-id.",
    )
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument(
        "--prompt-source",
        choices=["random", "real"],
        default="random",
        help=(
            "Source of prompts. 'random' (default) uses the mlx_lm.benchmark "
            "random-token contract (seed=0, uniform over vocab) and preserves "
            "prompt-hash parity with mlx_lm rows. 'real' loads a JSONL suite "
            "via --real-prompt-suite, tokenizes it with the model's tokenizer, "
            "and runs AX rows only (mlx_lm.benchmark and llama-bench cannot "
            "consume external prompts). Use 'real' to measure n-gram and "
            "decode behavior on workload-shaped inputs."
        ),
    )
    parser.add_argument(
        "--real-prompt-suite",
        type=Path,
        help=(
            "JSONL file with real prompt cases (one JSON object per line "
            "with id, category, prompt, and optional max_tokens). Required "
            "when --prompt-source=real."
        ),
    )
    parser.add_argument(
        "--no-real-prompt-chat-template",
        dest="real_prompt_chat_template",
        action="store_false",
        default=True,
        help=(
            "Encode --real-prompt-suite prompts as raw text instead of "
            "applying the tokenizer's chat template. The default applies "
            "the template, which is required for instruction-tuned models "
            "(e.g. Gemma 4 IT) to avoid immediate EOS at decode step 0. "
            "Use this flag only for base / non-IT models."
        ),
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        default=True,
        help=(
            "Pass enable_thinking=False to apply_chat_template. For Qwen3-series "
            "models this pre-fills the <think></think> block so the model skips "
            "reasoning and generates answers directly. Matches MTPLX's default "
            "enable_thinking=false runtime setting."
        ),
    )
    parser.add_argument(
        "--generation-tokens", type=int, default=DEFAULT_GENERATION_TOKENS
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MLX_LM_RANDOM_SEED,
        help=(
            "Sampling seed sent to AX generation requests. The default preserves "
            "the historical mlx_lm.benchmark-compatible seed 0 contract."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument(
        "--inter-case-cooldown",
        type=float,
        default=0.0,
        help=(
            "Extra sleep (seconds) between prompt cases within a suite, "
            "applied after each case except the last. Helps prevent GPU "
            "thermal throttling when running multiple cases back-to-back."
        ),
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prefill-scaling-output",
        type=Path,
        help=(
            "After saving --output, also build and validate an "
            "ax.mlx_prefill_scaling.v1 artifact at this path."
        ),
    )
    parser.add_argument(
        "--prefill-scaling-min-context-tokens",
        type=int,
        default=1024,
        help="Minimum prompt/context length admitted into the prefill-scaling artifact.",
    )
    parser.add_argument(
        "--skip-prefill-scaling-validate",
        action="store_true",
        help="Build --prefill-scaling-output without running the scaling artifact validator.",
    )
    parser.add_argument("--skip-ax-engine", action="store_true")
    parser.add_argument(
        "--skip-mlx-lm",
        action="store_true",
        help=(
            "Skip the mlx_lm.benchmark baseline. Useful when the run only "
            "needs an external baseline (e.g. llama.cpp Metal) and the "
            "mlx_lm rows have already been captured elsewhere. Conflicts "
            "with --reuse-reference-results-from."
        ),
    )
    parser.add_argument(
        "--no-build-ax-engine",
        action="store_true",
        help=(
            "Do not run `cargo build -p ax-engine-server --release` before AX rows. "
            "Use only when the release binary freshness is managed externally."
        ),
    )
    parser.add_argument(
        "--reuse-reference-results-from",
        type=Path,
        help=(
            "Reuse mlx_lm rows from an existing artifact and rerun "
            "only AX rows. This keeps AX refreshes apple-to-apple with the "
            "published reference rows and prompt contract."
        ),
    )
    parser.add_argument("--axengine-port", type=int, default=AXENGINE_PORT)
    parser.add_argument(
        "--ax-direct",
        dest="ax_direct",
        action="store_true",
        help=(
            "Run only the direct same-policy AX row. This is the harness "
            "baseline for apple-to-apple AX comparisons."
        ),
    )
    parser.add_argument(
        "--ax-ngram-accel",
        dest="ax_ngram_accel",
        action="store_true",
        help=(
            "Run only AX default n-gram policy rows. Row claim status reports "
            "whether acceleration was effective or fell back without drafts."
        ),
    )
    parser.add_argument(
        "--ax-enable-prefix-cache",
        action="store_true",
        help=(
            "Keep AX MLX prefix cache enabled for explicit prefix-cache experiments. "
            "By default the inference-stack benchmark disables it so prefill "
            "throughput and TTFT remain cold-prefill measurements rather than "
            "warm-cache hits seeded by AX warmup/repetition runs."
        ),
    )
    parser.add_argument(
        "--ax-compare-policies",
        dest="ax_compare_policies",
        action="store_true",
        help=(
            "Run ax-engine twice: direct same-policy first, then n-gram "
            f"acceleration. Emits {AX_ENGINE_DIRECT_KEY} and "
            f"{AX_ENGINE_NGRAM_ACCEL_KEY}."
        ),
    )
    parser.add_argument(
        "--ax-sampling",
        type=json.loads,
        default=None,
        metavar="JSON",
        help=(
            "JSON object of extra sampling knobs sent in every AX HTTP request "
            '(e.g. \'{"temperature": 0.6, "top_p": 0.95, "top_k": 20}\'). '
            "Omit for greedy (default). Non-greedy rows are labeled "
            "sampling_not_distribution_exact and cannot be promoted as "
            "distribution-exact baselines."
        ),
    )
    parser.add_argument(
        "--ax-mtp-max-depth",
        type=int,
        default=None,
        help=(
            "Set AX_MLX_MTP_MAX_DEPTH for AX server rows. Use this to sweep "
            "MTP depth caps against MTPLX reference models."
        ),
    )
    parser.add_argument(
        "--ax-mtp-disable-ngram-stacking",
        action="store_true",
        help=(
            "Pass --mlx-mtp-disable-ngram-stacking to AX server rows. This keeps "
            "MTP enabled but forces the MTP verify loop to draft from the MTP "
            "head only, producing ax_engine_mlx_pure_mtp rows."
        ),
    )
    parser.add_argument(
        "--ax-mtp-fast-tail-topk-sampling",
        action="store_true",
        help=(
            "Set AX_MLX_MTP_FAST_TAIL_TOPK_SAMPLING=1 for AX server rows. "
            "Diagnostic only: samples the correction token from top-k logits on GPU "
            "and does not apply top-p filtering."
        ),
    )
    parser.add_argument(
        "--ax-gemma4-assistant-mtp",
        action="store_true",
        help=(
            "Run the AX row with Gemma4 Assistant MTP enabled. Emits "
            f"{AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY} and records "
            "ax_mlx_gemma4_assistant_mtp route metadata."
        ),
    )
    parser.add_argument(
        "--ax-gemma4-moe-profile",
        action="store_true",
        help=(
            "Enable opt-in Gemma4 MoE decode-layer profiling for AX rows. "
            "This inserts timing barriers and is for diagnosis, not headline throughput."
        ),
    )
    parser.add_argument(
        "--ax-linear-attention-profile",
        action="store_true",
        help=(
            "Enable opt-in linear-attention stage profiling for AX rows without "
            "requiring the long-prompt gated-delta profile matrix. This inserts "
            "timing barriers and is for diagnosis, not headline throughput."
        ),
    )
    parser.add_argument(
        "--ax-pack-linear-attention-projections",
        action="store_true",
        help=(
            "Enable experimental loader-time packing of split Qwen linear-attention "
            "qkv/z/a/b projections into qkvz/ba for AX rows. This is an opt-in "
            "diagnostic TTFT experiment and does not change the model artifact."
        ),
    )
    parser.add_argument(
        "--ax-pack-dense-ffn-gate-up",
        action="store_true",
        help=(
            "Enable experimental loader-time packing of split dense FFN gate/up "
            "projections into the existing gate_up_packed fast path for AX rows. "
            "This is an opt-in prefill probe and does not change the model artifact."
        ),
    )
    parser.add_argument(
        "--ax-qwen-dense-ffn-gate-up-matvec-metal",
        action="store_true",
        help=(
            "Set AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL=1 for AX rows. "
            "This enables the decode-only Qwen split gate/up custom Metal "
            "matvec+SwiGLU route without loader-time row-concatenating the weights."
        ),
    )
    parser.add_argument(
        "--ax-compare-linear-attention-projection-pack",
        action="store_true",
        help=(
            "Run direct AX rows twice for the same prompts: first the default "
            "split linear-attention projections, then the experimental loader-time "
            f"packed row emitted as {AX_ENGINE_LINEAR_ATTENTION_PACK_KEY}. "
            "Also enables AX_MLX_LINEAR_ATTENTION_PROFILE=1 so the output can be "
            "checked by the pack-promotion gate."
        ),
    )
    parser.add_argument(
        "--ax-compare-dense-ffn-gate-up-pack",
        action="store_true",
        help=(
            "Run direct AX rows twice for the same prompts: first the default "
            "split dense FFN gate/up projections, then the experimental loader-time "
            f"packed row emitted as {AX_ENGINE_DENSE_FFN_PACK_KEY}."
        ),
    )
    parser.add_argument(
        "--ax-compare-direct-linear-attention-post-input-route",
        action="store_true",
        help=(
            "Run direct AX rows twice for the same prompts: first "
            "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=1 as the isolated baseline, "
            "then both AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=1 and "
            "AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT=1 emitted as "
            f"{AX_ENGINE_DIRECT_LINEAR_ATTENTION_POST_INPUT_KEY}. Use with Qwen "
            "linear-attention models before considering post-input route promotion."
        ),
    )
    parser.add_argument(
        "--ax-decode-profile",
        action="store_true",
        help=(
            "Enable opt-in direct decode stage profiling for AX rows. This "
            "materializes lazy graphs between stages, disables production decode "
            "pipelining, and is intended for hotspot diagnosis only."
        ),
    )
    parser.add_argument(
        "--ax-prefill-profile",
        action="store_true",
        help=(
            "Enable opt-in prompt prefill stage profiling for AX rows. This "
            "materializes lazy graphs between stages and is intended for "
            "long-context hotspot diagnosis only."
        ),
    )
    parser.add_argument(
        "--gateddelta-prefill-profile",
        action="store_true",
        help=(
            "Run the evidence-first Qwen/GatedDelta long-prompt prefill profile. "
            "This requires a linear-attention MLX manifest, direct AX rows, and "
            "the 512,2048,8192,32768 prompt-token matrix."
        ),
    )
    parser.add_argument(
        "--gateddelta-prefill-profile-report-output",
        type=Path,
        help=(
            "After saving and validating --gateddelta-prefill-profile --output, "
            "render a Markdown stage-profile report at this path."
        ),
    )
    parser.add_argument(
        "--experimental-mlx-kv-compression",
        choices=["disabled", "turboquant-shadow", "turboquant-fused-experimental"],
        default="disabled",
        help=(
            "Pass the AX server experimental MLX KV compression flag through "
            "for AX rows. Default is disabled."
        ),
    )
    parser.add_argument(
        "--experimental-mlx-kv-compression-hot-window-tokens",
        type=int,
        help="Pass through the AX server experimental KV compression hot-window token count.",
    )
    parser.add_argument(
        "--experimental-mlx-kv-compression-min-context-tokens",
        type=int,
        help="Pass through the AX server experimental KV compression minimum context token count.",
    )
    parser.add_argument(
        "--prompt-artifact-root",
        type=Path,
        help="Directory for canonical mlx_lm random-token prompt JSON files",
    )
    parser.add_argument(
        "--capture-output-token-ids",
        action="store_true",
        help=(
            "Record AX response output token IDs in each trial. This is opt-in "
            "because it increases artifact size and is intended for TurboQuant "
            "quality-gate evidence."
        ),
    )
    parser.add_argument(
        "--llama-cpp-bench",
        type=Path,
        help=(
            "Optional path to a Metal-enabled llama.cpp llama-bench binary. "
            "Requires --llama-cpp-gguf and emits shape-compatible external "
            "GGUF baseline rows."
        ),
    )
    parser.add_argument(
        "--llama-cpp-gguf",
        type=Path,
        help="GGUF model path for the optional llama.cpp Metal benchmark baseline.",
    )
    parser.add_argument(
        "--llama-cpp-n-gpu-layers",
        type=int,
        default=99,
        help="Value passed to llama-bench -ngl/--n-gpu-layers for Metal offload.",
    )
    parser.add_argument(
        "--llama-cpp-extra-args",
        help=(
            "Extra arguments appended to llama-bench, for example "
            "'-fa 1' (flash attention) or '-ctk q8_0 -ctv q8_0' (kv quantization). "
            "The Metal backend is selected at llama.cpp build time and verified at "
            "parse time via the row's 'backends' field; do not try to pass it here."
        ),
    )
    parser.add_argument(
        "--llama-cpp-decode-at-depth",
        action="store_true",
        help=(
            "For each optional llama.cpp Metal row, run an additional "
            "`llama-bench -p 0 -n <generation> -d <prompt>` pass and attach "
            "decode-at-depth metrics. This is required before llama.cpp can be "
            "included in ax.long_context_decode_at_depth.v1 artifacts."
        ),
    )
    parser.add_argument(
        "--peak-bandwidth-gb-s",
        type=float,
        default=None,
        help=(
            "Peak unified-memory read bandwidth for the host SoC in GB/s. "
            "When provided, bandwidth accounting rows include percent-of-peak. "
            "Example: M5 Max ~ 800."
        ),
    )
    parser.add_argument(
        "--peak-bandwidth-source",
        default=None,
        help="Short label for the peak bandwidth source (e.g. 'mlx_read_calibration').",
    )
    model_arg_explicit = arg_was_provided(sys.argv[1:], "--model")
    model_repo_id_arg_explicit = arg_was_provided(sys.argv[1:], "--model-repo-id")
    args = parser.parse_args()
    model_dir_explicit = args.model_dir is not None
    try:
        args.model_repo_id = normalize_model_repo_id_for_cache(
            model=args.model,
            model_repo_id=args.model_repo_id,
            model_arg_explicit=model_arg_explicit,
            model_repo_id_arg_explicit=model_repo_id_arg_explicit,
            model_dir_explicit=model_dir_explicit,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        resolved_model_dir = resolve_model_dir(
            args.model_dir,
            args.model_repo_id,
            args.hf_cache_root,
        )
    except RuntimeError as error:
        parser.error(str(error))
    args.model_dir = resolved_model_dir
    # Backfill model_repo_id from HF cache path when --model-dir was given without --model-repo-id.
    # Without this, the default placeholder ends up in the artifact's model_repo_id field.
    if not model_repo_id_arg_explicit and args.model_repo_id == DEFAULT_MODEL_REPO_ID:
        inferred = infer_hf_repo_id_from_path(resolved_model_dir)
        if inferred:
            args.model_repo_id = inferred
    if not model_arg_explicit:
        args.model = str(args.model_dir) if model_dir_explicit else args.model_repo_id
    if args.ax_mtp_max_depth is not None and args.ax_mtp_max_depth < 0:
        parser.error("--ax-mtp-max-depth must be >= 0")
    if args.ax_ngram_accel and args.ax_direct:
        parser.error("--ax-ngram-accel conflicts with --ax-direct")
    if args.ax_ngram_accel and args.ax_compare_policies:
        parser.error("--ax-ngram-accel conflicts with --ax-compare-policies")
    if args.ax_mtp_disable_ngram_stacking and args.ax_direct:
        parser.error(
            "--ax-mtp-disable-ngram-stacking requires an MTP/speculative AX row"
        )
    if args.ax_mtp_disable_ngram_stacking and args.skip_ax_engine:
        parser.error("--ax-mtp-disable-ngram-stacking requires AX rows")
    if args.ax_gemma4_assistant_mtp and args.skip_ax_engine:
        parser.error("--ax-gemma4-assistant-mtp requires AX rows")
    if args.ax_gemma4_assistant_mtp and (
        args.ax_ngram_accel
        or args.ax_compare_policies
        or args.ax_compare_linear_attention_projection_pack
        or args.ax_compare_dense_ffn_gate_up_pack
        or args.ax_compare_direct_linear_attention_post_input_route
    ):
        parser.error(
            "--ax-gemma4-assistant-mtp cannot be combined with other AX row-selection flags"
        )
    if args.ax_compare_linear_attention_projection_pack and args.skip_ax_engine:
        parser.error("--ax-compare-linear-attention-projection-pack requires AX rows")
    if args.ax_compare_linear_attention_projection_pack and (
        args.ax_ngram_accel or args.ax_compare_policies
    ):
        parser.error(
            "--ax-compare-linear-attention-projection-pack requires direct AX rows; "
            "do not combine it with --ax-ngram-accel or --ax-compare-policies"
        )
    if (
        args.ax_compare_linear_attention_projection_pack
        and args.ax_pack_linear_attention_projections
    ):
        parser.error(
            "--ax-compare-linear-attention-projection-pack already runs the packed row; "
            "do not combine it with --ax-pack-linear-attention-projections"
        )
    if args.ax_compare_dense_ffn_gate_up_pack and args.skip_ax_engine:
        parser.error("--ax-compare-dense-ffn-gate-up-pack requires AX rows")
    if args.ax_compare_dense_ffn_gate_up_pack and (
        args.ax_ngram_accel or args.ax_compare_policies
    ):
        parser.error(
            "--ax-compare-dense-ffn-gate-up-pack requires direct AX rows; "
            "do not combine it with --ax-ngram-accel or --ax-compare-policies"
        )
    if args.ax_compare_dense_ffn_gate_up_pack and args.ax_pack_dense_ffn_gate_up:
        parser.error(
            "--ax-compare-dense-ffn-gate-up-pack already runs the packed row; "
            "do not combine it with --ax-pack-dense-ffn-gate-up"
        )
    if (
        args.ax_compare_dense_ffn_gate_up_pack
        and args.ax_compare_linear_attention_projection_pack
    ):
        parser.error(
            "--ax-compare-dense-ffn-gate-up-pack and "
            "--ax-compare-linear-attention-projection-pack both run paired AX rows; "
            "run one comparison at a time"
        )
    try:
        validate_direct_linear_attention_post_input_route_compare_args(args)
    except ValueError as error:
        parser.error(str(error))
    if bool(args.llama_cpp_bench) != bool(args.llama_cpp_gguf):
        parser.error("--llama-cpp-bench and --llama-cpp-gguf must be provided together")
    if args.llama_cpp_decode_at_depth and not args.llama_cpp_bench:
        parser.error(
            "--llama-cpp-decode-at-depth requires --llama-cpp-bench and --llama-cpp-gguf"
        )
    if args.skip_mlx_lm and args.reuse_reference_results_from:
        parser.error("--skip-mlx-lm conflicts with --reuse-reference-results-from")
    if args.prompt_source == "real":
        if args.real_prompt_suite is None:
            parser.error("--prompt-source=real requires --real-prompt-suite")
        if not args.skip_mlx_lm:
            parser.error(
                "--prompt-source=real implies --skip-mlx-lm "
                "(mlx_lm.benchmark cannot consume external prompt text)"
            )
        if args.llama_cpp_bench:
            parser.error(
                "--prompt-source=real cannot be combined with --llama-cpp-bench "
                "(llama-bench generates its own internal prompts)"
            )
        if args.reuse_reference_results_from:
            parser.error(
                "--prompt-source=real cannot reuse mlx_lm reference rows "
                "(reference rows are not produced from real prompts)"
            )
        if args.gateddelta_prefill_profile:
            parser.error(
                "--prompt-source=real is incompatible with "
                "--gateddelta-prefill-profile, which requires fixed-length "
                "random prompts at the configured prefill ladder"
            )
        if args.prefill_scaling_output:
            parser.error(
                "--prompt-source=real cannot build the prefill-scaling "
                "artifact, which requires the random-token prompt ladder"
            )
    elif args.real_prompt_suite is not None:
        parser.error("--real-prompt-suite is only honored when --prompt-source=real")
    if args.prefill_scaling_output and not args.output:
        parser.error("--prefill-scaling-output requires --output")
    if args.gateddelta_prefill_profile_report_output and not args.output:
        parser.error("--gateddelta-prefill-profile-report-output requires --output")
    if (
        args.gateddelta_prefill_profile_report_output
        and not args.gateddelta_prefill_profile
    ):
        parser.error(
            "--gateddelta-prefill-profile-report-output requires --gateddelta-prefill-profile"
        )
    if args.gateddelta_prefill_profile:
        if args.skip_ax_engine:
            parser.error("--gateddelta-prefill-profile requires AX rows")
        if args.ax_ngram_accel or args.ax_compare_policies:
            parser.error(
                "--gateddelta-prefill-profile requires direct AX rows; do not combine "
                "it with --ax-ngram-accel or --ax-compare-policies"
            )
        if args.ax_gemma4_moe_profile:
            parser.error(
                "--gateddelta-prefill-profile conflicts with --ax-gemma4-moe-profile"
            )
        if args.experimental_mlx_kv_compression != "disabled":
            parser.error(
                "--gateddelta-prefill-profile requires KV compression disabled so "
                "prefill evidence is not mixed with compression experiments"
            )
        args.ax_direct = True

    try:
        if args.gateddelta_prefill_profile:
            prompt_lengths = normalize_gateddelta_prefill_profile_prompt_lengths(
                args.prompt_tokens
            )
        else:
            prompt_lengths = parse_prompt_lengths(args.prompt_tokens)
    except ValueError as error:
        parser.error(str(error))

    if args.prompt_artifact_root:
        prompt_artifact_root = args.prompt_artifact_root
    elif args.output:
        prompt_artifact_root = args.output.parent / f"{args.output.stem}-prompts"
    else:
        prompt_artifact_root = Path(
            tempfile.mkdtemp(prefix="ax-mlx-reference-prompts-")
        )
    print("\n=== AX Engine MLX inference stack ===", file=sys.stderr)
    print(f"  model: {args.model}", file=sys.stderr)
    print(f"  model_repo_id: {args.model_repo_id}", file=sys.stderr)
    print(f"  model_dir: {args.model_dir}", file=sys.stderr)
    print(f"  prompt_tokens: {prompt_lengths}", file=sys.stderr)
    print(f"  generation_tokens: {args.generation_tokens}", file=sys.stderr)
    print(
        f"  repetitions: {args.repetitions} + {args.warmup_repetitions} warmup for AX",
        file=sys.stderr,
    )
    print(
        "  ax_prefix_cache: "
        + (
            AX_PREFIX_CACHE_ENABLED_MODE
            if args.ax_enable_prefix_cache
            else AX_PREFIX_CACHE_DISABLED_MODE
        ),
        file=sys.stderr,
    )
    model_metadata = collect_model_metadata(args.model_dir)
    gateddelta_prefill_profile_contract: dict[str, Any] | None = None
    if args.gateddelta_prefill_profile:
        try:
            gateddelta_preflight = validate_gateddelta_prefill_profile_model(
                args.model_dir
            )
            model_metadata.update(
                {
                    "model_type": gateddelta_preflight.get("model_type"),
                    "model_family": gateddelta_preflight.get("model_family"),
                    "model_preflight_schema_version": gateddelta_preflight.get(
                        "schema_version"
                    ),
                    "linear_attention_enabled": True,
                    "linear_attention": gateddelta_preflight.get("linear_attention"),
                }
            )
            gateddelta_prefill_profile_contract = (
                build_gateddelta_prefill_profile_contract(
                    model_metadata,
                    prompt_lengths,
                )
            )
        except RuntimeError as error:
            parser.error(str(error))
    print(f"  prompt_artifact_root: {prompt_artifact_root}", file=sys.stderr)
    if gateddelta_prefill_profile_contract:
        print("  profile: gateddelta_prefill", file=sys.stderr)
    if args.llama_cpp_bench:
        print(f"  llama_cpp_bench: {args.llama_cpp_bench}", file=sys.stderr)
        print(f"  llama_cpp_gguf: {args.llama_cpp_gguf}", file=sys.stderr)

    if not args.skip_ax_engine:
        try:
            ensure_ax_engine_server_binary(build=not args.no_build_ax_engine)
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(1)

    if args.prompt_source == "real":
        prompts = build_real_prompts(
            args.real_prompt_suite,
            args.generation_tokens,
            args.model_dir,
            prompt_artifact_root,
            chat_template=args.real_prompt_chat_template,
            enable_thinking=args.enable_thinking,
        )
    else:
        prompts = build_reference_prompts(
            prompt_lengths,
            args.generation_tokens,
            args.model_dir,
            prompt_artifact_root,
        )

    results: list[dict[str, Any]] = []
    reused_reference_doc: dict[str, Any] | None = None
    if args.reuse_reference_results_from:
        reused_rows, reused_reference_doc = load_reused_reference_rows(
            args.reuse_reference_results_from,
            prompt_lengths=prompt_lengths,
            generation_tokens=args.generation_tokens,
        )
        validate_reused_reference_prompt_hashes(reused_rows, prompts)
        results.extend(reused_rows)
        print(
            f"  [reference] reused {len(reused_rows)} mlx_lm rows "
            f"from {args.reuse_reference_results_from}",
            file=sys.stderr,
        )

    procs: list[subprocess.Popen[Any]] = []
    try:
        if not args.reuse_reference_results_from and not args.skip_mlx_lm:
            for prompt_doc in prompts:
                prompt_tokens = int(prompt_doc["prompt_tokens"])
                results.append(
                    run_mlx_lm_benchmark(
                        args.model,
                        prompt_tokens,
                        args.generation_tokens,
                        args.repetitions,
                        args.cooldown,
                        args.prefill_step_size,
                        prompt_doc,
                    )
                )

        if args.llama_cpp_bench and args.llama_cpp_gguf:
            for prompt_doc in prompts:
                prompt_tokens = int(prompt_doc["prompt_tokens"])
                llama_cpp_row = run_llama_cpp_metal_benchmark(
                    args.llama_cpp_bench,
                    args.llama_cpp_gguf,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    repetitions=args.repetitions,
                    cooldown=args.cooldown,
                    n_gpu_layers=args.llama_cpp_n_gpu_layers,
                    prompt_doc=prompt_doc,
                    extra_args=args.llama_cpp_extra_args,
                )
                if args.llama_cpp_decode_at_depth:
                    attach_llama_cpp_decode_at_depth_benchmark(
                        llama_cpp_row,
                        args.llama_cpp_bench,
                        args.llama_cpp_gguf,
                        context_depth_tokens=prompt_tokens,
                        generation_tokens=args.generation_tokens,
                        repetitions=args.repetitions,
                        cooldown=args.cooldown,
                        n_gpu_layers=args.llama_cpp_n_gpu_layers,
                        extra_args=args.llama_cpp_extra_args,
                    )
                results.append(llama_cpp_row)

        if not args.skip_ax_engine:
            if args.axengine_port == 0:
                args.axengine_port = allocate_port()
            ax_run_configs = []
            if args.ax_compare_linear_attention_projection_pack:
                ax_run_configs = [
                    (
                        True,
                        False,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_DIRECT_KEY,
                    ),
                    (
                        True,
                        True,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_LINEAR_ATTENTION_PACK_KEY,
                    ),
                ]
            elif args.ax_compare_dense_ffn_gate_up_pack:
                ax_run_configs = [
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        False,
                        False,
                        False,
                        AX_ENGINE_DIRECT_KEY,
                    ),
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        True,
                        False,
                        False,
                        AX_ENGINE_DENSE_FFN_PACK_KEY,
                    ),
                ]
            elif args.ax_compare_direct_linear_attention_post_input_route:
                ax_run_configs = [
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        True,
                        False,
                        AX_ENGINE_DIRECT_LINEAR_ATTENTION_INPUTS_KEY,
                    ),
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        True,
                        True,
                        AX_ENGINE_DIRECT_LINEAR_ATTENTION_POST_INPUT_KEY,
                    ),
                ]
            elif args.ax_compare_policies:
                ax_run_configs = [
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_DIRECT_KEY,
                    ),
                    (
                        False,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_PURE_MTP_KEY
                        if args.ax_mtp_disable_ngram_stacking
                        else AX_ENGINE_NGRAM_ACCEL_KEY,
                    ),
                ]
            elif args.ax_ngram_accel:
                ax_run_configs = [
                    (
                        False,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_PURE_MTP_KEY
                        if args.ax_mtp_disable_ngram_stacking
                        else AX_ENGINE_NGRAM_ACCEL_KEY,
                    )
                ]
            elif args.ax_gemma4_assistant_mtp:
                ax_run_configs = [
                    (
                        False,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY
                        if args.ax_mtp_disable_ngram_stacking
                        else AX_ENGINE_GEMMA4_ASSISTANT_MTP_NGRAM_KEY,
                    )
                ]
            else:
                ax_run_configs = [
                    (
                        True,
                        args.ax_pack_linear_attention_projections,
                        args.ax_pack_dense_ffn_gate_up,
                        False,
                        False,
                        AX_ENGINE_DIRECT_KEY,
                    )
                ]

            for (
                direct_mode,
                pack_linear_attention_projections,
                pack_dense_ffn_gate_up,
                direct_linear_attention_inputs_route,
                direct_linear_attention_post_input_route,
                engine_key,
            ) in ax_run_configs:
                gemma4_assistant_mtp = engine_key in (
                    AX_ENGINE_GEMMA4_ASSISTANT_MTP_KEY,
                    AX_ENGINE_GEMMA4_ASSISTANT_MTP_NGRAM_KEY,
                )
                mtp_disable_ngram_stacking = (
                    bool(args.ax_mtp_disable_ngram_stacking) and not direct_mode
                )
                proc = start_axengine(
                    AX_ENGINE_SERVER,
                    args.model_dir,
                    args.axengine_port,
                    model_id=args.model,
                    direct_mode=direct_mode,
                    kv_compression=args.experimental_mlx_kv_compression,
                    kv_compression_hot_window_tokens=(
                        args.experimental_mlx_kv_compression_hot_window_tokens
                    ),
                    kv_compression_min_context_tokens=(
                        args.experimental_mlx_kv_compression_min_context_tokens
                    ),
                    gemma4_moe_profile=args.ax_gemma4_moe_profile,
                    linear_attention_profile=ax_linear_attention_profile_enabled(args),
                    prefill_profile=args.ax_prefill_profile,
                    decode_profile=args.ax_decode_profile,
                    pack_linear_attention_projections=pack_linear_attention_projections,
                    pack_dense_ffn_gate_up=pack_dense_ffn_gate_up,
                    qwen_dense_ffn_gate_up_matvec_metal=(
                        args.ax_qwen_dense_ffn_gate_up_matvec_metal
                    ),
                    direct_linear_attention_inputs_route=(
                        direct_linear_attention_inputs_route
                    ),
                    direct_linear_attention_post_input_route=(
                        direct_linear_attention_post_input_route
                    ),
                    gemma4_assistant_mtp=gemma4_assistant_mtp,
                    mtp_max_depth=args.ax_mtp_max_depth,
                    mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
                    mtp_fast_tail_topk_sampling=args.ax_mtp_fast_tail_topk_sampling,
                    prefill_chunk=args.prefill_step_size,
                    # Scheduler caps per-step prefill at max_batch_tokens. To
                    # let the runner emit one chunked_prefill call per request
                    # (matching mlx_lm.benchmark / mlx-swift-bench geometry),
                    # provision at least the longest configured prompt.
                    max_batch_tokens=max(max(prompt_lengths), args.prefill_step_size),
                    prefix_cache_enabled=args.ax_enable_prefix_cache,
                )
                procs.append(proc)
                if not wait_for_server(
                    f"http://127.0.0.1:{args.axengine_port}/health",
                    proc=proc,
                ):
                    stderr = process_stderr_snapshot(proc)
                    raise RuntimeError(
                        f"ax-engine-server did not become ready:\n{stderr}"
                    )
                for prompt_doc in prompts:
                    validate_prompt_doc(
                        prompt_doc,
                        prompt_tokens=int(prompt_doc["prompt_tokens"]),
                        generation_tokens=args.generation_tokens,
                    )
                    results.append(
                        bench_axengine(
                            args.axengine_port,
                            prompt_doc["token_ids"],
                            args.generation_tokens,
                            args.repetitions,
                            args.warmup_repetitions,
                            args.cooldown,
                            model_metadata=model_metadata,
                            direct_mode=direct_mode,
                            engine_key_override=engine_key,
                            kv_compression=args.experimental_mlx_kv_compression,
                            capture_output_token_ids=args.capture_output_token_ids,
                            server_pid=proc.pid,
                            prefix_cache_enabled=args.ax_enable_prefix_cache,
                            sampler=args.ax_sampling,
                            mtp_disable_ngram_stacking=mtp_disable_ngram_stacking,
                            gemma4_assistant_mtp=gemma4_assistant_mtp,
                            seed=args.seed,
                            prompt_source=prompt_doc.get("prompt_source", "random"),
                        )
                    )
                    results[-1]["prefill_step_size"] = args.prefill_step_size
                    results[-1]["prompt_token_ids_path"] = prompt_doc["token_ids_path"]
                    results[-1]["prompt_token_ids_sha256"] = prompt_doc[
                        "token_ids_sha256"
                    ]
                    results[-1]["prompt_source"] = prompt_doc.get(
                        "prompt_source", "random"
                    )
                    for real_key in (
                        "prompt_suite_id",
                        "prompt_case_id",
                        "prompt_category",
                        "prompt_text_sha256",
                        "case_max_tokens",
                    ):
                        if real_key in prompt_doc:
                            results[-1][real_key] = prompt_doc[real_key]
                    results[-1]["ax_linear_attention_projection_pack"] = bool(
                        pack_linear_attention_projections
                    )
                    results[-1]["ax_dense_ffn_gate_up_pack"] = bool(
                        pack_dense_ffn_gate_up
                    )
                    results[-1]["ax_mtp_disable_ngram_stacking"] = bool(
                        mtp_disable_ngram_stacking
                    )
                    results[-1]["ax_direct_linear_attention_inputs_route"] = bool(
                        direct_linear_attention_inputs_route
                    )
                    results[-1]["ax_direct_linear_attention_post_input_route"] = bool(
                        direct_linear_attention_post_input_route
                    )
                    if args.inter_case_cooldown > 0 and prompt_doc is not prompts[-1]:
                        print(
                            f"  [ax-engine] inter-case cooldown {args.inter_case_cooldown:.0f}s",
                            file=sys.stderr,
                        )
                        time.sleep(args.inter_case_cooldown)
                kill_proc(proc)
                procs.remove(proc)
                if (
                    (direct_mode and args.ax_compare_policies)
                    or (
                        not pack_linear_attention_projections
                        and args.ax_compare_linear_attention_projection_pack
                    )
                    or (
                        not pack_dense_ffn_gate_up
                        and args.ax_compare_dense_ffn_gate_up_pack
                    )
                    or (
                        direct_linear_attention_inputs_route
                        and not direct_linear_attention_post_input_route
                        and args.ax_compare_direct_linear_attention_post_input_route
                    )
                ):
                    time.sleep(3)  # brief cooldown between modes
    finally:
        for proc in procs:
            kill_proc(proc)

    attach_mlx_lm_baselines(results)

    llama_cpp_metal_present = bool(args.llama_cpp_bench) or any(
        cell.get("engine") == "llama_cpp_metal" for cell in results
    )

    doc = {
        "schema_version": "ax.mlx_inference_stack.v2",
        "claim_gate": {
            "schema_version": PHASE0_CLAIM_GATE_SCHEMA_VERSION,
            "scope": "mlx_inference_stack_public_readme",
            "requires_prompt_hash_parity": True,
            "prompt_hash_parity_scope": "mlx_lm_and_ax_engine_rows",
            "shape_only_external_rows": ["llama_cpp_metal"],
            "requires_runtime_identity": True,
            "requires_decode_policy_identity": True,
            "requires_prefill_decode_split": True,
            "forbidden_public_claims_without_artifacts": CLAIMS_REQUIRING_ARTIFACT_EVIDENCE,
        },
        "host": collect_host_metadata(),
        "build": collect_build_metadata(),
        "model": args.model,
        "model_repo_id": args.model_repo_id,
        "model_dir": str(args.model_dir),
        "model_dir_source": "explicit" if model_dir_explicit else "huggingface_cache",
        "hf_cache_root": str(args.hf_cache_root) if args.hf_cache_root else None,
        "model_config": model_metadata,
        "reference_contract": {
            "primary_reference": "mlx_lm.benchmark",
            "primary_reference_required": True,
            "external_gguf_reference": "llama.cpp Metal llama-bench",
            "external_gguf_reference_present": llama_cpp_metal_present,
            "external_gguf_reference_required": False,
            "retired_reference": "SwiftLM application server",
            "comparison_policy": (
                "Every non-baseline row is compared against the matching "
                "mlx_lm.benchmark row for the same random-token prompt and "
                "generation shape. ax_engine_mlx is the direct same-policy "
                "comparison baseline; ax_engine_mlx_ngram_accel rows are AX "
                "default n-gram policy rows whose ax_decode_claim_status "
                "distinguishes effective acceleration from no-draft fallback. "
                "llama_cpp_metal rows are shape-compatible external GGUF rows, "
                "not prompt-hash parity or repo-owned MLX throughput evidence."
            ),
            "prompt_contract": {
                "source": "mlx_lm.benchmark",
                "random_seed": MLX_LM_RANDOM_SEED,
                "distribution": "mx.random.randint(0, vocab_size, (1, prompt_tokens))",
                "batch_size": 1,
                "artifacts": [without_inline_tokens(prompt) for prompt in prompts],
            },
            "strictness": (
                "mlx_lm_prompt_algorithm_reproduced; "
                "llama_cpp_metal_shape_compatible_only"
            ),
        },
        "prompt_tokens": prompt_lengths,
        "generation_tokens": args.generation_tokens,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "cooldown": args.cooldown,
        "prefill_step_size": args.prefill_step_size,
        "ax_mtp_max_depth": args.ax_mtp_max_depth,
        "ax_mtp_disable_ngram_stacking": bool(args.ax_mtp_disable_ngram_stacking),
        "ax_mtp_fast_tail_topk_sampling": bool(args.ax_mtp_fast_tail_topk_sampling),
        "ax_prefix_cache_mode": (
            AX_PREFIX_CACHE_ENABLED_MODE
            if args.ax_enable_prefix_cache
            else AX_PREFIX_CACHE_DISABLED_MODE
        ),
        "prefill_ttft_measurement_contract": (
            "cold_prefill_no_cross_request_prefix_cache"
            if not args.ax_enable_prefix_cache
            else "prefix_cache_enabled_prefill_metrics_invalidated_on_hit"
        ),
        "prefill_timing_scope_contract": {
            "mlx_lm": "upstream_mlx_lm_response_stats",
            "ax_engine_primary_prefill": "server_sse_runner_time_us",
            "ax_engine_client_wall_ttft": "http_sse_first_output_token_observed_by_client",
            "long_greedy_ax_prefill_work": (
                "mlx_lm_style_cache_only_prefix_plus_final_prompt_token"
            ),
        },
        "concurrency": 1,
        "concurrent_prefill_overlap_classification": {
            "classification": "single_request_no_overlap",
            "continuous_batching_claim": False,
            "concurrency": 1,
        },
        "prefix_reuse_evidence": summarize_prefix_reuse_evidence(results),
        "run_stability_summary": summarize_artifact_run_stability(results),
        "ax_gemma4_moe_profile": bool(args.ax_gemma4_moe_profile),
        "ax_linear_attention_profile": ax_linear_attention_profile_enabled(args),
        "ax_linear_attention_projection_pack": bool(
            args.ax_pack_linear_attention_projections
            or args.ax_compare_linear_attention_projection_pack
        ),
        "ax_linear_attention_projection_pack_compare": bool(
            args.ax_compare_linear_attention_projection_pack
        ),
        "ax_dense_ffn_gate_up_pack": bool(
            args.ax_pack_dense_ffn_gate_up or args.ax_compare_dense_ffn_gate_up_pack
        ),
        "ax_dense_ffn_gate_up_pack_compare": bool(
            args.ax_compare_dense_ffn_gate_up_pack
        ),
        "ax_qwen_dense_ffn_gate_up_matvec_metal": bool(
            args.ax_qwen_dense_ffn_gate_up_matvec_metal
        ),
        "ax_direct_linear_attention_post_input_route_compare": bool(
            args.ax_compare_direct_linear_attention_post_input_route
        ),
        "ax_prefill_profile": bool(args.ax_prefill_profile),
        "ax_decode_profile": bool(args.ax_decode_profile),
        "results": results,
    }
    doc["bandwidth_accounting"] = build_bandwidth_accounting(
        args.model_dir,
        results,
        peak_bandwidth_gb_s=args.peak_bandwidth_gb_s,
        peak_bandwidth_source=args.peak_bandwidth_source,
    )
    if gateddelta_prefill_profile_contract:
        doc["gateddelta_prefill_profile"] = gateddelta_prefill_profile_contract
    if args.reuse_reference_results_from:
        doc["ax_only_refresh"] = {
            "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
            "reference_results_source": str(args.reuse_reference_results_from),
            "ax_reference_regression_summary": summarize_ax_only_refresh_regression(
                results=results,
                reference_doc=reused_reference_doc,
            ),
            "reference_rows_reused": len(
                [cell for cell in results if cell.get("engine") == "mlx_lm"]
            ),
            "ax_rows_refreshed": len(
                [
                    cell
                    for cell in results
                    if str(cell.get("engine", "")).startswith("ax_engine")
                ]
            ),
        }
        if reused_reference_doc:
            doc["reference_contract"]["reused_reference_artifact_schema_version"] = (
                reused_reference_doc.get("schema_version")
            )
            doc["reference_contract"]["reused_reference_artifact_build"] = (
                reused_reference_doc.get("build", {})
            )

    print_summary(doc)
    print(json.dumps(doc, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nSaved to {args.output}", file=sys.stderr)
        if gateddelta_prefill_profile_contract:
            checked = validate_gateddelta_prefill_profile_output(args.output)
            print(
                "Validated GatedDelta prefill profile artifact: "
                f"{len(checked)} shape groups",
                file=sys.stderr,
            )
            if args.gateddelta_prefill_profile_report_output:
                render_gateddelta_prefill_profile_output(
                    args.output,
                    args.gateddelta_prefill_profile_report_output,
                )
                print(
                    "Saved GatedDelta prefill profile report to "
                    f"{args.gateddelta_prefill_profile_report_output}",
                    file=sys.stderr,
                )
    if args.prefill_scaling_output:
        from build_mlx_prefill_scaling_artifact import (
            build_prefill_scaling_artifact,
        )
        from check_mlx_prefill_scaling_artifact import (
            validate_prefill_scaling_artifact,
        )

        scaling_doc = build_prefill_scaling_artifact(
            args.output,
            min_context_tokens=args.prefill_scaling_min_context_tokens,
        )
        args.prefill_scaling_output.parent.mkdir(parents=True, exist_ok=True)
        args.prefill_scaling_output.write_text(json.dumps(scaling_doc, indent=2) + "\n")
        if not args.skip_prefill_scaling_validate:
            validate_prefill_scaling_artifact(args.prefill_scaling_output)
        print(
            f"Saved prefill scaling artifact to {args.prefill_scaling_output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
