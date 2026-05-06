#!/usr/bin/env python3
"""Benchmark AX Engine MLX inference against MLX reference runtimes.

The primary reference is upstream `mlx_lm.benchmark`, not the older SwiftLM
application harness. `mlx-swift-lm` is supported as an optional command adapter
because the reference package exposes libraries and benchmark helpers, but no
repo-stable inference benchmark CLI.

Every comparison run includes `mlx_lm.benchmark`. If that baseline fails, the
run fails instead of emitting AX-only numbers.

Examples:
  cargo build -p ax-engine-server --release

  python3 scripts/bench_mlx_inference_stack.py \
    --model-dir .internal/models/Qwen3.5-9B-MLX-4bit \
    --prompt-tokens 512,2048 \
    --generation-tokens 128 \
    --repetitions 5 \
    --cooldown 5

Optional mlx-swift-lm adapter:
  python3 scripts/bench_mlx_inference_stack.py \
    --mlx-swift-lm-command './.internal/tools/mlx-swift-bench \
      --model {model} --prompt-tokens {prompt_tokens} \
      --generation-tokens {generation_tokens} --trials {trials} --delay {delay} \
      --prefill-step-size {prefill_step_size} \
      --prompt-token-ids {prompt_token_ids_path}'

The optional mlx-swift-lm adapter is treated as a secondary reference only when
it follows the mlx-swift-lm BenchmarkHelpers/MLXLMCommon generation contract and
uses the prompt token JSON emitted by this script. The command template may use:
  {model}, {prompt_tokens}, {generation_tokens}, {trials}, {delay},
  {prefill_step_size}, {random_seed}, {batch_size}, {prompt_token_ids_path},
  {prompt_token_ids_sha256}

The adapter command must print JSON with either:
  {"prompt_tps": 123.4, "generation_tps": 56.7, "peak_memory": 12.3}
or:
  {"prefill_tok_s": 123.4, "decode_tok_s": 56.7, "peak_memory_gb": 12.3}
or trial rows:
  {"trials": [{"prefill_tok_s": 123.4, "decode_tok_s": 56.7}]}
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
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_ENGINE_SERVER = REPO_ROOT / "target/release/ax-engine-server"
DEFAULT_MODEL_DIR = REPO_ROOT / ".internal/models/Qwen3.5-9B-MLX-4bit"
DEFAULT_MODEL_ID = str(DEFAULT_MODEL_DIR)
DEFAULT_PROMPT_TOKENS = "512,2048"
DEFAULT_GENERATION_TOKENS = 128
DEFAULT_REPETITIONS = 5
DEFAULT_COOLDOWN = 5.0
AXENGINE_PORT = 8091
MLX_LM_RANDOM_SEED = 0

AX_ENGINE_DIRECT_KEY = "ax_engine_mlx"
AX_ENGINE_NGRAM_ACCEL_KEY = "ax_engine_mlx_ngram_accel"

AX_NGRAM_TELEMETRY_KEYS = [
    "ax_ngram_draft_attempts",
    "ax_ngram_draft_tokens",
    "ax_ngram_accepted_tokens",
    "ax_ngram_rejected_tokens",
    "ax_ngram_full_accepts",
    "ax_ngram_partial_rejects",
    "ax_ngram_complete_misses",
    "ax_ngram_no_draft_steps",
    "ax_ngram_cooldown_steps",
    "ax_ngram_cooldown_events",
    "ax_ngram_cooldown_steps_scheduled",
    "ax_ngram_request_disable_events",
    "ax_ngram_request_disabled_steps",
]
AX_NGRAM_ACCEPT_RATE_KEY = "ax_ngram_accept_rate_micros"

AX_MLX_TELEMETRY_KEYS = [
    "ax_mlx_prefill_steps",
    "ax_mlx_prefill_wall_us",
    "ax_mlx_decode_steps",
    "ax_mlx_decode_wall_us",
    "ax_mlx_direct_bootstrap_steps",
    "ax_mlx_direct_bootstrap_wall_us",
    "ax_mlx_direct_pipeline_steps",
    "ax_mlx_direct_pipeline_wall_us",
    "ax_mlx_single_decode_steps",
    "ax_mlx_single_decode_wall_us",
    "ax_mlx_ngram_decode_steps",
    "ax_mlx_ngram_decode_wall_us",
    "ax_mlx_bonus_tokens",
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
    "ax_mlx_kv_compression_fused_decode_fallbacks",
    "ax_mlx_kv_compression_fused_decode_fallback_reason",
    "ax_mlx_kv_compression_fused_decode_ready_candidates",
    "ax_mlx_kv_compression_fused_decode_blocked_prefill_only",
    "ax_mlx_kv_compression_fused_decode_blocked_attention_kind",
    "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer",
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset",
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim",
    "ax_mlx_kv_compression_fused_decode_blocked_gqa",
    "ax_mlx_kv_compression_fused_decode_blocked_missing_storage",
]


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
        memory_gb = round(int(memory_bytes) / (1024 ** 3))
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
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    return {
        "commit": commit,
        "build_profile": "release",
        "server_binary": str(AX_ENGINE_SERVER),
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

    return metadata


def native_manifest_has_linear_attention(linear_attention: Any) -> bool:
    if not isinstance(linear_attention, dict):
        return False
    return any(value is not None for value in linear_attention.values())


def ax_decode_policy(
    model_metadata: dict[str, Any], *, direct_mode: bool
) -> str:
    if direct_mode:
        return "direct_no_ngram_acceleration"
    if model_metadata.get("linear_attention_enabled"):
        return "ngram_acceleration_linear_attention_branch_recompute"
    return "ngram_acceleration_kv_trim"


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

def wait_for_server(url: str, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_proc(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


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
    path = artifact_root / f"prompt-{prompt_tokens}-gen-{generation_tokens}-{token_hash[:12]}.json"
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
        prompts.append(prompt_doc)
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
        raise RuntimeError(f"mlx_lm.benchmark failed with exit={result.returncode}:\n{combined}")
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
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
        }
    )
    return cell


def start_axengine(
    binary: Path,
    model_dir: Path,
    port: int,
    *,
    direct_mode: bool,
    kv_compression: str = "disabled",
    kv_compression_hot_window_tokens: int | None = None,
    kv_compression_min_context_tokens: int | None = None,
    gemma4_moe_profile: bool = False,
) -> subprocess.Popen[Any]:
    cmd = [
        str(binary),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--port",
        str(port),
    ]
    if direct_mode:
        cmd.append("--disable-ngram-acceleration")
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
    if gemma4_moe_profile:
        env["AX_MLX_GEMMA4_MOE_PROFILE"] = "1"
    print(f"  [ax-engine] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)


def extract_ax_ngram_telemetry(route: dict[str, Any] | None) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    telemetry = {key: int(decisions.get(key, 0)) for key in AX_NGRAM_TELEMETRY_KEYS}
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


def extract_ax_mlx_telemetry(route: dict[str, Any] | None) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    return {key: int(decisions.get(key, 0)) for key in AX_MLX_TELEMETRY_KEYS}


def extract_ax_mlx_gemma4_moe_profile(
    route: dict[str, Any] | None,
) -> dict[str, int]:
    if not route:
        return {}
    decisions = route.get("crossover_decisions") or {}
    if "ax_mlx_gemma4_moe_profile_enabled" not in decisions:
        return {}
    return {
        key: int(decisions.get(key, 0))
        for key in AX_MLX_GEMMA4_MOE_PROFILE_KEYS
    }


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
        key: int(decisions.get(key, 0))
        for key in AX_MLX_KV_COMPRESSION_TELEMETRY_KEYS
    }


def route_with_more_decisions(
    candidate: dict[str, Any] | None, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not candidate:
        return current
    if not current:
        return candidate
    candidate_count = len(candidate.get("crossover_decisions") or {})
    current_count = len(current.get("crossover_decisions") or {})
    if candidate_count > current_count:
        return candidate
    if candidate_count == current_count:
        candidate_total = sum(
            int(value) for value in (candidate.get("crossover_decisions") or {}).values()
        )
        current_total = sum(
            int(value) for value in (current.get("crossover_decisions") or {}).values()
        )
        if candidate_total > current_total:
            return candidate
    return current


def summarize_telemetry(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ngram_acceleration_telemetry") or {}).items():
            if key == AX_NGRAM_ACCEPT_RATE_KEY:
                continue
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
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def summarize_ax_mlx_gemma4_moe_profile(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("ax_mlx_gemma4_moe_profile") or {}).items():
            if key == "ax_mlx_gemma4_moe_profile_enabled":
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


def is_ax_prefill_step(step: dict[str, Any], *, seen_prefill: bool) -> bool:
    scheduled_tokens = int(step.get("scheduled_tokens") or 0)
    return not seen_prefill or scheduled_tokens > 1


def axengine_one_run(
    port: int,
    tokens: list[int],
    generation_tokens: int,
    *,
    capture_output_token_ids: bool = False,
) -> dict[str, Any]:
    payload = json.dumps(
        {"input_tokens": tokens, "max_output_tokens": generation_tokens}
    ).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
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
    current_event = ""
    final_route: dict[str, Any] | None = None

    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if current_event == "step":
            step = obj.get("step", {})
            runner_us = int(step.get("runner_time_us", 0))
            output_tokens = int(obj.get("request", {}).get("output_len", output_tokens))
            final_route = route_with_more_decisions(
                step.get("route")
                or obj.get("request", {}).get("route"),
                final_route,
            )
            if is_ax_prefill_step(step, seen_prefill=seen_prefill):
                prefill_us += runner_us
                seen_prefill = True
            else:
                decode_us += runner_us
        elif current_event == "response":
            response_tokens = obj.get("response", {}).get("output_tokens", [])
            output_tokens = len(response_tokens) or output_tokens
            if capture_output_token_ids and isinstance(response_tokens, list):
                output_token_ids = [int(token) for token in response_tokens]
            final_route = route_with_more_decisions(
                obj.get("response", {}).get("route"),
                final_route,
            )

    conn.close()
    prompt_tokens = len(tokens)
    prefill_s = prefill_us / 1_000_000
    decode_s = decode_us / 1_000_000
    measured_decode_tokens = max(output_tokens - 1, 0)
    run: dict[str, Any] = {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tok_s": prompt_tokens / prefill_s if prefill_s > 0 else 0.0,
        "decode_tok_s": measured_decode_tokens / decode_s if decode_s > 0 else 0.0,
        "output_tokens": float(output_tokens),
    }
    if capture_output_token_ids:
        run["output_token_ids"] = output_token_ids
    telemetry = extract_ax_ngram_telemetry(final_route)
    if telemetry:
        run["ngram_acceleration_telemetry"] = telemetry
    mlx_telemetry = extract_ax_mlx_telemetry(final_route)
    if mlx_telemetry:
        run["ax_mlx_telemetry"] = mlx_telemetry
    gemma4_moe_profile = extract_ax_mlx_gemma4_moe_profile(final_route)
    if gemma4_moe_profile:
        run["ax_mlx_gemma4_moe_profile"] = gemma4_moe_profile
    compression_telemetry = extract_ax_mlx_kv_compression_telemetry(final_route)
    if compression_telemetry:
        run["kv_compression_telemetry"] = compression_telemetry
    return run


def summarize_runs(runs: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [run[key] for run in runs]
    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _trial_metric(trial: dict[str, Any], primary: str, fallback: str) -> float:
    value = trial.get(primary, trial.get(fallback))
    if value is None:
        raise RuntimeError(
            f"mlx-swift-lm trial JSON must include {primary} or {fallback}"
        )
    return float(value)


def bench_axengine(
    port: int,
    tokens: list[int],
    generation_tokens: int,
    repetitions: int,
    cooldown: float,
    *,
    model_metadata: dict[str, Any],
    direct_mode: bool = False,
    kv_compression: str = "disabled",
    capture_output_token_ids: bool = False,
) -> dict[str, Any]:
    engine_key = AX_ENGINE_DIRECT_KEY if direct_mode else AX_ENGINE_NGRAM_ACCEL_KEY
    decode_policy = ax_decode_policy(
        model_metadata, direct_mode=direct_mode
    )
    print(
        f"  [ax-engine/{engine_key}] prompt={len(tokens)} "
        f"generation={generation_tokens} policy={decode_policy} "
        f"kv_compression={kv_compression}",
        file=sys.stderr,
    )
    axengine_one_run(port, tokens, generation_tokens)
    if cooldown > 0:
        time.sleep(cooldown)

    runs = []
    for index in range(repetitions):
        run = axengine_one_run(
            port,
            tokens,
            generation_tokens,
            capture_output_token_ids=capture_output_token_ids,
        )
        runs.append(run)
        print(
            f"    rep {index + 1}: prefill={run['prefill_tok_s']:.1f} tok/s "
            f"decode={run['decode_tok_s']:.1f} tok/s out={run['output_tokens']:.0f}",
            file=sys.stderr,
        )
        if cooldown > 0 and index < repetitions - 1:
            time.sleep(cooldown)

    row = {
        "engine": engine_key,
        "method": "server_sse_runner_time_us",
        "timing_scope": "ax_engine_runner_time_us",
        "ax_decode_policy": decode_policy,
        "ax_decode_claim_status": (
            "direct_same_policy_baseline"
            if direct_mode
            else "ngram_acceleration_effective_throughput"
        ),
        "prompt_contract": "mlx_lm_random_tokens_seed_0",
        "random_seed": MLX_LM_RANDOM_SEED,
        "batch_size": 1,
        "prompt_tokens": len(tokens),
        "generation_tokens": generation_tokens,
        "prefill_tok_s": summarize_runs(runs, "prefill_tok_s"),
        "decode_tok_s": summarize_runs(runs, "decode_tok_s"),
        "prefill_s": summarize_runs(runs, "prefill_s"),
        "decode_s": summarize_runs(runs, "decode_s"),
        "ngram_acceleration_telemetry": summarize_telemetry(runs),
        "ax_mlx_telemetry": summarize_ax_mlx_telemetry(runs),
        "ax_mlx_gemma4_moe_profile": summarize_ax_mlx_gemma4_moe_profile(runs),
        "trials": runs,
    }
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
    if compression_summary:
        row["kv_compression_telemetry"] = compression_summary
    return row


def parse_swift_adapter_json(stdout: str) -> dict[str, Any]:
    payload = json.loads(stdout)
    prefill = payload.get("prefill_tok_s", payload.get("prompt_tps"))
    decode = payload.get("decode_tok_s", payload.get("generation_tps"))
    memory = payload.get("peak_memory_gb", payload.get("peak_memory"))
    trials = payload.get("trials")
    if trials:
        prefill_values = [
            _trial_metric(trial, "prefill_tok_s", "prompt_tps") for trial in trials
        ]
        decode_values = [
            _trial_metric(trial, "decode_tok_s", "generation_tps") for trial in trials
        ]
        parsed: dict[str, Any] = {
            "prefill_tok_s": summarize_values(prefill_values),
            "decode_tok_s": summarize_values(decode_values),
            "trials": trials,
        }
        memory_values = [
            trial.get("peak_memory_gb", trial.get("peak_memory")) for trial in trials
        ]
        if all(value is not None for value in memory_values):
            parsed["peak_memory_gb"] = summarize_values(
                [float(value) for value in memory_values]
            )
        return parsed

    if prefill is None or decode is None:
        raise RuntimeError("mlx-swift-lm adapter JSON must include prefill/decode throughput")
    parsed = {
        "prefill_tok_s": {"median": float(prefill)},
        "decode_tok_s": {"median": float(decode)},
    }
    if memory is not None:
        parsed["peak_memory_gb"] = {"median": float(memory)}
    return parsed


def run_mlx_swift_lm_adapter(
    command_template: str,
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
    command = command_template.format(
        model=model,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        trials=repetitions,
        delay=cooldown,
        prefill_step_size=prefill_step_size,
        random_seed=MLX_LM_RANDOM_SEED,
        batch_size=1,
        prompt_tokens_path=prompt_doc["token_ids_path"],
        prompt_token_ids_path=prompt_doc["token_ids_path"],
        prompt_token_ids_sha256=prompt_doc["token_ids_sha256"],
    )
    argv = shlex.split(command)
    print(f"  [mlx-swift-lm] {command}", file=sys.stderr)
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mlx-swift-lm adapter failed with exit={result.returncode}:\n"
            f"{result.stdout}{result.stderr}"
        )
    metrics = parse_swift_adapter_json(result.stdout)
    cell: dict[str, Any] = {
        "engine": "mlx_swift_lm",
        "method": "mlx_swift_lm_benchmark_adapter",
        "timing_scope": "external_adapter_reported",
        "prompt_contract": "mlx_lm_random_tokens_seed_0",
        "secondary_reference_role": "mlx-swift-lm BenchmarkHelpers/MLXLMCommon generation adapter",
        "random_seed": MLX_LM_RANDOM_SEED,
        "batch_size": 1,
        "prefill_step_size": prefill_step_size,
        "prompt_token_ids_path": prompt_doc["token_ids_path"],
        "prompt_token_ids_sha256": prompt_doc["token_ids_sha256"],
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tok_s": metrics["prefill_tok_s"],
        "decode_tok_s": metrics["decode_tok_s"],
    }
    if "peak_memory_gb" in metrics:
        cell["peak_memory_gb"] = metrics["peak_memory_gb"]
    if "trials" in metrics:
        cell["trials"] = metrics["trials"]
    return cell


def metric_value(cell: dict[str, Any], metric: str) -> float:
    data = cell.get(metric, {})
    if "median" in data:
        return float(data["median"])
    if "mean" in data:
        return float(data["mean"])
    return 0.0


def attach_mlx_lm_baselines(results: list[dict[str, Any]]) -> None:
    baselines = {
        (int(cell["prompt_tokens"]), int(cell["generation_tokens"])): cell
        for cell in results
        if cell.get("engine") == "mlx_lm"
    }

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
        if engine not in {"mlx_lm", "mlx_swift_lm"}:
            continue
        key = (int(cell["prompt_tokens"]), int(cell["generation_tokens"]))
        if key not in wanted:
            continue
        rows.append(copy.deepcopy(cell))
        if engine == "mlx_lm":
            seen_mlx_lm.add(key)

    missing = sorted(wanted - seen_mlx_lm)
    if missing:
        raise RuntimeError(
            f"{path} is missing mlx_lm reference rows for prompt/generation pairs: {missing}"
        )
    return rows, doc


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
    print("\n" + "=" * 88)
    print("AX Engine MLX inference stack benchmark")
    print("=" * 88)
    print(
        f"{'Engine':<18} {'Prompt tok':>10} {'Prefill tok/s':>14} "
        f"{'Decode tok/s':>13} {'Decode vs mlx_lm':>16}  Method"
    )
    print("-" * 88)
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
            f"{ratio_text:>16}  {cell['method']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AX MLX against MLX reference runtimes")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--generation-tokens", type=int, default=DEFAULT_GENERATION_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-ax-engine", action="store_true")
    parser.add_argument(
        "--reuse-reference-results-from",
        type=Path,
        help=(
            "Reuse mlx_lm/mlx_swift_lm rows from an existing artifact and rerun "
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
            "Run only the direct same-policy AX row. This is the default "
            "apple-to-apple AX comparison."
        ),
    )
    parser.add_argument(
        "--ax-ngram-accel",
        dest="ax_ngram_accel",
        action="store_true",
        help="Run only AX n-gram acceleration effective-throughput rows.",
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
        "--ax-gemma4-moe-profile",
        action="store_true",
        help=(
            "Enable opt-in Gemma4 MoE decode-layer profiling for AX rows. "
            "This inserts timing barriers and is for diagnosis, not headline throughput."
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
        "--mlx-swift-lm-command",
        help=(
            "Optional mlx-swift-lm BenchmarkHelpers/MLXLMCommon generation "
            "adapter command template that returns benchmark JSON"
        ),
    )
    args = parser.parse_args()
    if args.model == DEFAULT_MODEL_ID and args.model_dir != DEFAULT_MODEL_DIR:
        args.model = str(args.model_dir)
    if args.ax_ngram_accel and args.ax_direct:
        parser.error("--ax-ngram-accel conflicts with --ax-direct")
    if args.ax_ngram_accel and args.ax_compare_policies:
        parser.error("--ax-ngram-accel conflicts with --ax-compare-policies")
    if args.reuse_reference_results_from and args.mlx_swift_lm_command:
        parser.error("--reuse-reference-results-from conflicts with --mlx-swift-lm-command")

    if not args.skip_ax_engine and not AX_ENGINE_SERVER.exists():
        print(
            f"ERROR: ax-engine-server not found at {AX_ENGINE_SERVER}. "
            "Run: cargo build -p ax-engine-server --release",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_lengths = [int(item.strip()) for item in args.prompt_tokens.split(",") if item.strip()]
    if args.prompt_artifact_root:
        prompt_artifact_root = args.prompt_artifact_root
    elif args.output:
        prompt_artifact_root = args.output.parent / f"{args.output.stem}-prompts"
    else:
        prompt_artifact_root = Path(tempfile.mkdtemp(prefix="ax-mlx-reference-prompts-"))
    print("\n=== AX Engine MLX inference stack ===", file=sys.stderr)
    print(f"  model: {args.model}", file=sys.stderr)
    print(f"  model_dir: {args.model_dir}", file=sys.stderr)
    print(f"  prompt_tokens: {prompt_lengths}", file=sys.stderr)
    print(f"  generation_tokens: {args.generation_tokens}", file=sys.stderr)
    print(f"  repetitions: {args.repetitions} + 1 warmup for AX", file=sys.stderr)
    print(f"  prompt_artifact_root: {prompt_artifact_root}", file=sys.stderr)
    model_metadata = collect_model_metadata(args.model_dir)

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
            f"  [reference] reused {len(reused_rows)} mlx_lm/mlx_swift_lm rows "
            f"from {args.reuse_reference_results_from}",
            file=sys.stderr,
        )

    procs: list[subprocess.Popen[Any]] = []
    try:
        if not args.reuse_reference_results_from:
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

        if args.mlx_swift_lm_command:
            for prompt_doc in prompts:
                prompt_tokens = int(prompt_doc["prompt_tokens"])
                results.append(
                    run_mlx_swift_lm_adapter(
                        args.mlx_swift_lm_command,
                        args.model,
                        prompt_tokens,
                        args.generation_tokens,
                        args.repetitions,
                        args.cooldown,
                        args.prefill_step_size,
                        prompt_doc,
                    )
                )

        if not args.skip_ax_engine:
            modes = []
            if args.ax_compare_policies:
                modes = [True, False]  # direct first, then n-gram acceleration
            elif args.ax_ngram_accel:
                modes = [False]
            else:
                modes = [True]

            for direct_mode in modes:
                proc = start_axengine(
                    AX_ENGINE_SERVER,
                    args.model_dir,
                    args.axengine_port,
                    direct_mode=direct_mode,
                    kv_compression=args.experimental_mlx_kv_compression,
                    kv_compression_hot_window_tokens=(
                        args.experimental_mlx_kv_compression_hot_window_tokens
                    ),
                    kv_compression_min_context_tokens=(
                        args.experimental_mlx_kv_compression_min_context_tokens
                    ),
                    gemma4_moe_profile=args.ax_gemma4_moe_profile,
                )
                procs.append(proc)
                if not wait_for_server(f"http://127.0.0.1:{args.axengine_port}/health"):
                    stderr = proc.stderr.read(2000).decode(errors="replace") if proc.stderr else ""
                    raise RuntimeError(f"ax-engine-server did not become ready:\n{stderr}")
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
                            args.cooldown,
                            model_metadata=model_metadata,
                            direct_mode=direct_mode,
                            kv_compression=args.experimental_mlx_kv_compression,
                            capture_output_token_ids=args.capture_output_token_ids,
                        )
                    )
                    results[-1]["prefill_step_size"] = args.prefill_step_size
                    results[-1]["prompt_token_ids_path"] = prompt_doc["token_ids_path"]
                    results[-1]["prompt_token_ids_sha256"] = prompt_doc["token_ids_sha256"]
                kill_proc(proc)
                procs.remove(proc)
                if direct_mode and args.ax_compare_policies:
                    time.sleep(3)  # brief cooldown between modes
    finally:
        for proc in procs:
            kill_proc(proc)

    attach_mlx_lm_baselines(results)

    secondary_reference_present = bool(args.mlx_swift_lm_command) or any(
        cell.get("engine") == "mlx_swift_lm" for cell in results
    )

    doc = {
        "schema_version": "ax.mlx_inference_stack.v2",
        "host": collect_host_metadata(),
        "build": collect_build_metadata(),
        "model": args.model,
        "model_dir": str(args.model_dir),
        "model_config": model_metadata,
        "reference_contract": {
            "primary_reference": "mlx_lm.benchmark",
            "primary_reference_required": True,
            "secondary_reference": "mlx-swift-lm BenchmarkHelpers/MLXLMCommon generation adapter",
            "secondary_reference_present": secondary_reference_present,
            "secondary_reference_required": False,
            "retired_reference": "SwiftLM application server",
            "comparison_policy": (
                "Every non-baseline row is compared against the matching "
                "mlx_lm.benchmark row for the same random-token prompt and "
                "generation shape. ax_engine_mlx is the default direct "
                "same-policy comparison; ax_engine_mlx_ngram_accel rows are "
                "n-gram acceleration effective-throughput rows."
            ),
            "secondary_reference_policy": (
                "mlx-swift-lm rows are admitted only through an explicit "
                "BenchmarkHelpers/MLXLMCommon generation adapter that reads "
                "the prompt token JSON emitted by this harness and reports "
                "prefill/decode metrics for the same random-token prompt/decode shape."
            ),
            "prompt_contract": {
                "source": "mlx_lm.benchmark",
                "random_seed": MLX_LM_RANDOM_SEED,
                "distribution": "mx.random.randint(0, vocab_size, (1, prompt_tokens))",
                "batch_size": 1,
                "artifacts": [without_inline_tokens(prompt) for prompt in prompts],
            },
            "strictness": "same_prompt_tokens_for_ax_and_swift_adapter; mlx_lm_prompt_algorithm_reproduced",
        },
        "prompt_tokens": prompt_lengths,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "prefill_step_size": args.prefill_step_size,
        "concurrency": 1,
        "ax_gemma4_moe_profile": bool(args.ax_gemma4_moe_profile),
        "results": results,
    }
    if args.reuse_reference_results_from:
        doc["ax_only_refresh"] = {
            "method": "reuse_existing_reference_rows_and_rerun_ax_engine_rows",
            "reference_results_source": str(args.reuse_reference_results_from),
            "reference_rows_reused": len(
                [
                    cell
                    for cell in results
                    if cell.get("engine") in {"mlx_lm", "mlx_swift_lm"}
                ]
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
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
