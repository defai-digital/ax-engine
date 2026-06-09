#!/usr/bin/env python3
"""Benchmark Gemma 4 assistant-MTP on AX Engine.

This is the Gemma-specific companion to the Qwen3.6 fair MTP harness. Gemma 4
uses an assistant drafter model instead of a fused ``mtp.safetensors`` sidecar,
so this script prepares or locates ``ax_gemma4_assistant_mtp.json`` model
directories and runs AX Engine in both assistant-MTP-only and assistant
MTP+n-gram modes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import prepare_gemma4_assistant_mtp as prep_gemma

REPO_ROOT = SCRIPT_DIR.parents[0]
AX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "gemma4-assistant-mtp"
HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)

ENGINE_KEYS = {
    "direct": "ax_engine_mlx",
    "mtp": "ax_engine_gemma4_assistant_mtp",
    "mtp-ngram": "ax_engine_gemma4_assistant_mtp_ngram",
}

# PRD survival thresholds (PRD-2026-06-09-gemma4-12b-mtp-speedup R3/R4).
# Assistant-MTP keeps a default speed claim only if it beats same-artifact direct
# decode by this margin on the prompt-suite aggregate, with no suite regressing by
# more than MAX_SUITE_REGRESSION. MTP+n-gram keeps a default only if it beats pure
# assistant-MTP by NGRAM_KEEP_DEFAULT_MIN_GAIN under the same regression guard.
KEEP_DEFAULT_MIN_GAIN = 0.05
NGRAM_KEEP_DEFAULT_MIN_GAIN = 0.03
MAX_SUITE_REGRESSION = 0.03


@dataclass(frozen=True)
class Gemma4Profile:
    key: str
    label: str
    target_model_id: str
    assistant_model_id: str
    target_ref: str
    assistant_ref: str
    prepared_slug: str
    depth: int = 1


@dataclass(frozen=True)
class BenchProfile:
    key: str
    label: str
    mode: str
    env: dict[str, str]
    experimental: bool = False
    depth: int | None = None


BENCH_PROFILES = {
    "direct": BenchProfile(
        key="direct",
        label="direct decode (no assistant MTP / no n-gram)",
        mode="direct",
        env={},
    ),
    "assistant_mtp_default": BenchProfile(
        key="assistant_mtp_default",
        label="assistant MTP default",
        mode="mtp",
        env={},
    ),
    "assistant_mtp_ngram_default": BenchProfile(
        key="assistant_mtp_ngram_default",
        label="assistant MTP + n-gram default",
        mode="mtp-ngram",
        env={},
    ),
    "assistant_mtp_ngram_hurt_margin_015": BenchProfile(
        key="assistant_mtp_ngram_hurt_margin_015",
        label="assistant MTP + n-gram hurt margin 0.15",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_NGRAM_HURT_MARGIN": "0.15"},
    ),
    "assistant_mtp_ngram_ctx2_support1": BenchProfile(
        key="assistant_mtp_ngram_ctx2_support1",
        label="assistant MTP + n-gram ctx2 support1",
        mode="mtp-ngram",
        env={
            "AX_MLX_MTP_NGRAM_MIN_CONTEXT_LEN": "2",
            "AX_MLX_MTP_NGRAM_MIN_SUPPORT": "1",
        },
    ),
    "assistant_mtp_ngram_ctx4_support4": BenchProfile(
        key="assistant_mtp_ngram_ctx4_support4",
        label="assistant MTP + n-gram ctx4 support4",
        mode="mtp-ngram",
        env={
            "AX_MLX_MTP_NGRAM_MIN_CONTEXT_LEN": "4",
            "AX_MLX_MTP_NGRAM_MIN_SUPPORT": "4",
        },
    ),
    "assistant_mtp_ngram_confidence050": BenchProfile(
        key="assistant_mtp_ngram_confidence050",
        label="assistant MTP + n-gram confidence 0.50",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_NGRAM_CONFIDENCE_THRESHOLD": "0.50"},
    ),
    "assistant_mtp_ngram_safety_disable_all": BenchProfile(
        key="assistant_mtp_ngram_safety_disable_all",
        label="assistant MTP + n-gram safety disable all",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_NGRAM_SAFETY_MODE": "disable-all"},
    ),
    "assistant_mtp_gate095": BenchProfile(
        key="assistant_mtp_gate095",
        label="assistant MTP confidence gate 0.95",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE": "0.95"},
    ),
    "assistant_mtp_gate090": BenchProfile(
        key="assistant_mtp_gate090",
        label="assistant MTP confidence gate 0.90 (Qwen-style throughput-first)",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE": "0.90"},
    ),
    "assistant_mtp_gate085": BenchProfile(
        key="assistant_mtp_gate085",
        label="assistant MTP confidence gate 0.85 (hard-workload speculation)",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE": "0.85"},
    ),
    "assistant_mtp_deep099": BenchProfile(
        key="assistant_mtp_deep099",
        label="assistant MTP deep gate 0.99",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DEEP_DRAFT_MIN_CONFIDENCE": "0.99"},
    ),
    "assistant_mtp_deep095": BenchProfile(
        key="assistant_mtp_deep095",
        label="assistant MTP deep gate 0.95",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DEEP_DRAFT_MIN_CONFIDENCE": "0.95"},
    ),
    "assistant_mtp_depth1": BenchProfile(
        key="assistant_mtp_depth1",
        label="assistant MTP depth 1",
        mode="mtp",
        env={},
        depth=1,
    ),
    "assistant_mtp_depth2": BenchProfile(
        key="assistant_mtp_depth2",
        label="assistant MTP depth 2",
        mode="mtp",
        env={},
        depth=2,
    ),
    "assistant_mtp_gpu_confidence": BenchProfile(
        key="assistant_mtp_gpu_confidence",
        label="assistant MTP GPU-exact confidence",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_CONFIDENCE_MODE": "gpu-exact"},
        experimental=True,
    ),
    "assistant_mtp_ngram_gpu_confidence": BenchProfile(
        key="assistant_mtp_ngram_gpu_confidence",
        label="assistant MTP + n-gram GPU-exact confidence",
        mode="mtp-ngram",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_CONFIDENCE_MODE": "gpu-exact"},
        experimental=True,
    ),
    "target_softmax_topk256_experimental": BenchProfile(
        key="target_softmax_topk256_experimental",
        label="target softmax top-k 256 experimental",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_TARGET_SOFTMAX_MODE": "topk-256"},
        experimental=True,
    ),
    "combined_speed_experimental": BenchProfile(
        key="combined_speed_experimental",
        label="combined speed experimental",
        mode="mtp-ngram",
        env={
            "AX_MLX_MTP_NGRAM_HURT_MARGIN": "0.15",
            "AX_MLX_MTP_NGRAM_MIN_CONTEXT_LEN": "2",
            "AX_MLX_MTP_NGRAM_MIN_SUPPORT": "1",
            "AX_MLX_MTP_NGRAM_CONFIDENCE_THRESHOLD": "0.50",
            "AX_MLX_MTP_TARGET_SOFTMAX_MODE": "topk-256",
            "AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE": "0.90",
        },
        experimental=True,
    ),
    "utility_gate_candidate": BenchProfile(
        key="utility_gate_candidate",
        label="utility gate candidate",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_NGRAM_GATE_POLICY": "utility"},
    ),
}

DEFAULT_PROFILE_BY_MODE = {
    "direct": "direct",
    "mtp": "assistant_mtp_default",
    "mtp-ngram": "assistant_mtp_ngram_default",
}


GEMMA4_PROFILES = {
    "12b-4bit": Gemma4Profile(
        key="12b-4bit",
        label="Gemma 4 12B 4-bit",
        target_model_id="gemma-4-12b-it",
        assistant_model_id="gemma-4-12b-it-assistant",
        target_ref="mlx-community/gemma-4-12B-it-4bit",
        assistant_ref="mlx-community/gemma-4-12B-it-assistant-4bit",
        prepared_slug="models--ax-local--gemma-4-12b-it-4bit-assistant-mtp",
        depth=2,
    ),
    "12b-4bit-ffn4": Gemma4Profile(
        key="12b-4bit-ffn4",
        label="Gemma 4 12B 4-bit-FFN",
        target_model_id="gemma-4-12b-it",
        assistant_model_id="gemma-4-12b-it-assistant",
        target_ref="ax-local/gemma-4-12B-it-4bit-ffn4",
        assistant_ref="mlx-community/gemma-4-12B-it-assistant-4bit",
        prepared_slug="models--ax-local--gemma-4-12b-it-4bit-ffn4-assistant-mtp",
        depth=2,
    ),
    "26b-a4b-4bit": Gemma4Profile(
        key="26b-a4b-4bit",
        label="Gemma 4 26B A4B 4-bit",
        target_model_id="gemma-4-26b-a4b-it",
        assistant_model_id="gemma-4-26b-a4b-it-assistant",
        target_ref="mlx-community/gemma-4-26b-a4b-it-4bit",
        assistant_ref="google/gemma-4-26b-a4b-it-assistant",
        prepared_slug="models--ax-local--gemma-4-26b-a4b-it-assistant-mtp",
    ),
    "31b-4bit": Gemma4Profile(
        key="31b-4bit",
        label="Gemma 4 31B 4-bit",
        target_model_id="gemma-4-31b-it",
        assistant_model_id="gemma-4-31b-it-assistant",
        target_ref="mlx-community/gemma-4-31b-it-4bit",
        assistant_ref="google/gemma-4-31b-it-assistant",
        prepared_slug="models--ax-local--gemma-4-31b-it-assistant-mtp",
    ),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_name_path_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        result[name.strip()] = Path(path).expanduser()
    return result


def select_bench_profiles(*, modes: list[str], profile_keys: list[str]) -> list[BenchProfile]:
    if profile_keys:
        unknown = [key for key in profile_keys if key not in BENCH_PROFILES]
        if unknown:
            raise ValueError(f"unknown profile(s): {', '.join(unknown)}")
        return [BENCH_PROFILES[key] for key in profile_keys]

    unknown_modes = [mode for mode in modes if mode not in DEFAULT_PROFILE_BY_MODE]
    if unknown_modes:
        raise ValueError(f"unknown mode(s): {', '.join(unknown_modes)}")
    return [BENCH_PROFILES[DEFAULT_PROFILE_BY_MODE[mode]] for mode in modes]


def prepared_dir(profile: Gemma4Profile, hf_cache: Path = HF_CACHE) -> Path:
    return hf_cache / profile.prepared_slug / "snapshots" / "v1"


def resolve_model_dir(
    profile: Gemma4Profile,
    *,
    overrides: dict[str, Path],
    prepare_missing: bool,
    hf_cache: Path,
) -> Path:
    if profile.key in overrides:
        return overrides[profile.key].resolve()
    out_dir = prepared_dir(profile, hf_cache)
    if (out_dir / prep_gemma.CONTRACT_FILE).exists():
        return out_dir.resolve()
    if not prepare_missing:
        raise FileNotFoundError(
            f"{profile.key}: prepared Gemma4 assistant-MTP dir not found at {out_dir}. "
            "Run scripts/prepare_gemma4_assistant_mtp.py first or pass --model-dir."
        )
    return prep_gemma.prepare(
        target=profile.target_ref,
        assistant=profile.assistant_ref,
        target_model_id=profile.target_model_id,
        assistant_model_id=profile.assistant_model_id,
        output=out_dir,
        max_depth=profile.depth,
    )


def run_subprocess(
    cmd: list[str],
    *,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if env_overrides:
        env_text = " ".join(
            f"{key}={value}" for key, value in sorted(env_overrides.items())
        )
        print(f"  env: {env_text}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=f"See {log_path}\n\nLast log lines:\n{tail}",
        )
    print(f"  log: {log_path}", flush=True)


def build_ax_cmd(
    *,
    python: Path,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    mode: str,
    max_tokens: int,
    repetitions: int,
    cooldown: float,
    inter_case_cooldown: float,
    sampling: dict[str, Any],
    depth: int,
    no_build: bool,
) -> list[str]:
    """Build the bench_mlx_inference_stack invocation for one suite row.

    ``mode`` selects the decode policy on the *same* model package:

    - ``direct``: native MLX decode with no assistant drafter, no MTP, and no
      n-gram stacking. This is the same-artifact baseline the PRD survival test
      compares against; it must record ``ax_engine_mlx`` rows and zero assistant
      draft tokens.
    - ``mtp``: assistant-MTP with n-gram stacking disabled.
    - ``mtp-ngram``: assistant-MTP with n-gram stacking enabled.
    """
    cmd = [
        str(python),
        str(AX_BENCH_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--prompt-source",
        "real",
        "--real-prompt-suite",
        str(suite_file),
        "--generation-tokens",
        str(max_tokens),
        "--repetitions",
        str(repetitions),
        "--cooldown",
        str(cooldown),
        "--inter-case-cooldown",
        str(inter_case_cooldown),
        "--ax-sampling",
        json.dumps(sampling, sort_keys=True),
        "--skip-mlx-lm",
        "--capture-output-token-ids",
        "--output",
        str(output_path),
    ]
    if mode != "direct":
        cmd += ["--ax-gemma4-assistant-mtp", "--ax-mtp-max-depth", str(depth)]
        if mode == "mtp":
            cmd.append("--ax-mtp-disable-ngram-stacking")
    if no_build:
        cmd.append("--no-build-ax-engine")
    return cmd


def run_ax_suite(
    *,
    python: Path,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    mode: str,
    max_tokens: int,
    repetitions: int,
    cooldown: float,
    inter_case_cooldown: float,
    sampling: dict[str, Any],
    depth: int,
    no_build: bool,
    env_overrides: dict[str, str],
) -> Path:
    cmd = build_ax_cmd(
        python=python,
        suite_file=suite_file,
        output_path=output_path,
        model_dir=model_dir,
        mode=mode,
        max_tokens=max_tokens,
        repetitions=repetitions,
        cooldown=cooldown,
        inter_case_cooldown=inter_case_cooldown,
        sampling=sampling,
        depth=depth,
        no_build=no_build,
    )
    run_subprocess(
        cmd,
        log_path=output_path.with_suffix(".log"),
        env_overrides=env_overrides,
    )
    return output_path


def telemetry_int(telemetry: dict[str, Any], key: str) -> int:
    return int(telemetry.get(key, 0) or 0)


def stat_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def row_metric(row: dict[str, Any], key: str) -> float | None:
    metric = row.get(key)
    if isinstance(metric, dict) and metric.get("median") is not None:
        return float(metric["median"])
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def summarize_artifact(path: Path, *, expected_engine: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = [
        row for row in payload.get("results", []) if row.get("engine") == expected_engine
    ]
    if not rows:
        raise ValueError(f"{path}: no rows for expected engine {expected_engine}")

    decode = [v for row in rows if (v := row_metric(row, "decode_tok_s")) is not None]
    prefill = [v for row in rows if (v := row_metric(row, "prefill_tok_s")) is not None]
    ttft = [v for row in rows if (v := row_metric(row, "ttft_ms")) is not None]

    assistant_drafted = 0
    assistant_accepted = 0
    generic_mtp_drafted = 0
    generic_mtp_accepted = 0
    ngram_drafted = 0
    ngram_accepted = 0
    ngram_hit_steps = 0
    source_mtp_submitted = 0
    source_mtp_accepted = 0
    source_mtp_rejected = 0
    source_mtp_cascade_rejected = 0
    source_assistant_submitted = 0
    source_assistant_accepted = 0
    source_assistant_rejected = 0
    source_assistant_cascade_rejected = 0
    ngram_proposed = 0
    ngram_rejected = 0
    ngram_cascade_rejected = 0
    ngram_utility_gated_steps = 0
    ngram_utility_insufficient_sample_steps = 0
    ngram_safety_disabled_steps = 0
    ngram_safety_tightened_steps = 0
    ngram_lookup_wall_us = 0
    target_softmax_wall_us = 0
    verify_tokens = 0
    emitted_tokens = 0
    utility_baseline_costs: list[int] = []
    utility_stacked_costs: list[int] = []
    gate_policies: list[int] = []
    safety_reasons: list[int] = []
    assistant_confidence_modes: list[int] = []
    claim_statuses: list[str] = []
    effective_routes: list[str] = []
    affine_min_bits: int | None = None
    affine_max_bits: int | None = None
    affine_tensor_count = 0
    affine_4bit_count = 0
    affine_8bit_count = 0
    for row in rows:
        # Affine bit summary is constant per model load (set at startup, not
        # accumulated per step), so max-merge counts and max bits, min-merge min
        # bits across the trial rows. These prove which target package actually
        # ran and gate the same-artifact parity check (PRD R2).
        mlx_telemetry = row.get("ax_mlx_telemetry") or {}
        if "ax_mlx_affine_min_bits" in mlx_telemetry:
            bits = telemetry_int(mlx_telemetry, "ax_mlx_affine_min_bits")
            affine_min_bits = bits if affine_min_bits is None else min(affine_min_bits, bits)
        if "ax_mlx_affine_max_bits" in mlx_telemetry:
            bits = telemetry_int(mlx_telemetry, "ax_mlx_affine_max_bits")
            affine_max_bits = bits if affine_max_bits is None else max(affine_max_bits, bits)
        affine_tensor_count = max(
            affine_tensor_count, telemetry_int(mlx_telemetry, "ax_mlx_affine_tensor_count")
        )
        affine_4bit_count = max(
            affine_4bit_count, telemetry_int(mlx_telemetry, "ax_mlx_affine_4bit_count")
        )
        affine_8bit_count = max(
            affine_8bit_count, telemetry_int(mlx_telemetry, "ax_mlx_affine_8bit_count")
        )
        assistant = row.get("ax_mlx_gemma4_assistant_mtp") or {}
        assistant_drafted += int(
            assistant.get("ax_mlx_gemma4_assistant_mtp_draft_tokens", 0) or 0
        )
        assistant_accepted += int(
            assistant.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens", 0) or 0
        )
        if assistant.get("ax_mlx_gemma4_assistant_mtp_confidence_mode") is not None:
            assistant_confidence_modes.append(
                telemetry_int(assistant, "ax_mlx_gemma4_assistant_mtp_confidence_mode")
            )
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        generic_mtp_drafted += telemetry_int(telemetry, "ax_mtp_draft_tokens")
        generic_mtp_accepted += telemetry_int(telemetry, "ax_mtp_accepted_tokens")
        mtp_ngram_drafted = telemetry_int(telemetry, "ax_mtp_ngram_submitted_tokens")
        mtp_ngram_accepted = telemetry_int(telemetry, "ax_mtp_ngram_accepted_tokens")
        if mtp_ngram_accepted == 0:
            mtp_ngram_accepted = telemetry_int(
                telemetry,
                "ax_mtp_ngram_submitted_accepted_tokens",
            )
        row_ngram_hit_steps = telemetry_int(telemetry, "ax_mtp_ngram_hit_steps")
        if mtp_ngram_drafted > 0 or mtp_ngram_accepted > 0 or row_ngram_hit_steps > 0:
            ngram_drafted += mtp_ngram_drafted
            ngram_accepted += mtp_ngram_accepted
        else:
            ngram_drafted += telemetry_int(telemetry, "ax_ngram_draft_tokens")
            ngram_accepted += telemetry_int(telemetry, "ax_ngram_accepted_tokens")
        ngram_hit_steps += row_ngram_hit_steps
        source_mtp_submitted += telemetry_int(telemetry, "ax_mtp_source_mtp_submitted_tokens")
        source_mtp_accepted += telemetry_int(telemetry, "ax_mtp_source_mtp_accepted_tokens")
        source_mtp_rejected += telemetry_int(telemetry, "ax_mtp_source_mtp_rejected_tokens")
        source_mtp_cascade_rejected += telemetry_int(
            telemetry,
            "ax_mtp_source_mtp_cascade_rejected_tokens",
        )
        source_assistant_submitted += telemetry_int(
            telemetry,
            "ax_mtp_source_assistant_submitted_tokens",
        )
        source_assistant_accepted += telemetry_int(
            telemetry,
            "ax_mtp_source_assistant_accepted_tokens",
        )
        source_assistant_rejected += telemetry_int(
            telemetry,
            "ax_mtp_source_assistant_rejected_tokens",
        )
        source_assistant_cascade_rejected += telemetry_int(
            telemetry,
            "ax_mtp_source_assistant_cascade_rejected_tokens",
        )
        ngram_proposed += telemetry_int(telemetry, "ax_mtp_ngram_proposed_tokens")
        ngram_rejected += telemetry_int(telemetry, "ax_mtp_ngram_rejected_tokens")
        ngram_cascade_rejected += telemetry_int(
            telemetry,
            "ax_mtp_ngram_cascade_rejected_tokens",
        )
        ngram_utility_gated_steps += telemetry_int(
            telemetry,
            "ax_mtp_ngram_utility_gated_steps",
        )
        ngram_utility_insufficient_sample_steps += telemetry_int(
            telemetry,
            "ax_mtp_ngram_utility_insufficient_sample_steps",
        )
        ngram_safety_disabled_steps += telemetry_int(
            telemetry,
            "ax_mtp_ngram_safety_disabled_steps",
        )
        ngram_safety_tightened_steps += telemetry_int(
            telemetry,
            "ax_mtp_ngram_safety_tightened_steps",
        )
        ngram_lookup_wall_us += telemetry_int(telemetry, "ax_mtp_ngram_lookup_wall_us")
        target_softmax_wall_us += telemetry_int(telemetry, "ax_mtp_target_softmax_wall_us")
        verify_tokens += telemetry_int(telemetry, "ax_mtp_verify_tokens")
        emitted_tokens += telemetry_int(telemetry, "ax_mtp_emitted_tokens")
        utility_baseline_costs.append(
            telemetry_int(
                telemetry,
                "ax_mtp_ngram_utility_baseline_cost_per_emitted_token_us",
            )
        )
        utility_stacked_costs.append(
            telemetry_int(
                telemetry,
                "ax_mtp_ngram_utility_stacked_cost_per_emitted_token_us",
            )
        )
        gate_policies.append(telemetry_int(telemetry, "ax_mtp_ngram_gate_policy"))
        safety_reasons.append(telemetry_int(telemetry, "ax_mtp_ngram_safety_reason"))
        if row.get("ax_decode_claim_status"):
            claim_statuses.append(str(row["ax_decode_claim_status"]))
        if row.get("ax_decode_effective_route"):
            effective_routes.append(str(row["ax_decode_effective_route"]))

    return {
        "artifact": str(path),
        "engine": expected_engine,
        "case_count": len(rows),
        "decode_tok_s_median": stat_median(decode),
        "prefill_tok_s_median": stat_median(prefill),
        "ttft_ms_median": stat_median(ttft),
        "assistant_draft_tokens": assistant_drafted,
        "assistant_accepted_tokens": assistant_accepted,
        "assistant_accept_rate": (
            assistant_accepted / assistant_drafted if assistant_drafted > 0 else None
        ),
        "mtp_draft_tokens": generic_mtp_drafted,
        "mtp_accepted_tokens": generic_mtp_accepted,
        "mtp_accept_rate": (
            generic_mtp_accepted / generic_mtp_drafted
            if generic_mtp_drafted > 0
            else None
        ),
        "ngram_draft_tokens": ngram_drafted,
        "ngram_accepted_tokens": ngram_accepted,
        "ngram_accept_rate": ngram_accepted / ngram_drafted if ngram_drafted > 0 else None,
        "ngram_hit_steps": ngram_hit_steps,
        "source_mtp_submitted_tokens": source_mtp_submitted,
        "source_mtp_accepted_tokens": source_mtp_accepted,
        "source_mtp_rejected_tokens": source_mtp_rejected,
        "source_mtp_cascade_rejected_tokens": source_mtp_cascade_rejected,
        "source_assistant_submitted_tokens": source_assistant_submitted,
        "source_assistant_accepted_tokens": source_assistant_accepted,
        "source_assistant_rejected_tokens": source_assistant_rejected,
        "source_assistant_cascade_rejected_tokens": source_assistant_cascade_rejected,
        "ngram_proposed_tokens": ngram_proposed,
        "ngram_rejected_tokens": ngram_rejected,
        "ngram_cascade_rejected_tokens": ngram_cascade_rejected,
        "ngram_utility_gated_steps": ngram_utility_gated_steps,
        "ngram_utility_insufficient_sample_steps": ngram_utility_insufficient_sample_steps,
        "ngram_safety_disabled_steps": ngram_safety_disabled_steps,
        "ngram_safety_tightened_steps": ngram_safety_tightened_steps,
        "ngram_lookup_wall_us": ngram_lookup_wall_us,
        "target_softmax_wall_us": target_softmax_wall_us,
        "verify_tokens": verify_tokens,
        "emitted_tokens": emitted_tokens,
        "utility_baseline_cost_per_emitted_token_us_median": stat_median(
            [float(v) for v in utility_baseline_costs if v > 0]
        ),
        "utility_stacked_cost_per_emitted_token_us_median": stat_median(
            [float(v) for v in utility_stacked_costs if v > 0]
        ),
        "gate_policies": sorted(set(gate_policies)),
        "safety_reasons": sorted(set(safety_reasons)),
        "assistant_confidence_modes": sorted(set(assistant_confidence_modes)),
        "claim_statuses": sorted(set(claim_statuses)),
        "effective_routes": sorted(set(effective_routes)),
        "affine_tensor_count": affine_tensor_count,
        "affine_min_bits": affine_min_bits,
        "affine_max_bits": affine_max_bits,
        "affine_4bit_count": affine_4bit_count,
        "affine_8bit_count": affine_8bit_count,
        "build": payload.get("build", {}),
    }


def classify_vs_direct(
    delta: float | None,
    worst_suite_delta: float | None,
    parity: bool,
    drafted: bool,
) -> str:
    """Classify an assistant-MTP profile against same-artifact direct decode.

    Implements the PRD-2026-06-09 R3 survival criterion. ``delta`` is the
    aggregate decode tok/s gain over direct; ``worst_suite_delta`` is the
    most negative per-suite delta. Returns one of the PRD R6 statuses.
    """
    if delta is None:
        return "retest"
    if not parity:
        # A mixed-artifact comparison cannot decide MTP viability (PRD R2).
        return "retest"
    if not drafted:
        # Route metadata proves the decode policy never drafted (PRD R3).
        return "reject"
    no_bad_regression = worst_suite_delta is None or worst_suite_delta >= -MAX_SUITE_REGRESSION
    if delta >= KEEP_DEFAULT_MIN_GAIN and no_bad_regression:
        return "keep-default"
    if delta >= 0:
        return "keep-opt-in"
    return "remove-claim"


def classify_ngram_vs_mtp(
    delta: float | None,
    worst_suite_delta: float | None,
) -> str:
    """Classify an MTP+n-gram profile against pure assistant-MTP (PRD R4)."""
    if delta is None:
        return "retest"
    no_bad_regression = worst_suite_delta is None or worst_suite_delta >= -MAX_SUITE_REGRESSION
    if delta >= NGRAM_KEEP_DEFAULT_MIN_GAIN and no_bad_regression:
        return "keep-default"
    if delta >= 0:
        return "keep-opt-in"
    return "remove-claim"


def compare_suite_decodes(
    profile_suites: dict[str, float | None],
    baseline_suites: dict[str, float | None],
) -> dict[str, Any]:
    """Compare two profiles' per-suite decode medians over their shared suites.

    The aggregate delta uses the median of each side over the common suites; the
    worst-suite delta drives the regression guard.
    """
    common = sorted(
        suite
        for suite, value in profile_suites.items()
        if suite in baseline_suites
        and value is not None
        and baseline_suites[suite] is not None
    )
    if not common:
        return {
            "delta": None,
            "worst_suite_delta": None,
            "per_suite_delta": {},
            "profile_agg": None,
            "baseline_agg": None,
            "suites": [],
        }
    per_suite: dict[str, float | None] = {}
    for suite in common:
        base = baseline_suites[suite]
        per_suite[suite] = (profile_suites[suite] - base) / base if base else None
    profile_agg = statistics.median([profile_suites[suite] for suite in common])
    baseline_agg = statistics.median([baseline_suites[suite] for suite in common])
    delta = (profile_agg - baseline_agg) / baseline_agg if baseline_agg else None
    valid_deltas = [value for value in per_suite.values() if value is not None]
    worst = min(valid_deltas) if valid_deltas else None
    return {
        "delta": delta,
        "worst_suite_delta": worst,
        "per_suite_delta": per_suite,
        "profile_agg": profile_agg,
        "baseline_agg": baseline_agg,
        "suites": common,
    }


def _index_rows_by_profile(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        model = row["model"]
        profile = row["profile"]
        profiles = index.setdefault(model, {})
        entry = profiles.setdefault(
            profile,
            {
                "mode": row["mode"],
                "suites": {},
                "affine_max_bits": None,
                "affine_8bit_count": None,
                "affine_4bit_count": None,
                "affine_tensor_count": None,
                "drafted": False,
                "model_dir": row.get("model_dir"),
            },
        )
        entry["suites"][row["suite"]] = row.get("decode_tok_s_median")
        if (row.get("assistant_draft_tokens") or 0) > 0:
            entry["drafted"] = True
        for key in (
            "affine_max_bits",
            "affine_8bit_count",
            "affine_4bit_count",
            "affine_tensor_count",
        ):
            value = row.get(key)
            if value is not None:
                current = entry[key]
                entry[key] = value if current is None else max(current, value)
    return index


def _select_mtp_baseline(profiles: dict[str, dict[str, Any]]) -> str | None:
    if "assistant_mtp_default" in profiles:
        return "assistant_mtp_default"
    for key in sorted(profiles):
        if profiles[key]["mode"] == "mtp":
            return key
    return None


def build_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build same-artifact survival comparisons from summary rows.

    Produces, per model: each non-direct profile vs same-artifact direct decode,
    and each MTP+n-gram profile vs pure assistant-MTP. Flags artifact-parity
    mismatches (PRD R2) and missing draft participation (PRD R3) as warnings, and
    classifies every comparison with the PRD R6 taxonomy.
    """
    index = _index_rows_by_profile(rows)
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    parity_ok = True
    for model in sorted(index):
        profiles = index[model]
        direct = next(
            (entry for entry in profiles.values() if entry["mode"] == "direct"), None
        )
        mtp_baseline_key = _select_mtp_baseline(profiles)
        if direct is None:
            warnings.append(
                f"{model}: no direct-decode row; assistant-MTP cannot be judged "
                "against same-artifact direct decode (PRD R1/R3)."
            )
        for profile_key in sorted(profiles):
            entry = profiles[profile_key]
            if entry["mode"] == "direct":
                continue
            if direct is not None:
                result = compare_suite_decodes(entry["suites"], direct["suites"])
                parity = (entry["affine_max_bits"], entry["affine_8bit_count"]) == (
                    direct["affine_max_bits"],
                    direct["affine_8bit_count"],
                )
                if not parity:
                    parity_ok = False
                    warnings.append(
                        f"{model}/{profile_key}: artifact parity mismatch vs direct "
                        f"(direct max_bits={direct['affine_max_bits']} "
                        f"8bit={direct['affine_8bit_count']}, profile "
                        f"max_bits={entry['affine_max_bits']} "
                        f"8bit={entry['affine_8bit_count']}); same-artifact survival "
                        "verdict is not valid (PRD R2)."
                    )
                if not entry["drafted"]:
                    warnings.append(
                        f"{model}/{profile_key}: route metadata shows zero assistant "
                        "draft tokens; decode policy did not actually run (PRD R3)."
                    )
                comparisons.append(
                    {
                        "model": model,
                        "profile": profile_key,
                        "mode": entry["mode"],
                        "baseline": "direct",
                        "decode_tok_s_agg": result["profile_agg"],
                        "baseline_tok_s_agg": result["baseline_agg"],
                        "delta_vs_baseline": result["delta"],
                        "worst_suite_delta": result["worst_suite_delta"],
                        "per_suite_delta": result["per_suite_delta"],
                        "parity_ok": parity,
                        "drafted": entry["drafted"],
                        "affine_max_bits": entry["affine_max_bits"],
                        "affine_8bit_count": entry["affine_8bit_count"],
                        "classification": classify_vs_direct(
                            result["delta"],
                            result["worst_suite_delta"],
                            parity,
                            entry["drafted"],
                        ),
                    }
                )
            if (
                entry["mode"] == "mtp-ngram"
                and mtp_baseline_key is not None
                and mtp_baseline_key != profile_key
            ):
                base = profiles[mtp_baseline_key]
                result = compare_suite_decodes(entry["suites"], base["suites"])
                comparisons.append(
                    {
                        "model": model,
                        "profile": profile_key,
                        "mode": entry["mode"],
                        "baseline": "assistant_mtp",
                        "baseline_profile": mtp_baseline_key,
                        "decode_tok_s_agg": result["profile_agg"],
                        "baseline_tok_s_agg": result["baseline_agg"],
                        "delta_vs_baseline": result["delta"],
                        "worst_suite_delta": result["worst_suite_delta"],
                        "per_suite_delta": result["per_suite_delta"],
                        "classification": classify_ngram_vs_mtp(
                            result["delta"], result["worst_suite_delta"]
                        ),
                    }
                )
    return {"parity_ok": parity_ok, "warnings": warnings, "comparisons": comparisons}


def fmt_number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def fmt_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def fmt_bits(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Gemma 4 Assistant MTP Benchmark",
        "",
        f"Output: `{output_dir}`",
        "",
        "| Model | Suite | Profile | Mode | Depth | Decode tok/s | Affine max-bits | 8-bit tensors | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {model} | {suite} | {profile} | {mode} | {depth} | {decode} | {max_bits} | {eightbit} | {assistant} | {mtp} | {ngram} | {hits} | {utility_gates} | {safety_tightens} |".format(
                model=row["model_label"],
                suite=row["suite"],
                profile=row["profile"],
                mode=row["mode"],
                depth=row["depth"],
                decode=fmt_number(row["decode_tok_s_median"]),
                max_bits=fmt_bits(row.get("affine_max_bits")),
                eightbit=fmt_bits(row.get("affine_8bit_count")),
                assistant=fmt_pct(row["assistant_accept_rate"]),
                mtp=fmt_pct(row["mtp_accept_rate"]),
                ngram=fmt_pct(row["ngram_accept_rate"]),
                hits=row["ngram_hit_steps"],
                utility_gates=row["ngram_utility_gated_steps"],
                safety_tightens=row["ngram_safety_tightened_steps"],
            )
        )

    comparison = summary.get("comparison") or {}
    comparisons = comparison.get("comparisons") or []
    if comparisons:
        parity_note = (
            "All compared profiles share the direct-baseline target artifact."
            if comparison.get("parity_ok")
            else "**Artifact parity mismatch detected — survival verdicts marked `retest`.**"
        )
        lines += [
            "",
            "## Same-artifact survival comparison",
            "",
            parity_note,
            "",
            "| Model | Profile | Mode | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Parity | Drafted | Classification |",
            "|---|---|---|---|---:|---:|---:|---:|:---:|:---:|---|",
        ]
        for entry in comparisons:
            lines.append(
                "| {model} | {profile} | {mode} | {baseline} | {decode} | {baseline_decode} | {delta} | {worst} | {parity} | {drafted} | {cls} |".format(
                    model=entry["model"],
                    profile=entry["profile"],
                    mode=entry["mode"],
                    baseline=entry.get("baseline_profile") or entry["baseline"],
                    decode=fmt_number(entry.get("decode_tok_s_agg")),
                    baseline_decode=fmt_number(entry.get("baseline_tok_s_agg")),
                    delta=fmt_delta(entry.get("delta_vs_baseline")),
                    worst=fmt_delta(entry.get("worst_suite_delta")),
                    parity="—" if "parity_ok" not in entry else ("yes" if entry["parity_ok"] else "NO"),
                    drafted="—" if "drafted" not in entry else ("yes" if entry["drafted"] else "NO"),
                    cls=entry["classification"],
                )
            )
        warnings = comparison.get("warnings") or []
        if warnings:
            lines += ["", "### Warnings", ""]
            lines += [f"- {warning}" for warning in warnings]

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="26b-a4b-4bit,31b-4bit",
        help=f"Comma-separated model keys. Choices: {', '.join(GEMMA4_PROFILES)}",
    )
    parser.add_argument(
        "--modes",
        default="mtp,mtp-ngram",
        help="Comma-separated legacy modes used when --profiles is omitted: mtp, mtp-ngram.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help=f"Comma-separated benchmark profiles. Choices: {', '.join(BENCH_PROFILES)}",
    )
    parser.add_argument(
        "--suites",
        default="flappy,long_code,python_modules_long",
        help="Comma-separated prompt suites under benchmarks/prompts/mtp-suites.",
    )
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Prepared model override as MODEL_KEY=PATH. May be repeated.",
    )
    parser.add_argument("--hf-cache", type=Path, default=HF_CACHE)
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=30.0)
    parser.add_argument("--inter-case-cooldown", type=float, default=10.0)
    parser.add_argument(
        "--sampling",
        type=json.loads,
        default={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    )
    parser.add_argument("--no-build-ax-engine", action="store_true")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override the assistant draft depth for every profile (default: each "
        "profile's own depth). The runtime caps this by the prepared bundle's "
        "contract max_depth, so the bundle must be prepared with --max-depth >= "
        "this value for it to take effect.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-suite artifacts instead of rerunning them.",
    )
    args = parser.parse_args()
    if args.depth is not None and args.depth < 1:
        parser.error("--depth must be >= 1")

    model_keys = parse_csv(args.models)
    modes = parse_csv(args.modes)
    profile_keys = parse_csv(args.profiles)
    suites = parse_csv(args.suites)
    overrides = parse_name_path_map(args.model_dir)
    for key in model_keys:
        if key not in GEMMA4_PROFILES:
            parser.error(f"unknown model key {key!r}")
    try:
        bench_profiles = select_bench_profiles(modes=modes, profile_keys=profile_keys)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or DEFAULT_OUTPUT_BASE / (
        f"{date.today().isoformat()}-gemma4-assistant-mtp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_key in model_keys:
        profile = GEMMA4_PROFILES[model_key]
        effective_depth = args.depth if args.depth is not None else profile.depth
        model_dir = resolve_model_dir(
            profile,
            overrides=overrides,
            prepare_missing=not args.no_prepare,
            hf_cache=args.hf_cache,
        )
        for suite in suites:
            suite_file = args.suites_dir / f"{suite}.jsonl"
            if not suite_file.exists():
                raise FileNotFoundError(f"missing prompt suite {suite_file}")
            for bench_profile in bench_profiles:
                mode = bench_profile.mode
                row_depth = bench_profile.depth or effective_depth
                artifact_path = output_dir / model_key / suite / f"{bench_profile.key}.json"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                if args.resume and artifact_path.exists():
                    print(f"[resume] using existing {artifact_path}", flush=True)
                else:
                    run_ax_suite(
                        python=args.python,
                        suite_file=suite_file,
                        output_path=artifact_path,
                        model_dir=model_dir,
                        mode=mode,
                        max_tokens=args.max_tokens,
                        repetitions=args.repetitions,
                        cooldown=args.cooldown,
                        inter_case_cooldown=args.inter_case_cooldown,
                        sampling=args.sampling,
                        depth=row_depth,
                        no_build=args.no_build_ax_engine,
                        env_overrides=bench_profile.env,
                    )
                row = summarize_artifact(artifact_path, expected_engine=ENGINE_KEYS[mode])
                row.update(
                    {
                        "model": model_key,
                        "model_label": profile.label,
                        "suite": suite,
                        "mode": mode,
                        "profile": bench_profile.key,
                        "profile_label": bench_profile.label,
                        "profile_experimental": bench_profile.experimental,
                        "profile_env": bench_profile.env,
                        "depth": row_depth,
                        "model_dir": str(model_dir),
                    }
                )
                rows.append(row)

    comparison = build_comparisons(rows)
    summary = {
        "schema": "ax.gemma4_assistant_mtp_benchmark.v2",
        "models": model_keys,
        "modes": modes,
        "profiles": [profile.key for profile in bench_profiles],
        "suites": suites,
        "sampling": args.sampling,
        "max_tokens": args.max_tokens,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "inter_case_cooldown": args.inter_case_cooldown,
        "rows": rows,
        "comparison": comparison,
    }
    write_summary(output_dir, summary)
    for warning in comparison["warnings"]:
        print(f"[warn] {warning}", file=sys.stderr, flush=True)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
