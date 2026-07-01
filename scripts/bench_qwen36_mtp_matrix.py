#!/usr/bin/env python3
"""Plan, run, and summarize the Qwen3.6 MTP-only benchmark matrix."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
MTPLX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mtplx_prompt_suites.py"
RAPID_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_rapid_mlx_prompt_suites.py"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "mtp-qwen36-matrix"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
DEFAULT_MTPLX_PYTHON = Path("/opt/homebrew/var/mtplx/venv-1.0.4/bin/python")
DEFAULT_RAPID_PYTHON = Path("/opt/homebrew/var/mtplx/venv-1.0.4/bin/python")
DEFAULT_RAPID_SOURCE = REPO_ROOT / ".internal" / "reference" / "Rapid-MLX"
DEFAULT_LIGHTNING_SOURCE = REPO_ROOT / ".internal" / "reference" / "lightning-mlx"
DEFAULT_MTPLX_SOURCE = REPO_ROOT / ".internal" / "reference" / "MTPLX"
HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)
DEFAULT_PEER_CACHES = tuple(
    path
    for path in (HF_CACHE, Path("/Volumes/Ext4T/models/hub"))
    if path.exists()
)

DEFAULT_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
DEFAULT_SUITES = ("flappy", "long_code", "python_modules_long")
ENGINES = ("ax_engine", "mtplx", "lightning_mlx", "rapid_mlx", "omlx")
MODELS = ("27b", "35b-a3b")
BITS = (4, 6)
BENCHMARK_CONTRACTS = ("apples-to-apples", "peer-optimized", "lightning-optimized")
PEER_OPTIMIZED_DEFAULTS = {
    "max_tokens": 512,
    "prompt_limit": 3,
    "prefill_step_size": 8192,
    "mtplx_profile": "performance-cold",
}
AX_MTP_ENGINES = {"ax_engine_mlx_ngram_accel", "ax_engine_mlx_pure_mtp"}

# Degeneracy gate: reject runs where a short repeating cycle dominates output.
DEGENERACY_MAX_CYCLE_LEN = 8
DEGENERACY_COVERAGE_THRESHOLD = 0.50  # 50% of tokens in a repeating cycle = degenerate
DEGENERACY_PERIODIC_COVERAGE_THRESHOLD = 0.45

# MTP head provenance — tracks which artifact each engine loads.
MTP_HEAD_PROVENANCE: dict[str, dict[str, str]] = {
    "ax_engine": {
        "source": "Qwen/Qwen3.6-27B (official BF16 mtp.* tensors)",
        "packaging": "ax-local/Qwen3.6-27B-MTP sidecar",
        "mtp_precision": "bf16 (extracted with RMSNorm +1.0 delta correction)",
        "draft_lm_head": "bf16 (matching base)",
    },
    "mtplx": {
        "source": "Qwen/Qwen3.6-27B (re-quantized by Youssofal)",
        "packaging": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        "mtp_precision": "INT4 prequantized sidecar (mtp/weights.safetensors)",
        "draft_lm_head": "3-bit affine, group_size=64",
    },
    "lightning_mlx": {
        "source": "Qwen/Qwen3.6-27B (re-quantized by Youssofal)",
        "packaging": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        "mtp_precision": "INT4 prequantized sidecar (mtp/weights.safetensors)",
        "draft_lm_head": "3-bit affine, group_size=64",
    },
}
NGRAM_ZERO_KEYS = (
    "ax_ngram_accepted_tokens",
    "ax_ngram_draft_tokens",
    "ax_ngram_rejected_tokens",
    "ax_mtp_ngram_accepted_tokens",
    "ax_mtp_ngram_proposed_tokens",
    "ax_mtp_ngram_submitted_tokens",
    "ax_mtp_ngram_submitted_accepted_tokens",
    "ax_mtp_ngram_hit_steps",
    "ax_mtp_ngram_attempt_steps",
)


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    model: str
    bits: int
    ax_model_dir: Path
    mtp_depth: int
    mtp_head_id: str = ""


@dataclass(frozen=True)
class Lane:
    target: Target
    engine: str
    suite: str
    status: str
    output_path: Path
    command: list[str] | None
    reason: str | None = None
    metric_contract: dict[str, str] | None = None


TARGETS: dict[str, Target] = {
    "27b-4bit": Target(
        key="27b-4bit",
        label="Qwen3.6 27B 4-bit",
        model="27b",
        bits=4,
        ax_model_dir=HF_CACHE / "models--ax-local--Qwen3.6-27B-MTP" / "snapshots" / "v1",
        mtp_depth=3,
        mtp_head_id="ax-local/Qwen3.6-27B-MTP (BF16 from Qwen/Qwen3.6-27B)",
    ),
    "27b-6bit": Target(
        key="27b-6bit",
        label="Qwen3.6 27B 6-bit",
        model="27b",
        bits=6,
        ax_model_dir=HF_CACHE
        / "models--ax-local--mlx-community--Qwen3.6-27B-6bit-MTP"
        / "snapshots"
        / "v1",
        mtp_depth=3,
        mtp_head_id="ax-local/Qwen3.6-27B-6bit-MTP",
    ),
    "35b-a3b-4bit": Target(
        key="35b-a3b-4bit",
        label="Qwen3.6 35B-A3B 4-bit",
        model="35b-a3b",
        bits=4,
        ax_model_dir=HF_CACHE / "models--ax-local--Qwen3.6-35B-MTP" / "snapshots" / "v1",
        mtp_depth=1,
        mtp_head_id="ax-local/Qwen3.6-35B-MTP (BF16 from Qwen/Qwen3.6-35B-A3B)",
    ),
    "35b-a3b-6bit": Target(
        key="35b-a3b-6bit",
        label="Qwen3.6 35B-A3B 6-bit",
        model="35b-a3b",
        bits=6,
        ax_model_dir=HF_CACHE
        / "models--ax-local--mlx-community--Qwen3.6-35B-A3B-6bit-MTP"
        / "snapshots"
        / "v1",
        mtp_depth=1,
        mtp_head_id="ax-local/Qwen3.6-35B-A3B-6bit-MTP",
    ),
}

MTPLX_MODEL_IDS = {
    "27b-4bit": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "35b-a3b-4bit": "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
    "35b-a3b-6bit": "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
}

MTPLX_QUANT_POLICIES = {
    "27b-4bit": "prequantized-int4",
}

LIGHTNING_MODELS = dict(MTPLX_MODEL_IDS)
RAPID_MODELS: dict[str, str] = {}


def detect_degenerate_output(
    token_ids: list[int],
    max_cycle_len: int = DEGENERACY_MAX_CYCLE_LEN,
    coverage_threshold: float = DEGENERACY_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Check whether output collapses into a short repeating token cycle.

    Scans cycle lengths 1..max_cycle_len and tries to find a repeating pattern
    that covers >= coverage_threshold of the output.  Works by sliding a window
    across the sequence and counting consecutive cycle matches.  A run with
    750+ consecutive whitespace tokens (e.g. ``248045, 554, 248046, 198``) will
    be flagged.
    """
    n = len(token_ids)
    if n < max_cycle_len * 2:
        return {"is_degenerate": False, "cycle": None, "coverage": 0.0}
    for cycle_len in range(1, max_cycle_len + 1):
        best_coverage = 0.0
        best_cycle: list[int] | None = None
        # Try every candidate window across the full sequence.
        for start in range(n - cycle_len):
            candidate = token_ids[start : start + cycle_len]
            # Count consecutive forward matches from this position.
            matches = 0
            pos = start
            while pos + cycle_len <= n:
                if token_ids[pos : pos + cycle_len] == candidate:
                    matches += 1
                    pos += cycle_len
                else:
                    break
            coverage = (matches * cycle_len) / n
            if coverage > best_coverage:
                best_coverage = coverage
                best_cycle = candidate
            if best_coverage >= coverage_threshold:
                return {
                    "is_degenerate": True,
                    "cycle": best_cycle,
                    "cycle_length": cycle_len,
                    "method": "consecutive_cycle",
                    "coverage": round(best_coverage, 4),
                }
    periodic = detect_periodic_degenerate_output(token_ids, max_cycle_len=max_cycle_len)
    if periodic["is_degenerate"]:
        return periodic
    return {"is_degenerate": False, "cycle": None, "coverage": 0.0}


def detect_periodic_degenerate_output(
    token_ids: list[int],
    max_cycle_len: int = DEGENERACY_MAX_CYCLE_LEN,
    coverage_threshold: float = DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    n = len(token_ids)
    if n < max_cycle_len * 2:
        return {"is_degenerate": False, "cycle": None, "coverage": 0.0}
    best: tuple[float, int, list[int]] = (0.0, 0, [])
    for cycle_len in range(1, max_cycle_len + 1):
        cycle: list[int] = []
        covered = 0
        for phase in range(cycle_len):
            phase_tokens = token_ids[phase:n:cycle_len]
            if not phase_tokens:
                continue
            token, count = collections.Counter(phase_tokens).most_common(1)[0]
            cycle.append(int(token))
            covered += int(count)
        coverage = covered / n
        if coverage > best[0]:
            best = (coverage, cycle_len, cycle)
    if best[0] >= coverage_threshold:
        return {
            "is_degenerate": True,
            "cycle": best[2],
            "cycle_length": best[1],
            "method": "periodic_cycle",
            "coverage": round(best[0], 4),
        }
    return {"is_degenerate": False, "cycle": None, "coverage": 0.0}


def check_ax_output_degeneracy(artifact: dict[str, Any]) -> dict[str, Any]:
    """Run the degeneracy gate on all AX output trials.

    Returns a summary dict with per-case degeneracy results and an overall
    ``degenerate`` flag that is True if any case fails the gate.
    """
    cases: list[dict[str, Any]] = []
    any_degenerate = False
    for row in artifact.get("results", []):
        if row.get("engine") not in AX_MTP_ENGINES:
            continue
        for trial in row.get("trials", []):
            token_ids = trial.get("output_token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                continue
            result = detect_degenerate_output(token_ids)
            result["prompt_case_id"] = row.get("prompt_case_id")
            result["token_count"] = len(token_ids)
            cases.append(result)
            if result["is_degenerate"]:
                any_degenerate = True
    return {
        "degenerate": any_degenerate,
        "gate": "output_entropy_cycle_coverage",
        "threshold": DEGENERACY_COVERAGE_THRESHOLD,
        "periodic_threshold": DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
        "max_cycle_len": DEGENERACY_MAX_CYCLE_LEN,
        "cases": cases,
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def numeric_samples(value: Any) -> list[float]:
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list):
            return [float(v) for v in values if isinstance(v, int | float)]
        med = value.get("median")
        return [float(med)] if isinstance(med, int | float) else []
    if isinstance(value, int | float):
        return [float(value)]
    return []


def metric_median(row: dict[str, Any], metric: str) -> float | None:
    return median(numeric_samples(row.get(metric)))


def shared_int_value(rows: list[dict[str, Any]], key: str) -> int | None:
    values = {
        int(value)
        for row in rows
        if isinstance((value := row.get(key)), int | float)
    }
    return next(iter(values)) if len(values) == 1 else None


def telemetry_sum(artifact: dict[str, Any], key: str) -> int:
    total = 0
    for row in artifact.get("results", []):
        if row.get("prompt_case_id") is None:
            continue
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        value = telemetry.get(key, 0)
        if isinstance(value, int | float):
            total += int(value)
    return total


def ax_accept_rate(artifact: dict[str, Any]) -> float | None:
    accepted = telemetry_sum(artifact, "ax_mtp_accepted_tokens")
    drafted = telemetry_sum(artifact, "ax_mtp_draft_tokens")
    return accepted / drafted if drafted else None


def validate_ax_pure_mtp(artifact: dict[str, Any], artifact_path: Path) -> None:
    non_zero = {
        key: telemetry_sum(artifact, key)
        for key in NGRAM_ZERO_KEYS
        if telemetry_sum(artifact, key) != 0
    }
    if non_zero:
        raise RuntimeError(f"{artifact_path} is not pure MTP; n-gram telemetry={non_zero}")


def summarize_ax_artifact(artifact: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    validate_ax_pure_mtp(artifact, artifact_path)
    rows = [
        row
        for row in artifact.get("results", [])
        if row.get("engine") in AX_MTP_ENGINES and row.get("prompt_case_id")
    ]
    degeneracy = check_ax_output_degeneracy(artifact)
    # Collect client-observed TTFT for cross-engine comparability.
    client_ttft_values: list[float] = []
    for row in rows:
        for trial in row.get("trials", []):
            v = trial.get("client_wall_ttft_ms")
            if isinstance(v, int | float) and v > 0:
                client_ttft_values.append(float(v))
    return {
        "status": "ok" if rows else "no_data",
        "decode_tok_s": median(
            [v for row in rows if (v := metric_median(row, "decode_tok_s")) is not None]
        ),
        "prefill_tok_s": median(
            [v for row in rows if (v := metric_median(row, "prefill_tok_s")) is not None]
        ),
        "ttft_ms": median(
            [v for row in rows if (v := metric_median(row, "ttft_ms")) is not None]
        ),
        "client_wall_ttft_ms": median(client_ttft_values),
        "accept_rate": ax_accept_rate(artifact),
        "random_seed": shared_int_value(rows, "random_seed"),
        "seed": shared_int_value(rows, "seed"),
        "case_count": len(rows),
        "prefill_source": "ax_engine_runner_prefill_time",
        "ttft_source": "ax_engine_runner_prefill_time",
        "ttft_scope": "runner_internal",
        "client_ttft_scope": "client_http_wall",
        "accept_source": "ax_engine_mtp_telemetry",
        "degeneracy_gate": degeneracy,
    }


def measured_runs(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in case.get("runs", [])
        if isinstance(run, dict) and bool(run.get("measured"))
    ]


def summarize_mtplx_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in artifact.get("results", []) if case.get("prompt_id")]
    decode_values: list[float] = []
    accept_values: list[float] = []
    prefill_values: list[float] = []
    ttft_values: list[float] = []
    for case in cases:
        summary = case.get("summary") or {}
        decode_values.extend(numeric_samples(summary.get("decode_tok_s")))
        accept_values.extend(numeric_samples(summary.get("accept_rate")))
        prompt_tokens = case.get("prompt_tokens")
        if not isinstance(prompt_tokens, int | float) or prompt_tokens <= 0:
            continue
        for run in measured_runs(case):
            prompt_eval_time_s = run.get("prompt_eval_time_s")
            if isinstance(prompt_eval_time_s, int | float) and prompt_eval_time_s > 0:
                prefill_values.append(float(prompt_tokens) / float(prompt_eval_time_s))
                ttft_values.append(float(prompt_eval_time_s) * 1000.0)
    return {
        "status": "ok" if cases else "no_data",
        "decode_tok_s": median(decode_values),
        "prefill_tok_s": median(prefill_values),
        "ttft_ms": median(ttft_values),
        "accept_rate": median(accept_values),
        "case_count": len(cases),
        "prefill_source": "prompt_tokens_over_prompt_eval_time_s",
        "ttft_source": "prompt_eval_time_s",
        "ttft_scope": "server_prompt_eval",
        "accept_source": "mtplx_draft_stats",
    }


def summarize_lightning_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in artifact.get("results", []) if case.get("prompt_id")]
    log_lines = [str(line) for line in artifact.get("server_log_tail", [])]
    mtp_disabled = any(
        "MTP install skipped" in line
        or "model has no MTP head" in line
        or "MTP validation failed" in line
        for line in log_lines
    )
    decode_values: list[float] = []
    accept_values: list[float] = []
    prefill_values: list[float] = []
    ttft_values: list[float] = []
    for case in cases:
        summary = case.get("summary") or {}
        decode_values.extend(numeric_samples(summary.get("decode_tok_s")))
        accept_values.extend(numeric_samples(summary.get("accept_rate")))
        for run in measured_runs(case):
            ttft_s = run.get("ttft_s")
            prompt_tokens = run.get("prompt_tokens")
            if isinstance(ttft_s, int | float) and ttft_s > 0:
                ttft_values.append(float(ttft_s) * 1000.0)
                if isinstance(prompt_tokens, int | float) and prompt_tokens > 0:
                    prefill_values.append(float(prompt_tokens) / float(ttft_s))
    return {
        "status": "mtp_disabled" if mtp_disabled else "ok" if cases else "no_data",
        "decode_tok_s": median(decode_values),
        "prefill_tok_s": median(prefill_values),
        "ttft_ms": median(ttft_values),
        "accept_rate": median(accept_values),
        "case_count": len(cases),
        "prefill_source": "prompt_tokens_over_client_ttft_s",
        "ttft_source": "client_stream_ttft_s",
        "ttft_scope": "client_http_wall",
        "accept_source": "lightning_request_telemetry",
    }


def summarize_artifact(engine: str, artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file():
        return {"status": "missing"}
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema") == "ax.mtp_engine_error.v1":
        return {"status": "error", "error": artifact.get("error")}
    if engine == "ax_engine":
        return summarize_ax_artifact(artifact, artifact_path)
    if engine == "mtplx":
        return summarize_mtplx_artifact(artifact)
    if engine in {"lightning_mlx", "rapid_mlx"}:
        return summarize_lightning_artifact(artifact)
    return {"status": "unsupported"}


def target_keys(models: list[str], bits: list[int]) -> list[str]:
    keys: list[str] = []
    for model in MODELS:
        if model not in models:
            continue
        for bit in BITS:
            if bit in bits:
                keys.append(f"{model}-{bit}bit")
    return keys


def suite_path(args: argparse.Namespace, suite: str) -> Path:
    return args.suites_dir / f"{suite}.jsonl"


def prompt_suite_path(args: argparse.Namespace, suite: str) -> Path:
    source = suite_path(args, suite)
    if args.prompt_limit is None:
        return source
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    if len(lines) < args.prompt_limit:
        raise ValueError(
            f"{source} contains {len(lines)} prompts, fewer than --prompt-limit {args.prompt_limit}"
        )
    subset = args.output_dir / "prompt-subsets" / f"{suite}-first{args.prompt_limit}.jsonl"
    subset.parent.mkdir(parents=True, exist_ok=True)
    subset.write_text("\n".join(lines[: args.prompt_limit]) + "\n")
    return subset


def output_path(args: argparse.Namespace, target: Target, engine: str, suite: str) -> Path:
    return args.output_dir / target.key / suite / f"{engine}.json"


def ax_command(args: argparse.Namespace, target: Target, suite: str, output: Path) -> list[str]:
    cmd = [
        str(args.ax_python),
        str(AX_BENCH_SCRIPT),
        "--model-dir",
        str(target.ax_model_dir),
        "--prompt-source",
        "real",
        "--real-prompt-suite",
        str(prompt_suite_path(args, suite)),
        "--generation-tokens",
        str(args.max_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--ax-sampling",
        json.dumps(args.sampling, separators=(",", ":")),
        "--skip-mlx-lm",
        "--no-thinking",
        "--capture-output-token-ids",
        "--ax-ngram-accel",
        "--ax-mtp-disable-ngram-stacking",
        "--ax-mtp-max-depth",
        str(target.mtp_depth),
        "--output",
        str(output),
    ]
    if args.prefill_step_size is not None:
        cmd.extend(["--prefill-step-size", str(args.prefill_step_size)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.no_build_ax_engine:
        cmd.append("--no-build-ax-engine")
    return cmd


def mtplx_command(args: argparse.Namespace, target: Target, suite: str, output: Path) -> list[str] | None:
    model = MTPLX_MODEL_IDS.get(target.key)
    if model is None:
        return None
    model_path = resolve_hf_snapshot(model, args.peer_caches)
    model_arg = str(model_path or model)
    cmd = [
        str(args.mtplx_python),
        str(MTPLX_BENCH_SCRIPT),
        "--model",
        model_arg,
        "--suite",
        suite,
        "--prompts",
        str(prompt_suite_path(args, suite)),
        "--output",
        str(output),
        "--depth",
        str(target.mtp_depth),
        "--temperature",
        str(args.sampling["temperature"]),
        "--top-p",
        str(args.sampling["top_p"]),
        "--top-k",
        str(args.sampling["top_k"]),
        "--max-tokens",
        str(args.max_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--ignore-eos",
        "--profile",
        args.mtplx_profile,
        "--mtp-quant-mode",
        "cyankiwi",
        "--disable-thinking",
        "--allow-unverified-model",
    ]
    if policy := MTPLX_QUANT_POLICIES.get(target.key):
        cmd.extend(["--mtp-quant-policy", policy])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def lightning_command(args: argparse.Namespace, target: Target, suite: str, output: Path) -> list[str] | None:
    model = LIGHTNING_MODELS.get(target.key)
    if model is None:
        return None
    model_path = resolve_hf_snapshot(model, args.peer_caches)
    model_arg = str(model_path or model)
    cmd = [
        str(args.rapid_python),
        str(RAPID_BENCH_SCRIPT),
        "--model",
        model_arg,
        "--suite",
        suite,
        "--prompts",
        str(prompt_suite_path(args, suite)),
        "--output",
        str(output),
        "--rapid-source",
        str(args.lightning_source),
        "--lightning-source",
        str(args.lightning_source),
        "--rapid-mtp-patch",
        "lightning",
        "--lightning-mode",
        "--depth",
        str(target.mtp_depth),
        "--temperature",
        str(args.sampling["temperature"]),
        "--top-p",
        str(args.sampling["top_p"]),
        "--top-k",
        str(args.sampling["top_k"]),
        "--max-tokens",
        str(args.max_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--ignore-eos",
        "--require-full-output-tokens",
        "--port",
        str(args.base_port),
        "--disable-thinking",
    ]
    if args.lightning_mtp_draft_temperature is not None:
        cmd.extend(["--mtp-draft-temperature", str(args.lightning_mtp_draft_temperature)])
    if args.lightning_mtp_optimistic:
        cmd.append("--mtp-optimistic")
    if args.lightning_disable_prefix_cache:
        cmd.append("--disable-prefix-cache")
    if args.prefill_step_size is not None:
        cmd.extend(["--prefill-step-size", str(args.prefill_step_size)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def rapid_command(args: argparse.Namespace, target: Target, suite: str, output: Path) -> list[str] | None:
    model = RAPID_MODELS.get(target.key)
    if model is None:
        return None
    model_path = resolve_hf_snapshot(model, args.peer_caches)
    model_arg = str(model_path or model)
    return [
        str(args.rapid_python),
        str(RAPID_BENCH_SCRIPT),
        "--model",
        model_arg,
        "--suite",
        suite,
        "--prompts",
        str(prompt_suite_path(args, suite)),
        "--output",
        str(output),
        "--rapid-source",
        str(args.rapid_source),
        "--lightning-source",
        str(args.lightning_source),
        "--rapid-mtp-patch",
        "lightning",
        "--depth",
        str(target.mtp_depth),
        "--temperature",
        str(args.sampling["temperature"]),
        "--top-p",
        str(args.sampling["top_p"]),
        "--top-k",
        str(args.sampling["top_k"]),
        "--max-tokens",
        str(args.max_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--ignore-eos",
        "--require-full-output-tokens",
        "--port",
        str(args.base_port),
        "--disable-thinking",
    ]


def unsupported_reason(engine: str, target: Target) -> str:
    if engine in {"mtplx", "lightning_mlx", "rapid_mlx"} and target.key == "27b-6bit":
        return f"{engine} catalog has no official Qwen3.6 27B 6-bit MTP artifact."
    if engine == "rapid_mlx":
        return "Rapid-MLX starts with the shared Qwen3.6 artifacts, but its scheduler skips MTP install for this generation flow; running it would measure non-MTP decode."
    if engine == "omlx":
        return "oMLX has no repo-owned Qwen3.6 MTP prompt-suite adapter yet; native MTP needs prepared MTP weights and a dedicated runner."
    return "No matching pure-MTP runner/model artifact is configured for this lane."


def resolve_hf_snapshot(repo_id: str, cache_roots: list[Path]) -> Path | None:
    escaped = "models--" + repo_id.replace("/", "--")
    candidates: list[Path] = []
    for root in cache_roots:
        snapshots = root / escaped / "snapshots"
        if not snapshots.is_dir():
            continue
        for child in snapshots.iterdir():
            if (child / "config.json").is_file():
                candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def command_for_lane(args: argparse.Namespace, target: Target, engine: str, suite: str, output: Path) -> list[str] | None:
    if engine == "ax_engine":
        return ax_command(args, target, suite, output)
    if engine == "mtplx":
        return mtplx_command(args, target, suite, output)
    if engine == "lightning_mlx":
        return lightning_command(args, target, suite, output)
    return None


def metric_contract(engine: str) -> dict[str, str]:
    if engine == "ax_engine":
        return {
            "decode_tok_s": "measured",
            "prefill_tok_s": "measured",
            "ttft_ms": "measured",
            "accept_rate": "measured",
            "ttft_scope": "runner_internal",
            "client_ttft_scope": "client_http_wall",
        }
    if engine == "mtplx":
        return {
            "decode_tok_s": "measured",
            "prefill_tok_s": "derived_from_prompt_eval_time_s",
            "ttft_ms": "prompt_eval_time_s",
            "accept_rate": "measured",
            "ttft_scope": "server_prompt_eval",
        }
    if engine in {"lightning_mlx", "rapid_mlx"}:
        return {
            "decode_tok_s": "measured",
            "prefill_tok_s": "approx_prompt_tokens_over_client_ttft_s",
            "ttft_ms": "client_stream_ttft_s",
            "accept_rate": "measured_if_server_exposes_request_telemetry",
            "ttft_scope": "client_http_wall",
        }
    return {
        "decode_tok_s": "unavailable",
        "prefill_tok_s": "unavailable",
        "ttft_ms": "unavailable",
        "accept_rate": "unavailable",
        "ttft_scope": "unavailable",
    }


def build_lanes(args: argparse.Namespace) -> list[Lane]:
    lanes: list[Lane] = []
    for key in target_keys(args.models, args.bits):
        target = TARGETS[key]
        for suite in args.suites:
            for engine in args.engines:
                output = output_path(args, target, engine, suite)
                cmd = command_for_lane(args, target, engine, suite, output)
                lanes.append(
                    Lane(
                        target=target,
                        engine=engine,
                        suite=suite,
                        status="supported" if cmd is not None else "unsupported",
                        output_path=output,
                        command=cmd,
                        reason=None if cmd is not None else unsupported_reason(engine, target),
                        metric_contract=metric_contract(engine),
                    )
                )
    return lanes


def lane_to_dict(lane: Lane) -> dict[str, Any]:
    return {
        "target": asdict(lane.target) | {"ax_model_dir": str(lane.target.ax_model_dir)},
        "engine": lane.engine,
        "suite": lane.suite,
        "status": lane.status,
        "output_path": str(lane.output_path),
        "command": lane.command,
        "reason": lane.reason,
        "metric_contract": lane.metric_contract,
        "mtp_head": MTP_HEAD_PROVENANCE.get(lane.engine),
    }


def write_plan(args: argparse.Namespace, lanes: list[Lane]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ax.qwen36_mtp_matrix.plan.v1",
        "created_at": date.today().isoformat(),
        "contract": {
            "models": args.models,
            "bits": args.bits,
            "engines": args.engines,
            "suites": args.suites,
            "benchmark_contract": args.benchmark_contract,
            "mode": "mtp",
            "forbidden_modes": ["direct", "mtp-ngram", "ngram"],
            "metrics": ["decode_tok_s", "prefill_tok_s", "ttft_ms", "accept_rate"],
            "sampling": args.sampling,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "prompt_limit": args.prompt_limit,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "inter_case_cooldown_s": args.inter_case_cooldown,
            "prefill_step_size": args.prefill_step_size,
            "mtplx_profile": args.mtplx_profile,
            "lightning_mtp_optimistic": args.lightning_mtp_optimistic,
            "lightning_disable_prefix_cache": args.lightning_disable_prefix_cache,
            "lightning_prefix_cache_policy": (
                "enabled_explicitly"
                if args.lightning_enable_prefix_cache
                else "disabled_for_cold_prefill"
                if args.lightning_disable_prefix_cache
                else "enabled_default"
            ),
            "lightning_mtp_draft_temperature": args.lightning_mtp_draft_temperature,
            "ax_mtp_optimistic": args.ax_mtp_optimistic,
            "degeneracy_gate": {
                "max_cycle_len": DEGENERACY_MAX_CYCLE_LEN,
                "coverage_threshold": DEGENERACY_COVERAGE_THRESHOLD,
                "periodic_coverage_threshold": DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
            },
            "mtp_head_provenance": {
                engine: prov
                for engine in args.engines
                if (prov := MTP_HEAD_PROVENANCE.get(engine))
            },
        },
        "lanes": [lane_to_dict(lane) for lane in lanes],
    }
    (args.output_dir / "plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_plan_markdown(args.output_dir / "plan.md", lanes)


def write_plan_markdown(path: Path, lanes: list[Lane]) -> None:
    lines = [
        "# Qwen3.6 MTP Benchmark Matrix Plan",
        "",
        "Scope: Qwen3.6 27B and 35B-A3B, 4-bit and 6-bit, MTP-only.",
        "Required metrics: decode tok/s, prefill tok/s, TTFT ms, and MTP accept rate.",
        "",
        "| Target | Suite | Engine | Status | Metric contract | Reason / command |",
        "|---|---|---|---|---|---|",
    ]
    for lane in lanes:
        metrics = ", ".join(
            f"{key}={value}" for key, value in (lane.metric_contract or {}).items()
        )
        detail = lane.reason or shlex.join(lane.command or [])
        lines.append(
            "| {target} | `{suite}` | `{engine}` | {status} | {metrics} | `{detail}` |".format(
                target=lane.target.label,
                suite=lane.suite,
                engine=lane.engine,
                status=lane.status,
                metrics=metrics,
                detail=detail.replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def mtplx_env(args: argparse.Namespace) -> dict[str, str] | None:
    if not args.mtplx_source.is_dir():
        return None
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(args.mtplx_source)
        if not existing
        else str(args.mtplx_source) + os.pathsep + existing
    )
    return env


def ax_env(args: argparse.Namespace) -> dict[str, str]:
    """Build an environment dict that explicitly sets AX_MLX_MTP_OPTIMISTIC.

    This makes the AX optimistic mode choice visible and reproducible in the
    benchmark plan/summary metadata rather than relying on the compiled-in
    default.  When ``args.ax_mtp_optimistic`` is False, the env var is set to
    ``"0"`` which forces the full rejection-sampling path.
    """
    env = os.environ.copy()
    env["AX_MLX_MTP_OPTIMISTIC"] = "1" if args.ax_mtp_optimistic else "0"
    return env


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        if env and env.get("PYTHONPATH"):
            log.write(f"PYTHONPATH={env['PYTHONPATH']}\n")
        log.write("$ " + shlex.join(cmd) + "\n\n")
        log.flush()
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        elapsed = time.perf_counter() - started
        log.write(f"\n[exit {result.returncode} after {elapsed:.1f}s]\n")
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"command failed; see {log_path}\n{tail}")


def write_error_artifact(
    path: Path,
    *,
    engine: str,
    error: Exception,
    command: list[str],
    log_path: Path,
) -> None:
    payload = {
        "schema": "ax.mtp_engine_error.v1",
        "engine": engine,
        "error": str(error),
        "command": command,
        "log_path": str(log_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute_lanes(args: argparse.Namespace, lanes: list[Lane]) -> None:
    for lane in lanes:
        if lane.status != "supported" or lane.command is None:
            print(f"[skip] {lane.target.key} {lane.suite} {lane.engine}: {lane.reason}", flush=True)
            continue
        if args.skip_existing and lane.output_path.is_file():
            print(f"[skip-existing] {lane.output_path}", flush=True)
            continue
        lane.output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[run] {lane.target.key} {lane.suite} {lane.engine}", flush=True)
        log_path = lane.output_path.with_suffix(".log")
        try:
            if lane.engine == "ax_engine":
                env = ax_env(args)
            elif lane.engine == "mtplx":
                env = mtplx_env(args)
            else:
                env = None
            run_logged(lane.command, log_path, env=env)
        except Exception as exc:
            write_error_artifact(
                lane.output_path,
                engine=lane.engine,
                error=exc,
                command=lane.command,
                log_path=log_path,
            )
            print(f"[error] {lane.target.key} {lane.suite} {lane.engine}: {exc}", flush=True)


def build_summary(args: argparse.Namespace, lanes: list[Lane]) -> dict[str, Any]:
    rows = []
    for lane in lanes:
        metrics = summarize_artifact(lane.engine, lane.output_path) if lane.status == "supported" else {"status": "unsupported"}
        rows.append(
            {
                "target": lane.target.key,
                "model": lane.target.model,
                "bits": lane.target.bits,
                "model_label": lane.target.label,
                "suite": lane.suite,
                "engine": lane.engine,
                "status": lane.status,
                "reason": lane.reason,
                "artifact": str(lane.output_path),
                "metrics": metrics,
                "metric_contract": lane.metric_contract,
            }
        )
    return {
        "schema": "ax.qwen36_mtp_matrix.summary.v1",
        "created_at": date.today().isoformat(),
        "contract": {
            "models": args.models,
            "bits": args.bits,
            "engines": args.engines,
            "suites": args.suites,
            "benchmark_contract": args.benchmark_contract,
            "mode": "mtp",
            "metrics": ["decode_tok_s", "prefill_tok_s", "ttft_ms", "accept_rate"],
            "sampling": args.sampling,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "prompt_limit": args.prompt_limit,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "inter_case_cooldown_s": args.inter_case_cooldown,
            "prefill_step_size": args.prefill_step_size,
            "mtplx_profile": args.mtplx_profile,
            "lightning_mtp_optimistic": args.lightning_mtp_optimistic,
            "lightning_disable_prefix_cache": args.lightning_disable_prefix_cache,
            "lightning_prefix_cache_policy": (
                "enabled_explicitly"
                if args.lightning_enable_prefix_cache
                else "disabled_for_cold_prefill"
                if args.lightning_disable_prefix_cache
                else "enabled_default"
            ),
            "lightning_mtp_draft_temperature": args.lightning_mtp_draft_temperature,
            "ax_mtp_optimistic": args.ax_mtp_optimistic,
            "degeneracy_gate": {
                "max_cycle_len": DEGENERACY_MAX_CYCLE_LEN,
                "coverage_threshold": DEGENERACY_COVERAGE_THRESHOLD,
                "periodic_coverage_threshold": DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
            },
            "mtp_head_provenance": {
                engine: prov
                for engine in args.engines
                if (prov := MTP_HEAD_PROVENANCE.get(engine))
            },
        },
        "rows": rows,
    }


def fmt_number(value: Any, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}" if isinstance(value, int | float) else "-"


def fmt_percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, int | float) else "-"


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.6 MTP Benchmark Matrix Summary",
        "",
        "| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        status = metrics.get("status") or row["status"]
        # Flag degenerate rows.
        degen = metrics.get("degeneracy_gate", {})
        if degen.get("degenerate"):
            status += " [DEGENERATE OUTPUT]"
        lines.append(
            "| {target} | `{suite}` | `{engine}` | {decode} tok/s | {prefill} tok/s | {ttft} ms | {accept} | {status} |".format(
                target=row["model_label"],
                suite=row["suite"],
                engine=row["engine"],
                decode=fmt_number(metrics.get("decode_tok_s")),
                prefill=fmt_number(metrics.get("prefill_tok_s")),
                ttft=fmt_number(metrics.get("ttft_ms"), 0),
                accept=fmt_percent(metrics.get("accept_rate")),
                status=status,
            )
        )
    provenance = summary.get("contract", {}).get("mtp_head_provenance", {})
    seed = summary["contract"].get("seed")
    seed_label = str(seed) if seed is not None else "engine defaults"
    seed_note = (
        f"- Seed: `{seed_label}` (forwarded to AX, MTPLX, and lightning runner commands)."
        if seed is not None
        else "- Seed: `engine defaults` (AX defaults to seed 0; MTPLX/lightning use their runner defaults)."
    )
    lines += [
        "",
        "Notes:",
        "",
        "- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.",
        "- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.",
        "- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.",
        f"- AX MTP optimistic verify: {'ON (skip full softmax on accepted drafts)' if summary['contract'].get('ax_mtp_optimistic', True) else 'OFF (full rejection sampling)'}.",
        seed_note,
        "",
        "**Measurement scope (TTFT / prefill):**",
        "",
        "- AX `ttft_ms` / `prefill_tok_s`: measured inside the MLX runner (excludes HTTP/SSE overhead). `client_wall_ttft_ms` is also recorded for cross-engine parity.",
        "- MTPLX: derived from server-side `prompt_eval_time_s`.",
        "- Lightning: client-observed HTTP stream TTFT (includes local HTTP overhead).",
        "- **Only `decode_tok_s` is measured at the same scope across all engines.** Cross-engine prefill/TTFT comparisons should use `client_wall_ttft_ms` where available.",
        "",
        "**MTP head provenance:**",
        "",
    ]
    for engine, prov in provenance.items():
        lines.append(f"- `{engine}`: {prov.get('packaging', 'unknown')} (MTP precision: {prov.get('mtp_precision', 'unknown')}, draft LM head: {prov.get('draft_lm_head', 'unknown')})")
    lines += [
        "",
        "- Different MTP head artifacts across engines means this is a **production-configuration comparison**, not an apples-to-apples MTP weight test.",
        f"- Degeneracy gate: rejects runs where a consecutive repeating token cycle (length \u2264{DEGENERACY_MAX_CYCLE_LEN}) covers \u2265{DEGENERACY_COVERAGE_THRESHOLD*100:.0f}% of output tokens, or a phase-aligned periodic cycle covers \u2265{DEGENERACY_PERIODIC_COVERAGE_THRESHOLD*100:.0f}%.",
        "- Unsupported peer lanes are listed in `plan.md` with the exact support reason.",
    ]
    path.write_text("\n".join(lines) + "\n")


def positive_ints(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    invalid = [value for value in parsed if value not in BITS]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported bit widths: {invalid}")
    return parsed


def arg_was_provided(argv: list[str], flag: str) -> bool:
    return any(value == flag or value.startswith(f"{flag}=") for value in argv)


def apply_benchmark_contract(args: argparse.Namespace, argv: list[str]) -> None:
    if args.benchmark_contract == "lightning-optimized":
        args.benchmark_contract = "peer-optimized"
    if args.lightning_enable_prefix_cache:
        args.lightning_disable_prefix_cache = False
    elif not arg_was_provided(argv, "--lightning-disable-prefix-cache"):
        args.lightning_disable_prefix_cache = True
    if args.benchmark_contract != "peer-optimized":
        return
    if not arg_was_provided(argv, "--max-tokens"):
        args.max_tokens = PEER_OPTIMIZED_DEFAULTS["max_tokens"]
    if not arg_was_provided(argv, "--prompt-limit"):
        args.prompt_limit = PEER_OPTIMIZED_DEFAULTS["prompt_limit"]
    if not arg_was_provided(argv, "--prefill-step-size"):
        args.prefill_step_size = PEER_OPTIMIZED_DEFAULTS["prefill_step_size"]
    if not arg_was_provided(argv, "--mtplx-profile"):
        args.mtplx_profile = PEER_OPTIMIZED_DEFAULTS["mtplx_profile"]
    if not arg_was_provided(argv, "--lightning-mtp-draft-temperature"):
        args.lightning_mtp_draft_temperature = None
    if not arg_was_provided(argv, "--lightning-mtp-optimistic"):
        args.lightning_mtp_optimistic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_argv = sys.argv[1:]
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--bits", nargs="+", type=int, choices=BITS, default=list(BITS))
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hf-cache", type=Path, default=HF_CACHE)
    parser.add_argument(
        "--peer-caches",
        nargs="+",
        type=Path,
        default=list(DEFAULT_PEER_CACHES),
        help="HF cache roots used to resolve peer model ids to local snapshots.",
    )
    parser.add_argument(
        "--benchmark-contract",
        choices=BENCHMARK_CONTRACTS,
        default="apples-to-apples",
        help=(
            "`apples-to-apples` preserves the README prompt-suite contract and disables "
            "cross-request prefix cache for cold-prefill parity. "
            "`peer-optimized` applies the peer maintainer short-benchmark profile: "
            "3 prompts, 512 max tokens, no prefix cache, prefill step 8192, "
            "single sequence/batches, MTPLX performance-cold, and lightning optimistic MTP. "
            "`lightning-optimized` is accepted as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Limit each prompt suite to the first N prompts for benchmark profiles.",
    )
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for all engines. When unset, each engine uses its own "
            "default (AX: fixed seed 0; lightning: incrementing per repetition). "
            "Set this to force seed parity for cross-engine reproducibility."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--inter-case-cooldown", type=float, default=10.0)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING["temperature"])
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--ax-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--mtplx-python",
        type=Path,
        default=DEFAULT_MTPLX_PYTHON if DEFAULT_MTPLX_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument(
        "--rapid-python",
        type=Path,
        default=DEFAULT_RAPID_PYTHON if DEFAULT_RAPID_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument("--rapid-source", type=Path, default=DEFAULT_RAPID_SOURCE)
    parser.add_argument("--lightning-source", type=Path, default=DEFAULT_LIGHTNING_SOURCE)
    parser.add_argument("--mtplx-source", type=Path, default=DEFAULT_MTPLX_SOURCE)
    parser.add_argument("--mtplx-profile", default="stable")
    parser.add_argument("--lightning-mtp-draft-temperature", type=float, default=0.5)
    parser.add_argument("--lightning-mtp-optimistic", action="store_true")
    parser.add_argument(
        "--ax-mtp-optimistic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Control AX Engine MTP optimistic verify mode. "
            "Default ON matches AX runtime default (skip full-vocab softmax and "
            "rejection sampling when drafts are accepted). "
            "Use --no-ax-mtp-optimistic for strict rejection-sampling parity with "
            "peer engines that always compute full softmax."
        ),
    )
    lightning_prefix_cache = parser.add_mutually_exclusive_group()
    lightning_prefix_cache.add_argument("--lightning-disable-prefix-cache", action="store_true")
    lightning_prefix_cache.add_argument(
        "--lightning-enable-prefix-cache",
        action="store_true",
        help="Opt into lightning cross-request prefix cache for explicit warm-cache experiments.",
    )
    parser.add_argument("--base-port", type=int, default=18765)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-build-ax-engine", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run supported lanes. Without this flag the script only writes plan files.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Build summary.json/summary.md from existing artifacts without running lanes.",
    )
    args = parser.parse_args()
    args.output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-qwen36-mtp-matrix"
    )
    apply_benchmark_contract(args, raw_argv)
    if args.prompt_limit is not None and args.prompt_limit <= 0:
        parser.error("--prompt-limit must be positive")
    if args.prefill_step_size is not None and args.prefill_step_size <= 0:
        parser.error("--prefill-step-size must be positive")
    args.sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.hf_cache != HF_CACHE:
        for key, target in list(TARGETS.items()):
            if target.ax_model_dir.is_relative_to(HF_CACHE):
                TARGETS[key] = Target(
                    key=target.key,
                    label=target.label,
                    model=target.model,
                    bits=target.bits,
                    ax_model_dir=args.hf_cache / target.ax_model_dir.relative_to(HF_CACHE),
                    mtp_depth=target.mtp_depth,
                )
    return args


def main() -> int:
    args = parse_args()
    lanes = build_lanes(args)
    write_plan(args, lanes)

    if args.execute:
        execute_lanes(args, lanes)

    if args.execute or args.summarize_existing:
        summary = build_summary(args, lanes)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_summary_markdown(args.output_dir / "summary.md", summary)

    print(f"Wrote plan to {args.output_dir / 'plan.md'}")
    if args.execute or args.summarize_existing:
        print(f"Wrote summary to {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
