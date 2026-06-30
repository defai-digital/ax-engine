#!/usr/bin/env python3
"""Plan, run, and summarize the Qwen3.6 MTP-only benchmark matrix."""

from __future__ import annotations

import argparse
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
DEFAULT_MTPLX_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
DEFAULT_RAPID_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
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
AX_MTP_ENGINES = {"ax_engine_mlx_ngram_accel", "ax_engine_mlx_pure_mtp"}
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
    ),
    "35b-a3b-4bit": Target(
        key="35b-a3b-4bit",
        label="Qwen3.6 35B-A3B 4-bit",
        model="35b-a3b",
        bits=4,
        ax_model_dir=HF_CACHE / "models--ax-local--Qwen3.6-35B-MTP" / "snapshots" / "v1",
        mtp_depth=1,
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
        "accept_rate": ax_accept_rate(artifact),
        "case_count": len(rows),
        "prefill_source": "ax_engine_runner_prefill_time",
        "ttft_source": "ax_engine_runner_prefill_time",
        "accept_source": "ax_engine_mtp_telemetry",
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
        str(suite_path(args, suite)),
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
        str(suite_path(args, suite)),
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
    return cmd


def lightning_command(args: argparse.Namespace, target: Target, suite: str, output: Path) -> list[str] | None:
    model = LIGHTNING_MODELS.get(target.key)
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
        str(suite_path(args, suite)),
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
        "--mtp-draft-temperature",
        str(args.lightning_mtp_draft_temperature),
        "--disable-thinking",
    ]


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
        str(suite_path(args, suite)),
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
        }
    if engine == "mtplx":
        return {
            "decode_tok_s": "measured",
            "prefill_tok_s": "derived_from_prompt_eval_time_s",
            "ttft_ms": "prompt_eval_time_s",
            "accept_rate": "measured",
        }
    if engine in {"lightning_mlx", "rapid_mlx"}:
        return {
            "decode_tok_s": "measured",
            "prefill_tok_s": "approx_prompt_tokens_over_client_ttft_s",
            "ttft_ms": "client_stream_ttft_s",
            "accept_rate": "measured_if_server_exposes_request_telemetry",
        }
    return {
        "decode_tok_s": "unavailable",
        "prefill_tok_s": "unavailable",
        "ttft_ms": "unavailable",
        "accept_rate": "unavailable",
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
            "mode": "mtp",
            "forbidden_modes": ["direct", "mtp-ngram", "ngram"],
            "metrics": ["decode_tok_s", "prefill_tok_s", "ttft_ms", "accept_rate"],
            "sampling": args.sampling,
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "inter_case_cooldown_s": args.inter_case_cooldown,
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
            env = mtplx_env(args) if lane.engine == "mtplx" else None
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
            "mode": "mtp",
            "metrics": ["decode_tok_s", "prefill_tok_s", "ttft_ms", "accept_rate"],
            "sampling": args.sampling,
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
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
        lines.append(
            "| {target} | `{suite}` | `{engine}` | {decode} tok/s | {prefill} tok/s | {ttft} ms | {accept} | {status} |".format(
                target=row["model_label"],
                suite=row["suite"],
                engine=row["engine"],
                decode=fmt_number(metrics.get("decode_tok_s")),
                prefill=fmt_number(metrics.get("prefill_tok_s")),
                ttft=fmt_number(metrics.get("ttft_ms"), 0),
                accept=fmt_percent(metrics.get("accept_rate")),
                status=metrics.get("status") or row["status"],
            )
        )
    lines += [
        "",
        "Notes:",
        "",
        "- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.",
        "- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.",
        "- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.",
        "- Unsupported peer lanes are listed in `plan.md` with the exact support reason.",
    ]
    path.write_text("\n".join(lines) + "\n")


def positive_ints(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    invalid = [value for value in parsed if value not in BITS]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported bit widths: {invalid}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--inter-case-cooldown", type=float, default=10.0)
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
