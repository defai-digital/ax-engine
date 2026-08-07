#!/usr/bin/env python3
"""Run AX-only MTP-vs-direct benchmarks for supported 6-bit MTP packages."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "speculative" / "mtp-6bit"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
README_PATH = REPO_ROOT / "docs" / "PERFORMANCE-RESULTS.md"

GENERATED_TOKENS = 1000
REPETITIONS = 5
WARMUP_REPETITIONS = 2
COOLDOWN_S = 15.0
INTER_CASE_COOLDOWN_S = 10.0
MTP_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
DEFAULT_SUITES = ("flappy", "long_code", "python_modules_long")
MTP_6BIT_EXACT_SCHEMA = "ax.mtp_6bit_ax_comparison_summary.v4"
MTP_6BIT_EXACT_CLAIM_TYPE = "exact_mtp_comparison"
MLX_INFERENCE_STACK_SCHEMA = "ax.mlx_inference_stack.v2"
MAX_PUBLICATION_LOAD_AVERAGE = 2.0
MAX_PUBLICATION_PROCESS_CPU_PERCENT = 50.0
DEFAULT_LOAD_WAIT_TIMEOUT_S = 900.0
DEFAULT_LOAD_POLL_INTERVAL_S = 5.0
MTP_SAMPLER_SIGNATURE = "sampling[temperature=0.6,top_p=0.95,top_k=20]"
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
    mode: str
    model_dir: Path
    mtp_depth: int
    assistant_mtp: bool = False


@dataclass(frozen=True)
class ArtifactBuildIdentity:
    engine_version: str
    commit: str


def _resolve_mtp_model_dir(*candidates: str) -> Path:
    """Prefer Ext4T publication paths; fall back to local HF hub cache.

    mbp-m5 hosts often keep the same `models--ax-local--*` packages under
    ``~/.cache/huggingface/hub`` without mounting Ext4T.
    """
    home_hub = Path.home() / ".cache" / "huggingface" / "hub"
    expanded: list[Path] = []
    for raw in candidates:
        path = Path(raw).expanduser()
        expanded.append(path)
        # Also try HF hub cache sibling when given an Ext4T hub path.
        # If path looks like .../models--X/snapshots/v1, try hub/models--X/snapshots/*
        parts = path.parts
        if "models--" in str(path):
            for part in parts:
                if part.startswith("models--"):
                    hub_base = home_hub / part / "snapshots"
                    if hub_base.is_dir():
                        # Prefer exact snapshot name, else newest
                        exact = hub_base / path.name
                        if exact.is_dir():
                            expanded.append(exact)
                        else:
                            snaps = sorted(
                                [p for p in hub_base.iterdir() if p.is_dir()],
                                key=lambda p: p.stat().st_mtime,
                                reverse=True,
                            )
                            expanded.extend(snaps)
                    break
    for path in expanded:
        if (
            path.is_dir()
            and (path / "config.json").is_file()
            and (path / "model-manifest.json").is_file()
        ):
            return path
    # Return first candidate for error messages in validate_model_dir
    return Path(candidates[0]).expanduser()


SUPPORTED_TARGETS = (
    Target(
        key="qwen3.6-27b-6bit",
        label="Qwen3.6 27B",
        mode="Qwen fused sidecar",
        model_dir=_resolve_mtp_model_dir(
            "/Volumes/Ext4T/models/hub/models--ax-local--mlx-community--Qwen3.6-27B-6bit-MTP/snapshots/v1"
        ),
        mtp_depth=3,
    ),
    Target(
        key="qwen3.6-35b-a3b",
        label="Qwen3.6 35B-A3B",
        mode="Qwen fused sidecar",
        model_dir=_resolve_mtp_model_dir(
            "/Volumes/Ext4T/models/hub/models--ax-local--mlx-community--Qwen3.6-35B-A3B-6bit-MTP/snapshots/v1"
        ),
        mtp_depth=1,
    ),
    Target(
        key="gemma-4-12b",
        label="Gemma 4 12B",
        mode="Gemma assistant-MTP",
        model_dir=_resolve_mtp_model_dir(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-12b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
    Target(
        key="gemma-4-26b",
        label="Gemma 4 26B",
        mode="Gemma assistant-MTP",
        model_dir=_resolve_mtp_model_dir(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-26b-a4b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
    Target(
        key="gemma-4-31b",
        label="Gemma 4 31B",
        mode="Gemma assistant-MTP",
        model_dir=_resolve_mtp_model_dir(
            "/Volumes/Ext4T/models/hub/models--ax-local--gemma-4-31b-it-assistant-mtp/snapshots/v1"
        ),
        mtp_depth=2,
        assistant_mtp=True,
    ),
)
TARGETS_BY_KEY = {target.key: target for target in SUPPORTED_TARGETS}


def existing_artifact_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(artifact.get("results")) and not publication_condition_reasons(
        "existing",
        artifact,
    )


def validate_model_dir(path: Path) -> None:
    missing = []
    for name in ("config.json", "model-manifest.json"):
        if not (path / name).is_file():
            missing.append(name)
    if not any(path.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        raise FileNotFoundError(f"{path} is missing {', '.join(missing)}")


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        log.write(f"\n[exit {result.returncode} after {elapsed:.1f}s]\n")
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"command failed; see {log_path}\n{tail}")


def build_server() -> None:
    cmd = ["cargo", "build", "-p", "ax-engine-server", "--release"]
    print("[build] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def bench_cmd(
    *,
    target: Target,
    suite: str,
    mode: str,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--model-dir",
        str(target.model_dir),
        "--prompt-source",
        "real",
        "--real-prompt-suite",
        str(args.suites_dir / f"{suite}.jsonl"),
        "--generation-tokens",
        str(args.generated_tokens),
        "--repetitions",
        str(args.repetitions),
        "--warmup-repetitions",
        str(args.warmup_repetitions),
        "--cooldown",
        str(args.cooldown),
        "--inter-case-cooldown",
        str(args.inter_case_cooldown),
        "--max-load-average",
        str(getattr(args, "max_load_average", MAX_PUBLICATION_LOAD_AVERAGE)),
        "--max-top-process-cpu-percent",
        str(
            getattr(
                args,
                "max_top_process_cpu_percent",
                MAX_PUBLICATION_PROCESS_CPU_PERCENT,
            )
        ),
        "--load-average-wait-timeout",
        str(
            getattr(
                args,
                "load_wait_timeout",
                DEFAULT_LOAD_WAIT_TIMEOUT_S,
            )
        ),
        "--load-average-poll-interval",
        str(
            getattr(
                args,
                "load_poll_interval",
                DEFAULT_LOAD_POLL_INTERVAL_S,
            )
        ),
        "--ax-sampling",
        json.dumps(MTP_SAMPLING, separators=(",", ":")),
        "--skip-mlx-lm",
        "--no-thinking",
        "--capture-output-token-ids",
        "--no-build-ax-engine",
        "--output",
        str(output_path),
    ]
    if mode == "direct":
        cmd.append("--ax-direct")
    elif target.assistant_mtp:
        cmd.extend(
            [
                "--ax-gemma4-assistant-mtp",
                "--ax-mtp-disable-ngram-stacking",
                "--ax-mtp-max-depth",
                str(target.mtp_depth),
            ]
        )
    else:
        cmd.extend(
            [
                "--ax-ngram-accel",
                "--ax-mtp-disable-ngram-stacking",
                "--ax-mtp-max-depth",
                str(target.mtp_depth),
                "--ax-qwen-linear-mtp-exact",
            ]
        )
    if mode != "direct" and getattr(args, "approximate_speed_ceiling", False):
        cmd.append("--ax-mtp-approximate-optimistic")
    return cmd


def maybe_run_case(
    *,
    target: Target,
    suite: str,
    mode: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    if args.skip_existing and existing_artifact_ok(output_path):
        print(f"[skip] {target.key} {suite} {mode}: {output_path}", flush=True)
        return
    cmd = bench_cmd(
        target=target,
        suite=suite,
        mode=mode,
        output_path=output_path,
        args=args,
    )
    log_path = output_path.with_suffix(".log")
    print(f"[run] {target.key} {suite} {mode}", flush=True)
    print(f"      artifact: {output_path}", flush=True)
    print(f"      log:      {log_path}", flush=True)
    run_logged(cmd, log_path)


def metric_median(artifact: dict[str, Any], metric: str) -> float:
    values = [
        float(row[metric]["median"])
        for row in artifact.get("results", [])
        if row.get("prompt_case_id") is not None and row.get(metric, {}).get("median") is not None
    ]
    if not values:
        raise ValueError(f"artifact has no {metric} prompt-case medians")
    return float(statistics.median(values))


def telemetry_sum(artifact: dict[str, Any], key: str) -> int:
    total = 0
    for row in artifact.get("results", []):
        if row.get("prompt_case_id") is None:
            continue
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        total += int(telemetry.get(key, 0) or 0)
    return total


def accept_rate_pct(artifact: dict[str, Any]) -> float:
    accepted = telemetry_sum(artifact, "ax_mtp_accepted_tokens")
    drafted = telemetry_sum(artifact, "ax_mtp_draft_tokens")
    if drafted <= 0:
        accepted = telemetry_sum(artifact, "ax_mlx_gemma4_assistant_mtp_accepted_tokens")
        drafted = telemetry_sum(artifact, "ax_mlx_gemma4_assistant_mtp_draft_tokens")
    if drafted <= 0:
        modes = {
            (row.get("ax_mtp_correctness") or {}).get("effective_mode")
            for row in artifact.get("results", [])
            if row.get("prompt_case_id") is not None
        }
        if modes == {"direct_fallback"}:
            return 0.0
        raise ValueError("MTP artifact has no draft-token telemetry")
    return accepted / drafted * 100.0


def prompt_case_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in artifact.get("results", [])
        if isinstance(row, dict) and row.get("prompt_case_id") is not None
    ]


def draft_quality(artifact: dict[str, Any], *, assistant_mtp: bool) -> tuple[float, str]:
    if assistant_mtp:
        return accept_rate_pct(artifact), "verified_accept_rate"

    values = []
    for row in prompt_case_rows(artifact):
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        samples = int(telemetry.get("ax_mtp_mtp_only_accept_rate_ewma_samples", 0) or 0)
        if samples <= 0:
            continue
        raw_match = telemetry.get("ax_mtp_mtp_only_accept_rate_ewma_x1000")
        if not isinstance(raw_match, (int, float)):
            raise ValueError("Qwen MTP target-match EWMA telemetry is missing")
        match_pct = float(raw_match) / 10.0
        if not 0.0 <= match_pct <= 100.0:
            raise ValueError("Qwen MTP target-match EWMA telemetry is out of range")
        values.append(match_pct)
    if not values:
        raise ValueError("Qwen MTP artifact has no target-match EWMA telemetry")
    return float(statistics.median(values)), "target_argmax_match_ewma"


def mtp_coverage(artifact: dict[str, Any]) -> dict[str, float | int]:
    rows = prompt_case_rows(artifact)
    if not rows:
        raise ValueError("MTP artifact has no prompt-case rows")
    mtp_decode_steps = telemetry_sum(artifact, "ax_mtp_decode_steps")
    mtp_emitted_tokens = telemetry_sum(artifact, "ax_mtp_emitted_tokens")
    direct_fallback_steps = telemetry_sum(artifact, "ax_mtp_direct_fallback_steps")
    if min(mtp_decode_steps, mtp_emitted_tokens, direct_fallback_steps) < 0:
        raise ValueError("MTP artifact has negative route telemetry")
    decode_route_steps = mtp_decode_steps + direct_fallback_steps
    if decode_route_steps <= 0:
        raise ValueError("MTP artifact has no MTP or direct-fallback step telemetry")
    fallback_prompt_count = sum(
        1
        for row in rows
        if int(
            (row.get("ngram_acceleration_telemetry") or {}).get("ax_mtp_direct_fallback_steps", 0)
            or 0
        )
        > 0
    )
    return {
        "mtp_decode_steps": mtp_decode_steps,
        "mtp_emitted_tokens": mtp_emitted_tokens,
        "direct_fallback_steps": direct_fallback_steps,
        "decode_route_steps": decode_route_steps,
        "step_coverage_pct": mtp_decode_steps / decode_route_steps * 100.0,
        "fallback_prompt_count": fallback_prompt_count,
        "prompt_count": len(rows),
    }


def aggregate_ngram_telemetry(artifact: dict[str, Any]) -> dict[str, int]:
    return {key: telemetry_sum(artifact, key) for key in NGRAM_ZERO_KEYS}


def load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_exact_mtp_artifact(
    path: Path,
    artifact: dict[str, Any],
    *,
    require_qwen_linear_exact: bool = False,
) -> None:
    correctness = artifact.get("mtp_correctness_summary") or {}
    if correctness.get("publication_candidate") is not True:
        raise ValueError(
            f"{path} is not an exact MTP publication candidate: "
            f"{correctness.get('ineligible_rows') or 'missing correctness summary'}"
        )
    if require_qwen_linear_exact and (
        artifact.get("ax_qwen_linear_mtp_exact") is not True
        or artifact.get("ax_qwen_linear_mtp_exact_explicit_enable") is not True
    ):
        raise ValueError(
            f"{path} did not explicitly select the validated Qwen linear-MTP exact verifier profile"
        )


def validate_approximate_mtp_artifact(path: Path, artifact: dict[str, Any]) -> None:
    rows = [row for row in artifact.get("results", []) if row.get("prompt_case_id") is not None]
    allowed_modes = {"approximate_optimistic", "direct_fallback"}
    if not rows or any(
        (row.get("ax_mtp_correctness") or {}).get("effective_mode") not in allowed_modes
        for row in rows
    ):
        raise ValueError(
            f"{path} is not an effective approximate MTP speed ceiling or direct fallback"
        )
    if any(row.get("publication_candidate") is True for row in rows):
        raise ValueError(f"{path} incorrectly marks an approximate row publishable")


def exact_publication_methodology_reasons(direct: dict[str, Any], mtp: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for label, artifact in (("direct", direct), ("mtp", mtp)):
        if artifact.get("schema_version") != MLX_INFERENCE_STACK_SCHEMA:
            reasons.append(f"{label}_requires_current_artifact_schema")
        if int(artifact.get("warmup_repetitions", 0) or 0) < 2:
            reasons.append(f"{label}_requires_two_warmups")
        if int(artifact.get("repetitions", 0) or 0) < 5:
            reasons.append(f"{label}_requires_five_measurements")
        if float(artifact.get("cooldown", 0.0) or 0.0) < COOLDOWN_S:
            reasons.append(f"{label}_requires_15s_cooldown")
        if int(artifact.get("generation_tokens", 0) or 0) != GENERATED_TOKENS:
            reasons.append(f"{label}_requires_1000_generated_tokens")
        if artifact.get("ax_prefix_cache_mode") != "disabled_for_cold_prefill_benchmark":
            reasons.append(f"{label}_requires_cold_prefix_mode")
        build = artifact.get("build") or {}
        if build.get("git_tracked_dirty") is not False:
            reasons.append(f"{label}_requires_clean_tracked_build")
        if build.get("build_profile") != "release":
            reasons.append(f"{label}_requires_release_build")
        stability = artifact.get("run_stability_summary")
        if not isinstance(stability, dict) or stability.get(
            "publication_candidate"
        ) is not True:
            reasons.append(f"{label}_requires_stable_measurements")
        reasons.extend(publication_condition_reasons(label, artifact))
    return reasons


def publication_condition_reasons(label: str, artifact: dict[str, Any]) -> list[str]:
    window = artifact.get("benchmark_window")
    if not isinstance(window, dict):
        return [f"{label}_requires_benchmark_window"]
    reasons: list[str] = []
    for boundary in ("performance_conditions_start", "performance_conditions_end"):
        conditions = window.get(boundary)
        prefix = f"{label}_{boundary}"
        if not isinstance(conditions, dict):
            reasons.append(f"{prefix}_missing")
            continue
        load_average = conditions.get("load_average")
        one_minute = (
            load_average.get("one_minute")
            if isinstance(load_average, dict)
            else None
        )
        if (
            not isinstance(one_minute, (int, float))
            or isinstance(one_minute, bool)
            or not math.isfinite(float(one_minute))
            or float(one_minute) > MAX_PUBLICATION_LOAD_AVERAGE
        ):
            reasons.append(f"{prefix}_load_above_limit")
        if conditions.get("power_source") != "AC Power":
            reasons.append(f"{prefix}_requires_ac_power")
        for key in (
            "thermal_warning_recorded",
            "performance_warning_recorded",
            "cpu_power_status_recorded",
        ):
            if conditions.get(key) is not False:
                reasons.append(f"{prefix}_{key}_not_clear")
        top_processes = conditions.get("top_processes_cpu")
        cpu_values: list[float] = []
        if isinstance(top_processes, list):
            for process in top_processes:
                if not isinstance(process, dict):
                    continue
                cpu = process.get("cpu_percent")
                if (
                    isinstance(cpu, (int, float))
                    and not isinstance(cpu, bool)
                    and math.isfinite(float(cpu))
                ):
                    cpu_values.append(float(cpu))
        if (
            not cpu_values
            or max(cpu_values) > MAX_PUBLICATION_PROCESS_CPU_PERCENT
        ):
            reasons.append(f"{prefix}_process_cpu_above_limit")
    return reasons


def validate_exact_artifact_rows(
    path: Path,
    artifact: dict[str, Any],
    *,
    expected_engines: set[str],
    expected_suite: str,
) -> None:
    rows = artifact.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no exact benchmark rows")
    case_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} has a non-object benchmark row")
        case_id = row.get("prompt_case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError(f"{path} has a missing or duplicate prompt case")
        case_ids.add(case_id)
        if row.get("engine") not in expected_engines:
            raise ValueError(f"{path} has an unexpected engine for {case_id}")
        if row.get("prompt_source") != "real":
            raise ValueError(f"{path} did not use a real prompt suite for {case_id}")
        if row.get("prompt_suite_id") != expected_suite:
            raise ValueError(f"{path} has the wrong prompt suite for {case_id}")
        if row.get("generation_tokens") != GENERATED_TOKENS:
            raise ValueError(f"{path} has the wrong generation length for {case_id}")
        if row.get("sampler_settings") != MTP_SAMPLER_SIGNATURE:
            raise ValueError(f"{path} has the wrong sampler for {case_id}")
        for hash_key in ("prompt_text_sha256", "prompt_token_ids_sha256"):
            value = row.get(hash_key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{path} has no full {hash_key} for {case_id}")
        prompt_tokens = row.get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
        ):
            raise ValueError(f"{path} has an invalid prompt length for {case_id}")
        stability = row.get("run_stability")
        if not isinstance(stability, dict) or stability.get(
            "classification"
        ) != "stable_enough":
            raise ValueError(f"{path} has an unstable row for {case_id}")
        for metric in ("decode_tok_s", "prefill_tok_s", "ttft_ms"):
            metric_doc = row.get(metric)
            value = (
                metric_doc.get("median")
                if isinstance(metric_doc, dict)
                else metric_doc
            )
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{path} has an invalid {metric} for {case_id}")
        trials = row.get("trials")
        if not isinstance(trials, list) or len(trials) != REPETITIONS:
            raise ValueError(f"{path} does not have five trials for {case_id}")
        for trial in trials:
            if not isinstance(trial, dict):
                raise ValueError(f"{path} has an invalid trial for {case_id}")
            token_ids = trial.get("output_token_ids")
            output_tokens = trial.get("output_tokens")
            if (
                not isinstance(token_ids, list)
                or len(token_ids) != GENERATED_TOKENS
                or not isinstance(output_tokens, (int, float))
                or isinstance(output_tokens, bool)
                or float(output_tokens) != GENERATED_TOKENS
            ):
                raise ValueError(
                    f"{path} has an incomplete generated-token trial for {case_id}"
                )


def validate_exact_prompt_parity(
    direct_path: Path,
    direct: dict[str, Any],
    mtp_path: Path,
    mtp: dict[str, Any],
) -> None:
    if direct.get("model_dir") != mtp.get("model_dir"):
        raise ValueError(
            f"direct/MTP model packages differ: {direct_path} vs {mtp_path}"
        )
    direct_rows = {
        str(row.get("prompt_case_id")): row
        for row in direct.get("results", [])
        if isinstance(row, dict) and row.get("prompt_case_id") is not None
    }
    mtp_rows = {
        str(row.get("prompt_case_id")): row
        for row in mtp.get("results", [])
        if isinstance(row, dict) and row.get("prompt_case_id") is not None
    }
    if not direct_rows or direct_rows.keys() != mtp_rows.keys():
        raise ValueError(f"direct/MTP prompt cases differ: {direct_path} vs {mtp_path}")
    parity_fields = (
        "prompt_suite_id",
        "prompt_text_sha256",
        "prompt_token_ids_sha256",
        "prompt_tokens",
        "generation_tokens",
        "sampler_settings",
        "seed",
        "random_seed",
    )
    for case_id, direct_row in direct_rows.items():
        mtp_row = mtp_rows[case_id]
        if any(direct_row.get(field) != mtp_row.get(field) for field in parity_fields):
            raise ValueError(
                f"direct/MTP prompt or decode contract differs for {case_id}: "
                f"{direct_path} vs {mtp_path}"
            )


def artifact_build_identity(path: Path, artifact: dict[str, Any]) -> ArtifactBuildIdentity:
    build = artifact.get("build")
    if not isinstance(build, dict):
        raise ValueError(f"{path} has no build provenance")
    engine_version = build.get("engine_version")
    if not isinstance(engine_version, str) or re.fullmatch(
        r"\d+\.\d+\.\d+", engine_version
    ) is None:
        raise ValueError(f"{path} has no semantic measured engine version")
    commit = build.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{path} has no full measured build commit")
    return ArtifactBuildIdentity(engine_version=engine_version, commit=commit)


def matching_build_identity(
    direct_path: Path,
    direct: dict[str, Any],
    mtp_path: Path,
    mtp: dict[str, Any],
) -> ArtifactBuildIdentity:
    direct_identity = artifact_build_identity(direct_path, direct)
    mtp_identity = artifact_build_identity(mtp_path, mtp)
    if direct_identity != mtp_identity:
        raise ValueError(
            "direct/MTP measured build identity differs: "
            f"{direct_path} "
            f"({direct_identity.engine_version}, {direct_identity.commit}) vs "
            f"{mtp_path} ({mtp_identity.engine_version}, {mtp_identity.commit})"
        )
    return direct_identity


def validate_exact_seed_reproducibility(
    direct_path: Path,
    direct: dict[str, Any],
    mtp_path: Path,
    mtp: dict[str, Any],
) -> None:
    direct_rows = {
        str(row.get("prompt_case_id")): row
        for row in direct.get("results", [])
        if row.get("prompt_case_id") is not None
    }
    mtp_rows = {
        str(row.get("prompt_case_id")): row
        for row in mtp.get("results", [])
        if row.get("prompt_case_id") is not None
    }
    if not direct_rows or direct_rows.keys() != mtp_rows.keys():
        raise ValueError(f"direct/MTP prompt cases differ: {direct_path} vs {mtp_path}")
    for case_id, direct_row in direct_rows.items():
        direct_tokens = [trial.get("output_token_ids") for trial in direct_row.get("trials", [])]
        mtp_tokens = [
            trial.get("output_token_ids") for trial in mtp_rows[case_id].get("trials", [])
        ]
        if (
            not direct_tokens
            or not mtp_tokens
            or any(tokens != direct_tokens[0] for tokens in direct_tokens)
            or any(tokens != mtp_tokens[0] for tokens in mtp_tokens)
        ):
            raise ValueError(
                f"exact MTP seed-reproducibility oracle failed for {case_id}: "
                f"{direct_path} vs {mtp_path}"
            )


def build_summary(
    output_dir: Path,
    args: argparse.Namespace,
    targets: tuple[Target, ...],
    suites: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matrix_build_identity: ArtifactBuildIdentity | None = None
    for target in targets:
        for suite in suites:
            direct_path = output_dir / target.key / suite / "ax_direct.json"
            mtp_path = output_dir / target.key / suite / "ax_mtp.json"
            direct = load_artifact(direct_path)
            mtp = load_artifact(mtp_path)
            row_build_identity = matching_build_identity(
                direct_path, direct, mtp_path, mtp
            )
            if matrix_build_identity is None:
                matrix_build_identity = row_build_identity
            elif row_build_identity != matrix_build_identity:
                raise ValueError(
                    "MTP matrix mixes measured build identities: "
                    f"expected ({matrix_build_identity.engine_version}, "
                    f"{matrix_build_identity.commit}), found "
                    f"({row_build_identity.engine_version}, {row_build_identity.commit}) "
                    f"at {direct_path.parent}"
                )
            if args.approximate_speed_ceiling:
                validate_approximate_mtp_artifact(mtp_path, mtp)
                publication_reasons = ["approximate_optimistic_not_publishable"]
            else:
                validate_exact_mtp_artifact(
                    mtp_path,
                    mtp,
                    require_qwen_linear_exact=not target.assistant_mtp,
                )
                validate_exact_artifact_rows(
                    direct_path,
                    direct,
                    expected_engines={"ax_engine_mlx"},
                    expected_suite=suite,
                )
                validate_exact_artifact_rows(
                    mtp_path,
                    mtp,
                    expected_engines=(
                        {"ax_engine_gemma4_assistant_mtp"}
                        if target.assistant_mtp
                        else {"ax_engine_mlx_pure_mtp"}
                    ),
                    expected_suite=suite,
                )
                validate_exact_prompt_parity(direct_path, direct, mtp_path, mtp)
                validate_exact_seed_reproducibility(direct_path, direct, mtp_path, mtp)
                publication_reasons = exact_publication_methodology_reasons(direct, mtp)
            direct_decode = metric_median(direct, "decode_tok_s")
            mtp_decode = metric_median(mtp, "decode_tok_s")
            quality_pct, quality_kind = draft_quality(mtp, assistant_mtp=target.assistant_mtp)
            coverage = mtp_coverage(mtp)
            row = {
                "model_id": target.key,
                "model": target.label,
                "suite_id": suite,
                "suite": suite,
                "mode": target.mode,
                "depth": target.mtp_depth,
                "ax_direct_decode_tok_s": direct_decode,
                "ax_mtp_decode_tok_s": mtp_decode,
                "ax_mtp_speedup_x": mtp_decode / direct_decode,
                "ax_mtp_prefill_tok_s": metric_median(mtp, "prefill_tok_s"),
                "ax_mtp_ttft_ms": metric_median(mtp, "ttft_ms"),
                "ax_mtp_accept_rate_pct": accept_rate_pct(mtp),
                "ax_mtp_accept_rate_kind": (
                    "verified_accept_rate"
                    if target.assistant_mtp
                    else (
                        "optimistic_policy_accept_rate"
                        if args.approximate_speed_ceiling
                        else "verified_accept_rate"
                    )
                ),
                "ax_mtp_draft_quality_pct": quality_pct,
                "ax_mtp_draft_quality_kind": quality_kind,
                "ax_mtp_step_coverage_pct": coverage["step_coverage_pct"],
                "ax_mtp_decode_steps": coverage["mtp_decode_steps"],
                "ax_mtp_emitted_tokens": coverage["mtp_emitted_tokens"],
                "ax_mtp_direct_fallback_steps": coverage["direct_fallback_steps"],
                "ax_mtp_decode_route_steps": coverage["decode_route_steps"],
                "ax_mtp_fallback_prompt_count": coverage["fallback_prompt_count"],
                "prompt_count": coverage["prompt_count"],
                "publication_candidate": not publication_reasons,
                "publication_reasons": publication_reasons,
                "ax_mtp_ngram_telemetry": aggregate_ngram_telemetry(mtp),
                "artifact": str(mtp_path.relative_to(REPO_ROOT)),
                "mtplx": "N/A",
                "lightning_mlx": "N/A",
            }
            rows.append(row)
    if matrix_build_identity is None:
        raise ValueError("MTP summary requires at least one measured build identity")
    publication_candidate = bool(rows) and all(row["publication_candidate"] for row in rows)
    return {
        "schema": (
            "ax.mtp_6bit_approximate_diagnostic_summary.v2"
            if args.approximate_speed_ceiling
            else MTP_6BIT_EXACT_SCHEMA
        ),
        "publication_candidate": publication_candidate,
        "claim_type": (
            "approximate_optimistic_diagnostic"
            if args.approximate_speed_ceiling
            else MTP_6BIT_EXACT_CLAIM_TYPE
        ),
        "engine_version": matrix_build_identity.engine_version,
        "build_commit": matrix_build_identity.commit,
        "run_dir": str(output_dir.relative_to(REPO_ROOT)),
        "methodology": {
            "targets": [target.key for target in targets],
            "suites": list(suites),
            "generated_tokens": args.generated_tokens,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "inter_case_cooldown_s": args.inter_case_cooldown,
            "sampling": MTP_SAMPLING,
            "correctness_contract": (
                "explicit approximate optimistic speed ceiling; not exact and not publication eligible"
                if args.approximate_speed_ceiling
                else "distribution-exact MTP with deterministic-delta proposals, residual rejection correction, and per-mode seed reproducibility"
            ),
            "qwen_linear_mtp_verifier_profile": (
                "explicit validated exact profile for every Qwen MTP row; "
                "not applicable to Gemma assistant-MTP rows"
            ),
            "comparison": "AX MTP decode median divided by AX direct decode median for the same model package and prompt suite.",
            "mtp_ngram": "disabled; no MTP+n-gram rows are run or promoted",
        },
        "peer_support": {
            "mtplx": {
                "value": None,
                "label": "N/A",
                "reason": "Not run: this artifact is AX Engine only and compares each prepared 6-bit download-mtp package against the same package with MTP disabled.",
            },
            "lightning_mlx": {
                "value": None,
                "label": "N/A",
                "reason": "Not run: this artifact is AX Engine only and compares each prepared 6-bit download-mtp package against the same package with MTP disabled.",
            },
        },
        "rows": rows,
    }


def fmt_tok(value: float) -> str:
    return f"{value:.1f} tok/s"


def fmt_ms(value: float) -> str:
    return f"{value:.0f} ms"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def draft_quality_label(row: dict[str, Any]) -> str:
    suffix = (
        "match" if row["ax_mtp_draft_quality_kind"] == "target_argmax_match_ewma" else "verified"
    )
    return f"{float(row['ax_mtp_draft_quality_pct']):.1f}% {suffix}"


def table_lines(rows: list[dict[str, Any]], *, approximate_diagnostic: bool) -> list[str]:
    if approximate_diagnostic:
        lines = [
            "| Target | Suite | AX direct decode | Approx. MTP decode | Diagnostic ratio | Draft quality | MTP step coverage | Fallback prompts |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| `{model_id}` | `{suite_id}` | {direct} | {mtp} | {speedup:.2f}x | {quality} | {coverage:.1f}% | {fallback}/{prompts} |".format(
                    model_id=row["model_id"],
                    suite_id=row["suite_id"],
                    direct=fmt_tok(float(row["ax_direct_decode_tok_s"])),
                    mtp=fmt_tok(float(row["ax_mtp_decode_tok_s"])),
                    speedup=float(row["ax_mtp_speedup_x"]),
                    quality=draft_quality_label(row),
                    coverage=float(row["ax_mtp_step_coverage_pct"]),
                    fallback=int(row["ax_mtp_fallback_prompt_count"]),
                    prompts=int(row["prompt_count"]),
                )
            )
        return lines

    lines = [
        "| Target | Suite | AX direct decode | AX MTP decode | AX MTP/direct | "
        "AX MTP prefill | AX MTP TTFT | AX accept |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model_id}` | `{suite_id}` | {direct} | {mtp} | {speedup:.2f}x | {prefill} | {ttft} | {accept} |".format(
                model_id=row["model_id"],
                suite_id=row["suite_id"],
                direct=fmt_tok(float(row["ax_direct_decode_tok_s"])),
                mtp=fmt_tok(float(row["ax_mtp_decode_tok_s"])),
                speedup=float(row["ax_mtp_speedup_x"]),
                prefill=fmt_tok(float(row["ax_mtp_prefill_tok_s"])),
                ttft=fmt_ms(float(row["ax_mtp_ttft_ms"])),
                accept=fmt_pct(float(row["ax_mtp_accept_rate_pct"])),
            )
        )
    return lines


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    approximate_diagnostic = summary["claim_type"] == "approximate_optimistic_diagnostic"
    title = (
        "# 6-bit MTP AX approximate diagnostic"
        if approximate_diagnostic
        else "# 6-bit MTP AX comparison summary"
    )
    description = (
        "This non-publishable artifact records an explicit optimistic speed diagnostic. "
        "It does not establish exact-distribution MTP acceleration."
        if approximate_diagnostic
        else "This artifact compares exact AX MTP decode with AX direct decode."
    )
    ratio_label = "diagnostic ratio" if approximate_diagnostic else "comparison ratio"
    lines = [
        title,
        "",
        description,
        "",
        (
            f"Measured binary: AX Engine v{summary['engine_version']} at "
            f"`{summary['build_commit']}`."
        ),
        "",
        f"The {ratio_label} is `AX MTP decode tok/s / AX direct decode tok/s` for the same prepared `download-mtp` package and prompt suite. It is not a cross-model speed ranking.",
        "",
        *table_lines(summary["rows"], approximate_diagnostic=approximate_diagnostic),
        "",
        "This is an AX Engine only artifact. Peer engines are intentionally not run here; each row compares the prepared AX 6-bit `download-mtp` package against the same package with MTP disabled.",
        "",
        "Pure-MTP verification: all AX MTP rows have zero n-gram accepted, proposed, submitted, and hit-step telemetry.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def validate_readme_publication_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema") != MTP_6BIT_EXACT_SCHEMA:
        raise ValueError("README update requires the exact MTP comparison schema")
    if summary.get("publication_candidate") is not True:
        raise ValueError("README update requires a publication-candidate MTP summary")
    if summary.get("claim_type") != MTP_6BIT_EXACT_CLAIM_TYPE:
        raise ValueError("README update requires an exact MTP comparison claim")
    engine_version = summary.get("engine_version")
    if not isinstance(engine_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", engine_version):
        raise ValueError("README update requires a semantic engine_version")
    build_commit = summary.get("build_commit")
    if not isinstance(build_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", build_commit
    ) is None:
        raise ValueError("README update requires a full measured build_commit")

    methodology = summary.get("methodology")
    if not isinstance(methodology, dict):
        raise ValueError("README update requires recorded MTP methodology")
    expected_methodology = {
        "generated_tokens": GENERATED_TOKENS,
        "repetitions": REPETITIONS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "sampling": MTP_SAMPLING,
    }
    for key, expected in expected_methodology.items():
        if methodology.get(key) != expected:
            raise ValueError(f"README update requires methodology {key}={expected!r}")

    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("README update requires non-empty exact MTP rows")
    expected_rows = {
        (target.key, suite) for target in SUPPORTED_TARGETS for suite in DEFAULT_SUITES
    }
    actual_rows: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("README update requires object-valued MTP rows")
        row_key = (str(row.get("model_id")), str(row.get("suite_id")))
        if row_key in actual_rows:
            raise ValueError(f"README update found duplicate MTP row {row_key!r}")
        actual_rows.add(row_key)
        if row.get("publication_candidate") is not True:
            raise ValueError(f"README update found non-publishable MTP row {row_key!r}")
        if row.get("publication_reasons") != []:
            raise ValueError(f"README update found MTP publication reasons for {row_key!r}")
        try:
            direct = float(row["ax_direct_decode_tok_s"])
            mtp = float(row["ax_mtp_decode_tok_s"])
            speedup = float(row["ax_mtp_speedup_x"])
            prefill = float(row["ax_mtp_prefill_tok_s"])
            ttft = float(row["ax_mtp_ttft_ms"])
            accept = float(row["ax_mtp_accept_rate_pct"])
            coverage = float(row["ax_mtp_step_coverage_pct"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"README update found incomplete numeric MTP row {row_key!r}"
            ) from error
        metrics = (direct, mtp, speedup, prefill, ttft, accept, coverage)
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError(f"README update found non-finite MTP metric for {row_key!r}")
        if not all(value > 0.0 for value in (direct, mtp, speedup, prefill, ttft, accept)):
            raise ValueError(f"README update found non-positive MTP metric for {row_key!r}")
        if abs(speedup - mtp / direct) > 0.001:
            raise ValueError(
                f"README update found an inconsistent MTP/direct ratio for {row_key!r}"
            )
        if coverage != 100.0:
            raise ValueError(f"README update requires 100% MTP step coverage for {row_key!r}")
        if int(row.get("ax_mtp_fallback_prompt_count", -1)) != 0:
            raise ValueError(f"README update requires zero fallback prompts for {row_key!r}")
        if int(row.get("ax_mtp_direct_fallback_steps", -1)) != 0:
            raise ValueError(f"README update requires zero direct fallback steps for {row_key!r}")
        ngram = row.get("ax_mtp_ngram_telemetry")
        if not isinstance(ngram, dict) or any(ngram.get(key) != 0 for key in NGRAM_ZERO_KEYS):
            raise ValueError(f"README update requires zero n-gram telemetry for {row_key!r}")

    if actual_rows != expected_rows:
        missing = sorted(expected_rows - actual_rows)
        extra = sorted(actual_rows - expected_rows)
        raise ValueError(
            f"README update requires the complete supported MTP matrix; missing={missing}, extra={extra}"
        )

    run_dir = summary.get("run_dir")
    if not isinstance(run_dir, str) or re.search(r"(?:^|/)(\d{4}-\d{2}-\d{2})", run_dir) is None:
        raise ValueError("README update requires a dated MTP run_dir")


def render_readme_section(summary: dict[str, Any]) -> str:
    validate_readme_publication_summary(summary)
    rows = summary["rows"]
    run_dir = summary["run_dir"]
    run_link = f"../{run_dir}"
    date_match = re.search(r"(?:^|/)(\d{4}-\d{2}-\d{2})", run_dir)
    assert date_match is not None
    run_date = date_match.group(1)
    engine_version = summary["engine_version"]
    build_commit = summary["build_commit"]
    min_speedup = min(float(row["ax_mtp_speedup_x"]) for row in rows)
    max_speedup = max(float(row["ax_mtp_speedup_x"]) for row in rows)
    wins = sum(float(row["ax_mtp_speedup_x"]) > 1.0 for row in rows)
    ties = sum(float(row["ax_mtp_speedup_x"]) == 1.0 for row in rows)
    losses = len(rows) - wins - ties
    win_label = "win" if wins == 1 else "wins"
    tie_label = "tie" if ties == 1 else "ties"
    loss_label = "loss" if losses == 1 else "losses"
    lines = [
        f"#### AX Engine v{engine_version} 6-bit exact sampled-MTP comparison ({run_date})",
        "",
        "This AX Engine-only matrix compares each prepared 6-bit `download-mtp`",
        "package with MTP disabled and enabled. The enabled route uses",
        "distribution-exact sampled MTP with deterministic-delta proposals and",
        "residual rejection correction. Qwen rows explicitly select the validated",
        "linear-attention exact-verifier profile; this is not an optimistic speed",
        "ceiling or a cross-engine leaderboard.",
        "",
        f"Measured binary provenance: AX Engine v{engine_version}, commit `{build_commit}`.",
        "",
        (
            f"Across {len(rows)} target/suite rows: {wins} MTP {win_label}, "
            f"{ties} {tie_label}, and {losses} {loss_label}; MTP/direct ratios "
            f"span {min_speedup:.2f}x-{max_speedup:.2f}x."
        ),
        "Every row has 100% MTP step coverage, zero direct-fallback prompts or",
        "steps, and zero n-gram accepted, proposed, submitted, or hit-step",
        "telemetry.",
        "",
        (
            '<img src="assets/perf-mtp-6bit-ax-acceleration.svg" '
            f'alt="AX Engine v{engine_version} 6-bit exact sampled-MTP comparison '
            'of same-package direct and MTP decode throughput">'
        ),
        "",
        *table_lines(rows, approximate_diagnostic=False),
        "",
        "Methodology: sampled decode (`temperature=0.6`, `top_p=0.95`,",
        "`top_k=20`), 1,000 generated tokens, 2 warmups, 5 measured repetitions,",
        "and recorded cooldown. Prefill and TTFT are reported as context, not MTP",
        "decode-comparison claims, because speculative decoding starts after prompt",
        "prefill. Direct and MTP rows use the same package and prompt suite.",
        "",
        "Exactness is checked with per-mode seed reproducibility. Summary artifacts:",
        f"[`summary.md`]({run_link}/summary.md) and",
        f"[`summary.json`]({run_link}/summary.json).",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict[str, Any]) -> None:
    section = render_readme_section(summary)
    text = readme.read_text()
    section_match = re.search(r"^#### AX Engine(?: v\d+\.\d+\.\d+)? 6-bit", text, re.MULTILINE)
    if section_match is None:
        raise ValueError("README has no AX Engine 6-bit MTP section")
    section_start = section_match.start()
    section_end = text.find(
        "#### Qwen3.6 MTP peer decode comparison",
        section_start,
    )
    if section_end < 0:
        raise ValueError("README has no Qwen3.6 MTP peer section boundary")
    readme.write_text(text[:section_start] + section + "\n\n" + text[section_end:])


def parse_csv(value: str) -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in value.split(",") if entry.strip())
    if not entries:
        raise ValueError("comma-separated argument must not be empty")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-supported-mtp-ax-only-refresh",
    )
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument(
        "--targets",
        default=",".join(target.key for target in SUPPORTED_TARGETS),
        help="Comma-separated target keys to run.",
    )
    parser.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated prompt-suite ids to run.",
    )
    parser.add_argument("--generated-tokens", type=int, default=GENERATED_TOKENS)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--warmup-repetitions", type=int, default=WARMUP_REPETITIONS)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S)
    parser.add_argument("--inter-case-cooldown", type=float, default=INTER_CASE_COOLDOWN_S)
    parser.add_argument(
        "--max-load-average",
        type=float,
        default=MAX_PUBLICATION_LOAD_AVERAGE,
    )
    parser.add_argument(
        "--max-top-process-cpu-percent",
        type=float,
        default=MAX_PUBLICATION_PROCESS_CPU_PERCENT,
    )
    parser.add_argument(
        "--load-wait-timeout",
        type=float,
        default=DEFAULT_LOAD_WAIT_TIMEOUT_S,
    )
    parser.add_argument(
        "--load-poll-interval",
        type=float,
        default=DEFAULT_LOAD_POLL_INTERVAL_S,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--approximate-speed-ceiling",
        action="store_true",
        help=(
            "Run explicit optimistic MTP as a non-publishable approximate speed ceiling. "
            "Omit this flag for the distribution-exact MTP route."
        ),
    )
    parser.add_argument("--no-build-ax-engine", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0 or args.warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups must be non-negative")
    if args.max_load_average < 0 or args.max_top_process_cpu_percent < 0:
        raise ValueError("performance gate limits must be non-negative")
    if args.load_wait_timeout < 0 or args.load_poll_interval <= 0:
        raise ValueError("performance gate wait must be non-negative with positive polling")
    if args.update_readme and args.approximate_speed_ceiling:
        raise ValueError("approximate MTP speed ceilings cannot update README claims")
    args.output_dir = args.output_dir.resolve()
    target_keys = parse_csv(args.targets)
    unknown_targets = [key for key in target_keys if key not in TARGETS_BY_KEY]
    if unknown_targets:
        known = ", ".join(TARGETS_BY_KEY)
        raise ValueError(f"unknown target(s): {', '.join(unknown_targets)}; known: {known}")
    targets = tuple(TARGETS_BY_KEY[key] for key in target_keys)
    suites = parse_csv(args.suites)
    for target in targets:
        validate_model_dir(target.model_dir)
    for suite in suites:
        path = args.suites_dir / f"{suite}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.no_build_ax_engine:
        build_server()
    for target in targets:
        for suite in suites:
            suite_dir = args.output_dir / target.key / suite
            maybe_run_case(
                target=target,
                suite=suite,
                mode="direct",
                output_path=suite_dir / "ax_direct.json",
                args=args,
            )
            maybe_run_case(
                target=target,
                suite=suite,
                mode="mtp",
                output_path=suite_dir / "ax_mtp.json",
                args=args,
            )
    summary = build_summary(args.output_dir, args, targets, suites)
    write_summary_files(args.output_dir, summary)
    print(f"[summary] {args.output_dir / 'summary.json'}", flush=True)
    print(f"[summary] {args.output_dir / 'summary.md'}", flush=True)
    if args.update_readme:
        update_readme(README_PATH, summary)
        print(f"[readme] updated {README_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
