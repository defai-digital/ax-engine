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
    "mtp": "ax_engine_gemma4_assistant_mtp",
    "mtp-ngram": "ax_engine_gemma4_assistant_mtp_ngram",
}


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


BENCH_PROFILES = {
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
    "assistant_mtp_ngram_confidence050": BenchProfile(
        key="assistant_mtp_ngram_confidence050",
        label="assistant MTP + n-gram confidence 0.50",
        mode="mtp-ngram",
        env={"AX_MLX_MTP_NGRAM_CONFIDENCE_THRESHOLD": "0.50"},
    ),
    "assistant_mtp_gate095": BenchProfile(
        key="assistant_mtp_gate095",
        label="assistant MTP confidence gate 0.95",
        mode="mtp",
        env={"AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE": "0.95"},
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
    "mtp": "assistant_mtp_default",
    "mtp-ngram": "assistant_mtp_ngram_default",
}


GEMMA4_PROFILES = {
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
        "--ax-gemma4-assistant-mtp",
        "--ax-mtp-max-depth",
        str(depth),
        "--output",
        str(output_path),
    ]
    if mode == "mtp":
        cmd.append("--ax-mtp-disable-ngram-stacking")
    if no_build:
        cmd.append("--no-build-ax-engine")
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
    claim_statuses: list[str] = []
    effective_routes: list[str] = []
    for row in rows:
        assistant = row.get("ax_mlx_gemma4_assistant_mtp") or {}
        assistant_drafted += int(
            assistant.get("ax_mlx_gemma4_assistant_mtp_draft_tokens", 0) or 0
        )
        assistant_accepted += int(
            assistant.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens", 0) or 0
        )
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        generic_mtp_drafted += telemetry_int(telemetry, "ax_mtp_draft_tokens")
        generic_mtp_accepted += telemetry_int(telemetry, "ax_mtp_accepted_tokens")
        ngram_drafted += telemetry_int(telemetry, "ax_ngram_draft_tokens")
        ngram_accepted += telemetry_int(telemetry, "ax_ngram_accepted_tokens")
        ngram_drafted += telemetry_int(telemetry, "ax_mtp_ngram_submitted_tokens")
        ngram_accepted += telemetry_int(
            telemetry,
            "ax_mtp_ngram_submitted_accepted_tokens",
        )
        ngram_hit_steps += telemetry_int(telemetry, "ax_mtp_ngram_hit_steps")
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
        "claim_statuses": sorted(set(claim_statuses)),
        "effective_routes": sorted(set(effective_routes)),
        "build": payload.get("build", {}),
    }


def fmt_number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Gemma 4 Assistant MTP Benchmark",
        "",
        f"Output: `{output_dir}`",
        "",
        "| Model | Suite | Profile | Mode | Depth | Decode tok/s | Assistant accept | MTP accept | n-gram accept | n-gram hits | Utility gates | Safety tightens |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {model} | {suite} | {profile} | {mode} | {depth} | {decode} | {assistant} | {mtp} | {ngram} | {hits} | {utility_gates} | {safety_tightens} |".format(
                model=row["model_label"],
                suite=row["suite"],
                profile=row["profile"],
                mode=row["mode"],
                depth=row["depth"],
                decode=fmt_number(row["decode_tok_s_median"]),
                assistant=fmt_pct(row["assistant_accept_rate"]),
                mtp=fmt_pct(row["mtp_accept_rate"]),
                ngram=fmt_pct(row["ngram_accept_rate"]),
                hits=row["ngram_hit_steps"],
                utility_gates=row["ngram_utility_gated_steps"],
                safety_tightens=row["ngram_safety_tightened_steps"],
            )
        )
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
        "--resume",
        action="store_true",
        help="Reuse existing per-suite artifacts instead of rerunning them.",
    )
    args = parser.parse_args()

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
                        depth=profile.depth,
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
                        "depth": profile.depth,
                        "model_dir": str(model_dir),
                    }
                )
                rows.append(row)

    summary = {
        "schema": "ax.gemma4_assistant_mtp_benchmark.v1",
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
    }
    write_summary(output_dir, summary)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
