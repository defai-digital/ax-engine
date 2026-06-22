#!/usr/bin/env python3
"""Fair Qwen3.6 MTP benchmark across MTPLX and AX Engine."""

from __future__ import annotations

import argparse
import html
import json
import math
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
import bench_mtp_differential as diff
import check_mtp_sidecar_provenance as provenance_check

REPO_ROOT = SCRIPT_DIR.parents[0]
AX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
MTPLX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mtplx_prompt_suites.py"
RAPID_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_rapid_mlx_prompt_suites.py"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "mtp-fair"
DEFAULT_MTPLX_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
DEFAULT_RAPID_PYTHON = Path("/opt/homebrew/var/mtplx/venv-0.3.7/bin/python")
DEFAULT_LIGHTNING_SOURCE = REPO_ROOT / ".internal" / "reference" / "lightning-mlx"
HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
)
# Vendor-default labeling: each row is run with the vendor's recommended
# config for this model. Labels say "default" so reviewers don't expect
# strict apples-to-apples knob alignment.
#
# Lightning-MLX was removed from the active framework on 2026-06-03. The
# Lightning runner code (`run_rapid_mlx_suite`, `--lightning-*` CLI args)
# and reference source under `.internal/reference/lightning-mlx` are kept
# for ad-hoc probes but Lightning rows no longer appear in the matrix.
ENGINE_LABELS = {
    "ax_direct": "AX Engine v5.2.2 (direct same-artifact baseline)",
    "mtplx": "MTPLX 0.3.7 (default)",
    "mtplx_tuned": "MTPLX 0.3.7 (tuned)",
    "ax_engine": "AX Engine v5.2.2 (MTP-only)",
    "ax_engine_ngram": "AX Engine v5.2.2 (MTP + n-gram stacking)",
    "ax_engine_tuned": "AX Engine v5.2.2 (tuned best-of)",
}
VERSIONS_FOOTNOTE = (
    "Vendor-default config per row · MTPLX 0.3.7 · AX Engine v5.2.2"
)
ENGINE_COLORS = {
    "ax_direct": "#d97706",
    "mtplx": "#14532d",
    "mtplx_tuned": "#0f766e",
    "ax_engine": "#f97316",
    "ax_engine_ngram": "#eab308",
    "ax_engine_tuned": "#2563eb",
}
ENGINE_ORDER = [
    "ax_direct",
    "mtplx",
    "mtplx_tuned",
    "ax_engine",
    "ax_engine_ngram",
    "ax_engine_tuned",
]
AX_DIRECT_ENGINE_KEY = "ax_engine_mlx"
AX_DIRECT_FAIR_ENGINE = "ax_direct"
AX_FAIR_ENGINES = {AX_DIRECT_FAIR_ENGINE, "ax_engine", "ax_engine_ngram", "ax_engine_tuned"}
MTPLX_FAIR_ENGINES = {"mtplx", "mtplx_tuned"}
AX_RESULT_ENGINES = {AX_DIRECT_ENGINE_KEY, *diff.AX_MTP_ENGINES}
KEEP_DEFAULT_MIN_GAIN = 0.05
NGRAM_KEEP_DEFAULT_MIN_GAIN = 0.03
MAX_SUITE_REGRESSION = 0.03

# User-facing engine vendor names for --engines
ENGINE_VENDORS = ["mtplx", "ax"]
# User-facing decode mode names for --modes
ENGINE_MODES = ["direct", "mtp", "mtp-ngram", "tuned"]
# Maps (vendor, mode) -> internal engine key, or None if not yet supported.
# None entries are skipped at runtime with a warning; update this matrix
# when a vendor adds support for a new mode.
SUPPORT_MATRIX: dict[tuple[str, str], str | None] = {
    ("mtplx", "direct"): None,  # not a same-artifact AX direct baseline
    ("mtplx", "mtp"): "mtplx",
    ("mtplx", "mtp-ngram"): None,  # not supported yet
    ("mtplx", "tuned"): "mtplx_tuned",
    ("ax", "direct"): AX_DIRECT_FAIR_ENGINE,
    ("ax", "mtp"): "ax_engine",
    ("ax", "mtp-ngram"): "ax_engine_ngram",
    ("ax", "tuned"): "ax_engine_tuned",
}


def resolve_engines(vendors: list[str], modes: list[str]) -> list[str]:
    """Return ordered internal engine keys for the requested vendor × mode combos.

    Unsupported combinations (None in SUPPORT_MATRIX) are skipped with a warning.
    """
    keys: list[str] = []
    for vendor in vendors:
        for mode in modes:
            key = SUPPORT_MATRIX.get((vendor, mode))
            if key is None:
                print(
                    f"[skip] {vendor} does not support {mode} mode yet — "
                    "update SUPPORT_MATRIX when it does",
                    file=sys.stderr,
                )
            elif key not in keys:
                keys.append(key)
    return sorted(keys, key=lambda k: ENGINE_ORDER.index(k))


@dataclass(frozen=True)
class QwenProfile:
    key: str
    label: str
    model_key: str
    base_model_id: str
    source_model_id: str
    ax_local_slug: str
    lightning_alias: str
    native_depth: int
    is_moe: bool = False


@dataclass(frozen=True)
class BoxStats:
    values: tuple[float, ...]
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float


@dataclass(frozen=True)
class AxTuneCandidate:
    key: str
    policy: str
    depth: int | None


QWEN36_PROFILES = {
    "27b-4bit": QwenProfile(
        key="27b-4bit",
        label="Qwen3.6 27B 4-bit",
        model_key="27b",
        base_model_id="mlx-community/Qwen3.6-27B-4bit",
        source_model_id="Qwen/Qwen3.6-27B",
        ax_local_slug="models--ax-local--Qwen3.6-27B-MTP",
        lightning_alias="qwen3.6-27b",
        native_depth=3,
    ),
    "35b-a3b-4bit": QwenProfile(
        key="35b-a3b-4bit",
        label="Qwen3.6 35B-A3B 4-bit",
        model_key="35b",
        base_model_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        source_model_id="Qwen/Qwen3.6-35B-A3B",
        ax_local_slug="models--ax-local--Qwen3.6-35B-MTP",
        lightning_alias="qwen3.6-35b",
        native_depth=1,
        is_moe=True,
    ),
}


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile for empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def box_stats(values: list[float]) -> BoxStats:
    ordered = sorted(values)
    return BoxStats(
        values=tuple(values),
        minimum=ordered[0],
        q1=percentile(ordered, 0.25),
        median=percentile(ordered, 0.50),
        q3=percentile(ordered, 0.75),
        maximum=ordered[-1],
    )


def sidecar_dir(profile: QwenProfile, hf_cache: Path = HF_CACHE) -> Path:
    return hf_cache / profile.ax_local_slug / "snapshots" / "v1"


def validate_sidecar_for_profile(
    profile: QwenProfile, hf_cache: Path
) -> dict[str, Any]:
    model_dir = sidecar_dir(profile, hf_cache)
    try:
        _manifest_path, manifest = provenance_check.load_manifest(model_dir)
        return provenance_check.validate_manifest(
            manifest,
            strict_local=False,
            fair_base_only=True,
        )
    except provenance_check.ProvenanceError as exc:
        raise ValueError(
            f"invalid MTP sidecar for {profile.key}: {exc}. "
            f"Regenerate it with scripts/prepare_qwen36_mtp_sidecar.py --model {profile.model_key}"
        ) from exc


def suite_path_for(suite: str, suites_dir: Path) -> Path:
    return suites_dir / f"{suite}.jsonl"


def effective_depth(
    profile: QwenProfile, engines: list[str], policy: str, override: int | None
) -> int:
    if override is not None:
        return override
    if policy == "native":
        return profile.native_depth
    return profile.native_depth


def tunable_depths_for_profile(profile: QwenProfile) -> list[int]:
    return list(range(1, min(profile.native_depth, 3) + 1))


def ax_tune_candidates(profile: QwenProfile, policies: list[str]) -> list[AxTuneCandidate]:
    depths = tunable_depths_for_profile(profile)
    candidates: list[AxTuneCandidate] = []
    if "direct" in policies:
        candidates.append(AxTuneCandidate("direct", "direct", None))
    if "ngram" in policies:
        candidates.append(AxTuneCandidate("ngram", "ngram", 0))
    for depth in depths:
        if "mtp" in policies:
            candidates.append(AxTuneCandidate(f"mtp_d{depth}", "mtp", depth))
        if "mtp-ngram" in policies:
            candidates.append(AxTuneCandidate(f"mtp_ngram_d{depth}", "mtp-ngram", depth))
    return candidates


def tune_config_from_args(
    args: argparse.Namespace, *, depth: int, enable_thinking: bool
) -> diff.RunConfig:
    return diff.RunConfig(
        mode=args.mode,
        depth=depth,
        max_tokens=args.tune_max_tokens,
        repetitions=args.tune_repetitions,
        warmup_repetitions=args.tune_warmup_repetitions,
        cooldown_s=args.tune_cooldown,
        sampling=diff.sampling_for_mode(
            args.mode,
            {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        ),
        enable_thinking=enable_thinking,
    )


def run_subprocess(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def command_from_error(
    error: Exception, fallback: list[str] | None
) -> list[str] | None:
    if fallback is not None:
        return fallback
    command = getattr(error, "cmd", None)
    if command is None:
        return None
    if isinstance(command, list):
        return [str(part) for part in command]
    return [str(command)]


def write_error_artifact(
    path: Path,
    *,
    engine: str,
    error: Exception,
    command: list[str] | None,
    log_paths: list[Path] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ax.mtp_engine_error.v1",
        "engine": engine,
        "error": str(error),
        "command": command_from_error(error, command),
    }
    returncode = getattr(error, "returncode", None)
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if log_paths:
        payload["log_paths"] = [
            str(log_path) for log_path in log_paths if log_path.is_file()
        ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_ax_suite(
    *,
    python: Path,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    config: diff.RunConfig,
    no_build: bool,
    ax_policy: str = "mtp-ngram",
    inter_case_cooldown_s: float = 0.0,
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
        str(config.max_tokens),
        "--repetitions",
        str(config.repetitions),
        "--cooldown",
        str(config.cooldown_s),
        "--inter-case-cooldown",
        str(inter_case_cooldown_s),
        "--ax-sampling",
        json.dumps(config.sampling, sort_keys=True),
        "--skip-mlx-lm",
        "--capture-output-token-ids",
        "--output",
        str(output_path),
    ]
    if ax_policy == "direct":
        cmd.append("--ax-direct")
    else:
        mtp_depth = 0 if ax_policy == "ngram" else config.depth
        cmd += [
            "--ax-ngram-accel",
            "--ax-mtp-max-depth",
            str(mtp_depth),
        ]
    if ax_policy == "mtp":
        cmd.append("--ax-mtp-disable-ngram-stacking")
    if not config.enable_thinking:
        cmd.append("--no-thinking")
    if no_build:
        cmd.append("--no-build-ax-engine")
    run_subprocess(cmd)
    return output_path


def run_mtplx_suite(
    *,
    python: Path,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    config: diff.RunConfig,
    mtplx_profile: str = "stable",
    allow_unverified_model: bool = False,
    inter_case_cooldown_s: float = 0.0,
) -> Path:
    cmd = [
        str(python),
        str(MTPLX_BENCH_SCRIPT),
        "--model",
        str(model_dir),
        "--suite",
        suite,
        "--prompts",
        str(suite_file),
        "--output",
        str(output_path),
        "--depth",
        str(config.depth),
        "--temperature",
        str(config.sampling["temperature"]),
        "--top-p",
        str(config.sampling["top_p"]),
        "--top-k",
        str(config.sampling["top_k"]),
        "--max-tokens",
        str(config.max_tokens),
        "--repetitions",
        str(config.repetitions),
        "--warmup-repetitions",
        str(config.warmup_repetitions),
        "--cooldown",
        str(config.cooldown_s),
        "--inter-case-cooldown",
        str(inter_case_cooldown_s),
        "--profile",
        mtplx_profile,
    ]
    if not config.enable_thinking:
        cmd.append("--disable-thinking")
    if allow_unverified_model:
        cmd.append("--allow-unverified-model")
    run_subprocess(cmd)
    return output_path


def mtplx_best_depth_from_tune(payload: dict[str, Any], fallback_depth: int) -> int:
    best = payload.get("best") if isinstance(payload.get("best"), dict) else {}
    depth = best.get("depth")
    if isinstance(depth, int) and depth > 0:
        return depth
    return fallback_depth


def annotate_artifact(path: Path, key: str, value: dict[str, Any]) -> None:
    artifact = json.loads(path.read_text())
    artifact[key] = value
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def run_mtplx_tune(
    *,
    python: Path,
    output_path: Path,
    model_dir: Path,
    profile: QwenProfile,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    tune_dir = output_path.parent / "tune" / "mtplx"
    tune_dir.mkdir(parents=True, exist_ok=True)
    tune_output = tune_dir / "tune.json"
    state_path = tune_dir / "tuning-state.json"
    depths = tunable_depths_for_profile(profile)
    cmd = [
        str(python),
        "-m",
        "mtplx.cli",
        "tune",
        "--model",
        str(model_dir),
        "--depths",
        ",".join(str(depth) for depth in depths),
        "--max-tokens",
        str(args.tune_max_tokens),
        "--limit",
        str(args.tune_limit),
        "--seed",
        str(args.tune_seed),
        "--output-dir",
        str(tune_dir),
        "--output",
        str(tune_output),
        "--json",
        "--yes",
    ]
    if args.retune:
        cmd.append("--retune")
    env = {**os.environ, "MTPLX_TUNE_STATE": str(state_path)}
    run_subprocess(cmd, env=env)
    payload = json.loads(tune_output.read_text())
    selected_depth = mtplx_best_depth_from_tune(payload, profile.native_depth)
    return selected_depth, {
        "schema": "ax.qwen36_mtp_fair.tune.mtplx.v1",
        "strategy": "mtplx_cli_tune",
        "artifact": str(tune_output),
        "state_path": str(state_path),
        "retune": bool(args.retune),
        "candidate_depths": depths,
        "selected_depth": selected_depth,
        "best": payload.get("best"),
        "saved": payload.get("saved"),
        "save_skipped_reason": payload.get("save_skipped_reason"),
    }


def run_mtplx_tuned_suite(
    *,
    python: Path,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    config: diff.RunConfig,
    profile: QwenProfile,
    args: argparse.Namespace,
    mtplx_profile: str = "stable",
    allow_unverified_model: bool = False,
    inter_case_cooldown_s: float = 0.0,
) -> Path:
    selected_depth, tuning = run_mtplx_tune(
        python=python,
        output_path=output_path,
        model_dir=model_dir,
        profile=profile,
        args=args,
    )
    tuned_config = diff.RunConfig(
        mode=config.mode,
        depth=selected_depth,
        max_tokens=config.max_tokens,
        repetitions=config.repetitions,
        warmup_repetitions=config.warmup_repetitions,
        cooldown_s=config.cooldown_s,
        sampling=config.sampling,
        enable_thinking=config.enable_thinking,
    )
    result = run_mtplx_suite(
        python=python,
        suite=suite,
        suite_file=suite_file,
        output_path=output_path,
        model_dir=model_dir,
        config=tuned_config,
        mtplx_profile=mtplx_profile,
        allow_unverified_model=allow_unverified_model,
        inter_case_cooldown_s=inter_case_cooldown_s,
    )
    annotate_artifact(result, "tuning", tuning)
    return result


def run_rapid_mlx_suite(
    *,
    python: Path,
    lightning_source: Path,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    config: diff.RunConfig,
    port: int = 18765,
    enable_ngram: bool = False,
    mtp_optimistic: bool = False,
    mtp_draft_temperature: float = 0.5,
    ngram_auto_disable_mtp_threshold: float = 0.85,
    ngram_auto_disable_min_ngram: float = 0.50,
    inter_case_cooldown_s: float = 0.0,
) -> Path:
    cmd = [
        str(python),
        str(RAPID_BENCH_SCRIPT),
        "--model",
        str(model_dir),
        "--suite",
        suite,
        "--prompts",
        str(suite_file),
        "--output",
        str(output_path),
        "--rapid-source",
        str(lightning_source),
        "--rapid-mtp-patch",
        "none",
        "--lightning-mode",
        "--depth",
        str(config.depth),
        "--temperature",
        str(config.sampling["temperature"]),
        "--top-p",
        str(config.sampling["top_p"]),
        "--top-k",
        str(config.sampling["top_k"]),
        "--max-tokens",
        str(config.max_tokens),
        "--repetitions",
        str(config.repetitions),
        "--warmup-repetitions",
        str(config.warmup_repetitions),
        "--cooldown",
        str(config.cooldown_s),
        "--inter-case-cooldown",
        str(inter_case_cooldown_s),
        "--port",
        str(port),
        "--mtp-draft-temperature",
        str(mtp_draft_temperature),
    ]
    if config.enable_thinking:
        cmd.append("--enable-thinking")
    else:
        cmd.append("--disable-thinking")
    if mtp_optimistic:
        cmd.append("--mtp-optimistic")
    if enable_ngram:
        cmd.append("--enable-ngram")
        cmd += [
            "--ngram-auto-disable-mtp-threshold",
            str(ngram_auto_disable_mtp_threshold),
            "--ngram-auto-disable-min-ngram",
            str(ngram_auto_disable_min_ngram),
        ]
    run_subprocess(cmd)
    return output_path


def run_ax_tune_candidate(
    *,
    args: argparse.Namespace,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    candidate: AxTuneCandidate,
    enable_thinking: bool,
    inter_case_cooldown_s: float,
) -> Path:
    config = tune_config_from_args(
        args,
        depth=int(candidate.depth or 0),
        enable_thinking=enable_thinking,
    )
    return run_ax_suite(
        python=args.ax_python,
        suite=suite,
        suite_file=suite_file,
        output_path=output_path,
        model_dir=model_dir,
        config=config,
        no_build=args.no_build_ax_engine,
        ax_policy=candidate.policy,
        inter_case_cooldown_s=inter_case_cooldown_s,
    )


def selected_ax_tune_candidate(
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in candidate_rows
        if isinstance(row.get("decode_tok_s"), int | float)
        and str(row.get("status")) != "error"
    ]
    if not valid_rows:
        raise RuntimeError("AX tune did not produce any valid candidate rows")
    return max(valid_rows, key=lambda row: float(row["decode_tok_s"]))


def write_ax_tune_summary(
    path: Path,
    *,
    profile: QwenProfile,
    suite: str,
    candidates: list[AxTuneCandidate],
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "schema": "ax.qwen36_mtp_fair.tune.ax.v1",
        "strategy": "best_decode_tok_s",
        "model": profile.key,
        "suite": suite,
        "candidate_policies": args.ax_tune_policies,
        "candidate_depths": tunable_depths_for_profile(profile),
        "tune_max_tokens": args.tune_max_tokens,
        "tune_repetitions": args.tune_repetitions,
        "tune_warmup_repetitions": args.tune_warmup_repetitions,
        "tune_cooldown_s": args.tune_cooldown,
        "candidates": rows,
        "selected": selected,
        "candidate_keys": [candidate.key for candidate in candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_ax_tuned_suite(
    *,
    args: argparse.Namespace,
    suite: str,
    suite_file: Path,
    output_path: Path,
    model_dir: Path,
    config: diff.RunConfig,
    profile: QwenProfile,
    inter_case_cooldown_s: float = 0.0,
) -> Path:
    candidates = ax_tune_candidates(profile, args.ax_tune_policies)
    if not candidates:
        raise RuntimeError("AX tune has no candidate policies")
    tune_dir = output_path.parent / "tune" / "ax_engine"
    tune_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_path = tune_dir / f"{candidate.key}.json"
        if args.skip_existing and candidate_path.is_file():
            artifact_path = candidate_path
        else:
            artifact_path = run_ax_tune_candidate(
                args=args,
                suite=suite,
                suite_file=suite_file,
                output_path=candidate_path,
                model_dir=model_dir,
                candidate=candidate,
                enable_thinking=config.enable_thinking,
                inter_case_cooldown_s=0.0,
            )
        summary = summarize_engine_artifact("ax_engine_tuned", artifact_path)
        candidate_rows.append(
            {
                "key": candidate.key,
                "policy": candidate.policy,
                "depth": candidate.depth,
                "artifact": str(artifact_path),
                "status": summary.get("status"),
                "decode_tok_s": summary.get("decode_tok_s"),
                "accept_rate": summary.get("accept_rate"),
                "ngram_accept_rate": summary.get("ngram_accept_rate"),
                "ngram_hit_steps": summary.get("ngram_hit_steps"),
            }
        )
    selected = selected_ax_tune_candidate(candidate_rows)
    tune_summary = write_ax_tune_summary(
        tune_dir / "summary.json",
        profile=profile,
        suite=suite,
        candidates=candidates,
        rows=candidate_rows,
        selected=selected,
        args=args,
    )
    selected_config = diff.RunConfig(
        mode=config.mode,
        depth=int(selected.get("depth") or 0),
        max_tokens=config.max_tokens,
        repetitions=config.repetitions,
        warmup_repetitions=config.warmup_repetitions,
        cooldown_s=config.cooldown_s,
        sampling=config.sampling,
        enable_thinking=config.enable_thinking,
    )
    result = run_ax_suite(
        python=args.ax_python,
        suite=suite,
        suite_file=suite_file,
        output_path=output_path,
        model_dir=model_dir,
        config=selected_config,
        no_build=args.no_build_ax_engine,
        ax_policy=str(selected["policy"]),
        inter_case_cooldown_s=inter_case_cooldown_s,
    )
    annotate_artifact(result, "tuning", tune_summary)
    return result


def run_engine_suite(
    args: argparse.Namespace,
    *,
    engine: str,
    profile: QwenProfile,
    suite: str,
    depth: int,
    port: int,
) -> Path:
    model_dir = sidecar_dir(profile, args.hf_cache)
    lightning_model = (
        profile.lightning_alias
        if getattr(args, "lightning_model_source", "sidecar") == "optimized-alias"
        else str(model_dir)
    )
    suite_file = suite_path_for(suite, args.suites_dir)
    output_path = args.output_dir / profile.key / suite / f"{engine}.json"
    config_enable_thinking = (
        bool(getattr(args, "lightning_enable_thinking", False))
        if engine in ("lightning_mlx", "lightning_mtp_ngram")
        else bool(args.enable_thinking)
    )
    config = diff.RunConfig(
        mode=args.mode,
        depth=depth,
        max_tokens=args.max_tokens,
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup_repetitions,
        cooldown_s=args.cooldown,
        sampling=diff.sampling_for_mode(
            args.mode,
            {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        ),
        enable_thinking=config_enable_thinking,
    )
    if args.skip_existing and output_path.is_file():
        return output_path
    try:
        if engine == AX_DIRECT_FAIR_ENGINE:
            return run_ax_suite(
                python=args.ax_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                no_build=args.no_build_ax_engine,
                ax_policy="direct",
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "ax_engine":
            return run_ax_suite(
                python=args.ax_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                no_build=args.no_build_ax_engine,
                ax_policy="mtp",
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "ax_engine_ngram":
            return run_ax_suite(
                python=args.ax_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                no_build=args.no_build_ax_engine,
                ax_policy="mtp-ngram",
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "ax_engine_tuned":
            return run_ax_tuned_suite(
                args=args,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                profile=profile,
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "mtplx":
            return run_mtplx_suite(
                python=args.mtplx_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                mtplx_profile=args.mtplx_profile,
                allow_unverified_model=profile.is_moe,
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "mtplx_tuned":
            return run_mtplx_tuned_suite(
                python=args.mtplx_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                profile=profile,
                args=args,
                mtplx_profile=args.mtplx_profile,
                allow_unverified_model=profile.is_moe,
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "lightning_mlx":
            return run_rapid_mlx_suite(
                python=args.rapid_python,
                lightning_source=args.lightning_source,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=Path(lightning_model),
                config=config,
                port=args.base_port + 1,
                mtp_optimistic=args.lightning_mtp_optimistic,
                mtp_draft_temperature=args.lightning_mtp_draft_temperature,
                ngram_auto_disable_mtp_threshold=args.lightning_ngram_auto_disable_mtp_threshold,
                ngram_auto_disable_min_ngram=args.lightning_ngram_auto_disable_min_ngram,
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        if engine == "lightning_mtp_ngram":
            return run_rapid_mlx_suite(
                python=args.rapid_python,
                lightning_source=args.lightning_source,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=Path(lightning_model),
                config=config,
                port=args.base_port + 1,
                enable_ngram=True,
                mtp_optimistic=args.lightning_mtp_optimistic,
                mtp_draft_temperature=args.lightning_mtp_draft_temperature,
                ngram_auto_disable_mtp_threshold=args.lightning_ngram_auto_disable_mtp_threshold,
                ngram_auto_disable_min_ngram=args.lightning_ngram_auto_disable_min_ngram,
                inter_case_cooldown_s=args.inter_case_cooldown,
            )
        raise ValueError(f"unknown engine: {engine}")
    except Exception as exc:
        return write_error_artifact(
            output_path,
            engine=engine,
            error=exc,
            command=None,
        )


def summary_median(value: Any) -> float | None:
    if isinstance(value, dict):
        median_value = value.get("median")
        return float(median_value) if median_value is not None else None
    if value is None:
        return None
    return float(value)


def numeric_samples(value: Any) -> list[float]:
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list):
            return [float(v) for v in values if isinstance(v, int | float)]
        median_value = value.get("median")
        return [float(median_value)] if isinstance(median_value, int | float) else []
    if isinstance(value, int | float):
        return [float(value)]
    return []


def metric_samples_from_summary(artifact: dict[str, Any], metric: str) -> list[float]:
    summary = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    values = numeric_samples(summary.get(metric))
    if values:
        return values

    samples: list[float] = []
    for item in artifact.get("results", []):
        if metric in item:
            samples.extend(numeric_samples(item.get(metric)))
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        if metric in summary:
            samples.extend(numeric_samples(summary.get(metric)))
    return samples


def telemetry_accept_rate(telemetry: dict[str, Any]) -> float | None:
    drafted = int(telemetry.get("ax_mtp_draft_tokens", 0) or 0)
    accepted = int(telemetry.get("ax_mtp_accepted_tokens", 0) or 0)
    return accepted / drafted if drafted else None


def depth_accept_rate(run: dict[str, Any]) -> float | None:
    accepted = sum(int(v) for v in run.get("accepted_by_depth", []) if v is not None)
    drafted = sum(int(v) for v in run.get("drafted_by_depth", []) if v is not None)
    return accepted / drafted if drafted else None


def ax_cases_for_fair(
    artifact: dict[str, Any],
    *,
    allowed_row_engines: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    allowed = allowed_row_engines or AX_RESULT_ENGINES
    for row in artifact.get("results", []):
        if row.get("engine") not in allowed:
            continue
        prompt_id = row.get("prompt_case_id")
        if not prompt_id:
            continue
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        draft_tokens = int(telemetry.get("ax_mtp_draft_tokens", 0) or 0)
        accepted_tokens = int(telemetry.get("ax_mtp_accepted_tokens", 0) or 0)
        ngram_draft_tokens = int(telemetry.get("ax_ngram_draft_tokens", 0) or 0)
        ngram_accepted_tokens = int(telemetry.get("ax_ngram_accepted_tokens", 0) or 0)
        cases[str(prompt_id)] = {
            "prompt_id": str(prompt_id),
            "decode_tok_s": summary_median(row.get("decode_tok_s")),
            "accepted_tokens": accepted_tokens,
            "draft_tokens": draft_tokens,
            "accept_rate": accepted_tokens / draft_tokens if draft_tokens else None,
            "ngram_accept_rate": ngram_accepted_tokens / ngram_draft_tokens
            if ngram_draft_tokens
            else None,
            "ngram_hit_steps": int(telemetry.get("ax_mtp_ngram_hit_steps", 0) or 0),
        }
    return cases


def accept_rate_samples_for_engine(
    engine: str, artifact: dict[str, Any]
) -> list[float]:
    samples: list[float] = []
    if engine == AX_DIRECT_FAIR_ENGINE:
        return samples
    if engine in AX_FAIR_ENGINES:
        for row in artifact.get("results", []):
            if row.get("engine") not in diff.AX_MTP_ENGINES:
                continue
            row_samples = [
                rate
                for trial in row.get("trials", [])
                if isinstance(trial.get("ngram_acceleration_telemetry"), dict)
                for rate in [
                    telemetry_accept_rate(trial["ngram_acceleration_telemetry"])
                ]
                if rate is not None
            ]
            if row_samples:
                samples.extend(row_samples)
                continue
            telemetry = row.get("ngram_acceleration_telemetry") or {}
            rate = telemetry_accept_rate(telemetry)
            if rate is not None:
                samples.append(rate)
    elif engine in MTPLX_FAIR_ENGINES:
        for case in artifact.get("results", []):
            case_samples = [
                rate
                for run in case.get("runs", [])
                for rate in [depth_accept_rate(run)]
                if rate is not None
            ]
            if case_samples:
                samples.extend(case_samples)
            else:
                samples.extend(
                    numeric_samples((case.get("summary") or {}).get("accept_rate"))
                )
    elif engine in ("lightning_mlx", "lightning_mtp_ngram"):
        for case in artifact.get("results", []):
            case_samples = [
                float(run["mtp_acceptance_ratio"])
                for run in case.get("runs", [])
                if isinstance(run.get("mtp_acceptance_ratio"), int | float)
            ]
            if case_samples:
                samples.extend(case_samples)
            else:
                samples.extend(
                    numeric_samples((case.get("summary") or {}).get("accept_rate"))
                )
    return samples


def cases_for_engine(
    engine: str, artifact: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if artifact.get("schema") == "ax.mtp_engine_error.v1":
        return {}
    if engine == AX_DIRECT_FAIR_ENGINE:
        return ax_cases_for_fair(artifact, allowed_row_engines={AX_DIRECT_ENGINE_KEY})
    if engine in AX_FAIR_ENGINES:
        return ax_cases_for_fair(artifact)
    if engine in MTPLX_FAIR_ENGINES:
        return diff.mtplx_cases(artifact)
    if engine in ("lightning_mlx", "lightning_mtp_ngram"):
        return diff.rapid_mlx_cases(artifact)
    raise ValueError(f"unknown engine: {engine}")


def summarize_engine_artifact(engine: str, artifact_path: Path) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema") == "ax.mtp_engine_error.v1":
        return {
            "engine": engine,
            "status": "error",
            "artifact": str(artifact_path),
            "error": artifact.get("error"),
            "case_count": 0,
            "decode_tok_s": None,
            "accept_rate": None,
        }
    cases = cases_for_engine(engine, artifact)
    aggregate = (
        artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    )
    validations_passed = artifact.get(
        "validations_passed", aggregate.get("validations_passed")
    )
    validations_total = artifact.get(
        "validations_total", aggregate.get("validations_total")
    )
    decode_values = [
        float(case["decode_tok_s"])
        for case in cases.values()
        if case.get("decode_tok_s") is not None
    ]
    accept_values = [
        float(case["accept_rate"])
        for case in cases.values()
        if case.get("accept_rate") is not None
    ]
    ngram_accept_values = [
        float(case["ngram_accept_rate"])
        for case in cases.values()
        if case.get("ngram_accept_rate") is not None
    ]
    ngram_hit_steps = sum(
        int(case.get("ngram_hit_steps", 0) or 0) for case in cases.values()
    )
    mtp_draft_tokens = sum(
        int(case.get("draft_tokens", 0) or 0) for case in cases.values()
    )
    mtp_accepted_tokens = sum(
        int(case.get("accepted_tokens", 0) or 0) for case in cases.values()
    )
    decode_samples = (
        metric_samples_from_summary(artifact, "decode_tok_s") or decode_values
    )
    accept_samples = accept_rate_samples_for_engine(engine, artifact) or accept_values
    status = "ok" if decode_values else "no_valid_runs"
    if (
        status == "ok"
        and validations_total is not None
        and validations_passed is not None
        and int(validations_passed) < int(validations_total)
    ):
        status = "ok_validation_warnings"
    return {
        "engine": engine,
        "status": status,
        "artifact": str(artifact_path),
        "case_count": len(cases),
        "valid_case_count": len(decode_values),
        "validations_passed": validations_passed,
        "validations_total": validations_total,
        "decode_tok_s": median(decode_values),
        "accept_rate": median(accept_values),
        "decode_tok_s_samples": decode_samples,
        "accept_rate_samples": accept_samples,
        "mtp_draft_tokens": mtp_draft_tokens,
        "mtp_accepted_tokens": mtp_accepted_tokens,
        "tuning": artifact.get("tuning") if isinstance(artifact.get("tuning"), dict) else None,
        "ngram_accept_rate": median(ngram_accept_values)
        if ngram_accept_values
        else None,
        "ngram_hit_steps": ngram_hit_steps
        if engine in AX_FAIR_ENGINES
        else None,
    }


def ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right <= 0:
        return None
    return left / right


def compare_suite_decodes(
    profile_suites: dict[str, float | None],
    baseline_suites: dict[str, float | None],
) -> dict[str, Any]:
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
    valid_deltas = [value for value in per_suite.values() if value is not None]
    return {
        "delta": (profile_agg - baseline_agg) / baseline_agg if baseline_agg else None,
        "worst_suite_delta": min(valid_deltas) if valid_deltas else None,
        "per_suite_delta": per_suite,
        "profile_agg": profile_agg,
        "baseline_agg": baseline_agg,
        "suites": common,
    }


def classify_vs_direct(
    delta: float | None,
    worst_suite_delta: float | None,
    drafted: bool,
) -> str:
    if delta is None:
        return "retest"
    if not drafted:
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
    if delta is None:
        return "retest"
    no_bad_regression = worst_suite_delta is None or worst_suite_delta >= -MAX_SUITE_REGRESSION
    if delta >= NGRAM_KEEP_DEFAULT_MIN_GAIN and no_bad_regression:
        return "keep-default"
    if delta >= 0:
        return "keep-opt-in"
    return "remove-claim"


def build_survival_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        model = row["model"]
        engines = index.setdefault(model, {})
        for engine, summary in row["engines"].items():
            entry = engines.setdefault(
                engine,
                {
                    "suites": {},
                    "drafted": False,
                },
            )
            entry["suites"][row["suite"]] = summary.get("decode_tok_s")
            if int(summary.get("mtp_draft_tokens", 0) or 0) > 0:
                entry["drafted"] = True

    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    for model in sorted(index):
        engines = index[model]
        direct = engines.get(AX_DIRECT_FAIR_ENGINE)
        if direct is None:
            warnings.append(
                f"{model}: no AX direct row; AX MTP survival cannot be judged "
                "against same-artifact direct decode."
            )
        for engine in ("ax_engine", "ax_engine_ngram", "ax_engine_tuned"):
            entry = engines.get(engine)
            if entry is None or direct is None:
                continue
            result = compare_suite_decodes(entry["suites"], direct["suites"])
            comparisons.append(
                {
                    "model": model,
                    "engine": engine,
                    "baseline": AX_DIRECT_FAIR_ENGINE,
                    "decode_tok_s_agg": result["profile_agg"],
                    "baseline_tok_s_agg": result["baseline_agg"],
                    "delta_vs_baseline": result["delta"],
                    "worst_suite_delta": result["worst_suite_delta"],
                    "per_suite_delta": result["per_suite_delta"],
                    "drafted": entry["drafted"],
                    "classification": classify_vs_direct(
                        result["delta"],
                        result["worst_suite_delta"],
                        entry["drafted"],
                    ),
                }
            )
            if not entry["drafted"]:
                warnings.append(
                    f"{model}/{engine}: route metadata shows zero MTP draft tokens."
                )

        ngram = engines.get("ax_engine_ngram")
        mtp = engines.get("ax_engine")
        if ngram is not None and mtp is not None:
            result = compare_suite_decodes(ngram["suites"], mtp["suites"])
            comparisons.append(
                {
                    "model": model,
                    "engine": "ax_engine_ngram",
                    "baseline": "ax_engine",
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
    return {"warnings": warnings, "comparisons": comparisons}


def build_optimized_scenarios(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    models = sorted({entry["model"] for entry in comparisons})
    for model in models:
        candidates = [
            entry
            for entry in comparisons
            if entry["model"] == model
            and entry["baseline"] == AX_DIRECT_FAIR_ENGINE
            and entry["classification"] in ("keep-default", "keep-opt-in")
            and entry.get("decode_tok_s_agg") is not None
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda entry: float(entry["decode_tok_s_agg"]))
        scenarios.append(
            {
                "model": model,
                "engine": best["engine"],
                "decode_tok_s_agg": best["decode_tok_s_agg"],
                "baseline_tok_s_agg": best["baseline_tok_s_agg"],
                "delta_vs_direct": best["delta_vs_baseline"],
                "worst_suite_delta": best["worst_suite_delta"],
                "classification": best["classification"],
            }
        )
    return scenarios


def build_summary(
    args: argparse.Namespace, artifact_paths: dict[tuple[str, str, str], Path]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profiles = [QWEN36_PROFILES[key] for key in args.models]
    ax_engine_modes = {
        engine: mode
        for engine, mode in (
            ("ax_engine", "pure_mtp"),
            ("ax_engine_ngram", "mtp_ngram_stacked"),
            ("ax_engine_tuned", "tuned_best_of"),
        )
        if engine in args.engines
    }
    has_tuned_engines = any(engine.endswith("_tuned") for engine in args.engines)
    for profile in profiles:
        depth = effective_depth(profile, args.engines, args.depth_policy, args.depth)
        provenance = diff.model_provenance_for(sidecar_dir(profile, args.hf_cache))
        for suite in args.suites:
            engine_summaries = {
                engine: summarize_engine_artifact(
                    engine, artifact_paths[(profile.key, suite, engine)]
                )
                for engine in args.engines
            }
            ax_mtp_summary = engine_summaries.get("ax_engine")
            if ax_mtp_summary and ax_mtp_summary.get("status") != "error":
                ngram_hit_steps = int(ax_mtp_summary.get("ngram_hit_steps", 0) or 0)
                if ngram_hit_steps > 0:
                    raise RuntimeError(
                        "pure-MTP AX benchmark row observed n-gram draft hits; "
                        "AX_MLX_MTP_DISABLE_NGRAM_STACKING may not be honored "
                        f"for {profile.key}/{suite}: {ngram_hit_steps}"
                    )
            ax_direct_tok_s = (engine_summaries.get(AX_DIRECT_FAIR_ENGINE) or {}).get(
                "decode_tok_s"
            )
            ax_tok_s = (engine_summaries.get("ax_engine") or {}).get("decode_tok_s")
            ax_ngram_tok_s = (engine_summaries.get("ax_engine_ngram") or {}).get(
                "decode_tok_s"
            )
            mtplx_tok_s = (engine_summaries.get("mtplx") or {}).get("decode_tok_s")
            mtplx_tuned_tok_s = (engine_summaries.get("mtplx_tuned") or {}).get(
                "decode_tok_s"
            )
            ax_tuned_tok_s = (engine_summaries.get("ax_engine_tuned") or {}).get(
                "decode_tok_s"
            )
            lightning_tok_s = (engine_summaries.get("lightning_mlx") or {}).get(
                "decode_tok_s"
            )
            rows.append(
                {
                    "model": profile.key,
                    "model_label": profile.label,
                    "suite": suite,
                    "depth": depth,
                    "base_model_id": profile.base_model_id,
                    "source_model_id": profile.source_model_id,
                    "provenance": provenance,
                    "engines": engine_summaries,
                    "ratios": {
                        "ax_engine_vs_mtplx": ratio(ax_tok_s, mtplx_tok_s),
                        "ax_engine_vs_ax_direct": ratio(ax_tok_s, ax_direct_tok_s),
                        "ax_engine_ngram_vs_ax_direct": ratio(
                            ax_ngram_tok_s, ax_direct_tok_s
                        ),
                        "ax_engine_vs_lightning_mlx": ratio(ax_tok_s, lightning_tok_s),
                        "ax_engine_ngram_vs_mtplx": ratio(ax_ngram_tok_s, mtplx_tok_s),
                        "ax_engine_tuned_vs_mtplx_tuned": ratio(
                            ax_tuned_tok_s, mtplx_tuned_tok_s
                        ),
                        "ax_engine_tuned_vs_mtplx": ratio(ax_tuned_tok_s, mtplx_tok_s),
                        "mtplx_tuned_vs_mtplx": ratio(mtplx_tuned_tok_s, mtplx_tok_s),
                        "ax_engine_tuned_vs_ax_engine": ratio(
                            ax_tuned_tok_s, ax_tok_s
                        ),
                        "ax_engine_tuned_vs_ax_engine_ngram": ratio(
                            ax_tuned_tok_s, ax_ngram_tok_s
                        ),
                        "ax_engine_ngram_vs_lightning_mlx": ratio(
                            ax_ngram_tok_s, lightning_tok_s
                        ),
                        "ax_engine_ngram_vs_ax_engine": ratio(ax_ngram_tok_s, ax_tok_s),
                        "lightning_mlx_vs_mtplx": ratio(lightning_tok_s, mtplx_tok_s),
                    },
                }
            )
    summary = {
        "schema": "ax.qwen36_mtp_fair.v1",
        "created_at": date.today().isoformat(),
        "contract": {
            "models": args.models,
            "engines": args.engines,
            "suites": args.suites,
            "depth_policy": args.depth_policy,
            "mode": args.mode,
            "sampling": diff.sampling_for_mode(
                args.mode,
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                },
            ),
            "max_tokens": args.max_tokens,
            "repetitions": args.repetitions,
            "warmup_repetitions": args.warmup_repetitions,
            "cooldown_s": args.cooldown,
            "benchmark_contract": "tuned-best-of"
            if has_tuned_engines and all(engine.endswith("_tuned") for engine in args.engines)
            else ("mixed-fixed-and-tuned" if has_tuned_engines else "fixed-depth"),
            "tuning": {
                "enabled": has_tuned_engines,
                "mtplx_retune": bool(getattr(args, "retune", True)),
                "mtplx_candidate_depth_cap": 3,
                "ax_candidate_policies": getattr(
                    args,
                    "ax_tune_policies",
                    ["direct", "ngram", "mtp", "mtp-ngram"],
                ),
                "tune_max_tokens": getattr(args, "tune_max_tokens", None),
                "tune_repetitions": getattr(args, "tune_repetitions", None),
                "tune_warmup_repetitions": getattr(
                    args, "tune_warmup_repetitions", None
                ),
                "tune_cooldown_s": getattr(args, "tune_cooldown", None),
                "tune_limit": getattr(args, "tune_limit", None),
                "scope": "per model and prompt suite row",
            },
            "lightning_settings": {
                "source_profile": "lightning-mlx serve qwen3.6 MTPLX preset",
                "model_source": getattr(args, "lightning_model_source", "sidecar"),
                "optimized_aliases": {
                    profile.key: profile.lightning_alias for profile in profiles
                }
                if getattr(args, "lightning_model_source", "sidecar") == "optimized-alias"
                else None,
                "prefix_cache": "enabled",
                "max_num_seqs": 1,
                "prefill_batch_size": 1,
                "completion_batch_size": 1,
                "stream_interval": 1,
                "mtp_optimistic": getattr(args, "lightning_mtp_optimistic", False),
                "mtp_draft_temperature": getattr(
                    args, "lightning_mtp_draft_temperature", 0.5
                ),
                "enable_thinking": bool(getattr(args, "lightning_enable_thinking", False)),
                "ngram_num_draft_tokens": 6,
                "ngram_min_occurrences": 2,
                "ngram_acceptance_mode": "greedy",
                "ngram_hybrid_verify": True,
                "ngram_only_in_think": False,
                "ngram_skip_tool_calls": True,
                "ngram_self_tune": True,
                "ngram_self_tune_disable_threshold": 0.30,
                "ngram_auto_disable_mtp_threshold": getattr(
                    args, "lightning_ngram_auto_disable_mtp_threshold", 0.85
                ),
                "ngram_auto_disable_min_ngram": getattr(
                    args, "lightning_ngram_auto_disable_min_ngram", 0.50
                ),
            },
            "ax_pure_mtp": "ax_engine" in args.engines,
            "ax_engine_modes": ax_engine_modes,
            "fairness_rules": [
                "standard Qwen source MTP shards plus mlx-community 4-bit base only"
                if getattr(args, "lightning_model_source", "sidecar") == "sidecar"
                else "Lightning rows use upstream optimized qwen3.6 aliases and are not sidecar-fair rows",
                "Youssofal/samuelfaj optimized bundles are excluded"
                if getattr(args, "lightning_model_source", "sidecar") == "sidecar"
                else "Youssofal/samuelfaj optimized bundles are included for Lightning source-methodology rows",
                "same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown",
                "fixed-depth rows and tuned-best-of rows are separate benchmark contracts",
            ],
        },
        "rows": rows,
    }
    survival_comparison = build_survival_comparison(rows)
    summary["survival_comparison"] = survival_comparison
    summary["optimized_scenarios"] = build_optimized_scenarios(
        survival_comparison["comparisons"]
    )
    return summary


def fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def fmt_validation(engine_summary: dict[str, Any]) -> str:
    total = engine_summary.get("validations_total")
    passed = engine_summary.get("validations_passed")
    if total is None or passed is None:
        return "-"
    return f"{passed}/{total}"


def markdown_engine_label(engine: str) -> str:
    return {
        "ax_direct": "AX direct",
        "mtplx": "MTPLX",
        "mtplx_tuned": "MTPLX tuned",
        "lightning_mlx": "Light. MTP",
        "lightning_mtp_ngram": "Light. ngram+MTP",
        "ax_engine": "AX MTP",
        "ax_engine_ngram": "AX MTP+n-gram",
        "ax_engine_tuned": "AX tuned",
    }.get(engine, engine)


def fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.1f}%"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.6 MTP Fair Benchmark",
        "",
        "Contract:",
        "",
    ]
    contract = summary["contract"]
    for key in (
        "models",
        "engines",
        "suites",
        "depth_policy",
        "mode",
        "max_tokens",
        "repetitions",
    ):
        lines.append(f"- {key}: `{contract[key]}`")
    lines.append("")
    lines.append("Fairness rules:")
    lines.append("")
    for rule in contract["fairness_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    contract_engines = summary.get("contract", {}).get("engines") or []
    table_engines = [engine for engine in ENGINE_ORDER if engine in contract_engines]
    header_cols = ["Model", "Suite", "Depth"]
    for engine in table_engines:
        label = markdown_engine_label(engine)
        header_cols += [f"{label} tok/s", f"{label} accept"]
    lines.append("| " + " | ".join(header_cols) + " |")
    align = ["---", "---", "---:"] + ["---:"] * (len(header_cols) - 3)
    lines.append("| " + " | ".join(align) + " |")
    for row in summary["rows"]:
        engines = row["engines"]
        cells = [
            row["model_label"],
            row["suite"],
            str(row["depth"]),
        ]
        for engine in table_engines:
            e = engines.get(engine, {})
            cells += [
                fmt_number(e.get("decode_tok_s")),
                fmt_percent(e.get("accept_rate")),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    survival = summary.get("survival_comparison") or {}
    comparisons = survival.get("comparisons") or []
    if comparisons:
        lines += [
            "",
            "## Same-artifact AX survival comparison",
            "",
            "AX direct is the same sidecar package decoded without MTP or n-gram. "
            "Use this table to decide whether AX MTP should be a default and "
            "whether AX MTP+n-gram should remain opt-in.",
            "",
            "| Model | Engine | Baseline | Decode tok/s | Baseline tok/s | Δ vs baseline | Worst suite Δ | Drafted | Classification |",
            "|---|---|---|---:|---:|---:|---:|:---:|---|",
        ]
        for entry in comparisons:
            lines.append(
                "| {model} | {engine} | {baseline} | {decode} | {base_decode} | {delta} | {worst} | {drafted} | {classification} |".format(
                    model=entry["model"],
                    engine=markdown_engine_label(entry["engine"]),
                    baseline=markdown_engine_label(entry["baseline"]),
                    decode=fmt_number(entry.get("decode_tok_s_agg")),
                    base_decode=fmt_number(entry.get("baseline_tok_s_agg")),
                    delta=fmt_delta(entry.get("delta_vs_baseline")),
                    worst=fmt_delta(entry.get("worst_suite_delta")),
                    drafted="—" if "drafted" not in entry else ("yes" if entry["drafted"] else "NO"),
                    classification=entry["classification"],
                )
            )
        warnings = survival.get("warnings") or []
        if warnings:
            lines += ["", "### Warnings", ""]
            lines += [f"- {warning}" for warning in warnings]
    scenarios = summary.get("optimized_scenarios") or []
    if scenarios:
        lines += [
            "",
            "## Optimized AX scenario",
            "",
            "| Model | Engine | Decode tok/s | Direct tok/s | Δ vs direct | Worst suite Δ | Classification |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for entry in scenarios:
            lines.append(
                "| {model} | {engine} | {decode} | {direct} | {delta} | {worst} | {classification} |".format(
                    model=entry["model"],
                    engine=markdown_engine_label(entry["engine"]),
                    decode=fmt_number(entry.get("decode_tok_s_agg")),
                    direct=fmt_number(entry.get("baseline_tok_s_agg")),
                    delta=fmt_delta(entry.get("delta_vs_direct")),
                    worst=fmt_delta(entry.get("worst_suite_delta")),
                    classification=entry["classification"],
                )
            )
    lines.append("")
    lines.append("Artifacts:")
    lines.append("")
    for row in summary["rows"]:
        for engine, engine_summary in row["engines"].items():
            status = engine_summary.get("status")
            artifact = engine_summary.get("artifact")
            validation = fmt_validation(engine_summary)
            validation_note = f" validation `{validation}`" if validation != "-" else ""
            lines.append(
                f"- {row['model']} / {row['suite']} / {engine}: "
                f"`{status}`{validation_note} `{artifact}`"
            )
    path.write_text("\n".join(lines) + "\n")


def nice_axis_ceiling(max_value: float) -> float:
    if not math.isfinite(max_value) or max_value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(max_value))
    scaled = max_value / magnitude
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= step:
            return step * magnitude
    return 10.0 * magnitude


def axis_label(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.0f}%"
    if value >= 1000:
        return f"{value / 1000:g}k"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def point_label(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "ms":
        return f"{value:.0f}"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def chart_samples(
    engine_summary: dict[str, Any], metric: str, scale: float = 1.0
) -> list[float]:
    samples = engine_summary.get(f"{metric}_samples")
    values = (
        [float(v) * scale for v in samples if isinstance(v, int | float)]
        if isinstance(samples, list)
        else []
    )
    if values:
        return values
    value = engine_summary.get(metric)
    return [float(value) * scale] if isinstance(value, int | float) else []


def row_chart_groups(
    rows: list[dict[str, Any]],
    engines: list[str],
    metric: str,
    *,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    return [
        {
            "label": row["suite"],
            "values": {
                engine: chart_samples(row["engines"].get(engine, {}), metric, scale)
                for engine in engines
            },
        }
        for row in rows
    ]


def model_short_label(label: str) -> str:
    return label.removeprefix("Qwen3.6 ").removesuffix(" 4-bit")


def combined_suite_chart_group(
    rows: list[dict[str, Any]],
    engines: list[str],
    metric: str,
    *,
    scale: float = 1.0,
    label: str = "all suites",
) -> list[dict[str, Any]]:
    values_by_engine = {engine: [] for engine in engines}
    for row in rows:
        for engine in engines:
            values_by_engine[engine].extend(
                chart_samples(row["engines"].get(engine, {}), metric, scale)
            )
    return [{"label": label, "values": values_by_engine}]


def model_combined_suite_chart_groups(
    rows: list[dict[str, Any]],
    engines: list[str],
    metric: str,
    *,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    model_keys: list[str] = []
    for row in rows:
        model_key = row["model"]
        if model_key not in model_keys:
            model_keys.append(model_key)
    for model_key in model_keys:
        model_rows = [row for row in rows if row["model"] == model_key]
        groups.extend(
            combined_suite_chart_group(
                model_rows,
                engines,
                metric,
                scale=scale,
                label=model_short_label(model_rows[0]["model_label"]),
            )
        )
    return groups


def write_mtp_box_whisker_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    unit: str,
    direction_label: str,
    groups: list[dict[str, Any]],
    engines: list[str],
    width: int = 900,
    axis_max: float | None = None,
    lower_is_better: bool = False,
    footnote: str = "",
) -> None:
    height = 460
    left = 64
    right = 160
    header_right = 34
    top = 86
    bottom = 106
    plot_right = width - right
    plot_w = plot_right - left
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    gap = 8.0
    box_w = min(
        34.0,
        max(
            12.0,
            (group_w - 56 - gap * (len(engines) - 1)) / max(len(engines), 1),
        ),
    )

    stats_by_key: dict[tuple[int, str], BoxStats] = {}
    all_values: list[float] = []
    all_medians: list[float] = []
    for group_index, group in enumerate(groups):
        for engine in engines:
            values = [
                float(v) for v in group["values"].get(engine, []) if v is not None
            ]
            if not values:
                continue
            stats = box_stats(values)
            stats_by_key[(group_index, engine)] = stats
            all_values.extend(values)
            all_medians.append(stats.median)

    max_value = (
        axis_max
        if axis_max is not None
        else nice_axis_ceiling(max(all_values or [1.0]) * 1.08)
    )
    best_value = (
        min(all_medians)
        if lower_is_better and all_medians
        else max(all_medians or [0.0])
    )

    def fy(value: float) -> float:
        clamped = max(0.0, min(value, max_value))
        return top + plot_h - (clamped / max_value) * plot_h

    def engine_centers(group_x: float, count: int) -> list[float]:
        if count <= 1:
            return [group_x + group_w / 2]
        side_pad = min(76.0, max(44.0, group_w * 0.12))
        preferred_span = max(0.0, group_w - side_pad * 2)
        minimum_span = (box_w + gap) * (count - 1)
        max_span = max(0.0, group_w - box_w)
        span = min(max(preferred_span, minimum_span), max_span)
        start_x = group_x + (group_w - span) / 2
        step = span / (count - 1) if count > 1 else 0.0
        return [start_x + step * i for i in range(count)]

    direction_fill = "#dc2626" if lower_is_better else "#374151"
    best_line_label = "lowest median" if lower_is_better else "highest median"
    best_side_label = "lowest" if lower_is_better else "highest"
    best_label = f"{best_side_label}: {point_label(best_value, unit)}"
    engine_desc = ", ".join(ENGINE_LABELS.get(engine, engine) for engine in engines)
    group_desc = ", ".join(group["label"] for group in groups)
    unit_w = max(48, len(unit) * 7 + 24)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>Grouped box-and-whisker plot comparing {html.escape(engine_desc)} "
        f"across {html.escape(group_desc)}.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="24" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="16" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="24" y="46" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="11" fill="#4b5563">{html.escape(subtitle)}</text>',
        *(
            [
                f'<text x="24" y="62" font-family="Inter,Segoe UI,Arial,sans-serif" '
                f'font-size="10" fill="#6b7280">{html.escape(footnote)}</text>'
            ]
            if footnote
            else []
        ),
        f'<rect x="{width - header_right - unit_w}" y="13" width="{unit_w}" height="22" '
        f'rx="11" fill="#eef2ff" stroke="#c7d2fe"/>',
        f'<text x="{width - header_right - unit_w / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" '
        f'fill="#3730a3">{html.escape(unit)}</text>',
        f'<text x="{width - header_right}" y="52" text-anchor="end" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" '
        f'fill="{direction_fill}">{html.escape(direction_label)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="6" '
        f'fill="#ffffff" stroke="#dbe3ef"/>',
    ]

    for i in range(5):
        value = max_value * i / 4
        y = fy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#6b7280">'
            f"{axis_label(value, unit)}</text>"
        )

    if best_value > 0:
        best_y = fy(best_value)
        parts.append(
            f'<line x1="{left}" y1="{best_y:.1f}" x2="{plot_right}" y2="{best_y:.1f}" '
            f'stroke="#dc2626" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{plot_right + 8}" y="{max(top + 11, best_y - 5):.1f}" '
            f'text-anchor="start" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="11" font-weight="700" fill="#dc2626" '
            f'data-label="{html.escape(best_line_label)}">'
            f"{html.escape(best_label)}</text>"
        )

    dot_slots = 9
    dot_jitter = [
        (-0.36 + 0.72 * i / (dot_slots - 1)) * box_w for i in range(dot_slots)
    ]
    for group_index, group in enumerate(groups):
        group_x = left + group_w * group_index
        centers = engine_centers(group_x, len(engines))
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 62}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" '
            f'fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(engines):
            stats = stats_by_key.get((group_index, engine))
            if stats is None:
                continue
            x = centers[engine_index]
            color = ENGINE_COLORS[engine]
            y_min = fy(stats.minimum)
            y_q1 = fy(stats.q1)
            y_med = fy(stats.median)
            y_q3 = fy(stats.q3)
            y_max = fy(stats.maximum)
            box_top = min(y_q1, y_q3)
            box_h = max(abs(y_q3 - y_q1), 1.0)
            cap_left = x - box_w * 0.36
            cap_right = x + box_w * 0.36
            box_left = x - box_w / 2
            for vi, value in enumerate(stats.values):
                parts.append(
                    f'<circle cx="{x + dot_jitter[vi % len(dot_jitter)]:.1f}" '
                    f'cy="{fy(value):.1f}" r="1.45" fill="{color}" '
                    f'fill-opacity="0.46"/>'
                )
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y_max:.1f}" x2="{x:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_max:.1f}" x2="{cap_right:.1f}" y2="{y_max:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<line x1="{cap_left:.1f}" y1="{y_min:.1f}" x2="{cap_right:.1f}" y2="{y_min:.1f}" '
                    f'stroke="{color}" stroke-opacity="0.86" stroke-width="1.8"/>',
                    f'<rect x="{box_left:.1f}" y="{box_top:.1f}" width="{box_w:.1f}" '
                    f'height="{box_h:.1f}" rx="2" fill="{color}" fill-opacity="0.18" '
                    f'stroke="{color}" stroke-opacity="0.9" stroke-width="1.8"/>',
                    f'<line x1="{box_left:.1f}" y1="{y_med:.1f}" x2="{box_left + box_w:.1f}" '
                    f'y2="{y_med:.1f}" stroke="{color}" stroke-opacity="0.96" stroke-width="2.6"/>',
                    f'<text x="{box_left + box_w + 6:.1f}" y="{y_med + 4:.1f}" '
                    f'text-anchor="start" font-family="Inter,Segoe UI,Arial,sans-serif" '
                    f'font-size="11" font-weight="700" fill="#111827" '
                    f'stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                    f"{html.escape(point_label(stats.median, unit))}</text>",
                ]
            )

    legend_y = height - 18
    legend_step = max(118.0, (width - left - right) / max(len(engines), 1))
    legend_x = left
    for engine in engines:
        color = ENGINE_COLORS[engine]
        label = ENGINE_LABELS[engine]
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{color}" fill-opacity="0.72"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14:.1f}" y="{legend_y}" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">'
            f"{html.escape(label)}</text>"
        )
        legend_x += legend_step

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_decode_svg(path: Path, summary: dict[str, Any]) -> None:
    engines = [
        engine for engine in ENGINE_ORDER if engine in summary["contract"]["engines"]
    ]
    groups = model_combined_suite_chart_groups(summary["rows"], engines, "decode_tok_s")
    write_mtp_box_whisker_svg(
        path,
        title="Qwen3.6 MTP fair decode throughput",
        subtitle="All suites combined per model | box=IQR | whiskers=min/max | dots=runs",
        unit="tok/s",
        direction_label="Higher is better",
        groups=groups,
        engines=engines,
        width=max(900, len(groups) * 260),
        footnote=VERSIONS_FOOTNOTE,
    )


def write_decode_model_svg(
    path: Path, summary: dict[str, Any], model_key: str, y_max: float | None = None
) -> None:
    rows = [row for row in summary["rows"] if row["model"] == model_key]
    if not rows:
        return
    engines = [
        engine for engine in ENGINE_ORDER if engine in summary["contract"]["engines"]
    ]
    write_mtp_box_whisker_svg(
        path,
        title=f"{rows[0]['model_label']} MTP decode throughput",
        subtitle="All suites combined | box=IQR | whiskers=min/max | dots=runs",
        unit="tok/s",
        direction_label="Higher is better",
        groups=combined_suite_chart_group(rows, engines, "decode_tok_s"),
        engines=engines,
        axis_max=y_max,
        footnote=VERSIONS_FOOTNOTE,
    )


def write_accept_model_svg(path: Path, summary: dict[str, Any], model_key: str) -> None:
    rows = [row for row in summary["rows"] if row["model"] == model_key]
    if not rows:
        return
    engines = [
        engine
        for engine in ENGINE_ORDER
        if engine in summary["contract"]["engines"] and engine != AX_DIRECT_FAIR_ENGINE
    ]
    write_mtp_box_whisker_svg(
        path,
        title=f"{rows[0]['model_label']} MTP accept rate",
        subtitle="All suites combined | accepted/drafted | box=IQR | dots=runs",
        unit="%",
        direction_label="Higher is better",
        groups=combined_suite_chart_group(rows, engines, "accept_rate", scale=100.0),
        engines=engines,
        axis_max=100.0,
        footnote=VERSIONS_FOOTNOTE,
    )


def aggregate_metric_median(
    summary: dict[str, Any], model_key: str, engine: str, metric: str
) -> float | None:
    values: list[float] = []
    for row in summary["rows"]:
        if row["model"] != model_key:
            continue
        values.extend(chart_samples(row["engines"].get(engine, {}), metric))
    return statistics.median(values) if values else None


def build_decode_improvement_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_models: list[str] = []
    for row in summary["rows"]:
        model_key = row["model"]
        if model_key not in seen_models:
            seen_models.append(model_key)
    for model_key in seen_models:
        model_rows = [row for row in summary["rows"] if row["model"] == model_key]
        if not model_rows:
            continue
        mtplx = aggregate_metric_median(summary, model_key, "mtplx", "decode_tok_s")
        ax_mtp = aggregate_metric_median(summary, model_key, "ax_engine", "decode_tok_s")
        ax_ngram = aggregate_metric_median(
            summary, model_key, "ax_engine_ngram", "decode_tok_s"
        )
        if mtplx is None or ax_mtp is None or ax_ngram is None:
            continue
        rows.append(
            {
                "model": model_key,
                "label": model_short_label(model_rows[0]["model_label"]),
                "ratios": {
                    "ax_mtp_vs_mtplx": ax_mtp / mtplx - 1.0,
                    "ax_ngram_vs_mtplx": ax_ngram / mtplx - 1.0,
                    "ax_ngram_vs_ax_mtp": ax_ngram / ax_mtp - 1.0,
                },
                "medians": {
                    "mtplx": mtplx,
                    "ax_engine": ax_mtp,
                    "ax_engine_ngram": ax_ngram,
                },
            }
        )
    return rows


def write_decode_improvement_svg(path: Path, summary: dict[str, Any]) -> None:
    rows = build_decode_improvement_rows(summary)
    width = 900
    height = 360
    left = 72
    right = 34
    top = 82
    bottom = 84
    plot_w = width - left - right
    plot_h = height - top - bottom
    metrics = [
        ("ax_mtp_vs_mtplx", "AX MTP vs MTPLX", ENGINE_COLORS["ax_engine"]),
        ("ax_ngram_vs_mtplx", "AX MTP+ngram vs MTPLX", ENGINE_COLORS["ax_engine_ngram"]),
        ("ax_ngram_vs_ax_mtp", "ngram vs AX MTP", "#7c3aed"),
    ]
    values = [entry["ratios"][key] * 100.0 for entry in rows for key, _, _ in metrics]
    min_value = min([0.0, *values])
    max_value = max([1.0, *values])
    lower = -10.0 if min_value < 0 else 0.0
    upper = nice_axis_ceiling(max_value * 1.12)
    if upper <= 0:
        upper = 1.0
    group_w = plot_w / max(len(rows), 1)
    bar_gap = 10.0
    bar_w = min(44.0, max(18.0, (group_w - 90.0 - bar_gap * 2) / len(metrics)))

    def fx(group_index: int, metric_index: int) -> float:
        group_x = left + group_w * group_index
        span = bar_w * len(metrics) + bar_gap * (len(metrics) - 1)
        start = group_x + (group_w - span) / 2
        return start + metric_index * (bar_w + bar_gap)

    def fy(value: float) -> float:
        clamped = max(lower, min(value, upper))
        return top + plot_h - ((clamped - lower) / (upper - lower)) * plot_h

    zero_y = fy(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title>Qwen3.6 MTP decode improvement</title>",
        "<desc>Bar chart comparing aggregate sample-median decode improvement for AX MTP, "
        "AX MTP plus n-gram, and n-gram over pure AX MTP.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="24" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="16" font-weight="700" fill="#111827">Qwen3.6 MTP decode improvement</text>',
        '<text x="24" y="46" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="11" fill="#4b5563">Aggregate sample median over flappy, long_code, '
        'and python_modules_long | higher is better</text>',
        '<rect x="64" y="82" width="802" height="194" rx="6" fill="#ffffff" stroke="#dbe3ef"/>',
    ]
    for i in range(5):
        value = lower + (upper - lower) * i / 4
        y = fy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#6b7280">'
            f"{axis_label(value, '%')}</text>"
        )
    parts.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" '
        'stroke="#111827" stroke-opacity="0.45" stroke-width="1.2"/>'
    )
    for group_index, entry in enumerate(rows):
        group_x = left + group_w * group_index
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 54}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12" font-weight="700" '
            f'fill="#111827">{html.escape(entry["label"])}</text>'
        )
        for metric_index, (key, _label, color) in enumerate(metrics):
            value = entry["ratios"][key] * 100.0
            x = fx(group_index, metric_index)
            y_value = fy(value)
            y = min(y_value, zero_y)
            bar_h = max(abs(zero_y - y_value), 1.0)
            fill = color if value >= 0 else "#dc2626"
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="3" fill="{fill}" fill-opacity="0.78"/>'
            )
            label_y = y - 6 if value >= 0 else y + bar_h + 14
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" '
                f'font-weight="700" fill="#111827" stroke="#ffffff" stroke-width="3" '
                f'paint-order="stroke">{html.escape(point_label(value, "%"))}</text>'
            )
    legend_x = left
    legend_y = height - 18
    for _key, label, color in metrics:
        parts.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{color}" fill-opacity="0.78"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14:.1f}" y="{legend_y}" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">'
            f"{html.escape(label)}</text>"
        )
        legend_x += 210
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(QWEN36_PROFILES),
        default=sorted(QWEN36_PROFILES),
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=ENGINE_VENDORS,
        default=list(ENGINE_VENDORS),
        help="Engine vendors to include. Default: all (mtplx lightning ax).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ENGINE_MODES,
        default=["mtp"],
        help=(
            "Decode modes to include. Default: fixed-depth mtp only. "
            "mtp-ngram is preserved for historical/ad-hoc probes but is not "
            "part of the current MTP benchmark design."
        ),
    )
    parser.add_argument("--suites", nargs="+", default=["flappy", "long_code"])
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument("--hf-cache", type=Path, default=HF_CACHE)
    parser.add_argument(
        "--depth-policy", choices=["fair-shared", "native"], default="fair-shared"
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--mode", choices=["greedy", "sampled"], default="sampled")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=30.0)
    parser.add_argument("--tune-max-tokens", type=int, default=192)
    parser.add_argument("--tune-repetitions", type=int, default=1)
    parser.add_argument("--tune-warmup-repetitions", type=int, default=0)
    parser.add_argument("--tune-cooldown", type=float, default=0.0)
    parser.add_argument("--tune-limit", type=int, default=1)
    parser.add_argument("--tune-seed", type=int, default=0)
    parser.add_argument(
        "--retune",
        dest="retune",
        action="store_true",
        default=True,
        help="Force MTPLX tune to measure again before tuned MTPLX benchmark rows.",
    )
    parser.add_argument(
        "--no-retune",
        dest="retune",
        action="store_false",
        help="Allow MTPLX tune to reuse its saved recommendation for tuned rows.",
    )
    parser.add_argument(
        "--inter-case-cooldown",
        type=float,
        default=10.0,
        help=(
            "Extra sleep between prompt cases within a suite (seconds). "
            "Prevents GPU thermal throttling across cases. Default 10s."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=diff.DEFAULT_SAMPLING["temperature"]
    )
    parser.add_argument("--top-p", type=float, default=diff.DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=diff.DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--lightning-mtp-optimistic",
        dest="lightning_mtp_optimistic",
        action="store_true",
        default=False,
        help=(
            "Run Lightning-MLX serve with --mtp-optimistic. Default: disabled, "
            "matching the source qwen3.6 serve preset; raw lightning-mlx bench "
            "uses optimistic explicitly."
        ),
    )
    parser.add_argument(
        "--no-lightning-mtp-optimistic",
        dest="lightning_mtp_optimistic",
        action="store_false",
        help="Run Lightning-MLX without --mtp-optimistic.",
    )
    parser.add_argument(
        "--lightning-enable-thinking",
        dest="lightning_enable_thinking",
        action="store_true",
        default=False,
        help="Keep thinking enabled for Lightning-MLX serve. Default is disabled for benchmark comparability.",
    )
    parser.add_argument(
        "--lightning-disable-thinking",
        dest="lightning_enable_thinking",
        action="store_false",
        help="Pass --disable-thinking to the Lightning-MLX prompt-suite adapter.",
    )
    parser.add_argument(
        "--lightning-mtp-draft-temperature",
        type=float,
        default=0.5,
        help="Lightning-MLX --mtp-draft-temperature. Default follows its Qwen3.6 preset.",
    )
    parser.add_argument(
        "--lightning-ngram-auto-disable-mtp-threshold",
        type=float,
        default=0.85,
        help="Lightning-MLX n-gram auto-disable MTP threshold.",
    )
    parser.add_argument(
        "--lightning-ngram-auto-disable-min-ngram",
        type=float,
        default=0.50,
        help="Lightning-MLX n-gram auto-disable minimum n-gram threshold.",
    )
    parser.add_argument("--ax-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--ax-tune-policies",
        nargs="+",
        choices=["direct", "ngram", "mtp", "mtp-ngram"],
        default=["mtp"],
        help=(
            "AX policies swept for the tuned-best-of row. Default: mtp only. "
            "Direct, ngram, and mtp-ngram are historical/ad-hoc probes outside "
            "the current MTP benchmark design."
        ),
    )
    parser.add_argument(
        "--mtplx-python",
        type=Path,
        default=DEFAULT_MTPLX_PYTHON
        if DEFAULT_MTPLX_PYTHON.exists()
        else Path(sys.executable),
    )
    parser.add_argument(
        "--rapid-python",
        type=Path,
        default=DEFAULT_RAPID_PYTHON
        if DEFAULT_RAPID_PYTHON.exists()
        else Path(sys.executable),
    )
    parser.add_argument(
        "--lightning-source",
        type=Path,
        default=DEFAULT_LIGHTNING_SOURCE,
    )
    parser.add_argument(
        "--lightning-model-source",
        choices=["sidecar", "optimized-alias"],
        default="sidecar",
        help=(
            "Model identifier used for Lightning-MLX rows. 'sidecar' keeps the fair "
            "standard Qwen/mlx-community sidecar path; 'optimized-alias' uses the "
            "upstream lightning-mlx qwen3.6 aliases and matches their published setup."
        ),
    )
    parser.add_argument("--base-port", type=int, default=18765)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-build-ax-engine", action="store_true")
    parser.add_argument(
        "--mtplx-profile",
        default="stable",
        help=(
            "MTPLX runtime profile for the fair benchmark. Default is 'stable' because "
            "the 'sustained' profile requires MTPLX_PREFILL_OMLX_EXTERNAL=1, which "
            "expects the Youssofal bundle format and produces garbage output with "
            "standard mlx-community sidecars."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-qwen36-fair"
    )
    # Resolve vendor × mode combinations into ordered internal engine keys.
    args.engines = resolve_engines(args.engines, args.modes)
    if not args.engines:
        print(
            "No supported engine/mode combinations selected — nothing to run.",
            file=sys.stderr,
        )
        return 1
    if args.warmup_repetitions != 1 and any(
        e in args.engines
        for e in (
            AX_DIRECT_FAIR_ENGINE,
            "ax_engine",
            "ax_engine_ngram",
            "ax_engine_tuned",
        )
    ):
        raise ValueError(
            "AX prompt-suite benchmark has an implicit 1 warmup repetition"
        )
    for model_key in args.models:
        validate_sidecar_for_profile(QWEN36_PROFILES[model_key], args.hf_cache)
    for suite in args.suites:
        suite_file = suite_path_for(suite, args.suites_dir)
        if not suite_file.is_file():
            raise FileNotFoundError(f"missing suite: {suite_file}")

    artifact_paths: dict[tuple[str, str, str], Path] = {}
    for model_key in args.models:
        profile = QWEN36_PROFILES[model_key]
        depth = effective_depth(profile, args.engines, args.depth_policy, args.depth)
        for suite in args.suites:
            for engine in args.engines:
                artifact_paths[(profile.key, suite, engine)] = run_engine_suite(
                    args,
                    engine=engine,
                    profile=profile,
                    suite=suite,
                    depth=depth,
                    port=args.base_port,
                )

    # When --skip-existing is set, auto-include any engine from ENGINE_ORDER
    # that has complete artifacts on disk but was not in --engines. This prevents
    # partial re-runs (e.g. --engines lightning_mlx) from silently dropping other
    # engines from the summary and charts.
    if args.skip_existing:
        promoted: list[str] = []
        for engine in ENGINE_ORDER:
            if engine in args.engines:
                continue
            all_present = all(
                (
                    args.output_dir
                    / QWEN36_PROFILES[model_key].key
                    / suite
                    / f"{engine}.json"
                ).is_file()
                for model_key in args.models
                for suite in args.suites
            )
            if not all_present:
                continue
            for model_key in args.models:
                profile = QWEN36_PROFILES[model_key]
                for suite in args.suites:
                    artifact_paths[(profile.key, suite, engine)] = (
                        args.output_dir / profile.key / suite / f"{engine}.json"
                    )
            promoted.append(engine)
        if promoted:
            print(
                f"[skip-existing] auto-including engines with complete artifacts: "
                + ", ".join(promoted),
                file=sys.stderr,
            )
            args.engines = [
                e for e in ENGINE_ORDER if e in args.engines or e in promoted
            ]

    summary = build_summary(args, artifact_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    chart_svg = args.output_dir / "decode-tok-s.svg"
    improvement_svg = args.output_dir / "decode-improvement.svg"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary_md, summary)
    write_decode_svg(chart_svg, summary)
    write_decode_improvement_svg(improvement_svg, summary)
    decode_y_max: dict[str, float] = {"27b-4bit": 80.0, "35b-a3b-4bit": 300.0}
    model_chart_paths: list[Path] = []
    for model_key in args.models:
        model_chart = args.output_dir / f"decode-tok-s-{model_key}.svg"
        write_decode_model_svg(
            model_chart, summary, model_key, y_max=decode_y_max.get(model_key)
        )
        model_chart_paths.append(model_chart)
        accept_chart = args.output_dir / f"accept-rate-{model_key}.svg"
        write_accept_model_svg(accept_chart, summary, model_key)
        model_chart_paths.append(accept_chart)
    print(f"Saved summary: {summary_json}")
    print(f"Saved markdown: {summary_md}")
    print(f"Saved chart: {chart_svg}")
    print(f"Saved chart: {improvement_svg}")
    for model_chart in model_chart_paths:
        print(f"Saved chart: {model_chart}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
