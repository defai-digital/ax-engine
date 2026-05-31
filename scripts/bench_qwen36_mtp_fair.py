#!/usr/bin/env python3
"""Fair Qwen3.6 MTP benchmark across MTPLX and AX Engine."""

from __future__ import annotations

import argparse
import html
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
ENGINE_LABELS = {
    "mtplx": "MTPLX 0.3.7",
    "ax_engine": "AX Engine MTP",
    "lightning_mlx": "Lightning MLX",
}
ENGINE_COLORS = {
    "mtplx": "#a78bfa",
    "ax_engine": "#4ade80",
    "lightning_mlx": "#60a5fa",
}
ENGINE_ORDER = ["mtplx", "lightning_mlx", "ax_engine"]


@dataclass(frozen=True)
class QwenProfile:
    key: str
    label: str
    model_key: str
    base_model_id: str
    source_model_id: str
    ax_local_slug: str
    native_depth: int
    is_moe: bool = False


QWEN36_PROFILES = {
    "27b-4bit": QwenProfile(
        key="27b-4bit",
        label="Qwen3.6 27B 4-bit",
        model_key="27b",
        base_model_id="mlx-community/Qwen3.6-27B-4bit",
        source_model_id="Qwen/Qwen3.6-27B",
        ax_local_slug="models--ax-local--Qwen3.6-27B-MTP",
        native_depth=3,
    ),
    "35b-a3b-4bit": QwenProfile(
        key="35b-a3b-4bit",
        label="Qwen3.6 35B-A3B 4-bit",
        model_key="35b",
        base_model_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        source_model_id="Qwen/Qwen3.6-35B-A3B",
        ax_local_slug="models--ax-local--Qwen3.6-35B-MTP",
        native_depth=1,
        is_moe=True,
    ),
}


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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
    pure_mtp: bool = False,
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
        "--ax-ngram-accel",
        "--ax-sampling",
        json.dumps(config.sampling, sort_keys=True),
        "--skip-mlx-lm",
        "--capture-output-token-ids",
        "--ax-mtp-max-depth",
        str(config.depth),
        "--output",
        str(output_path),
    ]
    if not config.enable_thinking:
        cmd.append("--no-thinking")
    if no_build:
        cmd.append("--no-build-ax-engine")
    if pure_mtp:
        cmd.append("--ax-mtp-disable-ngram-stacking")
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
        "--profile",
        mtplx_profile,
    ]
    if not config.enable_thinking:
        cmd.append("--disable-thinking")
    if allow_unverified_model:
        cmd.append("--allow-unverified-model")
    run_subprocess(cmd)
    return output_path


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
        "--port",
        str(port),
    ]
    run_subprocess(cmd)
    return output_path


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
    suite_file = suite_path_for(suite, args.suites_dir)
    output_path = args.output_dir / profile.key / suite / f"{engine}.json"
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
        enable_thinking=bool(args.enable_thinking),
    )
    if args.skip_existing and output_path.is_file():
        return output_path
    try:
        if engine == "ax_engine":
            return run_ax_suite(
                python=args.ax_python,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                no_build=args.no_build_ax_engine,
                pure_mtp=args.pure_mtp,
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
            )
        if engine == "lightning_mlx":
            return run_rapid_mlx_suite(
                python=args.rapid_python,
                lightning_source=args.lightning_source,
                suite=suite,
                suite_file=suite_file,
                output_path=output_path,
                model_dir=model_dir,
                config=config,
                port=args.base_port + 1,
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


def cases_for_engine(
    engine: str, artifact: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if artifact.get("schema") == "ax.mtp_engine_error.v1":
        return {}
    if engine == "ax_engine":
        return diff.ax_cases(artifact)
    if engine == "mtplx":
        return diff.mtplx_cases(artifact)
    if engine == "lightning_mlx":
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
    ngram_hit_steps = sum(
        int(case.get("ngram_hit_steps", 0) or 0) for case in cases.values()
    )
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
        "ngram_hit_steps": ngram_hit_steps if engine == "ax_engine" else None,
    }


def ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right <= 0:
        return None
    return left / right


def build_summary(
    args: argparse.Namespace, artifact_paths: dict[tuple[str, str, str], Path]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profiles = [QWEN36_PROFILES[key] for key in args.models]
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
            if args.pure_mtp and "ax_engine" in engine_summaries:
                ngram_hit_steps = int(
                    engine_summaries["ax_engine"].get("ngram_hit_steps", 0) or 0
                )
                if ngram_hit_steps > 0:
                    raise RuntimeError(
                        "pure-MTP AX benchmark observed n-gram draft hits; "
                        "AX_MLX_MTP_DISABLE_NGRAM_STACKING may not be honored "
                        f"for {profile.key}/{suite}: {ngram_hit_steps}"
                    )
            ax_tok_s = (engine_summaries.get("ax_engine") or {}).get("decode_tok_s")
            mtplx_tok_s = (engine_summaries.get("mtplx") or {}).get("decode_tok_s")
            lightning_tok_s = (engine_summaries.get("lightning_mlx") or {}).get("decode_tok_s")
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
                        "ax_engine_vs_lightning_mlx": ratio(ax_tok_s, lightning_tok_s),
                        "lightning_mlx_vs_mtplx": ratio(lightning_tok_s, mtplx_tok_s),
                    },
                }
            )
    return {
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
            "ax_pure_mtp": bool(args.pure_mtp),
            "fairness_rules": [
                "standard Qwen source MTP shards plus mlx-community 4-bit base only",
                "Youssofal MTPLX bundles are excluded",
                "same prompt suite, max token cap, sampler, warmup, repetitions, and cooldown",
            ],
        },
        "rows": rows,
    }


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
        "ax_pure_mtp",
    ):
        lines.append(f"- {key}: `{contract[key]}`")
    lines.append("")
    lines.append("Fairness rules:")
    lines.append("")
    for rule in contract["fairness_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    has_lightning = "lightning_mlx" in (summary.get("contract", {}).get("engines") or [])
    if has_lightning:
        lines.append(
            "| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | Lightning tok/s | Lightning accept | AX tok/s | AX accept | AX/MTPLX | AX/Lightning |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(
            "| Model | Suite | Depth | MTPLX tok/s | MTPLX accept | AX tok/s | AX accept | AX/MTPLX |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        engines = row["engines"]
        mtplx = engines.get("mtplx", {})
        ax = engines.get("ax_engine", {})
        ratios = row["ratios"]
        if has_lightning:
            lightning = engines.get("lightning_mlx", {})
            lines.append(
                f"| {row['model_label']} | {row['suite']} | {row['depth']} | "
                f"{fmt_number(mtplx.get('decode_tok_s'))} | {fmt_percent(mtplx.get('accept_rate'))} | "
                f"{fmt_number(lightning.get('decode_tok_s'))} | {fmt_percent(lightning.get('accept_rate'))} | "
                f"{fmt_number(ax.get('decode_tok_s'))} | {fmt_percent(ax.get('accept_rate'))} | "
                f"{fmt_number(ratios.get('ax_engine_vs_mtplx'), 3)} | "
                f"{fmt_number(ratios.get('ax_engine_vs_lightning_mlx'), 3)} |"
            )
        else:
            lines.append(
                f"| {row['model_label']} | {row['suite']} | {row['depth']} | "
                f"{fmt_number(mtplx.get('decode_tok_s'))} | {fmt_percent(mtplx.get('accept_rate'))} | "
                f"{fmt_number(ax.get('decode_tok_s'))} | {fmt_percent(ax.get('accept_rate'))} | "
                f"{fmt_number(ratios.get('ax_engine_vs_mtplx'), 3)} |"
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


def write_decode_svg(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    groups = [
        {
            "label": f"{row['model'].replace('-4bit', '')} {row['suite']}",
            "values": {
                engine: (row["engines"].get(engine) or {}).get("decode_tok_s")
                for engine in ENGINE_ORDER
                if engine in summary["contract"]["engines"]
            },
        }
        for row in rows
    ]
    max_value = max(
        [
            float(value)
            for group in groups
            for value in group["values"].values()
            if value is not None
        ]
        or [1.0]
    )
    width = 920
    left = 190
    top = 54
    row_h = 70
    chart_w = width - left - 40
    height = top + row_h * len(groups) + 48
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="18" font-weight="700" fill="#111827">Qwen3.6 MTP fair decode throughput</text>',
        '<text x="896" y="30" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#4b5563">tok/s, higher is better</text>',
    ]
    for i in range(4):
        x = left + chart_w * i / 3
        value = max_value * i / 3
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 44}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#6b7280">{value:.0f}</text>'
        )
    for group_index, group in enumerate(groups):
        base_y = top + group_index * row_h
        parts.append(
            f'<text x="24" y="{base_y + 25}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12" font-weight="700" fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(ENGINE_ORDER):
            if engine not in group["values"]:
                continue
            value = group["values"][engine]
            y = base_y + 8 + engine_index * 18
            parts.append(
                f'<text x="{left - 8}" y="{y + 10}" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">{html.escape(ENGINE_LABELS[engine])}</text>'
            )
            bar_w = 0 if value is None else chart_w * float(value) / max_value
            color = ENGINE_COLORS[engine]
            parts.append(
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="12" rx="2" fill="{color}" fill-opacity="0.82"/>'
            )
            label = "-" if value is None else f"{float(value):.1f}"
            label_x = min(left + bar_w + 5, width - 38)
            parts.append(
                f'<text x="{label_x:.1f}" y="{y + 10}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" fill="#111827">{label}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_decode_model_svg(path: Path, summary: dict[str, Any], model_key: str) -> None:
    rows = [row for row in summary["rows"] if row["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    engines = [
        engine for engine in ENGINE_ORDER if engine in summary["contract"]["engines"]
    ]
    groups = [
        {
            "label": row["suite"],
            "values": {
                engine: (row["engines"].get(engine) or {}).get("decode_tok_s")
                for engine in engines
            },
        }
        for row in rows
    ]
    max_value = max(
        [
            float(value)
            for group in groups
            for value in group["values"].values()
            if value is not None
        ]
        or [1.0]
    )
    width = 760
    height = 430
    left = 60
    right = 28
    top = 68
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    bar_gap = 8
    bar_w = min(
        54, (group_w - 44 - bar_gap * (len(engines) - 1)) / max(len(engines), 1)
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="18" font-weight="700" fill="#111827">{html.escape(model_label)} MTP decode throughput</text>',
        '<text x="736" y="30" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#4b5563">tok/s, higher is better</text>',
    ]
    for i in range(4):
        y = top + plot_h - plot_h * i / 3
        value = max_value * i / 3
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#6b7280">{value:.0f}</text>'
        )
    for group_index, group in enumerate(groups):
        group_x = left + group_w * group_index
        bars_w = bar_w * len(engines) + bar_gap * max(len(engines) - 1, 0)
        start_x = group_x + (group_w - bars_w) / 2
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 34}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(engines):
            value = group["values"].get(engine)
            bar_h = 0 if value is None else plot_h * float(value) / max_value
            x = start_x + engine_index * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            color = ENGINE_COLORS[engine]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}" fill-opacity="0.86"/>'
            )
            label = "-" if value is None else f"{float(value):.1f}"
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 5, top + 10):.1f}" text-anchor="middle" '
                f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" fill="#111827">{label}</text>'
            )
    legend_y = height - 14
    legend_x = left
    for engine in engines:
        color = ENGINE_COLORS[engine]
        label = ENGINE_LABELS[engine]
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.86"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">{html.escape(label)}</text>'
        )
        legend_x += 130
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_accept_model_svg(path: Path, summary: dict[str, Any], model_key: str) -> None:
    rows = [row for row in summary["rows"] if row["model"] == model_key]
    if not rows:
        return
    model_label = rows[0]["model_label"]
    engines = [
        engine
        for engine in ENGINE_ORDER
        if engine in summary["contract"]["engines"]
    ]
    groups = [
        {
            "label": row["suite"],
            "values": {
                engine: (row["engines"].get(engine) or {}).get("accept_rate")
                for engine in engines
            },
        }
        for row in rows
    ]
    width = 760
    height = 430
    left = 60
    right = 28
    top = 68
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(len(groups), 1)
    bar_gap = 10
    bar_w = min(
        54, (group_w - 44 - bar_gap * (len(engines) - 1)) / max(len(engines), 1)
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="18" font-weight="700" fill="#111827">{html.escape(model_label)} MTP accept rate</text>',
        '<text x="736" y="30" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#4b5563">accepted/drafted, higher is better</text>',
    ]
    for i in range(4):
        y = top + plot_h - plot_h * i / 3
        value = i / 3
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#6b7280">{value * 100:.0f}%</text>'
        )
    for group_index, group in enumerate(groups):
        group_x = left + group_w * group_index
        bars_w = bar_w * len(engines) + bar_gap * max(len(engines) - 1, 0)
        start_x = group_x + (group_w - bars_w) / 2
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 34}" text-anchor="middle" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" font-weight="700" fill="#111827">{html.escape(group["label"])}</text>'
        )
        for engine_index, engine in enumerate(engines):
            value = group["values"].get(engine)
            bar_h = 0 if value is None else plot_h * max(0.0, min(float(value), 1.0))
            x = start_x + engine_index * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            color = ENGINE_COLORS[engine]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}" fill-opacity="0.86"/>'
            )
            label = "-" if value is None else f"{float(value) * 100:.1f}%"
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 5, top + 10):.1f}" text-anchor="middle" '
                f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" font-weight="700" fill="#111827">{label}</text>'
            )
    legend_y = height - 14
    legend_x = left
    for engine in engines:
        color = ENGINE_COLORS[engine]
        label = ENGINE_LABELS[engine]
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" fill="{color}" fill-opacity="0.86"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="#374151">{html.escape(label)}</text>'
        )
        legend_x += 130
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
        "--engines", nargs="+", choices=ENGINE_ORDER, default=ENGINE_ORDER
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
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--temperature", type=float, default=diff.DEFAULT_SAMPLING["temperature"]
    )
    parser.add_argument("--top-p", type=float, default=diff.DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=diff.DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--ax-python", type=Path, default=Path(sys.executable))
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
    parser.add_argument(
        "--pure-mtp",
        action="store_true",
        help=(
            "Pass --ax-mtp-disable-ngram-stacking to the AX subprocess so the MTP "
            "verify loop sources its draft only from the MTP head (no ADR-008 "
            "n-gram-first stacking). Use this to measure pure-MTP acceptance for "
            "fair comparison against MTPLX."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-qwen36-fair"
    )
    if args.warmup_repetitions != 1 and "ax_engine" in args.engines:
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

    summary = build_summary(args, artifact_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    chart_svg = args.output_dir / "decode-tok-s.svg"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary_md, summary)
    write_decode_svg(chart_svg, summary)
    model_chart_paths: list[Path] = []
    for model_key in args.models:
        model_chart = args.output_dir / f"decode-tok-s-{model_key}.svg"
        write_decode_model_svg(model_chart, summary, model_key)
        model_chart_paths.append(model_chart)
        accept_chart = args.output_dir / f"accept-rate-{model_key}.svg"
        write_accept_model_svg(accept_chart, summary, model_key)
        model_chart_paths.append(accept_chart)
    print(f"Saved summary: {summary_json}")
    print(f"Saved markdown: {summary_md}")
    print(f"Saved chart: {chart_svg}")
    for model_chart in model_chart_paths:
        print(f"Saved chart: {model_chart}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
