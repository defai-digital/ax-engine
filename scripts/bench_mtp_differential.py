#!/usr/bin/env python3
"""Build AX-vs-MTPLX MTP differential benchmark artifacts.

This script is a thin orchestrator over the existing AX and MTPLX prompt-suite
benchmarks. It can either run both engines or assemble a differential artifact
from already-captured artifacts.

The output is intended to answer three questions:

1. Did AX and MTPLX run the same benchmark contract?
2. Is AX faster under that contract?
3. In deterministic mode, where do token streams first diverge?

Examples:

  # Run a sampled throughput comparison on the repo MTP suites.
  python3 scripts/bench_mtp_differential.py \\
    --ax-model-dir /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed \\
    --mtplx-model /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed \\
    --suites flappy long_code \\
    --modes sampled \\
    --output-dir benchmarks/results/mtp-differential/$(date +%F)-speed

  # Assemble a differential artifact from existing suite artifacts.
  python3 scripts/bench_mtp_differential.py \\
    --suites flappy \\
    --ax-artifact flappy=benchmarks/results/mtp-compare/.../flappy.json \\
    --mtplx-artifact flappy=benchmarks/results/mtp-compare/.../mtplx-flappy.json \\
    --output-dir /tmp/mtp-diff

  # Add market-prompt panels after converting them to the repo JSONL suite shape.
  python3 scripts/bench_mtp_differential.py \\
    --ax-model-dir /path/to/model \\
    --mtplx-model /path/to/model \\
    --suite-file wildbench_lite=benchmarks/prompts/market/wildbench_lite.jsonl \\
    --suites wildbench_lite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
MTPLX_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mtplx_prompt_suites.py"
DEFAULT_SUITES_DIR = REPO_ROOT / "benchmarks" / "prompts" / "mtp-suites"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "benchmarks" / "results" / "mtp-differential"

DEFAULT_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
DEFAULT_MAX_TOKENS = 1000
DEFAULT_REPETITIONS = 5
DEFAULT_WARMUP_REPETITIONS = 1
DEFAULT_COOLDOWN = 15.0
AX_MTP_ENGINE = "ax_engine_mlx_ngram_accel"
SIDECAR_MANIFEST_FILE = "ax_mtp_sidecar_manifest.json"


@dataclass(frozen=True)
class RunConfig:
    mode: str
    depth: int
    max_tokens: int
    repetitions: int
    warmup_repetitions: int
    cooldown_s: float
    sampling: dict[str, int | float]
    enable_thinking: bool


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def token_sha256(tokens: list[int] | None) -> str | None:
    if tokens is None:
        return None
    return sha256_json([int(token) for token in tokens])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summary_median(value: Any) -> float | None:
    if isinstance(value, dict):
        median_value = value.get("median")
        return float(median_value) if median_value is not None else None
    if value is None:
        return None
    return float(value)


def first_diff_index(left: list[int] | None, right: list[int] | None) -> int | None:
    if left is None or right is None:
        return None
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    if len(left) != len(right):
        return limit
    return None


def speedup_ratio(ax_tok_s: float | None, mtplx_tok_s: float | None) -> float | None:
    if ax_tok_s is None or mtplx_tok_s is None or mtplx_tok_s <= 0:
        return None
    return ax_tok_s / mtplx_tok_s


def parse_name_path_map(entries: list[str], flag_name: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{flag_name} expects NAME=PATH entries, got: {entry}")
        name, raw_path = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{flag_name} entry has an empty suite name: {entry}")
        parsed[name] = Path(raw_path).expanduser()
    return parsed


def suite_path_for(suite: str, suites_dir: Path, explicit: dict[str, Path]) -> Path:
    if suite in explicit:
        return explicit[suite]
    return suites_dir / f"{suite}.jsonl"


def model_provenance_for(model_ref: Path | str | None) -> dict[str, Any]:
    if model_ref is None:
        return {"kind": "unspecified"}
    raw = str(model_ref)
    path = Path(raw).expanduser()
    if path.is_dir():
        manifest_path = path / SIDECAR_MANIFEST_FILE
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            return summarize_sidecar_manifest(manifest_path, manifest)
        return {
            "kind": "model_dir_without_sidecar_manifest",
            "path": raw,
            "sidecar_manifest_present": False,
        }
    if path.is_file() and path.name == SIDECAR_MANIFEST_FILE:
        manifest = load_json(path)
        return summarize_sidecar_manifest(path, manifest)
    return {"kind": "model_ref", "model": raw}


def summarize_sidecar_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    base = manifest.get("base") or {}
    source = manifest.get("source") or {}
    output = manifest.get("output") or {}
    transform = manifest.get("transform") or {}
    runtime = manifest.get("runtime") or {}
    output_mtp = output.get("mtp") or {}
    shards = source.get("mtp_shards") if isinstance(source.get("mtp_shards"), list) else []
    return {
        "kind": "ax_mtp_sidecar_manifest",
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "schema_version": manifest.get("schema_version"),
        "model_key": manifest.get("model_key"),
        "base_model_id": base.get("model_id"),
        "base_snapshot": base.get("snapshot"),
        "source_model_id": source.get("model_id"),
        "source_shard_count": len(shards),
        "source_shard_sha256": [
            shard.get("sha256") for shard in shards if isinstance(shard, dict)
        ],
        "output_mtp_sha256": output_mtp.get("sha256"),
        "output_mtp_size_bytes": output_mtp.get("size_bytes"),
        "transform_norm_policy": transform.get("norm_policy"),
        "mtp_depth_max": runtime.get("mtp_depth_max"),
        "mtp_tensor_count": runtime.get("mtp_tensor_count"),
        "recommended_draft_sampler": runtime.get("recommended_draft_sampler"),
        "sampler": runtime.get("sampler"),
    }


def sampling_for_mode(mode: str, sampled: dict[str, int | float]) -> dict[str, int | float]:
    if mode == "greedy":
        return {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    return dict(sampled)


def build_contract(
    config: RunConfig,
    suites: list[str],
    suite_files: dict[str, Path],
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite_fingerprints = {}
    for suite in suites:
        path = suite_files[suite]
        suite_fingerprints[suite] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    return {
        "mode": config.mode,
        "suites": suites,
        "suite_fingerprints": suite_fingerprints,
        "depth": config.depth,
        "max_tokens": config.max_tokens,
        "repetitions": config.repetitions,
        "warmup_repetitions": config.warmup_repetitions,
        "cooldown_s": config.cooldown_s,
        "sampling": config.sampling,
        "enable_thinking": config.enable_thinking,
        "model_provenance": model_provenance or {},
    }


def extract_depth_rates_from_telemetry(telemetry: dict[str, Any]) -> list[float | None]:
    rates: list[float | None] = []
    for depth in range(3):
        accepted = int(telemetry.get(f"ax_mtp_accepted_depth{depth}", 0) or 0)
        drafted = int(telemetry.get(f"ax_mtp_drafted_depth{depth}", 0) or 0)
        rates.append(accepted / drafted if drafted else None)
    return rates


def extract_mtplx_depth_rates(runs: list[dict[str, Any]]) -> list[float | None]:
    accepted = [0, 0, 0]
    drafted = [0, 0, 0]
    for run in runs:
        for index, value in enumerate(run.get("accepted_by_depth", [])[:3]):
            accepted[index] += int(value)
        for index, value in enumerate(run.get("drafted_by_depth", [])[:3]):
            drafted[index] += int(value)
    return [
        accepted[index] / drafted[index] if drafted[index] else None
        for index in range(3)
    ]


def first_trial_tokens(trials: list[dict[str, Any]], key: str) -> list[int] | None:
    for trial in trials:
        tokens = trial.get(key)
        if isinstance(tokens, list):
            return [int(token) for token in tokens]
    return None


def ax_cases(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in artifact.get("results", []):
        if row.get("engine") != AX_MTP_ENGINE:
            continue
        prompt_id = row.get("prompt_case_id")
        if not prompt_id:
            continue
        telemetry = row.get("ngram_acceleration_telemetry") or {}
        draft_tokens = int(telemetry.get("ax_mtp_draft_tokens", 0) or 0)
        accepted_tokens = int(telemetry.get("ax_mtp_accepted_tokens", 0) or 0)
        output_tokens = first_trial_tokens(row.get("trials", []), "output_token_ids")
        cases[str(prompt_id)] = {
            "prompt_id": str(prompt_id),
            "category": row.get("prompt_category"),
            "prompt_text_sha256": row.get("prompt_text_sha256"),
            "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
            "decode_tok_s": summary_median(row.get("decode_tok_s")),
            "client_wall_total_ms": summary_median(row.get("client_wall_total_ms")),
            "output_tokens": output_tokens,
            "output_token_sha256": token_sha256(output_tokens),
            "accepted_tokens": accepted_tokens,
            "draft_tokens": draft_tokens,
            "accept_rate": accepted_tokens / draft_tokens if draft_tokens else None,
            "acceptance_by_depth": extract_depth_rates_from_telemetry(telemetry),
            "ngram_hit_steps": int(telemetry.get("ax_mtp_ngram_hit_steps", 0) or 0),
            "mechanism_label": "mtp_ngram_stacked"
            if int(telemetry.get("ax_mtp_ngram_hit_steps", 0) or 0) > 0
            else "pure_mtp",
        }
    return cases


def mtplx_cases(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in artifact.get("results", []):
        prompt_id = case.get("prompt_id")
        if not prompt_id:
            continue
        runs = case.get("runs", [])
        output_tokens = first_trial_tokens(runs, "tokens")
        summary = case.get("summary", {})
        cases[str(prompt_id)] = {
            "prompt_id": str(prompt_id),
            "category": case.get("category"),
            "prompt_text_sha256": case.get("prompt_sha256"),
            "prompt_tokens": case.get("prompt_tokens"),
            "decode_tok_s": summary_median(summary.get("decode_tok_s")),
            "end_to_end_tok_s": summary_median(summary.get("end_to_end_tok_s")),
            "output_tokens": output_tokens,
            "output_token_sha256": token_sha256(output_tokens),
            "accepted_tokens": int(summary.get("accepted_drafts", 0) or 0),
            "draft_tokens": int(summary.get("drafted_tokens", 0) or 0),
            "accept_rate": summary.get("accept_rate"),
            "acceptance_by_depth": extract_mtplx_depth_rates(runs),
            "mechanism_label": "mtplx_mtp",
        }
    return cases


def artifact_dirty(artifact: dict[str, Any]) -> bool | None:
    build = artifact.get("build") or {}
    if "git_tracked_dirty" in build:
        return bool(build["git_tracked_dirty"])
    return None


def suite_warnings(
    *,
    suite: str,
    ax_artifact: dict[str, Any],
    mtplx_artifact: dict[str, Any],
    config: RunConfig,
) -> list[str]:
    warnings: list[str] = []
    if artifact_dirty(ax_artifact):
        warnings.append(f"{suite}: AX artifact was captured from a dirty tracked worktree")
    if artifact_dirty(mtplx_artifact):
        warnings.append(f"{suite}: MTPLX artifact was captured from a dirty tracked worktree")

    ax_reps = ax_artifact.get("repetitions")
    if ax_reps is not None and int(ax_reps) != config.repetitions:
        warnings.append(
            f"{suite}: AX repetitions={ax_reps} does not match contract={config.repetitions}"
        )
    mtplx_reps = mtplx_artifact.get("repetitions")
    if mtplx_reps is not None and int(mtplx_reps) != config.repetitions:
        warnings.append(
            f"{suite}: MTPLX repetitions={mtplx_reps} does not match contract={config.repetitions}"
        )
    mtplx_warmup = mtplx_artifact.get("warmup_repetitions")
    if mtplx_warmup is not None and int(mtplx_warmup) != config.warmup_repetitions:
        warnings.append(
            f"{suite}: MTPLX warmup={mtplx_warmup} does not match "
            f"contract={config.warmup_repetitions}"
        )
    ax_depth = ax_artifact.get("ax_mtp_max_depth")
    if ax_depth is not None and int(ax_depth) != config.depth:
        warnings.append(f"{suite}: AX depth={ax_depth} does not match contract={config.depth}")
    mtplx_depth = mtplx_artifact.get("depth")
    if mtplx_depth is not None and int(mtplx_depth) != config.depth:
        warnings.append(
            f"{suite}: MTPLX depth={mtplx_depth} does not match contract={config.depth}"
        )
    return warnings


def build_suite_diff(
    suite: str,
    ax_artifact_path: Path,
    mtplx_artifact_path: Path,
    config: RunConfig,
) -> dict[str, Any]:
    ax_artifact = load_json(ax_artifact_path)
    mtplx_artifact = load_json(mtplx_artifact_path)
    ax_by_id = ax_cases(ax_artifact)
    mtplx_by_id = mtplx_cases(mtplx_artifact)
    prompt_ids = sorted(set(ax_by_id) | set(mtplx_by_id))

    cases = []
    for prompt_id in prompt_ids:
        ax_case = ax_by_id.get(prompt_id)
        mtplx_case = mtplx_by_id.get(prompt_id)
        ax_tokens = ax_case.get("output_tokens") if ax_case else None
        mtplx_tokens = mtplx_case.get("output_tokens") if mtplx_case else None
        ax_token_hash = ax_case.get("output_token_sha256") if ax_case else None
        mtplx_token_hash = mtplx_case.get("output_token_sha256") if mtplx_case else None
        cases.append(
            {
                "prompt_id": prompt_id,
                "present": {
                    "ax": ax_case is not None,
                    "mtplx": mtplx_case is not None,
                },
                "ax": without_tokens(ax_case),
                "mtplx": without_tokens(mtplx_case),
                "speedup_ratio": speedup_ratio(
                    ax_case.get("decode_tok_s") if ax_case else None,
                    mtplx_case.get("decode_tok_s") if mtplx_case else None,
                ),
                "token_diff_first_index": first_diff_index(ax_tokens, mtplx_tokens),
                "output_token_sha256_equal": (
                    ax_token_hash == mtplx_token_hash
                    if ax_token_hash is not None and mtplx_token_hash is not None
                    else None
                ),
            }
        )

    ax_decode_values = [
        case["ax"]["decode_tok_s"]
        for case in cases
        if case.get("ax") and case["ax"].get("decode_tok_s") is not None
    ]
    mtplx_decode_values = [
        case["mtplx"]["decode_tok_s"]
        for case in cases
        if case.get("mtplx") and case["mtplx"].get("decode_tok_s") is not None
    ]
    ax_median = median([float(value) for value in ax_decode_values])
    mtplx_median = median([float(value) for value in mtplx_decode_values])

    return {
        "suite": suite,
        "artifacts": {
            "ax": str(ax_artifact_path),
            "mtplx": str(mtplx_artifact_path),
        },
        "warnings": suite_warnings(
            suite=suite,
            ax_artifact=ax_artifact,
            mtplx_artifact=mtplx_artifact,
            config=config,
        ),
        "summary": {
            "case_count": len(cases),
            "matched_case_count": sum(
                1 for case in cases if case["present"]["ax"] and case["present"]["mtplx"]
            ),
            "ax_decode_tok_s_median_of_cases": ax_median,
            "mtplx_decode_tok_s_median_of_cases": mtplx_median,
            "speedup_ratio": speedup_ratio(ax_median, mtplx_median),
            "token_identical_cases": sum(
                1 for case in cases if case.get("output_token_sha256_equal") is True
            ),
            "token_comparable_cases": sum(
                1 for case in cases if case.get("output_token_sha256_equal") is not None
            ),
        },
        "cases": cases,
    }


def without_tokens(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {key: value for key, value in case.items() if key != "output_tokens"}


def build_differential_artifact(
    *,
    config: RunConfig,
    suites: list[str],
    suite_files: dict[str, Path],
    ax_artifacts: dict[str, Path],
    mtplx_artifacts: dict[str, Path],
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_provenance = model_provenance or {}
    contract = build_contract(config, suites, suite_files, model_provenance)
    suite_results = [
        build_suite_diff(suite, ax_artifacts[suite], mtplx_artifacts[suite], config)
        for suite in suites
    ]
    warnings = [warning for suite in suite_results for warning in suite.get("warnings", [])]
    return {
        "schema": "ax.mtp_differential.v1",
        "created_at": date.today().isoformat(),
        "harness_contract_id": sha256_json(contract),
        "contract": contract,
        "model_provenance": model_provenance,
        "warnings": warnings,
        "summary": summarize_differential(suite_results),
        "suites": suite_results,
    }


def summarize_differential(suite_results: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        suite["summary"]["speedup_ratio"]
        for suite in suite_results
        if suite["summary"].get("speedup_ratio") is not None
    ]
    return {
        "suite_count": len(suite_results),
        "warnings_count": sum(len(suite.get("warnings", [])) for suite in suite_results),
        "speedup_ratio_median": median([float(value) for value in ratios]),
        "ax_wins": sum(1 for value in ratios if value > 1.0),
        "mtplx_wins": sum(1 for value in ratios if value < 1.0),
    }


def write_markdown_summary(path: Path, artifact: dict[str, Any]) -> None:
    lines = [
        "# MTP Differential Summary",
        "",
        f"Contract: `{artifact['harness_contract_id']}`",
        f"Mode: `{artifact['contract']['mode']}`",
        "",
    ]
    if artifact["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in artifact["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    provenance = artifact.get("model_provenance") or {}
    if provenance:
        lines.append("## Model Provenance")
        lines.append("")
        for label in ("ax", "mtplx"):
            model = provenance.get(label) or {}
            if not model:
                continue
            if model.get("kind") == "ax_mtp_sidecar_manifest":
                lines.append(
                    f"- {label}: `{model.get('base_model_id')}` + "
                    f"`{model.get('source_model_id')}` "
                    f"(sidecar `{model.get('output_mtp_sha256')}`)"
                )
            else:
                model_label = model.get("model", model.get("path", model.get("kind")))
                lines.append(f"- {label}: `{model_label}`")
        lines.append("")

    lines.append("## Suites")
    lines.append("")
    lines.append("| Suite | Matched cases | AX tok/s | MTPLX tok/s | AX / MTPLX | Token identical |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite in artifact["suites"]:
        summary = suite["summary"]
        ax_tok_s = format_number(summary.get("ax_decode_tok_s_median_of_cases"))
        mtplx_tok_s = format_number(summary.get("mtplx_decode_tok_s_median_of_cases"))
        ratio = format_number(summary.get("speedup_ratio"), digits=3)
        token_identical = (
            f"{summary['token_identical_cases']}/{summary['token_comparable_cases']}"
            if summary["token_comparable_cases"]
            else "-"
        )
        lines.append(
            f"| {suite['suite']} | {summary['matched_case_count']} | {ax_tok_s} | "
            f"{mtplx_tok_s} | {ratio} | {token_identical} |"
        )
    path.write_text("\n".join(lines) + "\n")


def format_number(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def run_subprocess(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_ax_suite(
    *,
    suite: str,
    suite_file: Path,
    output_path: Path,
    ax_model_dir: Path,
    config: RunConfig,
) -> Path:
    cmd = [
        sys.executable,
        str(AX_BENCH_SCRIPT),
        "--model-dir",
        str(ax_model_dir),
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
    run_subprocess(cmd)
    return output_path


def run_mtplx_suite(
    *,
    suite: str,
    suite_file: Path,
    output_path: Path,
    mtplx_model: str,
    config: RunConfig,
) -> Path:
    cmd = [
        sys.executable,
        str(MTPLX_BENCH_SCRIPT),
        "--model",
        mtplx_model,
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
    ]
    if not config.enable_thinking:
        cmd.append("--disable-thinking")
    run_subprocess(cmd)
    return output_path


def run_or_collect_artifacts(
    *,
    mode: str,
    suites: list[str],
    suite_files: dict[str, Path],
    config: RunConfig,
    output_dir: Path,
    ax_model_dir: Path | None,
    mtplx_model: str | None,
    ax_artifact_overrides: dict[str, Path],
    mtplx_artifact_overrides: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    ax_artifacts: dict[str, Path] = {}
    mtplx_artifacts: dict[str, Path] = {}
    for suite in suites:
        suite_out = output_dir / mode / suite
        suite_out.mkdir(parents=True, exist_ok=True)
        if suite in ax_artifact_overrides:
            ax_artifacts[suite] = ax_artifact_overrides[suite]
        else:
            if ax_model_dir is None:
                raise ValueError(f"missing --ax-model-dir or --ax-artifact for suite {suite}")
            ax_artifacts[suite] = run_ax_suite(
                suite=suite,
                suite_file=suite_files[suite],
                output_path=suite_out / "ax.json",
                ax_model_dir=ax_model_dir,
                config=config,
            )
        if suite in mtplx_artifact_overrides:
            mtplx_artifacts[suite] = mtplx_artifact_overrides[suite]
        else:
            if mtplx_model is None:
                raise ValueError(f"missing --mtplx-model or --mtplx-artifact for suite {suite}")
            mtplx_artifacts[suite] = run_mtplx_suite(
                suite=suite,
                suite_file=suite_files[suite],
                output_path=suite_out / "mtplx.json",
                mtplx_model=mtplx_model,
                config=config,
            )
    return ax_artifacts, mtplx_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ax-model-dir", type=Path)
    parser.add_argument("--mtplx-model")
    parser.add_argument("--suites", nargs="+", default=["flappy", "long_code"])
    parser.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    parser.add_argument(
        "--suite-file",
        action="append",
        default=[],
        help="Additional or overriding prompt suite path as NAME=PATH.",
    )
    parser.add_argument(
        "--ax-artifact",
        action="append",
        default=[],
        help="Use an existing AX artifact for a suite, as NAME=PATH.",
    )
    parser.add_argument(
        "--mtplx-artifact",
        action="append",
        default=[],
        help="Use an existing MTPLX artifact for a suite, as NAME=PATH.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["greedy", "sampled", "throughput"],
        default=["sampled"],
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=DEFAULT_WARMUP_REPETITIONS,
    )
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING["temperature"])
    parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_BASE / f"{date.today().isoformat()}-ax-vs-mtplx"
    suite_file_overrides = parse_name_path_map(args.suite_file, "--suite-file")
    ax_artifact_overrides = parse_name_path_map(args.ax_artifact, "--ax-artifact")
    mtplx_artifact_overrides = parse_name_path_map(args.mtplx_artifact, "--mtplx-artifact")

    suites = list(dict.fromkeys(args.suites))
    if args.warmup_repetitions != DEFAULT_WARMUP_REPETITIONS:
        suites_without_ax_artifact = [suite for suite in suites if suite not in ax_artifact_overrides]
        if suites_without_ax_artifact:
            raise ValueError(
                "AX bench_mlx_inference_stack.py has an implicit 1 warmup repetition; "
                "--warmup-repetitions must be 1 when running AX. "
                f"Suites without --ax-artifact: {', '.join(suites_without_ax_artifact)}"
            )
    suite_files = {
        suite: suite_path_for(suite, args.suites_dir, suite_file_overrides)
        for suite in suites
    }
    for suite, path in suite_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"suite {suite} not found: {path}")

    sampled = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    for mode in args.modes:
        config = RunConfig(
            mode=mode,
            depth=args.depth,
            max_tokens=args.max_tokens,
            repetitions=args.repetitions,
            warmup_repetitions=args.warmup_repetitions,
            cooldown_s=args.cooldown,
            sampling=sampling_for_mode(mode, sampled),
            enable_thinking=bool(args.enable_thinking),
        )
        ax_artifacts, mtplx_artifacts = run_or_collect_artifacts(
            mode=mode,
            suites=suites,
            suite_files=suite_files,
            config=config,
            output_dir=output_dir,
            ax_model_dir=args.ax_model_dir,
            mtplx_model=args.mtplx_model,
            ax_artifact_overrides=ax_artifact_overrides,
            mtplx_artifact_overrides=mtplx_artifact_overrides,
        )
        artifact = build_differential_artifact(
            config=config,
            suites=suites,
            suite_files=suite_files,
            ax_artifacts=ax_artifacts,
            mtplx_artifacts=mtplx_artifacts,
            model_provenance={
                "ax": model_provenance_for(args.ax_model_dir),
                "mtplx": model_provenance_for(args.mtplx_model),
            },
        )
        mode_dir = output_dir / mode
        write_json(mode_dir / "differential.json", artifact)
        write_markdown_summary(mode_dir / "summary.md", artifact)
        print(f"Saved differential artifact: {mode_dir / 'differential.json'}", flush=True)
        print(f"Saved summary: {mode_dir / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
