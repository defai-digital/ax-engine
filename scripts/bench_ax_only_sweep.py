#!/usr/bin/env python3
"""Run AX benchmarks across the README direct-mode rows while reusing exact
mlx_lm reference artifacts.

Per row this invokes scripts/bench_mlx_inference_stack.py with
  --ax-compare-policies, or --ax-direct for the strict peer-win gate
  --reuse-reference-results-from <prev mlx_lm JSON>
  --no-build-ax-engine  (uses existing target/release/ax-engine-server)
and skips both mlx_lm and llama.cpp (the former via --reuse, the latter by
not passing --llama-cpp-bench).

The default scope excludes two inventory-only 8-bit probes. Strict peer-win
sweeps also exclude README rows whose current upstream mlx_lm cannot load, and
record those unavailable peers explicitly. Writes per-row JSON plus a
fail-closed sweep_results.json and sweep_summary.md matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "llama_cpp_metal" / "inventory.json"
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"
DEFAULT_MAX_LOAD_AVERAGE = 2.0
DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT = 50.0
PEER_WIN_MATRIX_SCHEMA_VERSION = "ax.ax_mlx_lm_peer_win_matrix.v1"
PEER_WIN_SCHEMA_VERSION = "ax.ax_mlx_lm_peer_wins.v1"
PUBLICATION_MIN_COOLDOWN_SECONDS = 15.0
PUBLICATION_MIN_MEASUREMENT_REPETITIONS = 5
PUBLICATION_MIN_WARMUP_REPETITIONS = 2
MLX_INFERENCE_STACK_SCHEMA_VERSION = "ax.mlx_inference_stack.v2"


class AxOnlySweepError(RuntimeError):
    pass


class SweepInterrupted(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[ax-sweep] {msg}", flush=True)


def _command_output_lines(cmd: list[str]) -> list[str]:
    try:
        output = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def collect_performance_condition_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        metadata["load_average"] = {
            "one_minute": round(load_1m, 3),
            "five_minutes": round(load_5m, 3),
            "fifteen_minutes": round(load_15m, 3),
        }
    except Exception:
        pass

    battery_lines = _command_output_lines(["pmset", "-g", "batt"])
    if battery_lines:
        match = re.search(r"Now drawing from '([^']+)'", battery_lines[0])
        metadata["power_source"] = match.group(1) if match else battery_lines[0]
        if len(battery_lines) > 1:
            metadata["battery_status"] = battery_lines[1]

    thermal_lines = _command_output_lines(["pmset", "-g", "therm"])
    if thermal_lines:
        metadata["thermal_status_lines"] = thermal_lines[:10]
        metadata["thermal_warning_recorded"] = not any(
            "No thermal warning level has been recorded" in line
            for line in thermal_lines
        )
        metadata["performance_warning_recorded"] = not any(
            "No performance warning level has been recorded" in line
            for line in thermal_lines
        )
        metadata["cpu_power_status_recorded"] = not any(
            "No CPU power status has been recorded" in line for line in thermal_lines
        )

    return metadata


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def latest_hf_cache_snapshot(repo_id: str, cache_dir: Path) -> Path | None:
    repo_cache = cache_dir / f"models--{_slug_repo_id(repo_id)}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def missing_ax_model_artifacts(model_dir: Path) -> list[str]:
    missing = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not (model_dir / "model-manifest.json").is_file():
        missing.append("model-manifest.json")
    if not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    return missing


def _model_dir_args(model_dir: Path, repo_id: str | None) -> list[str]:
    args: list[str] = []
    if repo_id:
        args.extend(["--model-repo-id", repo_id])
    args.extend(["--model-dir", str(model_dir)])
    return args


def _hf_cache_snapshot_info(path: Path, repo_id: str | None) -> tuple[str, str] | None:
    expected_repo_cache = f"models--{_slug_repo_id(repo_id)}" if repo_id else None
    parts = path.expanduser().parts
    for index, part in enumerate(parts):
        if not part.startswith("models--"):
            continue
        if expected_repo_cache is not None and part != expected_repo_cache:
            continue
        if index + 2 >= len(parts) or parts[index + 1] != "snapshots":
            continue
        revision = parts[index + 2]
        if revision:
            return part, revision
    return None


def _resolve_reference_model_dir(
    reference_model_dir: str,
    *,
    repo_id: str | None,
    cache_dir: Path,
) -> Path | None:
    reference_path = Path(reference_model_dir).expanduser()
    candidates = [reference_path]
    snapshot_info = _hf_cache_snapshot_info(reference_path, repo_id)
    if snapshot_info is not None:
        repo_cache, revision = snapshot_info
        remapped = cache_dir / repo_cache / "snapshots" / revision
        if remapped not in candidates:
            candidates.append(remapped)

    for candidate in candidates:
        if candidate.is_dir() and not missing_ax_model_artifacts(candidate):
            return candidate
    return None


def _reference_model_dir_value(reference_artifact: Path) -> str | None:
    try:
        doc = json.loads(reference_artifact.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    value = doc.get("model_dir")
    if isinstance(value, str) and value.strip():
        return value
    model = doc.get("model")
    if isinstance(model, str) and model.startswith(("/", "~")):
        return model
    return None


def reference_model_args(
    row: dict[str, Any],
    *,
    cache_dir: Path,
    reference_artifact: Path | None,
) -> tuple[list[str] | None, str | None]:
    if reference_artifact is None or not reference_artifact.is_file():
        return None, None
    repo_id = row.get("mlx_repo_id")
    model_dir_value = _reference_model_dir_value(reference_artifact)
    if model_dir_value is None:
        return None, f"reference artifact has no reusable model_dir: {reference_artifact}"
    model_dir = _resolve_reference_model_dir(
        model_dir_value,
        repo_id=repo_id if isinstance(repo_id, str) else None,
        cache_dir=cache_dir,
    )
    if model_dir is None:
        return None, f"reference model_dir is not AX-ready or not found: {model_dir_value}"
    return _model_dir_args(model_dir, repo_id if isinstance(repo_id, str) else None), None


def resolve_model_args(
    row: dict[str, Any],
    cache_dir: Path,
    reference_artifact: Path | None = None,
) -> tuple[list[str] | None, str | None]:
    repo_id = row.get("mlx_repo_id")
    repo_id_arg = repo_id if isinstance(repo_id, str) and repo_id else None
    local_dir_value = row.get("mlx_local_dir")
    if local_dir_value:
        model_dir = REPO_ROOT / local_dir_value
        if model_dir.exists():
            missing = missing_ax_model_artifacts(model_dir)
            if missing:
                return (
                    None,
                    f"local MLX model dir for {row.get('slug')} missing "
                    f"{', '.join(missing)}",
                )
            return _model_dir_args(model_dir, repo_id_arg), None

    reference_args, _reference_error = reference_model_args(
        row,
        cache_dir=cache_dir,
        reference_artifact=reference_artifact,
    )
    if reference_args is not None:
        return reference_args, None

    if not repo_id_arg:
        return None, f"No mlx_local_dir or mlx_repo_id for {row.get('slug')}"
    snap = latest_hf_cache_snapshot(repo_id_arg, cache_dir)
    if snap is None:
        return None, f"No HF cache snapshot for {repo_id_arg}"
    # AX requires config.json + model-manifest.json + *.safetensors in the snapshot.
    missing = missing_ax_model_artifacts(snap)
    if missing:
        return None, f"MLX cache snapshot for {repo_id_arg} missing {', '.join(missing)}"
    return ["--model-repo-id", repo_id_arg, "--hf-cache-root", str(cache_dir)], None


def _row_slug(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        raise AxOnlySweepError("manifest row must be an object")
    slug = row.get("slug")
    if not isinstance(slug, str) or not slug:
        raise AxOnlySweepError("manifest row lacks non-empty slug")
    return slug


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    rows_filter: list[str] | None,
) -> list[dict[str, Any]]:
    if not rows:
        raise AxOnlySweepError("manifest contains no rows")

    seen: set[str] = set()
    duplicate_slugs: set[str] = set()
    for row in rows:
        slug = _row_slug(row)
        if slug in seen:
            duplicate_slugs.add(slug)
        seen.add(slug)
    if duplicate_slugs:
        raise AxOnlySweepError(
            "manifest contains duplicate slug(s): "
            + ", ".join(sorted(duplicate_slugs))
        )

    if rows_filter is None:
        return rows
    if not rows_filter:
        raise AxOnlySweepError("--rows-filter requires at least one slug")

    requested: set[str] = set()
    duplicate_filters: set[str] = set()
    for slug in rows_filter:
        if slug in requested:
            duplicate_filters.add(slug)
        requested.add(slug)
    if duplicate_filters:
        raise AxOnlySweepError(
            "--rows-filter contains duplicate slug(s): "
            + ", ".join(sorted(duplicate_filters))
        )

    missing = sorted(requested - seen)
    if missing:
        raise AxOnlySweepError(
            "--rows-filter references unknown slug(s): " + ", ".join(missing)
        )

    selected = [row for row in rows if _row_slug(row) in requested]
    if not selected:
        raise AxOnlySweepError("--rows-filter selected no rows")
    return selected


def readme_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("readme_direct_table") is not False]


def mlx_lm_peer_comparable_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("mlx_lm_peer_required") is not False]


def select_sweep_rows(
    rows: list[dict[str, Any]],
    rows_filter: list[str] | None,
    *,
    require_ax_multi_metric_peer_wins: bool,
) -> tuple[list[dict[str, Any]], str]:
    validated_rows = filter_manifest_rows(rows, rows_filter)
    if rows_filter is not None:
        return validated_rows, "filtered"

    readme_rows = readme_manifest_rows(validated_rows)
    if require_ax_multi_metric_peer_wins:
        return (
            mlx_lm_peer_comparable_rows(readme_rows),
            "readme_mlx_lm_comparable",
        )
    return readme_rows, "readme_direct_table"


def terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            proc.kill()
        proc.wait()


def run_row(
    row: dict[str, Any],
    *,
    output_dir: Path,
    bench_script: Path,
    prompt_tokens: str,
    generation_tokens: int,
    repetitions: int,
    warmup_repetitions: int,
    cooldown: float,
    model_args: list[str],
    reuse_ref_root: Path | None,
    ax_direct_only: bool = False,
    require_ax_multi_metric_peer_wins: bool = False,
    max_load_average: float | None = None,
    max_top_process_cpu_percent: float | None = None,
    load_average_wait_timeout: float | None = None,
    load_average_poll_interval: float | None = None,
) -> dict[str, Any]:
    slug = row["slug"]
    out_json = output_dir / f"{slug}.json"
    log_path = output_dir / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bench_script),
        *model_args,
        "--prompt-tokens", prompt_tokens,
        "--generation-tokens", str(generation_tokens),
        "--repetitions", str(repetitions),
        "--warmup-repetitions", str(warmup_repetitions),
        "--cooldown", str(cooldown),
        "--no-build-ax-engine",
        "--output", str(out_json),
    ]
    if ax_direct_only:
        cmd.extend(["--skip-mlx-lm", "--ax-direct"])
    else:
        if reuse_ref_root is None:
            return {
                "status": "skipped_no_reference",
                "note": "--reuse-reference-root is required outside --ax-direct-only",
            }
        ref_json = reuse_ref_root / f"{slug}.json"
        if not ref_json.is_file():
            return {"status": "skipped_no_reference", "note": f"missing {ref_json}"}
        cmd.extend(
            [
                (
                    "--ax-direct"
                    if require_ax_multi_metric_peer_wins
                    else "--ax-compare-policies"
                ),
                "--reuse-reference-results-from",
                str(ref_json),
            ]
        )
    if require_ax_multi_metric_peer_wins:
        cmd.append("--require-ax-multi-metric-peer-wins")
    if max_load_average is not None:
        cmd.extend(["--max-load-average", str(max_load_average)])
    if max_top_process_cpu_percent is not None:
        cmd.extend(
            [
                "--max-top-process-cpu-percent",
                str(max_top_process_cpu_percent),
            ]
        )
    if load_average_wait_timeout is not None:
        cmd.extend(["--load-average-wait-timeout", str(load_average_wait_timeout)])
    if load_average_poll_interval is not None:
        cmd.extend(["--load-average-poll-interval", str(load_average_poll_interval)])
    log(f"  invoke: {' '.join(cmd)}")
    with log_path.open("w") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait()
        except (KeyboardInterrupt, SweepInterrupted):
            terminate_process_tree(proc)
            raise
    if returncode != 0:
        return {
            "status": "bench_failed",
            "exit_code": returncode,
            "log_path": str(log_path),
            "output_path": str(out_json) if out_json.exists() else None,
        }
    if not out_json.exists():
        return {"status": "bench_failed_no_output", "log_path": str(log_path)}
    return {
        "status": "ok",
        "output_path": str(out_json),
        "log_path": str(log_path),
        "result_doc": json.loads(out_json.read_text()),
    }


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def status_counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{status}={count}" for status, count in counts.items())


def _increment_reason(
    summary: dict[str, Any],
    reason: str,
    count: int = 1,
) -> None:
    counts = summary["failure_reason_counts"]
    counts[reason] = int(counts.get(reason, 0)) + count
    summary["publication_candidate"] = False


def parse_prompt_token_csv(value: str) -> list[int]:
    try:
        prompt_tokens = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise AxOnlySweepError("--prompt-tokens must be comma-separated integers") from error
    if not prompt_tokens or any(token <= 0 for token in prompt_tokens):
        raise AxOnlySweepError("--prompt-tokens values must be positive")
    if len(prompt_tokens) != len(set(prompt_tokens)):
        raise AxOnlySweepError("--prompt-tokens values must be unique")
    return prompt_tokens


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _performance_condition_failure_reasons(
    conditions: Any,
    *,
    max_load_average: float,
    max_top_process_cpu_percent: float,
) -> list[str]:
    if not isinstance(conditions, dict):
        return ["missing_performance_conditions"]
    failures: list[str] = []
    load_average = conditions.get("load_average")
    one_minute = (
        _number(load_average.get("one_minute"))
        if isinstance(load_average, dict)
        else None
    )
    if one_minute is None:
        failures.append("missing_load_average")
    elif one_minute > max_load_average:
        failures.append("load_average_above_limit")

    top_processes = conditions.get("top_processes_cpu")
    top_cpu_values: list[float] = []
    if isinstance(top_processes, list):
        for process in top_processes:
            if not isinstance(process, dict):
                continue
            value = _number(process.get("cpu_percent"))
            if value is not None:
                top_cpu_values.append(value)
    if not top_cpu_values:
        failures.append("missing_top_process_cpu")
    elif max(top_cpu_values) > max_top_process_cpu_percent:
        failures.append("top_process_cpu_above_limit")

    if conditions.get("power_source") != "AC Power":
        failures.append("not_on_ac_power")
    for key in (
        "thermal_warning_recorded",
        "performance_warning_recorded",
        "cpu_power_status_recorded",
    ):
        if conditions.get(key) is not False:
            failures.append(f"{key}_not_clear")
    return failures


def publication_metadata_failure_reasons(
    metadata: Any,
    *,
    max_load_average: float,
    max_top_process_cpu_percent: float,
) -> list[str]:
    if not isinstance(metadata, dict):
        return ["missing_publication_metadata"]
    failures: list[str] = []
    if metadata.get("schema_version") != MLX_INFERENCE_STACK_SCHEMA_VERSION:
        failures.append("unexpected_artifact_schema")

    build = metadata.get("build")
    if not isinstance(build, dict):
        failures.append("missing_build_metadata")
    else:
        commit = build.get("commit")
        if not isinstance(commit, str) or not commit or commit == "unknown":
            failures.append("missing_build_commit")
        if build.get("build_profile") != "release":
            failures.append("non_release_build")
        if build.get("git_tracked_dirty") is not False:
            failures.append("dirty_tracked_build")

    repetitions = _number(metadata.get("repetitions"))
    if (
        repetitions is None
        or repetitions < PUBLICATION_MIN_MEASUREMENT_REPETITIONS
    ):
        failures.append("insufficient_measurement_repetitions")
    warmups = _number(metadata.get("warmup_repetitions"))
    if warmups is None or warmups < PUBLICATION_MIN_WARMUP_REPETITIONS:
        failures.append("insufficient_warmup_repetitions")
    cooldown = _number(metadata.get("cooldown"))
    if cooldown is None or cooldown < PUBLICATION_MIN_COOLDOWN_SECONDS:
        failures.append("insufficient_cooldown")

    window = metadata.get("benchmark_window")
    if not isinstance(window, dict):
        failures.append("missing_benchmark_window")
    else:
        for boundary in ("performance_conditions_start", "performance_conditions_end"):
            for reason in _performance_condition_failure_reasons(
                window.get(boundary),
                max_load_average=max_load_average,
                max_top_process_cpu_percent=max_top_process_cpu_percent,
            ):
                failures.append(f"{boundary}_{reason}")
    return failures


def engine_trial_failure_reasons(
    doc: dict[str, Any],
    *,
    engine: str,
    expected_cells: set[tuple[int, int]],
) -> list[str]:
    results = doc.get("results")
    if not isinstance(results, list):
        return ["missing_results"]
    failures: list[str] = []
    actual_cells: set[tuple[int, int]] = set()
    for row in results:
        if not isinstance(row, dict) or row.get("engine") != engine:
            continue
        try:
            key = (int(row["prompt_tokens"]), int(row["generation_tokens"]))
        except (KeyError, TypeError, ValueError):
            failures.append("invalid_trial_shape")
            continue
        if key in actual_cells:
            failures.append("duplicate_trial_shape")
            continue
        actual_cells.add(key)
        trials = row.get("trials")
        if not isinstance(trials, list) or len(trials) < PUBLICATION_MIN_MEASUREMENT_REPETITIONS:
            failures.append("insufficient_row_trials")
    if actual_cells != expected_cells:
        failures.append("trial_shape_mismatch")
    return failures


def summarize_peer_win_matrix(
    rows: list[dict[str, Any]],
    *,
    expected_slugs: list[str],
    prompt_tokens: list[int],
    generation_tokens: int,
    max_load_average: float | None = DEFAULT_MAX_LOAD_AVERAGE,
    max_top_process_cpu_percent: float | None = DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
) -> dict[str, Any]:
    expected_cells = {
        (prompt_length, generation_tokens) for prompt_length in prompt_tokens
    }
    summary: dict[str, Any] = {
        "schema_version": PEER_WIN_MATRIX_SCHEMA_VERSION,
        "scope": "readme_direct_ax_engine_mlx_vs_mlx_lm",
        "required_wins": {
            "prefill_tok_s": "ax_strictly_greater",
            "decode_tok_s": "ax_strictly_greater",
            "ttft_ms": "ax_strictly_lower",
        },
        "expected_slugs": expected_slugs,
        "expected_prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "expected_model_count": len(expected_slugs),
        "observed_model_count": 0,
        "strict_win_model_count": 0,
        "expected_cell_count": len(expected_slugs) * len(expected_cells),
        "strict_win_cell_count": 0,
        "performance_gate": {
            "max_load_average": max_load_average,
            "max_top_process_cpu_percent": max_top_process_cpu_percent,
        },
        "failure_reason_counts": {},
        "models": [],
        "publication_candidate": bool(expected_slugs and expected_cells),
    }
    if not expected_slugs:
        _increment_reason(summary, "no_expected_models")
    if not expected_cells:
        _increment_reason(summary, "no_expected_cells")
    if max_load_average is None or max_load_average > DEFAULT_MAX_LOAD_AVERAGE:
        _increment_reason(summary, "missing_or_relaxed_load_gate")
    if (
        max_top_process_cpu_percent is None
        or max_top_process_cpu_percent > DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT
    ):
        _increment_reason(summary, "missing_or_relaxed_top_process_cpu_gate")
    publication_max_load = (
        max_load_average
        if max_load_average is not None
        else DEFAULT_MAX_LOAD_AVERAGE
    )
    publication_max_top_cpu = (
        max_top_process_cpu_percent
        if max_top_process_cpu_percent is not None
        else DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT
    )

    rows_by_slug: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            _increment_reason(summary, "invalid_sweep_slug")
            continue
        if slug in rows_by_slug:
            _increment_reason(summary, "duplicate_sweep_slug")
            continue
        rows_by_slug[slug] = row

    unexpected_slugs = sorted(set(rows_by_slug) - set(expected_slugs))
    if unexpected_slugs:
        _increment_reason(summary, "unexpected_sweep_slug", len(unexpected_slugs))
        summary["unexpected_slugs"] = unexpected_slugs

    for slug in expected_slugs:
        row = rows_by_slug.get(slug)
        model_summary: dict[str, Any] = {
            "slug": slug,
            "classification": "not_strict_win",
            "failure_reasons": [],
        }
        failures = model_summary["failure_reasons"]
        if row is None:
            failures.append("missing_sweep_row")
        else:
            summary["observed_model_count"] += 1
            model_summary["status"] = row.get("status")
            model_summary["output_path"] = row.get("output_path")
            if row.get("status") != "ok":
                failures.append("sweep_row_not_ok")
            result_doc = row.get("result_doc")
            if not isinstance(result_doc, dict):
                failures.append("missing_result_doc")
            else:
                for reason in publication_metadata_failure_reasons(
                    result_doc,
                    max_load_average=publication_max_load,
                    max_top_process_cpu_percent=publication_max_top_cpu,
                ):
                    failures.append(f"current_{reason}")
                if result_doc.get("ax_prefix_cache_mode") != (
                    "disabled_for_cold_prefill_benchmark"
                ):
                    failures.append("current_prefix_cache_not_disabled")
                run_stability = result_doc.get("run_stability_summary")
                if not isinstance(run_stability, dict):
                    failures.append("current_missing_run_stability")
                elif run_stability.get("publication_candidate") is not True:
                    failures.append("current_unstable_ax_rows")
                for reason in engine_trial_failure_reasons(
                    result_doc,
                    engine="ax_engine_mlx",
                    expected_cells=expected_cells,
                ):
                    failures.append(f"ax_{reason}")
                for reason in engine_trial_failure_reasons(
                    result_doc,
                    engine="mlx_lm",
                    expected_cells=expected_cells,
                ):
                    failures.append(f"mlx_lm_{reason}")

                reference_contract = result_doc.get("reference_contract")
                reference_metadata = (
                    reference_contract.get(
                        "reused_reference_artifact_publication_metadata"
                    )
                    if isinstance(reference_contract, dict)
                    else None
                )
                for reason in publication_metadata_failure_reasons(
                    reference_metadata,
                    max_load_average=publication_max_load,
                    max_top_process_cpu_percent=publication_max_top_cpu,
                ):
                    failures.append(f"reference_{reason}")

                peer_summary = result_doc.get("ax_mlx_lm_peer_wins")
                if not isinstance(peer_summary, dict):
                    failures.append("missing_peer_win_summary")
                else:
                    model_summary["peer_win_schema_version"] = peer_summary.get(
                        "schema_version"
                    )
                    model_summary["pair_count"] = peer_summary.get("pair_count")
                    model_summary["strict_win_count"] = peer_summary.get(
                        "strict_win_count"
                    )
                    if peer_summary.get("schema_version") != PEER_WIN_SCHEMA_VERSION:
                        failures.append("unexpected_peer_win_schema")
                    if peer_summary.get("publication_candidate") is not True:
                        failures.append("peer_win_not_publication_candidate")
                    peer_rows = peer_summary.get("rows")
                    if not isinstance(peer_rows, list):
                        failures.append("missing_peer_win_rows")
                    else:
                        actual_cells: set[tuple[int, int]] = set()
                        strict_cells = 0
                        for peer_row in peer_rows:
                            if not isinstance(peer_row, dict):
                                failures.append("invalid_peer_win_row")
                                continue
                            try:
                                key = (
                                    int(peer_row["prompt_tokens"]),
                                    int(peer_row["generation_tokens"]),
                                )
                            except (KeyError, TypeError, ValueError):
                                failures.append("invalid_peer_win_shape")
                                continue
                            if key in actual_cells:
                                failures.append("duplicate_peer_win_shape")
                                continue
                            actual_cells.add(key)
                            if peer_row.get("classification") == "strict_win":
                                strict_cells += 1
                        if actual_cells != expected_cells:
                            failures.append("peer_win_shape_mismatch")
                        if strict_cells != len(expected_cells):
                            failures.append("incomplete_strict_win_cells")
                        model_summary["cells"] = peer_rows
                        model_summary["strict_win_cell_count"] = strict_cells

        if failures:
            for reason in sorted(set(failures)):
                _increment_reason(summary, reason)
        else:
            model_summary["classification"] = "strict_win"
            summary["strict_win_model_count"] += 1
            summary["strict_win_cell_count"] += len(expected_cells)
        summary["models"].append(model_summary)

    if summary["strict_win_model_count"] != summary["expected_model_count"]:
        summary["publication_candidate"] = False
    if summary["strict_win_cell_count"] != summary["expected_cell_count"]:
        summary["publication_candidate"] = False
    return summary


def peer_win_matrix_failure_reasons(summary: dict[str, Any]) -> list[str]:
    if summary.get("publication_candidate") is True:
        return []
    counts = summary.get("failure_reason_counts", {})
    if not isinstance(counts, dict):
        return ["publication_candidate=false"]
    reasons = [
        f"{reason}={int(count)}"
        for reason, count in sorted(counts.items())
        if int(count) > 0
    ]
    return reasons or ["publication_candidate=false"]


def fail_if_peer_win_matrix_not_publication_candidate(
    summary: dict[str, Any],
) -> None:
    reasons = peer_win_matrix_failure_reasons(summary)
    if not reasons:
        return
    print(
        "ERROR: README peer-win matrix is not a publication candidate: "
        + ", ".join(reasons),
        file=sys.stderr,
    )
    sys.exit(2)


def failed_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") != "ok"]


def fail_if_sweep_incomplete(rows: list[dict[str, Any]]) -> None:
    failed = failed_sweep_rows(rows)
    if not failed:
        return
    counts = status_counts(rows)
    print(
        "ERROR: AX-only sweep did not complete cleanly; "
        f"{len(failed)} row(s) were not ok: {status_counts_text(counts)}",
        file=sys.stderr,
    )
    sys.exit(2)


def build_sweep_doc(
    *,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    started: float,
    planned_slugs: list[str],
    performance_conditions_start: dict[str, Any],
) -> dict[str, Any]:
    elapsed = time.time() - started
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    performance_conditions_end = collect_performance_condition_metadata()
    failed_rows = failed_sweep_rows(summary_rows)
    peer_win_matrix = None
    if args.require_ax_multi_metric_peer_wins:
        peer_win_matrix = summarize_peer_win_matrix(
            summary_rows,
            expected_slugs=planned_slugs,
            prompt_tokens=parse_prompt_token_csv(args.prompt_tokens),
            generation_tokens=args.generation_tokens,
            max_load_average=args.max_load_average,
            max_top_process_cpu_percent=args.max_top_process_cpu_percent,
        )
    publication_candidate = not failed_rows and (
        peer_win_matrix is None
        or peer_win_matrix.get("publication_candidate") is True
    )
    sweep_doc = {
        "schema_version": "ax.ax_only_sweep.v1",
        "manifest_path": str(args.manifest),
        "scope": args.sweep_scope,
        "reuse_reference_root": (
            str(args.reuse_reference_root) if args.reuse_reference_root else None
        ),
        "model_snapshot_reference_root": (
            str(args.model_snapshot_reference_root)
            if args.model_snapshot_reference_root
            else None
        ),
        "ax_direct_only": bool(args.ax_direct_only),
        "require_ax_multi_metric_peer_wins": bool(
            args.require_ax_multi_metric_peer_wins
        ),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "cooldown": args.cooldown,
        "max_load_average": args.max_load_average,
        "max_top_process_cpu_percent": args.max_top_process_cpu_percent,
        "load_average_wait_timeout": args.load_average_wait_timeout,
        "load_average_poll_interval": args.load_average_poll_interval,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "benchmark_window": {
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(started),
            ),
            "finished_at": finished_at,
            "elapsed_seconds": round(elapsed, 1),
            "performance_conditions_start": performance_conditions_start,
            "performance_conditions_end": performance_conditions_end,
        },
        "readme_row_count": len(args.readme_slugs),
        "readme_slugs": args.readme_slugs,
        "mlx_lm_peer_unavailable_readme_rows": args.peer_unavailable_readme_rows,
        "planned_row_count": len(planned_slugs),
        "planned_slugs": planned_slugs,
        "completed_row_count": len(
            [
                row
                for row in summary_rows
                if row.get("status") not in {"running", "interrupted"}
            ]
        ),
        "publication_candidate": publication_candidate,
        "readme_peer_win_publication_candidate": bool(
            args.require_ax_multi_metric_peer_wins
            and args.sweep_scope == "readme_mlx_lm_comparable"
            and publication_candidate
        ),
        "failed_row_count": len(failed_rows),
        "status_counts": status_counts(summary_rows),
        "rows": summary_rows,
    }
    if peer_win_matrix is not None:
        sweep_doc["peer_win_matrix"] = peer_win_matrix
    return sweep_doc


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def sweep_row_note(row: dict[str, Any]) -> str:
    parts: list[str] = []
    note = row.get("note") or row.get("error")
    if note:
        parts.append(str(note))
    if row.get("exit_code") is not None:
        parts.append(f"exit_code={row['exit_code']}")
    if row.get("log_path"):
        parts.append(f"log={row['log_path']}")
    if row.get("output_path"):
        parts.append(f"output={row['output_path']}")
    return "; ".join(parts)


def write_sweep_outputs(
    *,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    started: float,
    planned_slugs: list[str],
    performance_conditions_start: dict[str, Any],
) -> dict[str, Any]:
    sweep_doc = build_sweep_doc(
        args=args,
        summary_rows=summary_rows,
        started=started,
        planned_slugs=planned_slugs,
        performance_conditions_start=performance_conditions_start,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_root / "sweep_results.json").write_text(
        json.dumps(sweep_doc, indent=2) + "\n"
    )
    log(f"wrote {args.output_root / 'sweep_results.json'}")

    md = [
        "# AX-only sweep summary",
        "",
        f"- scope: {sweep_doc['scope']}",
        f"- publication_candidate: {str(sweep_doc['publication_candidate']).lower()}",
        "- readme_peer_win_publication_candidate: "
        f"{str(sweep_doc['readme_peer_win_publication_candidate']).lower()}",
        f"- failed_row_count: {sweep_doc['failed_row_count']}",
        f"- status_counts: {status_counts_text(sweep_doc['status_counts'])}",
        f"- completed_row_count: {sweep_doc['completed_row_count']}/{len(planned_slugs)}",
        f"- elapsed: {sweep_doc['elapsed_seconds']:.0f}s",
    ]
    peer_win_matrix = sweep_doc.get("peer_win_matrix")
    if isinstance(peer_win_matrix, dict):
        md.extend(
            [
                "- strict_win_models: "
                f"{peer_win_matrix['strict_win_model_count']}/"
                f"{peer_win_matrix['expected_model_count']}",
                "- strict_win_cells: "
                f"{peer_win_matrix['strict_win_cell_count']}/"
                f"{peer_win_matrix['expected_cell_count']}",
            ]
        )
    unavailable_rows = sweep_doc["mlx_lm_peer_unavailable_readme_rows"]
    if unavailable_rows:
        md.append(
            "- mlx_lm_peer_unavailable_readme_rows: "
            + ", ".join(row["slug"] for row in unavailable_rows)
        )
    md.extend(
        [
            "",
            "| slug | status | notes |",
            "|---|---|---|",
        ]
    )
    for row in summary_rows:
        md.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(row["slug"]),
                    markdown_cell(row.get("status", "?")),
                    markdown_cell(sweep_row_note(row)),
                ]
            )
            + " |"
        )
    (args.output_root / "sweep_summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {args.output_root / 'sweep_summary.md'}")
    return sweep_doc


def _raise_sweep_interrupted(signum: int, _frame: Any) -> None:
    raise SweepInterrupted(f"received {signal.Signals(signum).name}")


def install_interrupt_handlers() -> dict[int, Any]:
    handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_sweep_interrupted)
    return handlers


def restore_interrupt_handlers(handlers: dict[int, Any]) -> None:
    for signum, handler in handlers.items():
        signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bench-script", type=Path, default=DEFAULT_BENCH_SCRIPT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-reference-root",
        type=Path,
        help=(
            "Directory containing per-row JSONs (<slug>.json) with mlx_lm "
            "baseline rows. Required unless --ax-direct-only is set."
        ),
    )
    parser.add_argument(
        "--model-snapshot-reference-root",
        type=Path,
        help=(
            "Directory containing per-row JSONs used only to resolve exact "
            "model_dir provenance. Defaults to --reuse-reference-root."
        ),
    )
    parser.add_argument(
        "--ax-direct-only",
        action="store_true",
        help=(
            "Run only the direct AX row for each model, without requiring or "
            "reusing mlx_lm reference rows."
        ),
    )
    parser.add_argument(
        "--require-ax-multi-metric-peer-wins",
        action="store_true",
        help=(
            "Require each row's direct AX result to strictly beat its reused "
            "mlx_lm reference in prefill, decode, and TTFT."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument("--rows-filter", nargs="*")
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument(
        "--max-load-average",
        type=float,
        default=DEFAULT_MAX_LOAD_AVERAGE,
        help=(
            "Forwarded to bench_mlx_inference_stack.py. Requires one-minute "
            "load average at or below this value before each row and AX repetition. "
            f"Default: {DEFAULT_MAX_LOAD_AVERAGE:.1f}."
        ),
    )
    parser.add_argument(
        "--no-load-gate",
        action="store_true",
        help="Disable the default publication performance gates for diagnostic sweeps.",
    )
    parser.add_argument(
        "--max-top-process-cpu-percent",
        type=float,
        default=DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
        help=(
            "Forwarded to bench_mlx_inference_stack.py. Requires the highest-CPU "
            "process to be at or below this value before each row and AX repetition. "
            f"Default: {DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT:.1f}."
        ),
    )
    parser.add_argument(
        "--load-average-wait-timeout",
        type=float,
        default=None,
        help="Forwarded to bench_mlx_inference_stack.py when --max-load-average is set.",
    )
    parser.add_argument(
        "--load-average-poll-interval",
        type=float,
        default=None,
        help="Forwarded to bench_mlx_inference_stack.py when --max-load-average is set.",
    )
    args = parser.parse_args()

    if not args.ax_direct_only and args.reuse_reference_root is None:
        parser.error("--reuse-reference-root is required unless --ax-direct-only")
    if args.ax_direct_only and args.require_ax_multi_metric_peer_wins:
        parser.error(
            "--require-ax-multi-metric-peer-wins requires --reuse-reference-root"
        )
    if args.no_load_gate:
        args.max_load_average = None
        args.max_top_process_cpu_percent = None
    if args.max_load_average is not None and args.max_load_average < 0.0:
        parser.error("--max-load-average must be non-negative")
    if (
        args.max_top_process_cpu_percent is not None
        and args.max_top_process_cpu_percent < 0.0
    ):
        parser.error("--max-top-process-cpu-percent must be non-negative")
    if (
        args.load_average_wait_timeout is not None
        and args.load_average_wait_timeout < 0.0
    ):
        parser.error("--load-average-wait-timeout must be non-negative")
    if (
        args.load_average_poll_interval is not None
        and args.load_average_poll_interval <= 0.0
    ):
        parser.error("--load-average-poll-interval must be positive")
    if args.warmup_repetitions < 0:
        parser.error("--warmup-repetitions must be non-negative")

    try:
        manifest = json.loads(args.manifest.read_text())
        raw_rows = manifest.get("rows")
        if not isinstance(raw_rows, list):
            raise AxOnlySweepError("manifest.rows must be an array")
        rows, args.sweep_scope = select_sweep_rows(
            raw_rows,
            args.rows_filter,
            require_ax_multi_metric_peer_wins=(
                args.require_ax_multi_metric_peer_wins
            ),
        )
        readme_rows = readme_manifest_rows(filter_manifest_rows(raw_rows, None))
        args.readme_slugs = [_row_slug(row) for row in readme_rows]
        args.peer_unavailable_readme_rows = [
            {
                "slug": _row_slug(row),
                "readme_model": row.get("readme_model"),
                "readme_quant": row.get("readme_quant"),
                "reason": row.get("prompt_source_note"),
            }
            for row in readme_rows
            if row.get("mlx_lm_peer_required") is False
        ]
        parse_prompt_token_csv(args.prompt_tokens)
    except AxOnlySweepError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    started = time.time()
    performance_conditions_start = collect_performance_condition_metadata()
    handlers = install_interrupt_handlers()
    planned_slugs = [_row_slug(row) for row in rows]

    try:
        model_snapshot_reference_root = (
            args.model_snapshot_reference_root or args.reuse_reference_root
        )
        for index, row in enumerate(rows, start=1):
            slug = row["slug"]
            log(f"({index}/{len(rows)}) {slug}")
            record: dict[str, Any] = {
                "slug": slug,
                "readme_model": row["readme_model"],
                "readme_quant": row["readme_quant"],
                "status": "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            summary_rows.append(record)
            write_sweep_outputs(
                args=args,
                summary_rows=summary_rows,
                started=started,
                planned_slugs=planned_slugs,
                performance_conditions_start=performance_conditions_start,
            )
            reference_artifact = (
                model_snapshot_reference_root / f"{slug}.json"
                if model_snapshot_reference_root is not None
                else None
            )
            model_args, err = resolve_model_args(
                row,
                args.cache_dir,
                reference_artifact=reference_artifact,
            )
            if model_args is None:
                record["status"] = "model_dir_missing"
                record["note"] = err
                log(f"  -> skip: {err}")
                write_sweep_outputs(
                    args=args,
                    summary_rows=summary_rows,
                    started=started,
                    planned_slugs=planned_slugs,
                    performance_conditions_start=performance_conditions_start,
                )
                continue
            result = run_row(
                row,
                output_dir=args.output_root,
                bench_script=args.bench_script,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                repetitions=args.repetitions,
                warmup_repetitions=args.warmup_repetitions,
                cooldown=args.cooldown,
                model_args=model_args,
                reuse_ref_root=args.reuse_reference_root,
                ax_direct_only=args.ax_direct_only,
                require_ax_multi_metric_peer_wins=(
                    args.require_ax_multi_metric_peer_wins
                ),
                max_load_average=args.max_load_average,
                max_top_process_cpu_percent=args.max_top_process_cpu_percent,
                load_average_wait_timeout=args.load_average_wait_timeout,
                load_average_poll_interval=args.load_average_poll_interval,
            )
            record.update(result)
            log(f"  -> {record.get('status')}")
            write_sweep_outputs(
                args=args,
                summary_rows=summary_rows,
                started=started,
                planned_slugs=planned_slugs,
                performance_conditions_start=performance_conditions_start,
            )
    except SweepInterrupted as error:
        for record in reversed(summary_rows):
            if record.get("status") == "running":
                record["status"] = "interrupted"
                record["note"] = str(error)
                break
        else:
            summary_rows.append(
                {
                    "slug": "sweep",
                    "readme_model": "",
                    "readme_quant": "",
                    "status": "interrupted",
                    "note": str(error),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
        write_sweep_outputs(
            args=args,
            summary_rows=summary_rows,
            started=started,
            planned_slugs=planned_slugs,
            performance_conditions_start=performance_conditions_start,
        )
    finally:
        restore_interrupt_handlers(handlers)

    sweep_doc = write_sweep_outputs(
        args=args,
        summary_rows=summary_rows,
        started=started,
        planned_slugs=planned_slugs,
        performance_conditions_start=performance_conditions_start,
    )
    fail_if_sweep_incomplete(summary_rows)
    if args.require_ax_multi_metric_peer_wins:
        fail_if_peer_win_matrix_not_publication_candidate(
            sweep_doc["peer_win_matrix"]
        )


if __name__ == "__main__":
    main()
