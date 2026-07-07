#!/usr/bin/env python3
"""Run AX direct + n-gram benchmarks across the 12 README rows, reusing the
mlx_lm baseline from a previous combined sweep so AX % deltas are computed
against fresh mlx_lm numbers without re-running mlx_lm.

Per row this invokes scripts/bench_mlx_inference_stack.py with
  --ax-compare-policies
  --reuse-reference-results-from <prev mlx_lm JSON>
  --no-build-ax-engine  (uses existing target/release/ax-engine-server)
and skips both mlx_lm and llama.cpp (the former via --reuse, the latter by
not passing --llama-cpp-bench).

Writes per-row JSON + sweep_results.json + sweep_summary.md.
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


def resolve_model_args(row: dict[str, Any], cache_dir: Path) -> tuple[list[str] | None, str | None]:
    local_dir_value = row.get("mlx_local_dir")
    if local_dir_value:
        model_dir = REPO_ROOT / local_dir_value
        if model_dir.exists():
            return ["--model-dir", str(model_dir)], None
    repo_id = row.get("mlx_repo_id")
    if not repo_id:
        return None, f"No mlx_local_dir or mlx_repo_id for {row.get('slug')}"
    snap = latest_hf_cache_snapshot(repo_id, cache_dir)
    if snap is None:
        return None, f"No HF cache snapshot for {repo_id}"
    # AX requires config.json + model-manifest.json + *.safetensors in the snapshot.
    missing = []
    if not (snap / "config.json").is_file():
        missing.append("config.json")
    if not (snap / "model-manifest.json").is_file():
        missing.append("model-manifest.json")
    if not any(snap.glob("*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        return None, f"MLX cache snapshot for {repo_id} missing {', '.join(missing)}"
    return ["--model-repo-id", repo_id, "--hf-cache-root", str(cache_dir)], None


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
    cooldown: float,
    model_args: list[str],
    reuse_ref_root: Path,
    max_load_average: float | None = None,
    max_top_process_cpu_percent: float | None = None,
    load_average_wait_timeout: float | None = None,
    load_average_poll_interval: float | None = None,
) -> dict[str, Any]:
    slug = row["slug"]
    out_json = output_dir / f"{slug}.json"
    log_path = output_dir / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ref_json = reuse_ref_root / f"{slug}.json"
    if not ref_json.is_file():
        return {"status": "skipped_no_reference", "note": f"missing {ref_json}"}

    cmd = [
        sys.executable,
        str(bench_script),
        *model_args,
        "--prompt-tokens", prompt_tokens,
        "--generation-tokens", str(generation_tokens),
        "--repetitions", str(repetitions),
        "--cooldown", str(cooldown),
        "--ax-compare-policies",
        "--reuse-reference-results-from", str(ref_json),
        "--no-build-ax-engine",
        "--output", str(out_json),
    ]
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
    planned_row_count: int,
    performance_conditions_start: dict[str, Any],
) -> dict[str, Any]:
    elapsed = time.time() - started
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    performance_conditions_end = collect_performance_condition_metadata()
    failed_rows = failed_sweep_rows(summary_rows)
    return {
        "schema_version": "ax.ax_only_sweep.v1",
        "manifest_path": str(args.manifest),
        "reuse_reference_root": str(args.reuse_reference_root),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
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
        "planned_row_count": planned_row_count,
        "completed_row_count": len(
            [
                row
                for row in summary_rows
                if row.get("status") not in {"running", "interrupted"}
            ]
        ),
        "publication_candidate": not failed_rows,
        "failed_row_count": len(failed_rows),
        "status_counts": status_counts(summary_rows),
        "rows": summary_rows,
    }


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
    planned_row_count: int,
    performance_conditions_start: dict[str, Any],
) -> dict[str, Any]:
    sweep_doc = build_sweep_doc(
        args=args,
        summary_rows=summary_rows,
        started=started,
        planned_row_count=planned_row_count,
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
        f"- publication_candidate: {str(sweep_doc['publication_candidate']).lower()}",
        f"- failed_row_count: {sweep_doc['failed_row_count']}",
        f"- status_counts: {status_counts_text(sweep_doc['status_counts'])}",
        f"- completed_row_count: {sweep_doc['completed_row_count']}/{planned_row_count}",
        f"- elapsed: {sweep_doc['elapsed_seconds']:.0f}s",
        "",
        "| slug | status | notes |",
        "|---|---|---|",
    ]
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
        required=True,
        help="Directory containing per-row JSONs (<slug>.json) with mlx_lm baseline.",
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

    try:
        manifest = json.loads(args.manifest.read_text())
        raw_rows = manifest.get("rows")
        if not isinstance(raw_rows, list):
            raise AxOnlySweepError("manifest.rows must be an array")
        rows = filter_manifest_rows(raw_rows, args.rows_filter)
    except AxOnlySweepError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    started = time.time()
    performance_conditions_start = collect_performance_condition_metadata()
    handlers = install_interrupt_handlers()

    try:
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
                planned_row_count=len(rows),
                performance_conditions_start=performance_conditions_start,
            )
            model_args, err = resolve_model_args(row, args.cache_dir)
            if model_args is None:
                record["status"] = "model_dir_missing"
                record["note"] = err
                log(f"  -> skip: {err}")
                write_sweep_outputs(
                    args=args,
                    summary_rows=summary_rows,
                    started=started,
                    planned_row_count=len(rows),
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
                cooldown=args.cooldown,
                model_args=model_args,
                reuse_ref_root=args.reuse_reference_root,
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
                planned_row_count=len(rows),
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
            planned_row_count=len(rows),
            performance_conditions_start=performance_conditions_start,
        )
    finally:
        restore_interrupt_handlers(handlers)

    fail_if_sweep_incomplete(summary_rows)


if __name__ == "__main__":
    main()
