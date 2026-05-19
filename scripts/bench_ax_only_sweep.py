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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "manifests" / "llama_cpp_metal" / "inventory.json"
DEFAULT_BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_mlx_inference_stack.py"


def log(msg: str) -> None:
    print(f"[ax-sweep] {msg}", flush=True)


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
    log(f"  invoke: {' '.join(cmd)}")
    with log_path.open("w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {
            "status": "bench_failed",
            "exit_code": result.returncode,
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=15.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    if args.rows_filter:
        rows = [r for r in rows if r["slug"] in set(args.rows_filter)]

    summary_rows: list[dict[str, Any]] = []
    started = time.time()

    for index, row in enumerate(rows, start=1):
        slug = row["slug"]
        log(f"({index}/{len(rows)}) {slug}")
        record: dict[str, Any] = {
            "slug": slug,
            "readme_model": row["readme_model"],
            "readme_quant": row["readme_quant"],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        model_args, err = resolve_model_args(row, args.cache_dir)
        if model_args is None:
            record["status"] = "model_dir_missing"
            record["note"] = err
            summary_rows.append(record)
            log(f"  -> skip: {err}")
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
        )
        record.update(result)
        summary_rows.append(record)
        log(f"  -> {record.get('status')}")

    elapsed = time.time() - started
    sweep_doc = {
        "schema_version": "ax.ax_only_sweep.v1",
        "manifest_path": str(args.manifest),
        "reuse_reference_root": str(args.reuse_reference_root),
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "repetitions": args.repetitions,
        "cooldown": args.cooldown,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_seconds": round(elapsed, 1),
        "rows": summary_rows,
    }
    (args.output_root / "sweep_results.json").write_text(json.dumps(sweep_doc, indent=2))
    log(f"wrote {args.output_root / 'sweep_results.json'}")

    md = ["# AX-only sweep summary", "", f"- elapsed: {elapsed:.0f}s", "",
          "| slug | status | notes |", "|---|---|---|"]
    for r in summary_rows:
        md.append(f"| {r['slug']} | {r.get('status','?')} | {r.get('note') or r.get('error') or ''} |")
    (args.output_root / "sweep_summary.md").write_text("\n".join(md) + "\n")
    log(f"wrote {args.output_root / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
