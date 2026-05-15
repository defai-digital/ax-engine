#!/usr/bin/env python3
"""F3 M4 — cross-restart disk prefix-cache equivalence harness.

Bridges the M2/M3 implementation against the PRD §8.2 acceptance:

    Process A: serve prompt P, exit
    Process B: serve prompt P (or P + suffix) — disk_hits must fire,
               and B's output tokens must match A's byte-for-byte.

The harness has two modes:

  --phase orchestrate  Default. Launches two subprocesses (phase A,
                       phase B) of this very script with
                       `--phase run-once`, then compares their
                       captured JSON. Each phase loads the model
                       freshly so the L1 in-memory cache cannot
                       satisfy a hit and the disk layer is exercised
                       end-to-end.

  --phase run-once     Worker mode. Loads the model, runs each
                       corpus prompt once, writes a JSON summary
                       (tokens + per-prompt telemetry) to the path
                       given by `--phase-artifact`.

Exit codes (ax-engine-bench convention):

  0  cross-restart equivalence held + at least one disk hit on phase B
  3  any prompt's tokens diverged (correctness)
  4  no disk hit observed on phase B (cache wire-up regressed)

Artifact schema: ax.disk_prefix_cache_cross_restart.v1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.disk_prefix_cache_cross_restart.v1"

# Block alignment for the disk-cache store path: M2 §2.5 stores only
# the largest valid prefix where prefix_len == full_block_tokens, so
# every prompt must round up to a multiple of block_size_tokens (16).
DEFAULT_BLOCK_SIZE = 16

# Telemetry surfaced by the runner. The disk_* keys land in M2/M3.
DISK_TELEMETRY_KEYS = [
    "ax_mlx_prefix_cache_hits",
    "ax_mlx_prefix_cache_misses",
    "ax_mlx_prefix_cache_stores",
    "ax_mlx_prefix_cache_entries",
    "ax_mlx_prefix_cache_disk_hits",
    "ax_mlx_prefix_cache_disk_misses",
    "ax_mlx_prefix_cache_disk_inserts",
    "ax_mlx_prefix_cache_disk_insert_bytes_kib",
    "ax_mlx_prefix_cache_disk_evictions",
]

DEFAULT_CORPUS = [
    {
        "id": "p1_short_factoid",
        "text": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "id": "p2_medium_explain",
        "text": (
            "Explain the difference between supervised and unsupervised learning, "
            "and give one concrete example for each. Keep the answer under 200 "
            "words and avoid bullet points."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--mlx-artifacts-dir", required=True, type=Path)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Disk-cache directory shared between phase A and phase B. "
            "Defaults to a fresh per-run tempdir under $TMPDIR."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/disk-prefix-cache-cross-restart"),
        help="Output JSON path or directory.",
    )
    p.add_argument("--max-output-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional JSON list of {id, text}. Defaults to a 2-prompt corpus.",
    )
    p.add_argument(
        "--pad-to-block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            "Right-pads each tokenized prompt up to a multiple of this. "
            "M2 only stores block-aligned snapshots, so the test corpus "
            "must be padded for the disk layer to fire."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["orchestrate", "run-once"],
        default="orchestrate",
    )
    p.add_argument(
        "--phase-artifact",
        type=Path,
        default=None,
        help="Internal: where the run-once worker writes its phase JSON.",
    )
    p.add_argument(
        "--phase-label",
        type=str,
        default=None,
        help="Internal: label embedded in the run-once worker output (A or B).",
    )
    return p.parse_args()


def load_corpus(args: argparse.Namespace) -> list[dict]:
    if args.corpus is None:
        return DEFAULT_CORPUS
    text = args.corpus.read_text()
    items = json.loads(text)
    if not isinstance(items, list):
        raise SystemExit("--corpus must point to a JSON list of {id, text}")
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise SystemExit("each --corpus item must be {id: str, text: str}")
    return items


def pad_to_block(tokens: list[int], block: int) -> list[int]:
    if block <= 0 or not tokens:
        return tokens
    remainder = len(tokens) % block
    if remainder == 0:
        return tokens
    need = block - remainder
    padding = (tokens * ((need // len(tokens)) + 1))[:need]
    return tokens + padding


def run_generate(session, tokens: list[int], max_output_tokens: int, seed: int) -> dict:
    output_tokens: list[int] = []
    crossover: dict[str, int] = {}
    for event in session.stream_generate(
        input_tokens=tokens,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        seed=seed,
        deterministic=True,
    ):
        if event.delta_tokens:
            output_tokens.extend(event.delta_tokens)
        if event.event == "response" and event.response is not None:
            if event.response.route and event.response.route.crossover_decisions:
                crossover = dict(event.response.route.crossover_decisions)
    return {
        "output_tokens": output_tokens,
        "telemetry": {k: crossover.get(k, 0) for k in DISK_TELEMETRY_KEYS},
    }


def run_once(args: argparse.Namespace) -> int:
    """Worker mode: load model, run each corpus prompt once, write JSON."""
    if args.phase_artifact is None:
        raise SystemExit("--phase-artifact is required in run-once mode")
    try:
        from ax_engine import Session
    except ImportError as e:
        raise SystemExit("ax_engine not importable; run `maturin develop` first") from e
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit("tokenizers not installed") from e

    if not args.mlx_artifacts_dir.is_dir():
        raise SystemExit(f"--mlx-artifacts-dir not found: {args.mlx_artifacts_dir}")
    tok_path = args.mlx_artifacts_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"tokenizer.json not found at {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    corpus = load_corpus(args)

    session = Session(
        model_id=args.model_id,
        mlx=True,
        mlx_model_artifacts_dir=str(args.mlx_artifacts_dir),
        deterministic=True,
    )
    # Allocator/code-cache warmup, kept short so it does not seed a
    # disk entry that the corpus comparison would have to filter out.
    _ = session.generate(
        input_tokens=tokenizer.encode("hi").ids,
        max_output_tokens=2,
        temperature=0.0,
        seed=args.seed,
    )

    per_prompt: list[dict] = []
    for item in corpus:
        prompt_id = item["id"]
        tokens = pad_to_block(
            tokenizer.encode(item["text"]).ids,
            args.pad_to_block_size,
        )
        if not tokens:
            raise SystemExit(f"prompt '{prompt_id}' tokenized to empty list")
        result = run_generate(session, tokens, args.max_output_tokens, args.seed)
        per_prompt.append(
            {
                "id": prompt_id,
                "prompt_preview": item["text"][:80],
                "prompt_token_count": len(tokens),
                "output_token_count": len(result["output_tokens"]),
                "output_tokens": result["output_tokens"],
                "telemetry": result["telemetry"],
            }
        )

    artifact = {
        "phase_label": args.phase_label or "?",
        "pid": os.getpid(),
        "per_prompt": per_prompt,
    }
    args.phase_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.phase_artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return 0


def list_axkv_files(cache_dir: Path) -> list[dict]:
    out: list[dict] = []
    if not cache_dir.is_dir():
        return out
    for entry in sorted(cache_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".axkv":
            stat = entry.stat()
            out.append(
                {
                    "name": entry.name,
                    "size_bytes": stat.st_size,
                    "mtime_unix": int(stat.st_mtime),
                }
            )
    return out


def spawn_phase(
    args: argparse.Namespace,
    phase_label: str,
    phase_artifact: Path,
    cache_dir: Path,
) -> subprocess.CompletedProcess:
    """Launch this same script as a fresh subprocess in run-once mode."""
    env = os.environ.copy()
    env["AX_MLX_PREFIX_CACHE_DIR"] = str(cache_dir)
    # Make sure the L2 path is not accidentally killed by a stale flag.
    env.pop("AX_MLX_PREFIX_CACHE_DISK_DISABLED", None)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "run-once",
        "--phase-label",
        phase_label,
        "--phase-artifact",
        str(phase_artifact),
        "--model-id",
        args.model_id,
        "--mlx-artifacts-dir",
        str(args.mlx_artifacts_dir),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--seed",
        str(args.seed),
        "--pad-to-block-size",
        str(args.pad_to_block_size),
    ]
    if args.corpus is not None:
        cmd += ["--corpus", str(args.corpus)]
    return subprocess.run(cmd, env=env, check=False)


def orchestrate(args: argparse.Namespace) -> tuple[dict, int]:
    """Drive phase A → exit → phase B, then diff their JSON outputs."""
    own_tempdir: tempfile.TemporaryDirectory | None = None
    if args.cache_dir is None:
        own_tempdir = tempfile.TemporaryDirectory(prefix="ax-disk-m4-")
        cache_dir = Path(own_tempdir.name)
    else:
        cache_dir = args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"cross-restart cache dir: {cache_dir}")
    workdir = Path(tempfile.mkdtemp(prefix="ax-disk-m4-work-"))
    phase_a_artifact = workdir / "phase_a.json"
    phase_b_artifact = workdir / "phase_b.json"

    print("=== phase A (cold; populates disk cache) ===")
    rc_a = spawn_phase(args, "A", phase_a_artifact, cache_dir).returncode
    if rc_a != 0:
        raise SystemExit(f"phase A failed with exit code {rc_a}")
    files_after_a = list_axkv_files(cache_dir)
    print(f"phase A wrote {len(files_after_a)} .axkv file(s) to {cache_dir}")

    if not files_after_a:
        raise SystemExit(
            "phase A produced no .axkv files — the store path did not "
            "fire. Check pad_to_block_size and model architecture; "
            "non-block-aligned prompts are not persisted."
        )

    print("=== phase B (fresh process; must hit L2) ===")
    rc_b = spawn_phase(args, "B", phase_b_artifact, cache_dir).returncode
    if rc_b != 0:
        raise SystemExit(f"phase B failed with exit code {rc_b}")
    files_after_b = list_axkv_files(cache_dir)

    artifact_a = json.loads(phase_a_artifact.read_text())
    artifact_b = json.loads(phase_b_artifact.read_text())

    by_id_a = {p["id"]: p for p in artifact_a["per_prompt"]}
    by_id_b = {p["id"]: p for p in artifact_b["per_prompt"]}

    per_prompt: list[dict] = []
    correctness_failures = 0
    total_disk_hits_b = 0
    for prompt_id, a in by_id_a.items():
        b = by_id_b.get(prompt_id)
        if b is None:
            correctness_failures += 1
            per_prompt.append(
                {
                    "id": prompt_id,
                    "tokens_match": False,
                    "reason": "missing in phase B",
                }
            )
            continue
        tokens_match = a["output_tokens"] == b["output_tokens"]
        first_div: int | None = None
        if not tokens_match:
            n = min(len(a["output_tokens"]), len(b["output_tokens"]))
            first_div = next(
                (i for i in range(n) if a["output_tokens"][i] != b["output_tokens"][i]),
                n,
            )
            correctness_failures += 1
        disk_hits_b = b["telemetry"].get("ax_mlx_prefix_cache_disk_hits", 0)
        total_disk_hits_b += disk_hits_b
        per_prompt.append(
            {
                "id": prompt_id,
                "tokens_match": tokens_match,
                "first_divergence_index": first_div,
                "prompt_token_count": a["prompt_token_count"],
                "output_token_count_a": a["output_token_count"],
                "output_token_count_b": b["output_token_count"],
                "telemetry_a": a["telemetry"],
                "telemetry_b": b["telemetry"],
                "disk_hits_b": disk_hits_b,
            }
        )

    verdict = "PASS"
    exit_code = 0
    if correctness_failures > 0:
        verdict = "FAIL_CORRECTNESS"
        exit_code = 3
    elif total_disk_hits_b == 0:
        verdict = "FAIL_NO_DISK_HIT"
        exit_code = 4

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.platform(),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
        },
        "config": {
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "pad_to_block_size": args.pad_to_block_size,
            "prompt_count": len(by_id_a),
            "corpus_source": "default" if args.corpus is None else str(args.corpus),
            "cache_dir": str(cache_dir),
        },
        "disk_state": {
            "files_after_phase_a": files_after_a,
            "files_after_phase_b": files_after_b,
        },
        "aggregate": {
            "prompts_total": len(by_id_a),
            "correctness_failures": correctness_failures,
            "total_disk_hits_phase_b": total_disk_hits_b,
            "verdict": verdict,
        },
        "per_prompt": per_prompt,
    }

    if own_tempdir is not None:
        own_tempdir.cleanup()
    return artifact, exit_code


def main() -> int:
    args = parse_args()
    if args.phase == "run-once":
        return run_once(args)
    artifact, exit_code = orchestrate(args)
    out_path = args.output
    if out_path.is_dir() or (not out_path.exists() and out_path.suffix != ".json"):
        out_path.mkdir(parents=True, exist_ok=True)
        safe_id = args.model_id.replace("/", "_").replace(" ", "_")
        date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_path / f"{safe_id}-{date_part}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    a = artifact["aggregate"]
    print(
        f"verdict: {a['verdict']} "
        f"({a['prompts_total'] - a['correctness_failures']}/{a['prompts_total']} "
        f"correctness; {a['total_disk_hits_phase_b']} disk_hits in phase B)"
    )
    if exit_code != 0:
        print("FAIL — see per_prompt for details.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
