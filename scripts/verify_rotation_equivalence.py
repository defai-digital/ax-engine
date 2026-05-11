#!/usr/bin/env python3
"""Verify that `AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION` does not change inference.

P1+ regression harness. Runs greedy decode on multiple fixed prompts under
both baseline (env unset) and tested (env set) modes, then reports the
token-equivalence breakdown.

Verdict (P1 identity rotation): `shared_prefix_ratio ≥ 0.90` PASS, else FAIL.
Pure token-exact match is too strict because matmul-based identity rotation
on bf16 activations introduces fp noise that occasionally flips an argmax
on marginal-confidence positions. Observed natural noise floor on Qwen3.5-9B
with the orthonormal Hadamard + bf16 matmul path is ratio ≈ 0.95 across
the built-in 5-prompt corpus (4/5 prompts match exactly; one prompt
diverges at a marginal-confidence token).

For P2 (real rotation + sub-4-bit quant): the acceptance is RELATIVE to
P1's noise floor — P2 ratio must be within ~0.05 of P1's ratio on the
same corpus. Absolute ratio ≥ 0.90 is the floor.

Usage:

    python scripts/verify_rotation_equivalence.py \\
        --model-id qwen3_dense \\
        --mlx-artifacts-dir .internal/models/Qwen3.5-9B-MLX-4bit \\
        --mode enable

    # Custom corpus (JSON list of {"id": str, "text": str}):
    python scripts/verify_rotation_equivalence.py ... --corpus my_prompts.json

Artifact schema: ax.rotation_equivalence.v2

    {
      "schema_version": "ax.rotation_equivalence.v2",
      "generated_at_utc": "...",
      "model": { "model_id": str, "artifacts_dir": str },
      "config": {
        "tested_mode": "shadow" | "enable" | ...,
        "max_output_tokens": int,
        "seed": int,
        "prompt_count": int,
        "corpus_source": "default" | "<file path>"
      },
      "aggregate": {
        "prompts_total": int,
        "prompts_matching_exactly": int,
        "shared_prefix_total_tokens": int,
        "tested_total_tokens": int,
        "shared_prefix_ratio": float,
        "verdict": "PASS" | "FAIL"
      },
      "per_prompt": [
        {
          "id": str,
          "prompt_preview": str,
          "baseline_token_count": int,
          "tested_token_count": int,
          "tokens_match": bool,
          "first_divergence_index": int | null,
          "shared_prefix_len": int
        }
      ]
    }

Exit codes:
- 0: PASS — shared_prefix_ratio ≥ PASS_THRESHOLD (default 0.90)
- 1: FAIL — ratio below threshold
- 2: usage / setup error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.rotation_equivalence.v2"
PASS_THRESHOLD = 0.90

# Built-in corpus covers multiple regimes that exercise different decoder
# behaviours. Stable identifiers so artifacts compare cleanly across runs.
DEFAULT_CORPUS = [
    {
        "id": "english_continuation",
        "text": (
            "The quick brown fox jumps over the lazy dog while the engine "
            "warms its caches and the operator records throughput metrics."
        ),
    },
    {
        "id": "instruction_short",
        "text": "Write one sentence about the moon.",
    },
    {
        "id": "code_continuation",
        "text": (
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return"
        ),
    },
    {
        "id": "repetitive_list",
        "text": (
            "List five fruits separated by commas: apple, banana, cherry, "
        ),
    },
    {
        "id": "long_narrative",
        "text": (
            "Margaret turned the brass key with steady fingers and pushed "
            "open the warehouse door, which groaned on hinges that had not "
            "seen oil for at least a decade, releasing a long breath of "
            "stale air into the cold morning."
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
        "--mode",
        default="shadow",
        choices=["shadow", "enable", "apply"],
        help="Rotation mode to test under AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION. "
        "`apply` mode applies a SINGLE rotation R(x) and is mathematically "
        "incorrect unless paired with an offline-rotated checkpoint — use it "
        "with this harness to confirm the activation rotation has real effect "
        "(tokens MUST diverge from baseline when no rotated checkpoint loaded).",
    )
    p.add_argument(
        "--corpus",
        default=None,
        type=Path,
        help="Optional path to a JSON file containing a list of "
        '{"id": str, "text": str} entries. Default uses the built-in corpus.',
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/results/rotation-equivalence"),
    )
    p.add_argument("--max-output-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_corpus(path: Path | None) -> tuple[list[dict], str]:
    if path is None:
        return DEFAULT_CORPUS, "default"
    if not path.is_file():
        raise SystemExit(f"--corpus file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(
        isinstance(e, dict) and "id" in e and "text" in e for e in data
    ):
        raise SystemExit(f"--corpus must be a JSON list of {{id,text}} objects")
    return data, str(path)


def run_decode(
    args: argparse.Namespace, rotation_env: str | None, prompt_text: str
) -> list[int]:
    """Spawn a subprocess to run greedy decode under a clean env.

    Subprocess isolation avoids the rotation-mode OnceLock cache
    contaminating subsequent runs.
    """
    harness = """
import os, sys, json
from ax_engine import Session
from tokenizers import Tokenizer
art = os.environ["AX_ART_DIR"]
tok = Tokenizer.from_file(f"{art}/tokenizer.json")
prompt = os.environ["AX_PROMPT_TEXT"]
ids = tok.encode(prompt).ids
s = Session(model_id=os.environ["AX_MODEL_ID"], mlx=True,
            mlx_model_artifacts_dir=art, deterministic=True)
r = s.generate(input_tokens=ids,
               max_output_tokens=int(os.environ["AX_MAX_OUTPUT"]),
               temperature=0.0,
               seed=int(os.environ["AX_SEED"]),
               deterministic=True)
print(json.dumps({"output_tokens": list(r.output_tokens)}))
"""
    env = dict(os.environ)
    env["AX_ART_DIR"] = str(args.mlx_artifacts_dir)
    env["AX_MODEL_ID"] = args.model_id
    env["AX_PROMPT_TEXT"] = prompt_text
    env["AX_MAX_OUTPUT"] = str(args.max_output_tokens)
    env["AX_SEED"] = str(args.seed)
    if rotation_env is None:
        env.pop("AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION", None)
    else:
        env["AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION"] = rotation_env
    proc = subprocess.run(
        [sys.executable, "-c", harness],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Decode subprocess failed (rotation_env={rotation_env!r}):\n"
            f"stderr:\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(
            f"No JSON output from decode (rotation_env={rotation_env!r}):\n"
            f"stderr:\n{proc.stderr}"
        )
    return list(json.loads(lines[-1])["output_tokens"])


def compare_pair(baseline: list[int], tested: list[int]) -> dict:
    shared_prefix_len = 0
    for a, b in zip(baseline, tested):
        if a != b:
            break
        shared_prefix_len += 1
    tokens_match = baseline == tested
    return {
        "baseline_token_count": len(baseline),
        "tested_token_count": len(tested),
        "tokens_match": tokens_match,
        "first_divergence_index": None if tokens_match else shared_prefix_len,
        "shared_prefix_len": shared_prefix_len,
    }


def main() -> int:
    args = parse_args()
    if not args.mlx_artifacts_dir.is_dir():
        print(f"--mlx-artifacts-dir not found: {args.mlx_artifacts_dir}", file=sys.stderr)
        return 2

    corpus, corpus_source = load_corpus(args.corpus)
    print(f"corpus: {corpus_source} ({len(corpus)} prompts)")

    per_prompt = []
    shared_prefix_total = 0
    tested_total = 0
    prompts_matching = 0

    for entry in corpus:
        pid = entry["id"]
        text = entry["text"]
        print(f"\n[{pid}] running baseline ...")
        baseline = run_decode(args, rotation_env=None, prompt_text=text)
        print(f"  baseline n={len(baseline)} first 8={baseline[:8]}")
        print(f"[{pid}] running tested (mode={args.mode}) ...")
        tested = run_decode(args, rotation_env=args.mode, prompt_text=text)
        print(f"  tested   n={len(tested)} first 8={tested[:8]}")
        cmp = compare_pair(baseline, tested)
        cmp["id"] = pid
        cmp["prompt_preview"] = text[:60]
        per_prompt.append(cmp)
        if cmp["tokens_match"]:
            prompts_matching += 1
        shared_prefix_total += cmp["shared_prefix_len"]
        tested_total += cmp["tested_token_count"]
        status = "MATCH" if cmp["tokens_match"] else f"DIVERGE@{cmp['first_divergence_index']}"
        print(f"[{pid}] {status} (shared={cmp['shared_prefix_len']}/{cmp['tested_token_count']})")

    shared_prefix_ratio = (
        shared_prefix_total / tested_total if tested_total > 0 else 0.0
    )
    verdict = "PASS" if shared_prefix_ratio >= PASS_THRESHOLD else "FAIL"

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "model_id": args.model_id,
            "artifacts_dir": str(args.mlx_artifacts_dir),
        },
        "config": {
            "tested_mode": args.mode,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
            "prompt_count": len(corpus),
            "corpus_source": corpus_source,
        },
        "aggregate": {
            "prompts_total": len(corpus),
            "prompts_matching_exactly": prompts_matching,
            "shared_prefix_total_tokens": shared_prefix_total,
            "tested_total_tokens": tested_total,
            "shared_prefix_ratio": round(shared_prefix_ratio, 6),
            "pass_threshold": PASS_THRESHOLD,
            "verdict": verdict,
        },
        "per_prompt": per_prompt,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    safe_id = args.model_id.replace("/", "_").replace(" ", "_")
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.output_root / f"{safe_id}-{args.mode}-{date_part}.json"
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    print(
        f"verdict: {verdict}  ({prompts_matching}/{len(corpus)} match exactly, "
        f"shared_prefix_ratio={shared_prefix_ratio:.4f})"
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
