#!/usr/bin/env python3
"""Diff AX text-only first-token logits against an ``mlx_lm`` reference.

This is the numerical companion to ``scripts/probe_mlx_model_support.py`` (which
only checks the support *contract* statically). It answers the load-bearing
question for VL / hybrid-attention Qwen3.5-MoE checkpoints such as
``Qwen/Qwen-AgentWorld-35B-A3B``: does AX's decode graph — which does **not**
implement multimodal M-RoPE (``mrope_section`` / ``mrope_interleaved``) — produce
the same logits as the reference for *text-only* input, where M-RoPE collapses to
ordinary 1-D RoPE?

How it works
------------
1. Reference: load the model with ``mlx_lm`` and forward the *same* token ids,
   text-only (no pixel values). ``mlx_lm.models.qwen3_5_moe.sanitize`` already
   drops the vision tower and remaps ``language_model.*``, so this exercises the
   exact text decoder AX's ``qwen3_5`` family mirrors. Take the last-position
   logits.
2. AX: run the ``logit_dump_probe`` binary with ``--dump=PATH`` to write AX's
   full last-position logit vector (raw little-endian f32) for the same ids.
3. Compare: top-1 / top-k agreement, log-softmax deltas, cosine, KL. A real
   RoPE/attention bug diverges hard on the *first* token; benign quant drift
   stays tiny. Run both sides on the **same** (ideally bf16, unquantized)
   artifact to keep quant noise out of the signal.

Usage
-----
    python scripts/probe_qwen3_5_logit_parity.py \
        --ax-model-dir  /path/to/ax-native-dir \
        --ref-model-dir /path/to/mlx-or-hf-dir \
        --token-ids "9707,11,358,1079" \
        --build

``--ref-model-dir`` defaults to ``--ax-model-dir``. Pass ``--prompt TEXT``
instead of ``--token-ids`` to tokenize with the reference tokenizer. Exits
non-zero if parity fails the thresholds (``--max-logprob-delta``, ``--max-kl``,
and top-1 agreement).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ax.qwen3_5_logit_parity.v1"


def _log_softmax(logits: "Any") -> "Any":
    import numpy as np

    m = logits.max()
    shifted = logits - m
    lse = m + math.log(float(np.exp(shifted).sum()))
    return logits - lse


def _reference_logits(ref_dir: Path, ids: list[int]) -> "Any":
    """Last-position logits [vocab] from the mlx_lm text decoder, as float32."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm.utils import load_model

    # load_model applies quantization + the family's sanitize() (which drops the
    # vision tower) and returns the bare model — no tokenizer needed for ids.
    try:
        model, _config = load_model(ref_dir)
    except Exception:
        # Vision checkpoints can carry weights the text model doesn't bind;
        # retry tolerant so leftover keys don't abort the reference forward.
        model, _config = load_model(ref_dir, strict=False)
    model.eval()

    out = model(mx.array([ids], dtype=mx.uint32))  # [1, L, vocab]
    last = out[0, -1].astype(mx.float32)
    mx.eval(last)
    return np.array(last, dtype=np.float32)


def _tokenize(ref_dir: Path, prompt: str) -> list[int]:
    from mlx_lm import load

    _model, tokenizer = load(str(ref_dir))
    return list(tokenizer.encode(prompt))


def _ax_logits(probe_bin: Path, ax_dir: Path, ids: list[int]) -> "Any":
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        dump_path = Path(tmp.name)
    id_arg = ",".join(str(i) for i in ids)
    cmd = [str(probe_bin), str(ax_dir), id_arg, "1", f"--dump={dump_path}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"logit_dump_probe failed (exit {proc.returncode})")
    raw = dump_path.read_bytes()
    dump_path.unlink(missing_ok=True)
    return np.frombuffer(raw, dtype="<f4").copy()


def _compare(ref: "Any", ax: "Any", topk: int) -> dict[str, Any]:
    import numpy as np

    report: dict[str, Any] = {}
    report["ref_vocab"] = int(ref.shape[0])
    report["ax_vocab"] = int(ax.shape[0])
    if ref.shape[0] != ax.shape[0]:
        report["fatal"] = "vocab size mismatch — cannot compare"
        return report

    ref_lp = _log_softmax(ref)
    ax_lp = _log_softmax(ax)

    ref_order = np.argsort(-ref)
    ax_order = np.argsort(-ax)
    ref_top = ref_order[:topk]
    ax_top = ax_order[:topk]

    report["ref_argmax"] = int(ref_order[0])
    report["ax_argmax"] = int(ax_order[0])
    report["top1_match"] = bool(ref_order[0] == ax_order[0])
    report["topk"] = topk
    report["topk_overlap"] = int(len(set(ref_top.tolist()) & set(ax_top.tolist())))
    # Where AX's argmax ranks in the reference ordering (0 == perfect).
    report["ax_argmax_rank_in_ref"] = int(np.where(ref_order == ax_order[0])[0][0])

    # Log-prob deltas restricted to the union of both top-k (the tokens that
    # actually matter for sampling); absolute logit offset is irrelevant.
    union = sorted(set(ref_top.tolist()) | set(ax_top.tolist()))
    deltas = np.abs(ref_lp[union] - ax_lp[union])
    report["max_logprob_delta_topk"] = float(deltas.max())
    report["mean_logprob_delta_topk"] = float(deltas.mean())
    report["max_abs_logit_delta"] = float(np.abs(ref - ax).max())

    cos = float(
        np.dot(ref, ax) / (np.linalg.norm(ref) * np.linalg.norm(ax) + 1e-12)
    )
    report["cosine"] = cos

    # KL(P_ref || P_ax) over the full vocab.
    p = np.exp(ref_lp)
    kl = float(np.sum(p * (ref_lp - ax_lp)))
    report["kl_ref_to_ax"] = kl
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ax-model-dir", required=True, type=Path)
    parser.add_argument("--ref-model-dir", type=Path, default=None)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--token-ids", help="comma/space separated token ids")
    g.add_argument("--prompt", help="text to tokenize with the reference tokenizer")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--probe-bin",
        type=Path,
        default=REPO_ROOT / "target/release/logit_dump_probe",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="cargo build --release --bin logit_dump_probe first",
    )
    parser.add_argument("--max-logprob-delta", type=float, default=0.5)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument("--out", type=Path, help="write JSON report to this path")
    args = parser.parse_args()

    ref_dir = args.ref_model_dir or args.ax_model_dir

    if args.build:
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "logit_dump_probe"],
            cwd=REPO_ROOT,
            check=True,
        )
    if not args.probe_bin.exists():
        raise SystemExit(
            f"probe binary not found: {args.probe_bin} (pass --build or --probe-bin)"
        )

    if args.token_ids:
        ids = [
            int(t)
            for t in args.token_ids.replace(",", " ").split()
            if t.strip()
        ]
    else:
        ids = _tokenize(ref_dir, args.prompt)
    if not ids:
        raise SystemExit("no token ids")

    ref = _reference_logits(ref_dir, ids)
    ax = _ax_logits(args.probe_bin, args.ax_model_dir, ids)
    metrics = _compare(ref, ax, args.topk)

    passed = (
        "fatal" not in metrics
        and metrics["top1_match"]
        and metrics["max_logprob_delta_topk"] <= args.max_logprob_delta
        and metrics["kl_ref_to_ax"] <= args.max_kl
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "ax_model_dir": str(args.ax_model_dir),
        "ref_model_dir": str(ref_dir),
        "num_prompt_tokens": len(ids),
        "token_ids": ids,
        "thresholds": {
            "max_logprob_delta": args.max_logprob_delta,
            "max_kl": args.max_kl,
        },
        "metrics": metrics,
        "verdict": "parity" if passed else "divergent",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
