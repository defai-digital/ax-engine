#!/usr/bin/env python3
"""Greedy text parity: AX Engine muse_glimmer route vs mlxcel reference.

Runs the same checkpoint through `ax-engine-server` (`/v1/generate`, greedy)
and `mlxcel generate --no-chat-template --temp 0`, then diffs the generated
token ids / text. The prompt is tokenized with the checkpoint's own
`tokenizer.json` with `add_special_tokens=True`, matching mlxcel's
`tokenize_prompt` rule for raw prompts.

mlxcel's CLI suppresses image/video/pad token ids with a -inf bias; a text
prompt's greedy continuation should never hit them, and the check fails
loudly if the reference stream contains one (the comparison would then be
suppression-sensitive and needs a suppression-aware AX lane instead).

Example:
  python3 scripts/check_muse_glimmer_parity.py \
    --model-dir /path/to/Muse-Glimmer-30B-4bit \
    --mlxcel-bin .internal/reference/mlxcel/target/release/mlxcel \
    --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "target/release/ax-engine-server"

MUSE_SUPPRESSED_IDS = {200_091, 200_092, 200_018}

DEFAULT_PROMPT = (
    "The three most important properties of a good benchmark are "
    "reproducibility, isolation, and honesty. Reproducibility means"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tokenize(model_dir: Path, prompt: str) -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return list(tok.encode(prompt, add_special_tokens=True).ids)


def detokenize(model_dir: Path, ids: list[int]) -> str:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return tok.decode(ids, skip_special_tokens=False)


def extract_mlxcel_generated_text(output: str) -> str:
    """Extract the decoded prompt+continuation from mlxcel CLI output."""
    normalized = output.replace("\r\n", "\n").replace("\r", "\n")
    marker = "Generating...\n"
    marker_at = normalized.find(marker)
    if marker_at < 0:
        raise RuntimeError("mlxcel output is missing the Generating... marker")
    generated = normalized[marker_at + len(marker) :]
    stats_at = generated.rfind("\n[Generated ")
    if stats_at < 0:
        raise RuntimeError("mlxcel output is missing the generated-token summary")
    generated = generated[:stats_at].rstrip("\n")
    if not generated:
        raise RuntimeError("mlxcel output contains no generated text")
    return generated


def reference_completion_ids(
    model_dir: Path,
    output: str,
    prompt_ids: list[int],
) -> list[int]:
    """Re-encode mlxcel's decoded stream and remove the known prompt prefix."""
    generated_text = extract_mlxcel_generated_text(output)
    generated_ids = tokenize(model_dir, generated_text)
    if generated_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "mlxcel decoded stream does not begin with the expected prompt token ids"
        )
    return generated_ids[len(prompt_ids) :]


def common_prefix_len(left: list[int], right: list[int]) -> int:
    """Return the ordered token-prefix length shared by two continuations."""
    return next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


def run_ax(model_dir: Path, prompt_ids: list[int], max_tokens: int) -> list[int]:
    port = free_port()
    cmd = [
        str(SERVER),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--model-id",
        "muse-parity",
        "--port",
        str(port),
        "--disable-ngram-acceleration",
        "--prefill-chunk",
        "2048",
        "--max-batch-tokens",
        "2048",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = (proc.stderr.read() or b"").decode(errors="replace")
                raise RuntimeError(f"ax-engine-server exited early:\n{stderr[-4000:]}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("ax-engine-server did not become ready in 600s")

        body = json.dumps(
            {
                "input_tokens": prompt_ids,
                "max_output_tokens": max_tokens,
                "sampling": {"seed": 0, "ignore_eos": False},
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.loads(resp.read())
        tokens = payload.get("output_tokens") or payload.get("tokens")
        if tokens is None:
            raise RuntimeError(f"unexpected /v1/generate payload keys: {list(payload)}")
        return [int(t) for t in tokens]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_mlxcel(
    mlxcel_bin: Path, model_dir: Path, prompt: str, max_tokens: int
) -> str:
    cmd = [
        str(mlxcel_bin),
        "generate",
        "-m",
        str(model_dir),
        "-p",
        prompt,
        "--no-chat-template",
        "--temp",
        "0",
        "--max-tokens",
        str(max_tokens),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"mlxcel generate failed ({result.returncode}):\n{result.stderr[-4000:]}"
        )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mlxcel-bin", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--min-match-tokens",
        type=int,
        default=48,
        help="Minimum shared greedy prefix length to PASS (near-tie argmax "
        "flips between engines' dtype policies can diverge late streams)",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.min_match_tokens <= 0 or args.min_match_tokens > args.max_tokens:
        parser.error("--min-match-tokens must be in 1..=--max-tokens")

    prompt_ids = tokenize(args.model_dir, args.prompt)
    print(f"prompt tokens: {len(prompt_ids)} first={prompt_ids[:8]}")

    ax_ids = run_ax(args.model_dir, prompt_ids, args.max_tokens)
    ax_text = detokenize(args.model_dir, ax_ids)
    print(f"AX tokens ({len(ax_ids)}): {ax_ids[:24]}")
    print(f"AX text: {ax_text[:400]!r}")

    mlxcel_out = run_mlxcel(args.mlxcel_bin, args.model_dir, args.prompt, args.max_tokens)
    print(f"mlxcel raw output tail:\n{mlxcel_out[-1200:]}")

    mlxcel_ids = reference_completion_ids(args.model_dir, mlxcel_out, prompt_ids)
    print(f"mlxcel tokens ({len(mlxcel_ids)}): {mlxcel_ids[:24]}")

    ax_suppressed = MUSE_SUPPRESSED_IDS.intersection(ax_ids)
    reference_suppressed = MUSE_SUPPRESSED_IDS.intersection(mlxcel_ids)
    if ax_suppressed or reference_suppressed:
        print(
            "FAIL: comparison is suppression-sensitive; "
            f"AX suppressed ids={sorted(ax_suppressed)}, "
            f"mlxcel suppressed ids={sorted(reference_suppressed)}"
        )
        return 1

    matched = common_prefix_len(ax_ids, mlxcel_ids)
    print(
        f"ordered greedy prefix: {matched} token(s) "
        f"(required {args.min_match_tokens})"
    )
    if matched >= args.min_match_tokens:
        print("PASS: ordered greedy token prefix matches")
        return 0
    print("FAIL: greedy token streams diverge before the required prefix")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
