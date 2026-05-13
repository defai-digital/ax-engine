#!/usr/bin/env python3
"""Compare ax-engine delegated output against mlx_lm.server direct output.

Phase A evidence helper for openai/gpt-oss-{20b,120b} (PRD:
.internal/planning/GPT-OSS-SUPPORT-PRD.md, W1.4 correctness gate).

Both ax-engine `/v1/completions` (when configured with
`--support-tier mlx_lm_delegated --mlx-lm-server-url <ref>`) and the same
`mlx_lm.server` instance expose OpenAI-compatible `/v1/completions`. When
the ax-engine route is a transparent forwarder, the two outputs should
be byte-identical at temperature=0. Any drift surfaces prompt-mangling,
sampling-config translation, or tokenization bugs in the delegated
forwarding path.

Output: one JSON file with per-prompt agreement, plus aggregate top-1
agreement ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PromptResult:
    index: int
    prompt_chars: int
    ref_text: str
    tested_text: str
    char_match: bool
    common_prefix_chars: int
    ref_latency_s: float
    tested_latency_s: float


def post_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, float]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    elapsed = time.monotonic() - started
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response from {base_url}: {payload!r}")
    text = choices[0].get("text", "")
    return text, elapsed


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ax-engine-url", required=True, help="ax-engine server base URL")
    parser.add_argument("--mlx-lm-server-url", required=True, help="mlx_lm.server base URL (reference)")
    parser.add_argument("--model", required=True, help="model id to send in the OpenAI request payload")
    parser.add_argument("--prompts", required=True, type=Path, help="JSONL with {\"prompt\": ...} per line")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="cap prompt count (0 = all)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.prompts.exists():
        print(f"prompts file not found: {args.prompts}", file=sys.stderr)
        return 2

    prompts: list[str] = []
    with args.prompts.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt")
            if not isinstance(p, str) or not p:
                continue
            prompts.append(p)
            if args.limit and len(prompts) >= args.limit:
                break

    if not prompts:
        print(f"no prompts loaded from {args.prompts}", file=sys.stderr)
        return 2

    results: list[PromptResult] = []
    for i, prompt in enumerate(prompts):
        try:
            ref_text, ref_lat = post_completion(args.mlx_lm_server_url, args.model, prompt, args.max_tokens)
            tested_text, tested_lat = post_completion(args.ax_engine_url, args.model, prompt, args.max_tokens)
        except (urllib.error.URLError, RuntimeError) as exc:
            print(f"prompt {i}: request failed: {exc}", file=sys.stderr)
            return 3

        prefix = common_prefix_len(ref_text, tested_text)
        results.append(
            PromptResult(
                index=i,
                prompt_chars=len(prompt),
                ref_text=ref_text,
                tested_text=tested_text,
                char_match=(ref_text == tested_text),
                common_prefix_chars=prefix,
                ref_latency_s=ref_lat,
                tested_latency_s=tested_lat,
            )
        )

    char_match_rate = sum(1 for r in results if r.char_match) / len(results)
    prefix_ratios = [
        (r.common_prefix_chars / max(1, len(r.ref_text))) for r in results if r.ref_text
    ]
    mean_prefix_ratio = statistics.mean(prefix_ratios) if prefix_ratios else 0.0

    summary = {
        "prompts_evaluated": len(results),
        "char_match_rate": char_match_rate,
        "mean_prefix_ratio": mean_prefix_ratio,
        "ref_latency_s_mean": statistics.mean(r.ref_latency_s for r in results),
        "tested_latency_s_mean": statistics.mean(r.tested_latency_s for r in results),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "ax_engine_url": args.ax_engine_url,
        "mlx_lm_server_url": args.mlx_lm_server_url,
        "prompts_file": str(args.prompts),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "per_prompt": [asdict(r) for r in results]},
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"char_match_rate={char_match_rate:.3f} mean_prefix_ratio={mean_prefix_ratio:.3f} n={len(results)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
