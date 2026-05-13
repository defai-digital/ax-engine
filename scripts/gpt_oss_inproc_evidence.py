#!/usr/bin/env python3
"""In-process Phase A evidence harness for openai/gpt-oss-{20b,120b}.

Phase A baseline: PRD GPT-OSS-SUPPORT-PRD.md W1.4. Because mlx_lm.server
crashes on every generation under Python 3.14 + mlx_lm 0.31.2
(RuntimeError: There is no Stream(gpu, N) in current thread — affects
all models, not gpt-oss-specific), this harness bypasses the HTTP server
and exercises mlx_lm.stream_generate directly to confirm the
model+weights load and decode correctly, capturing reference perf
numbers with prompt and generation timings separated.

The delegated route (ax-engine -> mlx_lm.server) cannot be exercised on
this host until the server bug is resolved upstream or Python is
downgraded. That is captured in the artifact `report.md`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class PromptRecord:
    index: int
    prompt: str
    prompt_chars: int
    out_text: str
    out_chars: int
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    total_seconds: float
    peak_memory_gb: float
    finish_reason: str


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    prompts: list[str] = []
    with args.prompts.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
                if args.limit and len(prompts) >= args.limit:
                    break
    if not prompts:
        print(f"no prompts in {args.prompts}", file=sys.stderr)
        return 2

    print(f"loading {args.model} ...", flush=True)
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    load_seconds = time.monotonic() - t0
    print(f"loaded in {load_seconds:.2f}s", flush=True)

    greedy_sampler = make_sampler(temp=0.0)
    records: list[PromptRecord] = []
    for i, prompt in enumerate(prompts):
        text_parts: list[str] = []
        last = None
        t0 = time.monotonic()
        for resp in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=greedy_sampler,
        ):
            text_parts.append(resp.text)
            last = resp
        total = time.monotonic() - t0
        if last is None:
            print(f"prompt {i}: no response produced", file=sys.stderr)
            return 3

        out_text = "".join(text_parts)
        peak_gb = float(last.peak_memory or 0.0)  # GenerationResponse.peak_memory is already GB

        rec = PromptRecord(
            index=i,
            prompt=prompt,
            prompt_chars=len(prompt),
            out_text=out_text,
            out_chars=len(out_text),
            prompt_tokens=last.prompt_tokens,
            generation_tokens=last.generation_tokens,
            prompt_tps=float(last.prompt_tps),
            generation_tps=float(last.generation_tps),
            total_seconds=total,
            peak_memory_gb=peak_gb,
            finish_reason=str(last.finish_reason or ""),
        )
        records.append(rec)
        print(
            f"prompt {i}: total={total:.2f}s "
            f"prompt={rec.prompt_tokens}tok @ {rec.prompt_tps:.1f}tps "
            f"gen={rec.generation_tokens}tok @ {rec.generation_tps:.1f}tps "
            f"peak={peak_gb:.1f}GB finish={rec.finish_reason}",
            flush=True,
        )

    gen_tps_values = [r.generation_tps for r in records if r.generation_tokens > 0]
    prompt_tps_values = [r.prompt_tps for r in records if r.prompt_tokens > 0]
    summary = {
        "model": args.model,
        "prompts_evaluated": len(records),
        "load_seconds": load_seconds,
        "generation_tps_mean": (sum(gen_tps_values) / len(gen_tps_values)) if gen_tps_values else 0.0,
        "generation_tps_p50": sorted(gen_tps_values)[len(gen_tps_values) // 2] if gen_tps_values else 0.0,
        "generation_tps_max": max(gen_tps_values) if gen_tps_values else 0.0,
        "generation_tps_min": min(gen_tps_values) if gen_tps_values else 0.0,
        "prompt_tps_mean": (sum(prompt_tps_values) / len(prompt_tps_values)) if prompt_tps_values else 0.0,
        "prompt_tps_p50": sorted(prompt_tps_values)[len(prompt_tps_values) // 2] if prompt_tps_values else 0.0,
        "peak_memory_gb_max": max((r.peak_memory_gb for r in records), default=0.0),
        "max_tokens_per_request": args.max_tokens,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "per_prompt": [asdict(r) for r in records]},
            indent=2,
            ensure_ascii=False,
        )
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
