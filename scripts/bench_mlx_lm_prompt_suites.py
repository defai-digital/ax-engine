"""mlx-lm direct autoregressive lane on the repository real-prompt suite.

Same-pack decode baseline for the peer table: greedy stream_generate, fixed
generated-token budget, chat template with thinking disabled, two warmups and
five measured repetitions per case. Decode tok/s is mlx-lm's own
generation_tps (excludes the prompt and the first generated token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--host-label", default="")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    cases = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    results = []
    all_values: list[float] = []
    for case in cases:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        values: list[float] = []
        prompt_values: list[float] = []
        completion_tokens: list[int] = []
        output_hashes: set[str] = set()
        for rep in range(args.warmup_repetitions + args.repetitions):
            last = None
            text = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt_ids,
                max_tokens=args.max_tokens,
                sampler=sampler,
            ):
                last = response
                text.append(response.text)
            mx.clear_cache()
            if rep >= args.warmup_repetitions and last is not None:
                values.append(float(last.generation_tps))
                prompt_values.append(float(last.prompt_tps))
                completion_tokens.append(int(last.generation_tokens))
                output_hashes.add(hashlib.sha256("".join(text).encode()).hexdigest())
            time.sleep(args.cooldown)
        row = {
            "prompt_id": case["id"],
            "engine": "mlx_lm",
            "method": "mlx_lm.generate stream_generate greedy",
            "max_tokens": args.max_tokens,
            "warmup": args.warmup_repetitions,
            "repetitions": args.repetitions,
            "prompt_tokens": len(prompt_ids),
            "decode_tok_s_values": values,
            "decode_tok_s_median": statistics.median(values),
            "prompt_tok_s_values": prompt_values,
            "prompt_tok_s_median": statistics.median(prompt_values),
            "completion_tokens": completion_tokens,
            "output_text_sha256": sorted(output_hashes),
        }
        results.append(row)
        all_values.extend(values)
        print(
            f"{case['id']}: decode median {row['decode_tok_s_median']:.2f} tok/s "
            f"(prompt {row['prompt_tok_s_median']:.1f} tok/s, {completion_tokens[0]} tokens)",
            flush=True,
        )
    payload = {
        "engine": "mlx_lm",
        "mlx_lm_version": mlx_lm.__version__,
        "mlx_version": mx.__version__,
        "host_label": args.host_label,
        "host": {
            "platform": platform.platform(),
            "hw_model": subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        },
        "pack": args.model,
        "sampling": {"temperature": 0.0, "enable_thinking": False},
        "results": results,
        "decode_tok_s_median_20": statistics.median(all_values),
        "decode_tok_s_count": len(all_values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"decode_tok_s_median_{len(all_values)} = {payload['decode_tok_s_median_20']:.2f}", flush=True)


if __name__ == "__main__":
    main()
