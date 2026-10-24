#!/usr/bin/env python3
"""mlx-lm direct-AR harness over a repo MTP prompt suite (greedy), prefill and decode.

Prefill is mlx-lm's own prompt_tps (prompt tokens over the prompt phase) and the
client time to the first token; decode is generation_tps (tokens after the first
over their time) next to the earlier first-token-excluded wall measure. Prompts
go through the chat template with thinking off, like the other runtimes.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--prompts', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--cooldown', type=float, default=3.0)
    args = ap.parse_args()
    cases = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    model, tokenizer = load(args.pack)
    sampler = make_sampler(temp=0.0)
    results = []
    for case in cases:
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
                                            add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, 'input_ids'):
            ids = ids.input_ids
        ids = list(ids)
        trials = []
        for i in range(args.warmup + args.reps):
            n, t0, t_first, last, text = 0, time.perf_counter(), None, None, ''
            for resp in stream_generate(model, tokenizer, prompt=ids, max_tokens=args.max_tokens, sampler=sampler):
                n += 1
                if n == 1:
                    t_first = time.perf_counter()
                last = resp
                text += resp.text
            t_end = time.perf_counter()
            decode_s = t_end - t_first if t_first else 0.0
            trial = {
                'ttft_s': (t_first - t0) if t_first else None,
                'prefill_tok_s': len(ids) / (t_first - t0) if t_first else None,
                'engine_prompt_tps': last.prompt_tps if last else None,
                'engine_generation_tps': last.generation_tps if last else None,
                'decode_tok_s': (n - 1) / decode_s if n >= 2 and decode_s > 0 else 0.0,
                'generated_tokens': n,
                'text': text,
            }
            print(f"{case['id']} run {i+1}/{args.warmup+args.reps} n={n} prompt={len(ids)} "
                  f"prefill={trial['engine_prompt_tps']:.1f} (client {trial['prefill_tok_s']:.1f}) "
                  f"decode={trial['decode_tok_s']:.2f} (engine {trial['engine_generation_tps']:.2f})", flush=True)
            if i >= args.warmup:
                trials.append(trial)
            time.sleep(args.cooldown)
        results.append({'prompt_id': case['id'], 'engine': 'mlx_lm', 'prompt_tokens': len(ids),
                        'method': 'mlx_lm.generate stream_generate greedy, chat template, thinking off',
                        'max_tokens': args.max_tokens, 'warmup': args.warmup, 'repetitions': args.reps,
                        'trials': trials,
                        'decode_tok_s_values': [t['decode_tok_s'] for t in trials]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'engine': 'mlx_lm', 'pack': args.pack, 'prompts': str(args.prompts),
                                       'results': results}, indent=2) + '\n')
    print('wrote', args.output, flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
