#!/usr/bin/env python3
"""Benchmark mlx-lm vs AX native engine on Qwen3.5-2B-bf16.

Usage:
    # MLX-LM only (no server needed):
    .venv/bin/python3 scripts/bench_mlx_vs_native.py --mlx-only

    # Both (start native server first on port 8080):
    .venv/bin/python3 scripts/bench_mlx_vs_native.py

    # Custom prompt / token count:
    .venv/bin/python3 scripts/bench_mlx_vs_native.py --max-tokens 256 --prompt "Explain quicksort"
"""

import argparse
import json
import time
import sys
import urllib.request
import urllib.error


MODEL_ID = "mlx-community/Qwen3.5-2B-bf16"
DEFAULT_PROMPT = "Write a short Python function that checks if a number is prime."
DEFAULT_MAX_TOKENS = 128
NATIVE_URL = "http://127.0.0.1:8080"
WARMUP_TOKENS = 8


def bench_mlx_lm(prompt: str, max_tokens: int, warmup: bool = True) -> dict:
    """Benchmark mlx-lm generate."""
    import mlx_lm

    print(f"\n{'='*60}")
    print("MLX-LM Benchmark")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Max tokens: {max_tokens}")
    print("Loading model...")

    t_load_start = time.perf_counter()
    model, tokenizer = mlx_lm.load(MODEL_ID)
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.2f}s")

    if warmup:
        print("Warming up...")
        _ = mlx_lm.generate(
            model, tokenizer, prompt="Hi", max_tokens=WARMUP_TOKENS, verbose=False
        )

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_tokens = len(tokenizer.encode(formatted))

    print(f"Input tokens: {input_tokens}")
    print(f"Generating {max_tokens} tokens...")

    t_start = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=True
    )
    t_total = time.perf_counter() - t_start

    output_tokens = len(tokenizer.encode(response))

    # mlx_lm.generate with verbose=True prints its own stats,
    # but we also compute ours for consistency
    tps = output_tokens / t_total if t_total > 0 else 0

    result = {
        "engine": "mlx-lm",
        "model": MODEL_ID,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": round(t_total, 3),
        "tokens_per_sec": round(tps, 1),
        "load_time_s": round(t_load, 3),
    }

    print(f"\n--- MLX-LM Results ---")
    print(f"Output tokens: {output_tokens}")
    print(f"Total time:    {t_total:.3f}s")
    print(f"Throughput:    {tps:.1f} tok/s")
    print(f"Load time:     {t_load:.2f}s")

    return result


def bench_native(prompt: str, max_tokens: int, base_url: str, warmup: bool = True) -> dict:
    """Benchmark AX native engine via /v1/generate endpoint."""
    print(f"\n{'='*60}")
    print("AX Native Engine Benchmark")
    print(f"{'='*60}")
    print(f"Server: {base_url}")
    print(f"Max tokens: {max_tokens}")

    # Check server is up
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read())
            model_data = models.get("data", [{}])[0]
            print(f"Model: {model_data.get('id', 'unknown')}")
            soc = model_data.get("runtime", {}).get("host", {}).get("detected_soc", "unknown")
            print(f"SoC: {soc}")
    except Exception as e:
        print(f"ERROR: Cannot reach native server at {base_url}: {e}")
        print("Start the server first:")
        print(f"  cargo run --release -p ax-engine-server -- --model-id qwen3_5 \\")
        print(f"    --native-model-artifacts-dir .internal/models/Qwen3.5-2B-bf16 \\")
        print(f"    --support-tier native-preview")
        return None

    # Format and tokenize prompt with chat template (same as mlx-lm uses)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=False,
    )
    # apply_chat_template may return a list of ints or a BatchEncoding
    if isinstance(encoded, dict):
        input_token_ids = list(encoded["input_ids"])
    else:
        input_token_ids = list(encoded)

    if warmup:
        print("Warming up...")
        warmup_ids = tokenizer.encode("Hello")
        _do_native_generate(base_url, warmup_ids, WARMUP_TOKENS)

    print(f"Input tokens: {len(input_token_ids)}")
    print(f"Generating {max_tokens} tokens...")

    t_start = time.perf_counter()
    resp_data = _do_native_generate(base_url, input_token_ids, max_tokens)
    t_total = time.perf_counter() - t_start

    if resp_data is None:
        print("ERROR: Generate request failed")
        return None

    input_tokens = len(resp_data.get("prompt_tokens", []))
    output_tokens = len(resp_data.get("output_tokens", []))
    output_text = resp_data.get("output_text", "")
    tps = output_tokens / t_total if t_total > 0 else 0

    result = {
        "engine": "ax-native",
        "model": resp_data.get("model_id", "unknown"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": round(t_total, 3),
        "tokens_per_sec": round(tps, 1),
    }

    print(f"\n--- AX Native Results ---")
    print(f"Input tokens:  {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total time:    {t_total:.3f}s")
    print(f"Throughput:    {tps:.1f} tok/s")
    if output_text:
        preview = output_text[:200]
        print(f"Output:        {preview}{'...' if len(output_text) > 200 else ''}")

    return result


def _do_native_generate(base_url: str, token_ids: list[int], max_tokens: int) -> dict | None:
    """Send a generate request to AX native /v1/generate endpoint with pre-tokenized input."""
    payload = json.dumps({
        "input_tokens": token_ids,
        "max_output_tokens": max_tokens,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "seed": 1234,
        },
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def print_comparison(mlx_result: dict, native_result: dict | None):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'MLX-LM':>15} {'AX Native':>15}")
    print(f"{'-'*55}")

    print(f"{'Output tokens':<25} {mlx_result['output_tokens']:>15} ", end="")
    if native_result:
        print(f"{native_result['output_tokens']:>15}")
    else:
        print(f"{'N/A':>15}")

    print(f"{'Total time (s)':<25} {mlx_result['total_time_s']:>15.3f} ", end="")
    if native_result:
        print(f"{native_result['total_time_s']:>15.3f}")
    else:
        print(f"{'N/A':>15}")

    print(f"{'Throughput (tok/s)':<25} {mlx_result['tokens_per_sec']:>15.1f} ", end="")
    if native_result:
        print(f"{native_result['tokens_per_sec']:>15.1f}")
        speedup = native_result['tokens_per_sec'] / mlx_result['tokens_per_sec'] if mlx_result['tokens_per_sec'] > 0 else 0
        print(f"\n{'Speedup (native/mlx)':<25} {speedup:>15.2f}x")
    else:
        print(f"{'N/A':>15}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mlx-lm vs AX native")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--mlx-only", action="store_true", help="Skip native engine")
    parser.add_argument("--native-only", action="store_true", help="Skip mlx-lm")
    parser.add_argument("--native-url", default=NATIVE_URL)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (reports best)")
    args = parser.parse_args()

    warmup = not args.no_warmup

    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs}")

    mlx_result = None
    native_result = None

    if not args.native_only:
        best_mlx = None
        for i in range(args.runs):
            if args.runs > 1:
                print(f"\n--- MLX-LM Run {i+1}/{args.runs} ---")
            r = bench_mlx_lm(args.prompt, args.max_tokens, warmup=warmup and i == 0)
            if best_mlx is None or r["tokens_per_sec"] > best_mlx["tokens_per_sec"]:
                best_mlx = r
        mlx_result = best_mlx

    if not args.mlx_only:
        best_native = None
        for i in range(args.runs):
            if args.runs > 1:
                print(f"\n--- Native Run {i+1}/{args.runs} ---")
            r = bench_native(args.prompt, args.max_tokens, args.native_url, warmup=warmup and i == 0)
            if r is None:
                break
            if best_native is None or r["tokens_per_sec"] > best_native["tokens_per_sec"]:
                best_native = r
        native_result = best_native

    if mlx_result and (native_result or args.mlx_only):
        if not args.mlx_only:
            print_comparison(mlx_result, native_result)
        else:
            print(f"\n{'='*60}")
            print("MLX-LM SUMMARY")
            print(f"{'='*60}")
            print(f"Throughput: {mlx_result['tokens_per_sec']:.1f} tok/s")
    elif native_result:
        print(f"\n{'='*60}")
        print("AX NATIVE SUMMARY")
        print(f"{'='*60}")
        print(f"Throughput: {native_result['tokens_per_sec']:.1f} tok/s")


if __name__ == "__main__":
    main()
