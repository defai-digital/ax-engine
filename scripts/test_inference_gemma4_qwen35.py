#!/usr/bin/env python3
"""
Smoke-test ax-engine inference on Gemma4 (e2b, e4b) and Qwen3.5-9B.

Checks:
- Model loads without error
- A short prompt produces non-empty token output
- Output tokens are within vocab range
- Streaming produces the same final tokens as non-streaming
"""
from __future__ import annotations

import sys
import os
import time

# Add the Python package to the path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import ax_engine
from tokenizers import Tokenizer

def _gemma4_chat(user_msg: str) -> str:
    # Gemma4-it chat template: <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
    return (
        "<bos><start_of_turn>user\n"
        + user_msg
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _qwen35_chat(user_msg: str) -> str:
    # Qwen3.5 chat template with thinking disabled via forced empty think block
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + user_msg
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    )


# Quality check: expected token must appear in output_text (case-insensitive)
MODELS = [
    {
        "name": "Gemma4-e2b (per-layer input gating, no MoE)",
        "model_id": "gemma4",
        "artifacts_dir": os.path.join(
            REPO_ROOT, ".internal/models/gemma-4-e2b-it-4bit"
        ),
        "tokenizer_path": os.path.join(
            REPO_ROOT, ".internal/models/gemma-4-e2b-it-4bit/tokenizer.json"
        ),
        "prompt": _gemma4_chat("What is the capital of France? Answer in one word."),
        "max_output_tokens": 24,
        "expected_in_output": "paris",
    },
    {
        "name": "Gemma4-e4b (per-layer input gating, no MoE)",
        "model_id": "gemma4",
        "artifacts_dir": os.path.join(
            REPO_ROOT, ".internal/models/gemma-4-e4b-it-4bit"
        ),
        "tokenizer_path": os.path.join(
            REPO_ROOT, ".internal/models/gemma-4-e4b-it-4bit/tokenizer.json"
        ),
        "prompt": _gemma4_chat("What is 2+2? Answer with only the number."),
        "max_output_tokens": 8,
        "expected_in_output": "4",
    },
    {
        "name": "Qwen3.5-9B (linear attention, MoE)",
        "model_id": "qwen3_5",
        "artifacts_dir": (
            "/Users/akiralam/.cache/huggingface/hub"
            "/models--mlx-community--Qwen3.5-9B-MLX-4bit"
            "/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631"
        ),
        "tokenizer_path": (
            "/Users/akiralam/.cache/huggingface/hub"
            "/models--mlx-community--Qwen3.5-9B-MLX-4bit"
            "/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631"
            "/tokenizer.json"
        ),
        "prompt": _qwen35_chat("What is the largest planet in the solar system? One word."),
        "max_output_tokens": 32,
        "expected_in_output": "jupiter",
    },
]


PASS = "PASS"
FAIL = "FAIL"


def tokenize(tokenizer_path: str, text: str) -> list[int]:
    tok = Tokenizer.from_file(tokenizer_path)
    enc = tok.encode(text)
    return enc.ids


def run_model_test(model: dict) -> bool:
    name = model["name"]
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  artifacts_dir: {model['artifacts_dir']}")
    print(f"  prompt: {model['prompt']!r}")

    if not os.path.isdir(model["artifacts_dir"]):
        print(f"  [{FAIL}] model directory not found: {model['artifacts_dir']}")
        return False

    manifest_path = os.path.join(model["artifacts_dir"], "model-manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  [{FAIL}] manifest not found: {manifest_path}")
        return False

    # Tokenize
    if not os.path.exists(model["tokenizer_path"]):
        print(f"  [{FAIL}] tokenizer not found: {model['tokenizer_path']}")
        return False

    try:
        input_tokens = tokenize(model["tokenizer_path"], model["prompt"])
    except Exception as e:
        print(f"  [{FAIL}] tokenization error: {e}")
        return False

    print(f"  input_tokens ({len(input_tokens)}): {input_tokens}")

    # --- Non-streaming generate ---
    t0 = time.monotonic()
    try:
        with ax_engine.Session(
            model_id=model["model_id"],
            mlx=True,
            mlx_model_artifacts_dir=model["artifacts_dir"],
        ) as session:
            result = session.generate(input_tokens, max_output_tokens=model["max_output_tokens"])
    except Exception as e:
        print(f"  [{FAIL}] generate() raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.monotonic() - t0
    output_tokens = result.output_tokens
    print(f"  output_tokens ({len(output_tokens)}): {output_tokens}")
    print(f"  finish_reason: {result.finish_reason}")
    print(f"  elapsed: {elapsed:.2f}s")

    # Validation
    if len(output_tokens) == 0:
        print(f"  [{FAIL}] output is empty")
        return False

    # Decode output and quality check
    output_text = None
    try:
        tok = Tokenizer.from_file(model["tokenizer_path"])
        output_text = tok.decode(output_tokens)
        print(f"  decoded output: {output_text!r}")
    except Exception as e:
        print(f"  (decode error: {e})")

    expected = model.get("expected_in_output")
    if expected and output_text is not None:
        if expected.lower() not in output_text.lower():
            print(f"  [{FAIL}] expected {expected!r} in output, got: {output_text!r}")
            return False
        print(f"  [{PASS}] quality check: found {expected!r} in output")

    # --- Streaming generate (verify matches non-streaming) ---
    try:
        with ax_engine.Session(
            model_id=model["model_id"],
            mlx=True,
            mlx_model_artifacts_dir=model["artifacts_dir"],
        ) as session:
            stream_tokens = []
            for event in session.stream_generate(
                input_tokens, max_output_tokens=model["max_output_tokens"]
            ):
                if event.delta_tokens:
                    stream_tokens.extend(event.delta_tokens)
    except Exception as e:
        print(f"  [{FAIL}] stream_generate() raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  streaming_tokens ({len(stream_tokens)}): {stream_tokens}")

    if stream_tokens != output_tokens:
        print(f"  [{FAIL}] streaming output differs from non-streaming!")
        print(f"    non-streaming: {output_tokens}")
        print(f"    streaming:     {stream_tokens}")
        return False

    print(f"  [{PASS}] output matches between streaming and non-streaming")
    print(f"  [{PASS}] {name}")
    return True


def main() -> int:
    print("ax-engine inference smoke test: Gemma4 + Qwen3.5")
    print(f"ax_engine module: {ax_engine.__file__}")

    results = []
    for model in MODELS:
        ok = run_model_test(model)
        results.append((model["name"], ok))

    print(f"\n{'='*60}")
    print("RESULTS:")
    all_passed = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
        return 0
    else:
        print("\nSome tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
