#!/usr/bin/env python3
"""Compare AX direct MLX generation against mlx_lm on local direct-support models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HUB = Path.home() / ".cache/huggingface/hub"
BENCH_BIN = REPO_ROOT / "target/release/ax-engine-bench"


@dataclass(frozen=True)
class ModelCase:
    slug: str
    model_id: str
    cache_name: str
    compare_tokens: int = 4


@dataclass(frozen=True)
class PromptCase:
    slug: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class AxMode:
    slug: str
    disable_ngram: bool


MODEL_CASES = [
    ModelCase("gemma-4-e2b-it-4bit", "mlx-community/gemma-4-e2b-it-4bit", "mlx-community--gemma-4-e2b-it-4bit"),
    ModelCase("gemma-4-e2b-it-5bit", "mlx-community/gemma-4-e2b-it-5bit", "mlx-community--gemma-4-e2b-it-5bit"),
    ModelCase("gemma-4-e2b-it-6bit", "mlx-community/gemma-4-e2b-it-6bit", "mlx-community--gemma-4-e2b-it-6bit"),
    ModelCase("gemma-4-e2b-it-8bit", "mlx-community/gemma-4-e2b-it-8bit", "mlx-community--gemma-4-e2b-it-8bit"),
    ModelCase("gemma-4-e4b-it-4bit", "mlx-community/gemma-4-e4b-it-4bit", "mlx-community--gemma-4-e4b-it-4bit"),
    ModelCase("gemma-4-26b-a4b-it-4bit", "mlx-community/gemma-4-26b-a4b-it-4bit", "mlx-community--gemma-4-26b-a4b-it-4bit"),
    ModelCase("gemma-4-31b-it-4bit", "mlx-community/gemma-4-31b-it-4bit", "mlx-community--gemma-4-31b-it-4bit"),
    ModelCase("qwen3-4b-4bit", "mlx-community/Qwen3-4B-4bit", "mlx-community--Qwen3-4B-4bit"),
    ModelCase("qwen3-5-9b-mlx-4bit", "mlx-community/Qwen3.5-9B-MLX-4bit", "mlx-community--Qwen3.5-9B-MLX-4bit"),
    ModelCase("qwen3-6-27b-4bit", "mlx-community/Qwen3.6-27B-4bit", "mlx-community--Qwen3.6-27B-4bit"),
    ModelCase("qwen3-6-27b-5bit", "mlx-community/Qwen3.6-27B-5bit", "mlx-community--Qwen3.6-27B-5bit"),
    ModelCase("qwen3-6-27b-6bit", "mlx-community/Qwen3.6-27B-6bit", "mlx-community--Qwen3.6-27B-6bit"),
    ModelCase("qwen3-6-27b-8bit", "mlx-community/Qwen3.6-27B-8bit", "mlx-community--Qwen3.6-27B-8bit"),
    ModelCase("qwen3-6-35b-a3b-4bit", "mlx-community/Qwen3.6-35B-A3B-4bit", "mlx-community--Qwen3.6-35B-A3B-4bit"),
    ModelCase("qwen3-coder-next-4bit", "mlx-community/Qwen3-Coder-Next-4bit", "mlx-community--Qwen3-Coder-Next-4bit"),
]


AX_MODES = [
    AxMode("direct", True),
    AxMode("ngram", False),
]


PROMPT_CASES = [
    PromptCase(
        "short_fact",
        [
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": "What is the capital city of Japan? Answer with just the city name."},
        ],
    ),
    PromptCase(
        "arithmetic_exact",
        [
            {"role": "system", "content": "Answer with only the requested final value."},
            {"role": "user", "content": "Compute 17 + 28. Answer with just the number."},
        ],
    ),
    PromptCase(
        "fixed_phrase",
        [
            {"role": "system", "content": "Follow the requested output exactly."},
            {"role": "user", "content": "Print exactly this text and nothing else: alpha | beta | gamma"},
        ],
    ),
    PromptCase(
        "short_code",
        [
            {"role": "system", "content": "You are a careful coding assistant."},
            {"role": "user", "content": "Write a Python function named add_one that returns x + 1. Keep it tiny."},
        ],
    ),
    PromptCase(
        "json_exact",
        [
            {"role": "system", "content": "Return compact JSON only."},
            {"role": "user", "content": "Return a JSON object with keys ok and count, where ok is true and count is 3."},
        ],
    ),
    PromptCase(
        "stop_short",
        [
            {"role": "system", "content": "Be brief and stop immediately after the requested token."},
            {"role": "user", "content": "Reply with exactly one word: Done"},
        ],
    ),
    PromptCase(
        "repeat_guard",
        [
            {"role": "system", "content": "Avoid runaway repetition; obey exact counts."},
            {"role": "user", "content": "Write the word echo exactly five times separated by spaces, then stop."},
        ],
    ),
    PromptCase(
        "ngram_pattern",
        [
            {"role": "system", "content": "Continue simple patterns exactly."},
            {
                "role": "user",
                "content": (
                    "Continue the pattern with the next six words only: "
                    "red blue green red blue green red blue green red blue"
                ),
            },
        ],
    ),
    PromptCase(
        "glm_qwen_regression",
        [
            {"role": "system", "content": "Answer directly and avoid repetition."},
            {"role": "user", "content": "Say exactly three comma-separated colors, then stop."},
        ],
    ),
]


def load_prompt_cases_from_manifest(path: Path) -> list[PromptCase]:
    payload = json.loads(path.read_text())
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"prompt manifest has no prompts: {path}")
    cases = []
    for index, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise SystemExit(f"prompt manifest entry {index} is not an object: {path}")
        slug = item.get("slug")
        messages = item.get("messages")
        if not isinstance(slug, str) or not slug:
            raise SystemExit(f"prompt manifest entry {index} has invalid slug: {path}")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"prompt manifest entry {slug} has invalid messages: {path}")
        normalized_messages = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise SystemExit(f"prompt {slug} message {message_index} is not an object: {path}")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise SystemExit(f"prompt {slug} message {message_index} is invalid: {path}")
            normalized_messages.append({"role": role, "content": content})
        cases.append(PromptCase(slug, normalized_messages))
    return cases


def latest_snapshot(cache_name: str, *, require_manifest: bool = False) -> Path | None:
    root = HF_HUB / f"models--{cache_name}" / "snapshots"
    if not root.is_dir():
        return None
    snapshots = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if require_manifest:
        snapshots = [p for p in snapshots if (p / "model-manifest.json").is_file()]
    return snapshots[-1] if snapshots else None


def tokenizer_stop_token_ids(tok: Any) -> set[int]:
    ids: set[int] = set()
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, list):
        ids.update(int(item) for item in eos if isinstance(item, int))
    for token in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<turn|>", "<|turn|>"):
        token_id = tok.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(tok, "unk_token_id", None):
            ids.add(token_id)
    return ids


def tokenize_prompt(model_dir: Path, prompt_case: PromptCase) -> tuple[list[int], str, set[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    text = tok.apply_chat_template(
        prompt_case.messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return tok.encode(text, add_special_tokens=False), text, tokenizer_stop_token_ids(tok)


def decode_tokens(model_dir: Path, tokens: list[int]) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    return tok.decode(tokens)


def run_ref_child(model_dir: Path, prompt_tokens: list[int], max_tokens: int) -> dict[str, Any]:
    code = r'''
import json
import sys
import mlx.core as mx
from pathlib import Path
from mlx_lm.utils import load_model
from mlx_lm.generate import generate_step

payload = json.loads(sys.stdin.read())
model, _config = load_model(Path(payload["model_dir"]), strict=False)
prompt = mx.array(payload["prompt_tokens"])
tokens = []
top_logprobs = []
for token, _logprobs in generate_step(prompt, model, max_tokens=payload["max_tokens"]):
    tokens.append(int(token))
    top_k = int(payload.get("top_logprobs") or 0)
    if top_k > 0:
        top_indices = mx.argpartition(-_logprobs, kth=top_k - 1)[:top_k]
        top_values = _logprobs[top_indices]
        mx.eval(top_indices, top_values)
        pairs = sorted(
            zip([int(item) for item in top_indices.tolist()], [float(item) for item in top_values.tolist()]),
            key=lambda item: item[1],
            reverse=True,
        )
        top_logprobs.append(pairs)
mx.synchronize()
result = {"tokens": tokens}
if top_logprobs:
    result["top_logprobs"] = top_logprobs
print(json.dumps(result))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(
            {
                "model_dir": str(model_dir),
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "top_logprobs": int(os.environ.get("AX_REF_TOP_LOGPROBS", "0") or "0"),
            }
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
    return json.loads(proc.stdout)


def run_ax(
    model_id: str,
    model_dir: Path,
    prompt_tokens: list[int],
    max_tokens: int,
    mode: AxMode,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("AX_ALLOW_UNSUPPORTED_HOST", "1")
    env["AX_NO_SPEC"] = "1" if mode.disable_ngram else "0"
    cmd = [
        str(BENCH_BIN),
        "generate",
        "--model-id",
        model_id,
        "--tokens",
        ",".join(str(t) for t in prompt_tokens),
        "--max-output-tokens",
        str(max_tokens),
        "--mlx",
        "--support-tier",
        "mlx_preview",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
    return json.loads(proc.stdout)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def normalize_prefix_text(text: str) -> str:
    return " ".join(text.split())


def whitespace_equivalent_prefix(
    model_dir: Path,
    ref_tokens: list[int],
    ax_tokens: list[int],
    required: int,
) -> bool:
    if required <= 0 or len(ref_tokens) < required or len(ax_tokens) < required:
        return False
    ref_text = normalize_prefix_text(decode_tokens(model_dir, ref_tokens[:required]))
    ax_text = normalize_prefix_text(decode_tokens(model_dir, ax_tokens[:required]))
    return bool(ref_text) and ref_text == ax_text


def run_prompt_case(
    case: ModelCase,
    model_dir: Path,
    prompt_case: PromptCase,
    max_tokens: int,
    compare_tokens: int | None,
    ax_modes: list[AxMode],
) -> dict[str, Any]:
    prompt_tokens, prompt_text, stop_token_ids = tokenize_prompt(model_dir, prompt_case)
    ref = run_ref_child(model_dir, prompt_tokens, max_tokens)
    result: dict[str, Any] = {
        "prompt_case": prompt_case.slug,
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": len(prompt_tokens),
        "prompt_text": prompt_text,
        "max_tokens": max_tokens,
        "compare_tokens": compare_tokens if compare_tokens is not None else case.compare_tokens,
        "reference": ref,
        "ax_modes": {},
    }
    if "error" in ref:
        result["status"] = "error"
        return result
    ref_tokens = [int(t) for t in ref["tokens"]]
    result["reference"]["text"] = decode_tokens(model_dir, ref_tokens)
    required = result["compare_tokens"]
    mode_statuses = []
    for mode in ax_modes:
        ax = run_ax(case.model_id, model_dir, prompt_tokens, max_tokens, mode)
        mode_result: dict[str, Any] = {"ax": ax}
        if "error" in ax:
            mode_result["status"] = "error"
            result["ax_modes"][mode.slug] = mode_result
            mode_statuses.append("error")
            continue
        ax_tokens = [int(t) for t in ax["output_tokens"]]
        prefix = common_prefix_len(ref_tokens, ax_tokens)
        mode_result["ax"]["text"] = decode_tokens(model_dir, ax_tokens)
        mode_result["common_prefix_tokens"] = prefix
        mode_result["required_prefix_tokens"] = required
        whitespace_prefix_equivalent = prefix < required and whitespace_equivalent_prefix(
            model_dir, ref_tokens, ax_tokens, required
        )
        mode_result["whitespace_prefix_equivalent"] = whitespace_prefix_equivalent
        stop_equivalent = (
            prefix == len(ax_tokens)
            and ax.get("finish_reason") == "stop"
            and len(ref_tokens) > len(ax_tokens)
            and int(ref_tokens[len(ax_tokens)]) in stop_token_ids
        )
        mode_result["stop_equivalent"] = stop_equivalent
        mode_result["status"] = (
            "pass" if prefix >= required or whitespace_prefix_equivalent or stop_equivalent else "fail"
        )
        result["ax_modes"][mode.slug] = mode_result
        mode_statuses.append(mode_result["status"])
    if any(status == "error" for status in mode_statuses):
        result["status"] = "error"
    elif any(status == "fail" for status in mode_statuses):
        result["status"] = "fail"
    else:
        result["status"] = "pass"
    return result


def run_case(
    case: ModelCase,
    prompt_cases: list[PromptCase],
    max_tokens: int,
    compare_tokens: int | None,
    ax_modes: list[AxMode],
    stop_on_first_prompt_failure: bool,
) -> dict[str, Any]:
    model_dir = latest_snapshot(case.cache_name, require_manifest=True)
    if model_dir is None:
        return {
            "slug": case.slug,
            "model_id": case.model_id,
            "status": "skipped",
            "reason": "missing_ax_ready_local_snapshot",
        }
    result: dict[str, Any] = {
        "slug": case.slug,
        "model_id": case.model_id,
        "model_dir": str(model_dir),
        "prompt_results": [],
    }
    for prompt_case in prompt_cases:
        prompt_result = run_prompt_case(case, model_dir, prompt_case, max_tokens, compare_tokens, ax_modes)
        result["prompt_results"].append(prompt_result)
        if prompt_result["status"] in {"fail", "error"} and stop_on_first_prompt_failure:
            break
    statuses = [r["status"] for r in result["prompt_results"]]
    if any(s == "error" for s in statuses):
        result["status"] = "error"
    elif any(s == "fail" for s in statuses):
        result["status"] = "fail"
    else:
        result["status"] = "pass"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="Comma-separated slugs or all.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--compare-tokens",
        type=int,
        default=None,
        help="Required matching output-token prefix length. Defaults to each model case setting.",
    )
    parser.add_argument(
        "--prompt-cases",
        default="all",
        help="Comma-separated prompt-case slugs or all.",
    )
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="JSON file with a top-level prompts array of slug/messages prompt cases.",
    )
    parser.add_argument(
        "--ax-modes",
        default="direct,ngram",
        help="Comma-separated AX modes to test: direct, ngram, or all.",
    )
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--ref-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    selected = MODEL_CASES
    if args.models != "all":
        wanted = {item.strip() for item in args.models.split(",") if item.strip()}
        selected = [case for case in MODEL_CASES if case.slug in wanted]
        missing = wanted - {case.slug for case in selected}
        if missing:
            raise SystemExit(f"unknown model slugs: {', '.join(sorted(missing))}")
    available_prompts = load_prompt_cases_from_manifest(args.prompt_manifest) if args.prompt_manifest else PROMPT_CASES
    selected_prompts = available_prompts
    if args.prompt_cases != "all":
        wanted_prompts = {item.strip() for item in args.prompt_cases.split(",") if item.strip()}
        selected_prompts = [case for case in available_prompts if case.slug in wanted_prompts]
        missing_prompts = wanted_prompts - {case.slug for case in selected_prompts}
        if missing_prompts:
            raise SystemExit(f"unknown prompt-case slugs: {', '.join(sorted(missing_prompts))}")
    selected_modes = AX_MODES
    if args.ax_modes != "all":
        wanted_modes = {item.strip() for item in args.ax_modes.split(",") if item.strip()}
        selected_modes = [mode for mode in AX_MODES if mode.slug in wanted_modes]
        missing_modes = wanted_modes - {mode.slug for mode in selected_modes}
        if missing_modes:
            raise SystemExit(f"unknown AX modes: {', '.join(sorted(missing_modes))}")

    results = []
    for case in selected:
        print(f"[io] {case.slug}", file=sys.stderr, flush=True)
        result = run_case(
            case,
            selected_prompts,
            args.max_tokens,
            args.compare_tokens,
            selected_modes,
            stop_on_first_prompt_failure=not args.continue_on_fail,
        )
        results.append(result)
        status = result["status"]
        print(f"[io] {case.slug}: {status}", file=sys.stderr, flush=True)
        if status == "fail":
            for prompt_result in result.get("prompt_results", []):
                if prompt_result["status"] == "fail":
                    for mode_slug, mode_result in prompt_result.get("ax_modes", {}).items():
                        if mode_result["status"] != "fail":
                            continue
                        print(
                            f"  {prompt_result['prompt_case']}[{mode_slug}]: "
                            f"prefix={mode_result['common_prefix_tokens']}/"
                            f"{mode_result['required_prefix_tokens']} "
                            f"ref={prompt_result['reference'].get('tokens')} "
                            f"ax={mode_result['ax'].get('output_tokens')}",
                            file=sys.stderr,
                            flush=True,
                        )
        if status in {"fail", "error"} and not args.continue_on_fail:
            break

    summary = {
        "schema_version": "ax.direct_model_io.v1",
        "prompt_cases": [case.slug for case in selected_prompts],
        "ax_modes": [mode.slug for mode in selected_modes],
        "max_tokens": args.max_tokens,
        "compare_tokens_override": args.compare_tokens,
        "results": results,
        "counts": {
            "pass": sum(1 for r in results if r["status"] == "pass"),
            "fail": sum(1 for r in results if r["status"] == "fail"),
            "error": sum(1 for r in results if r["status"] == "error"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
        },
    }
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0 if summary["counts"]["fail"] == 0 and summary["counts"]["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
