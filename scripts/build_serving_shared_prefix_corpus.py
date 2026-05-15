#!/usr/bin/env python3
"""Build a deterministic shared-prefix serving corpus.

The corpus is meant for online serving soak tests where the runtime must prove
that repeated long prompts exercise prefix-cache reuse. Rows use `input_tokens`
instead of natural language so the prompt shape is stable across tokenizers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("benchmarks/corpora/serving/disk_prefix_cache_soak.jsonl")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_prefix_tokens(length: int, *, start_token: int, token_span: int) -> list[int]:
    return [start_token + (index % token_span) for index in range(length)]


def build_suffix_tokens(length: int, *, variant: int, start_token: int, token_span: int) -> list[int]:
    offset = variant * 17
    return [start_token + ((offset + index) % token_span) for index in range(length)]


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    prefix = build_prefix_tokens(
        args.prefix_tokens,
        start_token=args.prefix_start_token,
        token_span=args.prefix_token_span,
    )
    rows: list[dict[str, Any]] = []
    for index in range(args.prompts):
        suffix = build_suffix_tokens(
            args.suffix_tokens,
            variant=index,
            start_token=args.suffix_start_token,
            token_span=args.suffix_token_span,
        )
        tokens = prefix + suffix
        rows.append(
            {
                "id": f"{args.id_prefix}_{index:03d}",
                "category": args.category,
                "input_tokens": tokens,
                "input_tokens_count": len(tokens),
                "max_output_tokens": args.max_output_tokens,
                "metadata": {
                    "generated_by": Path(__file__).name,
                    "shared_prefix_id": args.shared_prefix_id,
                    "shared_prefix_tokens": args.prefix_tokens,
                    "suffix_tokens": args.suffix_tokens,
                    "variant_index": index,
                    "purpose": "disk_prefix_cache_serving_soak",
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts", type=positive_int, default=8)
    parser.add_argument("--prefix-tokens", type=positive_int, default=8192)
    parser.add_argument("--suffix-tokens", type=positive_int, default=64)
    parser.add_argument("--max-output-tokens", type=positive_int, default=64)
    parser.add_argument("--category", default="disk_prefix_shared_long")
    parser.add_argument("--id-prefix", default="disk_prefix_soak")
    parser.add_argument("--shared-prefix-id", default="disk-prefix-cache-soak-v1")
    parser.add_argument("--prefix-start-token", type=positive_int, default=200)
    parser.add_argument("--prefix-token-span", type=positive_int, default=512)
    parser.add_argument("--suffix-start-token", type=positive_int, default=2000)
    parser.add_argument("--suffix-token-span", type=positive_int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = build_rows(args)
    write_jsonl(args.output, rows)
    print(
        "AX serving shared-prefix corpus written: "
        f"{args.output} ({len(rows)} prompts, "
        f"{args.prefix_tokens}+{args.suffix_tokens} tokens each)"
    )
    return 0


def main_with_args_for_test(argv: list[str]) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
