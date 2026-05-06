#!/usr/bin/env python3
"""Extract AX output-token vectors from MLX benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ax.turboquant_decode_outputs.v1"


class DecodeOutputBuildError(RuntimeError):
    """Raised when benchmark output cannot become quality evidence."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise DecodeOutputBuildError(f"{path} must contain a JSON object")
    return payload


def _matching_rows(
    doc: dict[str, Any],
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
    compression_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in doc.get("results", []):
        if not isinstance(row, dict):
            continue
        if row.get("engine") != engine:
            continue
        if int(row.get("prompt_tokens", -1)) != context_tokens:
            continue
        if int(row.get("generation_tokens", -1)) != generation_tokens:
            continue
        row_compression = row.get("experimental_mlx_kv_compression")
        if compression_mode == "disabled":
            if row_compression not in {None, "disabled"}:
                continue
        elif row_compression != compression_mode:
            continue
        rows.append(row)
    return rows


def _select_row(
    doc: dict[str, Any],
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
    compression_mode: str,
) -> dict[str, Any]:
    rows = _matching_rows(
        doc,
        engine=engine,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
        compression_mode=compression_mode,
    )
    if len(rows) != 1:
        raise DecodeOutputBuildError(
            "expected exactly one AX row for "
            f"engine={engine} prompt_tokens={context_tokens} "
            f"generation_tokens={generation_tokens} "
            f"compression_mode={compression_mode}, got {len(rows)}"
        )
    return rows[0]


def _trial_token_ids(
    trial: dict[str, Any],
    *,
    trial_index: int,
    generation_tokens: int,
    allow_short_output: bool,
) -> list[float]:
    raw_tokens = trial.get("output_token_ids")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise DecodeOutputBuildError(
            f"trial {trial_index} missing output_token_ids; rerun benchmark with "
            "--capture-output-token-ids"
        )
    tokens: list[float] = []
    for token_index, token in enumerate(raw_tokens):
        if not isinstance(token, int) or isinstance(token, bool):
            raise DecodeOutputBuildError(
                f"trial {trial_index} output_token_ids[{token_index}] must be an integer"
            )
        tokens.append(float(token))
    if not allow_short_output and len(tokens) != generation_tokens:
        raise DecodeOutputBuildError(
            f"trial {trial_index} output token count mismatch: "
            f"expected={generation_tokens} actual={len(tokens)}"
        )
    return tokens


def build_decode_outputs(
    benchmark: Path,
    *,
    engine: str,
    context_tokens: int,
    generation_tokens: int,
    compression_mode: str,
    allow_short_output: bool = False,
) -> dict[str, Any]:
    doc = _load_json(benchmark)
    row = _select_row(
        doc,
        engine=engine,
        context_tokens=context_tokens,
        generation_tokens=generation_tokens,
        compression_mode=compression_mode,
    )
    trials = row.get("trials")
    if not isinstance(trials, list) or not trials:
        raise DecodeOutputBuildError("selected row has no trial data")

    outputs = [
        _trial_token_ids(
            trial,
            trial_index=index,
            generation_tokens=generation_tokens,
            allow_short_output=allow_short_output,
        )
        for index, trial in enumerate(trials)
        if isinstance(trial, dict)
    ]
    if len(outputs) != len(trials):
        raise DecodeOutputBuildError("selected row contains non-object trial entries")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_benchmark": str(benchmark),
        "engine": engine,
        "prompt_tokens": context_tokens,
        "generation_tokens": generation_tokens,
        "compression_mode": compression_mode,
        "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
        "decode_outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", default="ax_engine_mlx")
    parser.add_argument("--context-tokens", required=True, type=int)
    parser.add_argument("--generation-tokens", required=True, type=int)
    parser.add_argument(
        "--compression-mode",
        default="disabled",
        choices=["disabled", "turboquant-shadow", "turboquant-fused-experimental"],
    )
    parser.add_argument(
        "--allow-short-output",
        action="store_true",
        help="Allow early-stop output vectors shorter than --generation-tokens.",
    )
    args = parser.parse_args(argv)

    try:
        doc = build_decode_outputs(
            args.benchmark,
            engine=args.engine,
            context_tokens=args.context_tokens,
            generation_tokens=args.generation_tokens,
            compression_mode=args.compression_mode,
            allow_short_output=args.allow_short_output,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2) + "\n")
    except (OSError, json.JSONDecodeError, DecodeOutputBuildError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ok: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
