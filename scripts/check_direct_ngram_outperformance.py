#!/usr/bin/env python3
"""Gate AX direct and n-gram decode rows against matching mlx_lm rows.

This checker is intentionally narrower than the README provenance checker. It
answers the active performance question directly: for every completed
mlx-inference artifact row, does AX direct decode beat mlx_lm, and does the AX
n-gram row both beat mlx_lm and prove effective accepted-draft throughput?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ax.mlx_inference_stack.v2"
MLX_LM_ENGINE = "mlx_lm"
AX_DIRECT_ENGINE = "ax_engine_mlx"
AX_NGRAM_ENGINE = "ax_engine_mlx_ngram_accel"
NGRAM_EFFECTIVE_STATUS = "ngram_acceleration_effective_throughput"
NGRAM_EFFECTIVE_ROUTE = "ngram_verified_bonus_tokens"


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class RowResult:
    artifact: Path
    prompt_tokens: int
    generation_tokens: int
    mlx_lm_decode_tok_s: float
    direct_decode_tok_s: float
    ngram_decode_tok_s: float
    direct_delta_pct: float
    ngram_delta_pct: float
    ngram_status: str
    ngram_route: str


def metric_median(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, dict):
        value = value.get("median")
    if not isinstance(value, (int, float)):
        raise GateError(f"{row.get('engine')} row lacks numeric {key}.median")
    return float(value)


def row_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("prompt_tokens", -1)), int(row.get("generation_tokens", -1))


def rows_by_engine_and_shape(
    artifact: dict[str, Any],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        engine = row.get("engine")
        if engine not in {MLX_LM_ENGINE, AX_DIRECT_ENGINE, AX_NGRAM_ENGINE}:
            continue
        prompt_tokens, generation_tokens = row_key(row)
        rows[(str(engine), prompt_tokens, generation_tokens)] = row
    return rows


def check_artifact(
    artifact_path: Path,
    *,
    min_delta_pct: float,
    require_effective_ngram: bool,
) -> list[RowResult]:
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"{artifact_path} has unsupported schema_version")

    indexed = rows_by_engine_and_shape(artifact)
    shapes = sorted(
        (prompt, generation)
        for engine, prompt, generation in indexed
        if engine == MLX_LM_ENGINE
    )
    if not shapes:
        raise GateError(f"{artifact_path} has no mlx_lm baseline rows")

    results: list[RowResult] = []
    for prompt_tokens, generation_tokens in shapes:
        baseline = indexed.get((MLX_LM_ENGINE, prompt_tokens, generation_tokens))
        direct = indexed.get((AX_DIRECT_ENGINE, prompt_tokens, generation_tokens))
        ngram = indexed.get((AX_NGRAM_ENGINE, prompt_tokens, generation_tokens))
        if baseline is None or direct is None or ngram is None:
            missing = [
                engine
                for engine, row in (
                    (MLX_LM_ENGINE, baseline),
                    (AX_DIRECT_ENGINE, direct),
                    (AX_NGRAM_ENGINE, ngram),
                )
                if row is None
            ]
            raise GateError(
                f"{artifact_path} prompt={prompt_tokens} gen={generation_tokens} "
                f"missing rows: {', '.join(missing)}"
            )

        baseline_decode = metric_median(baseline, "decode_tok_s")
        direct_decode = metric_median(direct, "decode_tok_s")
        ngram_decode = metric_median(ngram, "decode_tok_s")
        if baseline_decode <= 0:
            raise GateError(
                f"{artifact_path} prompt={prompt_tokens} gen={generation_tokens} "
                "has non-positive mlx_lm decode throughput"
            )
        direct_delta_pct = (direct_decode / baseline_decode - 1.0) * 100.0
        ngram_delta_pct = (ngram_decode / baseline_decode - 1.0) * 100.0
        ngram_status = str(ngram.get("ax_decode_claim_status", ""))
        ngram_route = str(ngram.get("ax_decode_effective_route", ""))

        if direct_delta_pct <= min_delta_pct:
            raise GateError(
                f"{artifact_path.name} prompt={prompt_tokens} direct decode "
                f"{direct_decode:.3f} tok/s is not > mlx_lm {baseline_decode:.3f} "
                f"tok/s by min_delta_pct={min_delta_pct:.2f} "
                f"(delta={direct_delta_pct:+.2f}%)"
            )
        if ngram_delta_pct <= min_delta_pct:
            raise GateError(
                f"{artifact_path.name} prompt={prompt_tokens} n-gram decode "
                f"{ngram_decode:.3f} tok/s is not > mlx_lm {baseline_decode:.3f} "
                f"tok/s by min_delta_pct={min_delta_pct:.2f} "
                f"(delta={ngram_delta_pct:+.2f}%)"
            )
        if require_effective_ngram and (
            ngram_status != NGRAM_EFFECTIVE_STATUS
            or ngram_route != NGRAM_EFFECTIVE_ROUTE
        ):
            raise GateError(
                f"{artifact_path.name} prompt={prompt_tokens} n-gram row is "
                f"not effective throughput: status={ngram_status!r} "
                f"route={ngram_route!r}"
            )

        results.append(
            RowResult(
                artifact=artifact_path,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                mlx_lm_decode_tok_s=baseline_decode,
                direct_decode_tok_s=direct_decode,
                ngram_decode_tok_s=ngram_decode,
                direct_delta_pct=direct_delta_pct,
                ngram_delta_pct=ngram_delta_pct,
                ngram_status=ngram_status,
                ngram_route=ngram_route,
            )
        )
    return results


def check_sweep_results(artifact_dir: Path) -> list[str]:
    sweep_path = artifact_dir / "sweep_results.json"
    if not sweep_path.exists():
        return []
    payload = json.loads(sweep_path.read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GateError(f"{sweep_path} has no rows list")
    failures: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status") != "ok":
            label = row.get("slug") or row.get("label") or "<unknown>"
            reason = row.get("reason") or row.get("error") or row.get("status")
            failures.append(f"{label}: {reason}")
    return failures


def check_artifact_dir(
    artifact_dir: Path,
    *,
    min_delta_pct: float,
    require_effective_ngram: bool,
    require_sweep_ok: bool,
) -> list[RowResult]:
    if not artifact_dir.is_dir():
        raise GateError(f"artifact directory does not exist: {artifact_dir}")
    if require_sweep_ok:
        sweep_failures = check_sweep_results(artifact_dir)
        if sweep_failures:
            raise GateError(
                "sweep_results.json contains non-ok rows:\n  "
                + "\n  ".join(sweep_failures)
            )

    checked: list[RowResult] = []
    artifact_paths = [
        path
        for path in sorted(artifact_dir.glob("*.json"))
        if path.name != "sweep_results.json"
    ]
    if not artifact_paths:
        raise GateError(f"{artifact_dir} has no benchmark artifact JSON files")
    for path in artifact_paths:
        checked.extend(
            check_artifact(
                path,
                min_delta_pct=min_delta_pct,
                require_effective_ngram=require_effective_ngram,
            )
        )
    return checked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check AX direct/ngram decode outperformance against mlx_lm artifacts."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--min-delta-pct",
        type=float,
        default=0.0,
        help="Require AX decode throughput to exceed mlx_lm by this percentage.",
    )
    parser.add_argument(
        "--allow-ngram-fallback",
        action="store_true",
        help=(
            "Only require n-gram row throughput to beat mlx_lm; do not require "
            "accepted-draft effective-throughput status."
        ),
    )
    parser.add_argument(
        "--allow-sweep-skips",
        action="store_true",
        help="Ignore non-ok rows in sweep_results.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        rows = check_artifact_dir(
            args.artifact_dir,
            min_delta_pct=args.min_delta_pct,
            require_effective_ngram=not args.allow_ngram_fallback,
            require_sweep_ok=not args.allow_sweep_skips,
        )
    except GateError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    worst_direct = min(rows, key=lambda row: row.direct_delta_pct)
    worst_ngram = min(rows, key=lambda row: row.ngram_delta_pct)
    print(
        "PASS: checked "
        f"{len(rows)} direct/ngram shapes across {args.artifact_dir}"
    )
    print(
        "worst direct: "
        f"{worst_direct.artifact.name} prompt={worst_direct.prompt_tokens} "
        f"delta={worst_direct.direct_delta_pct:+.2f}%"
    )
    print(
        "worst n-gram: "
        f"{worst_ngram.artifact.name} prompt={worst_ngram.prompt_tokens} "
        f"delta={worst_ngram.ngram_delta_pct:+.2f}% "
        f"status={worst_ngram.ngram_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
