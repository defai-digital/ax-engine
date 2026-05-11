#!/usr/bin/env python3
"""Aggregate multiple `ax.bw_profile.v1` artifacts into a single variance summary.

Formalizes the variance-characterization protocol established in
`DS4-REFERENCE-LEARNINGS-PRD.md` Phase 1: run `profile_decode_bandwidth.py`
N≥3 times sequentially on an idle machine with distinct `--run-tag`, then
aggregate. CV < 5% confirms a clean reading; larger CV indicates GPU
contention, thermal effects, or a model behaviour worth investigating.

Output schema: ax.bw_profile_variance.v1

    {
      "schema_version": "ax.bw_profile_variance.v1",
      "generated_at_utc": "...",
      "input_artifacts": [list of paths],
      "model": { "model_id": str, "weight_bytes_on_disk": int },
      "host": { "platform": str, "peak_bandwidth_gbps": float },
      "n_runs": int,
      "summary": {
        "<metric>": { "mean": float, "stddev": float, "cv_pct": float,
                       "min": float, "max": float, "values": [float, ...] }
      },
      "regime_summary": {
        "single_decode": { "<metric>": <stats>, ... } | null,
        "ngram_decode": { "<metric>": <stats>, ... } | null
      },
      "advisory": {
        "max_cv_pct": float,
        "high_variance_metrics": [<metric>, ...],   # CV > 5%
        "verdict": "tight" | "loose" | "polluted",
        "verdict_explanation": str
      }
    }

Usage:
    python scripts/aggregate_bw_variance.py \\
        benchmarks/results/bw-profile-variance/qwen3_dense-2026-05-11-run*.json \\
        --output benchmarks/results/bw-profile-variance/qwen3_dense-2026-05-11-summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "ax.bw_profile_variance.v1"
HIGH_VARIANCE_THRESHOLD_PCT = 5.0
NUMERIC_TOP_LEVEL_METRICS = [
    "decode_tokens_observed",
    "speculation_bonus_tokens",
    "forward_pass_count",
    "decode_wall_time_s",
    "decode_tok_s",
    "forward_pass_per_s",
    "mean_forward_pass_us",
    "median_step_event_us",
    "p95_step_event_us",
    "estimated_effective_bandwidth_gbps",
    "bandwidth_utilization_ratio",
]
REGIME_METRICS = [
    "steps",
    "wall_us",
    "us_per_step",
    "effective_bandwidth_gbps",
    "utilization_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("artifacts", nargs="+", type=Path, help="Input artifact JSON files (≥2).")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: prints to stdout only.",
    )
    return p.parse_args()


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    cv = (stdev / mean * 100) if mean != 0 else 0.0
    return {
        "mean": round(mean, 6),
        "stddev": round(stdev, 6),
        "cv_pct": round(cv, 3),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "values": [round(v, 6) for v in values],
    }


def collect_metrics(records: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = []
        for r in records:
            v = r.get(k)
            if v is None or isinstance(v, bool):
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if vals:
            out[k] = summarise(vals)
    return out


def aggregate(artifacts: list[Path]) -> dict:
    if len(artifacts) < 2:
        raise SystemExit("Need at least 2 artifacts to compute variance.")
    loaded = []
    for path in artifacts:
        if not path.is_file():
            raise SystemExit(f"Artifact not found: {path}")
        loaded.append(json.loads(path.read_text()))

    # Sanity: all artifacts must describe the same model + host.
    first_model = loaded[0]["model"]["model_id"]
    first_peak = loaded[0]["host"]["peak_bandwidth_gbps"]
    for i, a in enumerate(loaded[1:], 1):
        if a["model"]["model_id"] != first_model:
            raise SystemExit(
                f"Artifact {artifacts[i]} has model_id={a['model']['model_id']!r}; "
                f"expected {first_model!r}. Aggregation requires identical model."
            )
        if a["host"]["peak_bandwidth_gbps"] != first_peak:
            raise SystemExit(
                f"Artifact {artifacts[i]} has peak_bandwidth_gbps mismatch; "
                f"got {a['host']['peak_bandwidth_gbps']}, expected {first_peak}."
            )

    measurements = [a["measurements"] for a in loaded]
    summary = collect_metrics(measurements, NUMERIC_TOP_LEVEL_METRICS)

    def regime_records(name: str) -> list[dict] | None:
        records = []
        for a in loaded:
            rb = a.get("classification", {}).get("regime_breakdown", {})
            r = rb.get(name)
            if r is not None:
                records.append(r)
        return records if len(records) == len(loaded) else None

    single_records = regime_records("single_decode")
    ngram_records = regime_records("ngram_decode")
    regime_summary = {
        "single_decode": collect_metrics(single_records, REGIME_METRICS) if single_records else None,
        "ngram_decode": collect_metrics(ngram_records, REGIME_METRICS) if ngram_records else None,
    }

    all_cvs = []
    high_var = []
    for k, stats in summary.items():
        all_cvs.append(stats["cv_pct"])
        if stats["cv_pct"] > HIGH_VARIANCE_THRESHOLD_PCT:
            high_var.append(k)
    for regime_name, regime_stats in regime_summary.items():
        if not regime_stats:
            continue
        for k, stats in regime_stats.items():
            all_cvs.append(stats["cv_pct"])
            if stats["cv_pct"] > HIGH_VARIANCE_THRESHOLD_PCT:
                high_var.append(f"{regime_name}.{k}")

    max_cv = max(all_cvs) if all_cvs else 0.0
    if max_cv > 25:
        verdict = "polluted"
        explanation = (
            "CV > 25% on at least one metric. Likely cause: GPU contention "
            "(another process holding the GPU), thermal throttling, or a "
            "fundamental model behaviour change between runs. Re-run on an "
            "idle machine before promoting any classification."
        )
    elif max_cv > HIGH_VARIANCE_THRESHOLD_PCT:
        verdict = "loose"
        explanation = (
            f"CV between {HIGH_VARIANCE_THRESHOLD_PCT}% and 25% on some "
            "metrics. Acceptable for first-pass evidence but tighter "
            "measurement preferred before ADR-grade claims."
        )
    else:
        verdict = "tight"
        explanation = (
            f"All metrics CV ≤ {HIGH_VARIANCE_THRESHOLD_PCT}%. Variance is "
            "well-bounded; means are reproducible. Suitable for "
            "evidence-grade classification."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_artifacts": [str(p) for p in artifacts],
        "model": {
            "model_id": first_model,
            "weight_bytes_on_disk": loaded[0]["model"].get("weight_bytes_on_disk"),
        },
        "host": {
            "platform": loaded[0]["host"]["platform"],
            "peak_bandwidth_gbps": first_peak,
        },
        "n_runs": len(loaded),
        "summary": summary,
        "regime_summary": regime_summary,
        "advisory": {
            "max_cv_pct": round(max_cv, 3),
            "high_variance_metrics": sorted(high_var),
            "verdict": verdict,
            "verdict_explanation": explanation,
        },
    }


def main() -> int:
    args = parse_args()
    result = aggregate(args.artifacts)
    output_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"wrote {args.output}")
    print(
        f"  model: {result['model']['model_id']}  n_runs: {result['n_runs']}  "
        f"verdict: {result['advisory']['verdict']} (max CV {result['advisory']['max_cv_pct']}%)"
    )
    if result["advisory"]["high_variance_metrics"]:
        print(
            f"  high-variance metrics: {', '.join(result['advisory']['high_variance_metrics'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
