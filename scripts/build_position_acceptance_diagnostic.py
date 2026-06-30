#!/usr/bin/env python3
"""Position-wise conditional acceptance diagnostic (DSpark Figure-2) for AX speculative decode.

For each draft position ``k`` this reports the *conditional* probability that the
draft token at ``k`` is accepted **given the preceding prefix (1..k-1) was
accepted** -- DSpark's "position-wise conditional acceptance" (DSpark paper
§4.3.1, Figure 2). That metric removes the penalty of earlier rejections, unlike
the unconditional ``ax_mtp_accept_rate_depth{d}`` AX already emits
(``accepted_by_depth[d] / drafted_by_depth[d]``), which is depressed at later
positions purely because earlier ones failed. The conditional curve answers the
question prior "perf at floor" work kept needing: is a draft block
*capacity-limited* (low position-1 quality) or *decay-limited* (suffix collapse)?

Zero hot-path risk: a pure offline re-slice of counters the runner already emits.
No decode behaviour changes.

Data sources (route-decision keys; land at
``results[*].ngram_acceleration_telemetry`` in bench artifacts):
  MTP    (runner/mod.rs:1048-1053): ax_mtp_accepted_depth{0,1,2}, ax_mtp_drafted_depth{0,1,2}
  n-gram (runner_telemetry.rs:345-359): ax_ngram_accept_at_depth_{0..7} (accept-count
          histogram: bucket k = attempts that accepted exactly k tokens)

Acceptance is prefix-based (first rejection discards the rest; runner/mod.rs:539-541),
so for MTP:
  accepted_by_depth[d] = #cycles with accept_len > d   (depth d, 0-indexed, accepted)
  drafted_by_depth[d]  = #cycles with draft_len  > d   (depth d proposed at all)

Known limitations (surfaced in the report, not hidden):
  * MTP per-depth counters are hard-capped at 3 (depths 0-2); deeper draft
    positions are invisible even when ax_mtp_max_depth > 3.
  * The MTP per-depth counters merge all draft sources (MTP + n-gram + assistant);
    only aggregate token counts are source-split.
  * Because the confidence gate truncates draft length per cycle, the marginal
    counters bound rather than pin the conditional rate -- we report a rigorous
    [lower, upper] bracket that collapses to an exact value when there was no
    truncation at that depth. See ``mtp_conditional_acceptance``.

Usage:
  python3 scripts/build_position_acceptance_diagnostic.py ARTIFACT.json [ARTIFACT2.json ...]
  python3 scripts/build_position_acceptance_diagnostic.py --output report.json ARTIFACT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

MTP_DEPTHS = 3
NGRAM_BUCKETS = 8

# --------------------------------------------------------------------------- #
# Metric core (schema-independent; unit-tested)
# --------------------------------------------------------------------------- #


def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(1.0, x))


def mtp_conditional_acceptance(
    accepted_by_depth: list[int], drafted_by_depth: list[int]
) -> list[dict[str, Any]]:
    """Conditional acceptance per MTP depth from the survival/draft counters.

    Let, over decode cycles with prefix-based acceptance,
      A[d] = accepted_by_depth[d] = #{accept_len > d}   (depth d accepted)
      D[d] = drafted_by_depth[d]  = #{draft_len  > d}   (depth d proposed)
      N    = D[0]                 = #cycles that drafted at least one token

    The conditional acceptance at depth d is ``A[d] / R[d]`` where the denominator
    ``R[d] = #{prefix 0..d-1 accepted AND depth d drafted} = #{accept_len >= d AND
    draft_len >= d+1}`` -- the cycles that actually *reached* and *proposed* depth d.

    R[d] is an intersection of two events whose marginals we know
    (``prefix = A[d-1]`` and ``drafted = D[d]``), so it is bounded but not pinned:
      max(0, prefix + drafted - N)  <=  R[d]  <=  min(prefix, drafted)
    giving the reported bracket
      cond in [ A[d]/min(prefix, drafted) , A[d]/max(1, prefix + drafted - N) ].
    The bracket collapses (lower == upper) exactly when the draft length did not
    vary at this depth (no gate truncation among prefix-accepted cycles), making
    the conditional acceptance exact.

    DSpark's naive hazard ``A[d]/A[d-1]`` (prefix-survival ratio) is **not** emitted
    for the MTP path: when ``drafted < prefix`` (gate truncation), its denominator
    exceeds the largest feasible ``R[d] = min(prefix, drafted)``, so the point can
    fall *below* ``cond_lower`` and misrepresents the bracket. Consumers must use the
    ``[cond_lower, cond_upper]`` bracket instead.
    """
    n = drafted_by_depth[0] if drafted_by_depth else 0
    rows: list[dict[str, Any]] = []
    depths = min(len(accepted_by_depth), len(drafted_by_depth))
    for d in range(depths):
        a_d = accepted_by_depth[d]
        d_d = drafted_by_depth[d]
        prefix = n if d == 0 else accepted_by_depth[d - 1]
        unconditional = (a_d / d_d) if d_d > 0 else None
        if prefix <= 0:
            rows.append(
                {
                    "position": d + 1,
                    "reached_prefix": prefix,
                    "drafted": d_d,
                    "accepted": a_d,
                    "cond_lower": None,
                    "cond_upper": None,
                    "exact": False,
                    "unconditional": unconditional,
                }
            )
            continue
        lower_denom = min(prefix, d_d)
        upper_denom = max(1, prefix + d_d - n)
        cond_lower = _clamp01(a_d / lower_denom) if lower_denom > 0 else None
        cond_upper = _clamp01(a_d / upper_denom)
        exact = (
            cond_lower is not None
            and cond_upper is not None
            and abs(cond_lower - cond_upper) < 1e-9
        )
        rows.append(
            {
                "position": d + 1,
                "reached_prefix": prefix,
                "drafted": d_d,
                "accepted": a_d,
                "cond_lower": cond_lower,
                "cond_upper": cond_upper,
                "exact": exact,
                "unconditional": unconditional,
            }
        )
    return rows


def ngram_conditional_acceptance(histogram: list[int]) -> list[dict[str, Any]]:
    """Conditional acceptance per position from the n-gram accept-count histogram.

    ``histogram[k]`` = #draft attempts that accepted exactly ``k`` tokens (the last
    bucket saturates: it also holds attempts with >= len-1 accepted). The survival
    ``S[p] = #{accept_count >= p} = sum(histogram[p:])`` gives the prefix-survival
    hazard ``S[p] / S[p-1]``.

    n-gram telemetry records no per-depth draft-length counter, so the truncation
    correction cannot be bracketed here. The hazard is a **lower bound** on the
    true conditional acceptance (attempts whose draft ran out before position ``p``
    inflate the denominator); flagged ``hazard_lower_bound``.
    """
    rows: list[dict[str, Any]] = []
    s_prev = sum(histogram)  # S[0]
    for p in range(1, len(histogram)):
        s_p = sum(histogram[p:])
        cond = _clamp01(s_p / s_prev) if s_prev > 0 else None
        rows.append(
            {
                "position": p,
                "reached_prefix": s_prev,
                "survivors": s_p,
                "cond_point": cond,
                "kind": "hazard_lower_bound",
            }
        )
        s_prev = s_p
    return rows


def mean_accept_length_mtp(accepted_by_depth: list[int], n_cycles: int) -> Optional[float]:
    """Mean accepted-prefix length per cycle = sum_d #{accept_len > d} / N.

    Capped at the 3 tracked depths; undercounts if accepts ran deeper than depth 2.
    """
    if n_cycles <= 0:
        return None
    return sum(accepted_by_depth) / n_cycles


def mean_accept_length_ngram(histogram: list[int]) -> Optional[float]:
    total = sum(histogram)
    if total <= 0:
        return None
    return sum(k * c for k, c in enumerate(histogram)) / total


# --------------------------------------------------------------------------- #
# Artifact IO
# --------------------------------------------------------------------------- #

_SENTINEL_KEYS = (
    "ax_mtp_accepted_depth0",
    "ax_mtp_drafted_depth0",
    "ax_ngram_accept_at_depth_0",
    "ax_ngram_draft_attempts",
    "ax_mtp_accepted_tokens",
)


def _looks_like_route_map(d: dict[str, Any]) -> bool:
    return any(k in d for k in _SENTINEL_KEYS)


def find_route_decisions(obj: Any) -> Optional[dict[str, Any]]:
    """Fallback: locate the first route-decision map anywhere in an artifact."""
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if _looks_like_route_map(node):
                return {k: v for k, v in node.items() if isinstance(v, (int, float))}
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def aggregate_route_maps(maps: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum integer counters across telemetry dicts (one per result row)."""
    agg: dict[str, float] = {}
    for m in maps:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                agg[k] = agg.get(k, 0) + v
    return agg


def extract_ax_artifact(obj: Any) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Return (context, aggregated route map) from a bench artifact.

    Prefers the documented ``results[*].ngram_acceleration_telemetry`` location
    (summed across result rows; trials are NOT descended into, to avoid double
    counting). Falls back to a whole-tree search for other artifact shapes.
    """
    context: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in ("model", "ax_mtp_max_depth", "ax_decode_profile", "schema_version"):
            if key in obj:
                context[key] = obj[key]
        results = obj.get("results")
        if isinstance(results, list):
            per_result = [
                r["ngram_acceleration_telemetry"]
                for r in results
                if isinstance(r, dict)
                and isinstance(r.get("ngram_acceleration_telemetry"), dict)
            ]
            if per_result:
                context["result_rows"] = len(per_result)
                return context, aggregate_route_maps(per_result)

    return context, find_route_decisions(obj)


def _int(route: dict[str, Any], key: str) -> int:
    v = route.get(key, 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def analyze_route_decisions(route: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Build the per-source position-acceptance report from a route-decision map."""
    report: dict[str, Any] = {"context": context, "sources": {}, "warnings": []}

    max_depth = context.get("ax_mtp_max_depth")
    if isinstance(max_depth, int) and max_depth > MTP_DEPTHS:
        report["warnings"].append(
            f"ax_mtp_max_depth={max_depth} exceeds the {MTP_DEPTHS} tracked MTP depths; "
            "positions beyond 3 are not represented in the per-depth counters."
        )

    # --- MTP (Qwen / GLM fused head) -- merges all draft sources by position ---
    mtp_accepted = [_int(route, f"ax_mtp_accepted_depth{d}") for d in range(MTP_DEPTHS)]
    mtp_drafted = [_int(route, f"ax_mtp_drafted_depth{d}") for d in range(MTP_DEPTHS)]
    if any(mtp_drafted):
        n_cycles = mtp_drafted[0]
        computed_mean = mean_accept_length_mtp(mtp_accepted, n_cycles)
        # Cross-check against the artifact's own aggregate accepted/steps.
        decode_steps = _int(route, "ax_mtp_decode_steps")
        accepted_tokens = _int(route, "ax_mtp_accepted_tokens")
        crosscheck_mean = (accepted_tokens / decode_steps) if decode_steps > 0 else None
        report["sources"]["mtp"] = {
            "cycles": n_cycles,
            "mean_accept_length": computed_mean,
            "mean_accept_length_crosscheck": crosscheck_mean,
            "positions": mtp_conditional_acceptance(mtp_accepted, mtp_drafted),
            "accepted_by_depth": mtp_accepted,
            "drafted_by_depth": mtp_drafted,
            "accepted_source_mtp_tokens": _int(route, "ax_mtp_accepted_source_mtp_tokens"),
            "accepted_source_ngram_tokens": _int(route, "ax_mtp_accepted_source_ngram_tokens"),
            "note": "per-depth counters merge all draft sources (MTP+n-gram+assistant).",
        }

    # --- n-gram-only acceptance histogram ---
    histogram = [_int(route, f"ax_ngram_accept_at_depth_{k}") for k in range(NGRAM_BUCKETS)]
    if any(histogram):
        report["sources"]["ngram"] = {
            "attempts": sum(histogram),
            "mean_accept_length": mean_accept_length_ngram(histogram),
            "positions": ngram_conditional_acceptance(histogram),
            "accept_count_histogram": histogram,
        }

    # --- Gemma4 assistant MTP: only aggregate counters exist (no per-depth) ---
    g_draft = _int(route, "ax_mlx_gemma4_assistant_mtp_draft_tokens")
    g_accept = _int(route, "ax_mlx_gemma4_assistant_mtp_accepted_tokens")
    if g_draft > 0:
        report["sources"]["gemma4_assistant_mtp"] = {
            "draft_tokens": g_draft,
            "accepted_tokens": g_accept,
            "mean_accept_rate": g_accept / g_draft if g_draft else None,
            "positions": None,
            "note": (
                "no per-position curve: Gemma4AssistantMtpTelemetry records only "
                "aggregate draft/accepted tokens. Add an accepted/drafted_by_depth "
                "histogram (mirror NgramAccelerationTelemetry.accepts_by_depth) to enable."
            ),
        }

    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_pct(x: Optional[float]) -> str:
    return "   n/a" if x is None else f"{x * 100:5.1f}%"


def render_text(label: str, report: dict[str, Any]) -> str:
    lines = [f"=== {label} ==="]
    ctx = report.get("context", {})
    ctx_bits = [
        f"{k}={ctx[k]}"
        for k in ("model", "ax_mtp_max_depth", "ax_decode_profile", "result_rows")
        if k in ctx
    ]
    if ctx_bits:
        lines.append("  " + "  ".join(ctx_bits))
    for w in report.get("warnings", []):
        lines.append(f"  ! {w}")

    sources = report.get("sources", {})
    if not sources:
        lines.append("  (no speculative-decode counters found)")
        return "\n".join(lines)

    mtp = sources.get("mtp")
    if mtp:
        mean = mtp["mean_accept_length"]
        cross = mtp["mean_accept_length_crosscheck"]
        mean_s = f"{mean:.3f}" if mean is not None else "n/a"
        cross_s = f"{cross:.3f}" if cross is not None else "n/a"
        lines.append(
            f"  MTP  cycles={mtp['cycles']}  mean_accept_len={mean_s}  "
            f"(crosscheck accepted/steps={cross_s})"
        )
        lines.append(
            "    pos  conditional-accept (given prefix)   uncond   reached  drafted"
        )
        for r in mtp["positions"]:
            if r["cond_lower"] is None:
                if r["reached_prefix"] and r["reached_prefix"] > 0 and r["drafted"] == 0:
                    cell = "       (not drafted)         "
                else:
                    cell = "     (no prefix-accepted)    "
            elif r["exact"]:
                cell = f"  {_fmt_pct(r['cond_lower'])} (exact)            "
            else:
                cell = f"  [{_fmt_pct(r['cond_lower'])}, {_fmt_pct(r['cond_upper'])}]      "
            lines.append(
                f"    {r['position']:>3}  {cell}  {_fmt_pct(r['unconditional'])}  "
                f"{r['reached_prefix']:>7}  {r['drafted']:>7}"
            )
        lines.append(f"    note: {mtp['note']}")

    ng = sources.get("ngram")
    if ng:
        ml = ng["mean_accept_length"]
        ml_s = f"{ml:.3f}" if ml is not None else "n/a"
        lines.append(
            f"  n-gram  attempts={ng['attempts']}  mean_accept_len={ml_s}   "
            "(conditional = hazard, lower bound)"
        )
        lines.append("    pos  conditional-accept   reached  survivors")
        for r in ng["positions"]:
            lines.append(
                f"    {r['position']:>3}  {_fmt_pct(r['cond_point'])}              "
                f"{r['reached_prefix']:>7}  {r['survivors']:>7}"
            )

    g = sources.get("gemma4_assistant_mtp")
    if g:
        lines.append(
            f"  gemma4-assistant-mtp  draft={g['draft_tokens']} accepted={g['accepted_tokens']} "
            f"mean_accept_rate={_fmt_pct(g['mean_accept_rate'])}"
        )
        lines.append(f"    note: {g['note']}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Position-wise conditional acceptance diagnostic (DSpark Fig-2) for AX "
            "speculative decode. Pure offline re-slice of existing route-decision counters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("artifacts", nargs="+", type=Path, help="bench artifact JSON file(s)")
    parser.add_argument("--output", type=Path, help="write the structured report to this JSON path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    full_report: dict[str, Any] = {}
    exit_code = 0
    for path in args.artifacts:
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"=== {path} ===\n  ERROR: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        context, route = extract_ax_artifact(obj)
        if not route:
            report = {"context": context, "sources": {}, "warnings": []}
        else:
            report = analyze_route_decisions(route, context)
        full_report[str(path)] = report
        print(render_text(str(path), report))
        print()

    if args.output:
        Path(args.output).write_text(
            json.dumps(full_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
