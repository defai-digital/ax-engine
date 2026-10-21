#!/usr/bin/env python3
"""Offline verify-prefix simulator for MTP step traces.

Reads `AX_MLX_MTP_STEP_TRACE` JSONL captures and prices draft-truncation
policies before any runtime policy change:

* cost traces (one per fixed draft depth, production path) give the measured
  cycle wall per verify width;
* a calibration trace (non-zero draft gate, so the draft head's log-probs are
  recorded) gives per-cycle head confidences and the verifier's accepted
  prefix.

Greedy verification accepts a prefix independently of how many later drafts
were submitted, so a cycle that accepted `a` of `D` drafts emits
`min(a, k) + 1` tokens when only `k` drafts are verified. Truncating after
drafting does not refund the drafts already generated: the simulated cost of
verifying `k` of `D` drafts is `cost(k) + (D - k) * draft_cost`, where
`cost(k)` is the measured fixed-depth-`k` cycle (which already includes `k`
drafts).

Policies: `full` (verify every draft), `threshold:<t>` (verify the leading run
whose head probability is >= t), `cost` (choose `k` maximizing expected
tokens / cost with acceptance calibrated from head probability, cross-fitted
by request so a cycle is never scored by a model fitted on its own request),
and `oracle` (verify exactly the accepted prefix; an upper bound).

Deterministic: no randomness, stable ordering, stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BIN_EDGES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0000001)
DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class Cycle:
    request: int
    t_us: int
    ctx: int
    generated: int
    depth: int
    used: int
    accepted: int
    pure_mtp: bool
    temperature: float
    draft_us: int
    probs: tuple[float, ...]
    wall_us: int | None = None
    # False when any recorded log-prob was null (non-finite in the runtime).
    probs_finite: bool = True


def load_trace(path: Path) -> list[Cycle]:
    """Parse a trace and assign request ids and per-cycle walls.

    A record is written at the end of its cycle, after the next draft was
    built, so the gap to the previous record spans this cycle's verify plus
    the following draft; under a fixed draft depth that is one consistent
    cycle cost per verify width. A new request starts when the generated
    count does not increase. The wall is kept only when the two records are
    contiguous (the previous cycle emitted exactly `accepted + 1` tokens and
    nothing else ran in between): a non-speculative step in the gap writes
    no record and would otherwise be absorbed into this cycle's wall.
    """
    raw: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {error}") from error
    cycles: list[Cycle] = []
    request = -1
    previous: dict | None = None
    for record in raw:
        new_request = previous is None or int(record["generated"]) <= int(previous["generated"])
        wall: int | None = None
        if new_request:
            request += 1
        else:
            assert previous is not None
            contiguous = int(record["generated"]) == (
                int(previous["generated"]) + int(previous["accepted"]) + 1
            )
            if contiguous:
                wall = int(record["t_us"]) - int(previous["t_us"])
        log_probs = record.get("draft_log_probs", [])
        probs = tuple(math.exp(float(v)) if v is not None else 0.0 for v in log_probs)
        probs_finite = all(v is not None for v in log_probs)
        cycles.append(
            Cycle(
                request=request,
                t_us=int(record["t_us"]),
                ctx=int(record["ctx"]),
                generated=int(record["generated"]),
                depth=int(record["depth"]),
                used=int(record["used"]),
                accepted=int(record["accepted"]),
                pure_mtp=bool(record["pure_mtp"]),
                temperature=float(record.get("temperature") or 0.0),
                draft_us=int(record.get("draft_us", 0)),
                probs=probs,
                wall_us=wall,
                probs_finite=probs_finite,
            )
        )
        previous = record
    return cycles


def build_cost_table(cost_traces: Sequence[Sequence[Cycle]]) -> dict[int, float]:
    """Median cycle wall per verify width over all cost traces."""
    table: dict[int, float] = {}
    pooled: dict[int, list[int]] = {}
    for cycles in cost_traces:
        for cycle in cycles:
            if cycle.pure_mtp and cycle.wall_us is not None and cycle.wall_us > 0:
                pooled.setdefault(cycle.used, []).append(cycle.wall_us)
    for used, walls in sorted(pooled.items()):
        table[used] = float(statistics.median(walls))
    return table


def bin_index(prob: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= prob < edges[index + 1]:
            return index
    return len(edges) - 2


@dataclass
class Calibration:
    """Conditional acceptance per (position, probability bin), Beta(1,1) prior."""

    edges: Sequence[float]
    max_position: int
    accepts: dict[tuple[int, int], int] = field(default_factory=dict)
    trials: dict[tuple[int, int], int] = field(default_factory=dict)

    def key(self, position: int, prob: float) -> tuple[int, int]:
        return (min(position, self.max_position - 1), bin_index(prob, self.edges))

    def observe(self, cycle: Cycle) -> None:
        # Position i is observed only when every earlier draft was accepted;
        # positions past the first rejection carry no evidence.
        for position, prob in enumerate(cycle.probs[: cycle.used]):
            key = self.key(position, prob)
            self.trials[key] = self.trials.get(key, 0) + 1
            if position < cycle.accepted:
                self.accepts[key] = self.accepts.get(key, 0) + 1
            else:
                break

    def rate(self, position: int, prob: float) -> float:
        key = self.key(position, prob)
        return (self.accepts.get(key, 0) + 1.0) / (self.trials.get(key, 0) + 2.0)


def truncated_cost(k: int, depth: int, cost: dict[int, float], draft_cost_us: float) -> float:
    if k not in cost:
        raise KeyError(f"no measured cycle cost for verify width {k}")
    return cost[k] + max(depth - k, 0) * draft_cost_us


def cost_optimal_k(
    probs: Sequence[float],
    depth: int,
    calibration: Calibration,
    cost: dict[int, float],
    draft_cost_us: float,
) -> int:
    best_k = 0
    best_rate = -1.0
    survival = 1.0
    expected = 1.0
    for k in range(0, len(probs) + 1):
        if k > 0:
            survival *= calibration.rate(k - 1, probs[k - 1])
            expected += survival
        if k not in cost:
            continue
        rate = expected / truncated_cost(k, depth, cost, draft_cost_us)
        if rate > best_rate:
            best_rate = rate
            best_k = k
    return best_k


def threshold_k(probs: Sequence[float], threshold: float) -> int:
    k = 0
    for prob in probs:
        if prob < threshold:
            break
        k += 1
    return k


@dataclass
class PolicyResult:
    name: str
    tokens: int = 0
    wall_us: float = 0.0
    verified: int = 0
    cycles: int = 0

    @property
    def tok_per_s(self) -> float:
        return self.tokens / (self.wall_us / 1e6) if self.wall_us > 0 else 0.0


def eligible_cycles(cycles: Iterable[Cycle]) -> list[Cycle]:
    return [
        c
        for c in cycles
        if c.pure_mtp
        and c.temperature == 0.0
        and c.used >= 1
        and len(c.probs) >= c.used
        and c.probs_finite
    ]


def simulate(
    calibration_cycles: Sequence[Cycle],
    cost: dict[int, float],
    draft_cost_us: float,
    thresholds: Sequence[float],
    edges: Sequence[float],
) -> list[PolicyResult]:
    cycles = eligible_cycles(calibration_cycles)
    max_position = max((c.used for c in cycles), default=1)
    request_ids = sorted({c.request for c in cycles})
    fold_of = {request: index % 2 for index, request in enumerate(request_ids)}
    if len(fold_of) < 2:
        raise ValueError("cross-fitting needs at least two requests in the calibration trace")
    # Two-fold cross-fitting by request order.
    folds = {parity: Calibration(edges=edges, max_position=max_position) for parity in (0, 1)}
    for cycle in cycles:
        folds[fold_of[cycle.request]].observe(cycle)

    results = {
        "full": PolicyResult("full"),
        "cost": PolicyResult("cost"),
        "oracle": PolicyResult("oracle"),
    }
    for threshold in thresholds:
        results[f"threshold:{threshold:g}"] = PolicyResult(f"threshold:{threshold:g}")

    for cycle in cycles:
        if cycle.used not in cost:
            continue
        held_out = folds[1 - fold_of[cycle.request]]
        probs = cycle.probs[: cycle.used]
        accepted = min(cycle.accepted, cycle.used)
        # Every width >= accepted emits the same tokens; the oracle takes the
        # cheapest of them, so it can never score below `full`.
        oracle_k = min(
            (k for k in range(accepted, cycle.used + 1) if k in cost),
            key=lambda k: (truncated_cost(k, cycle.used, cost, draft_cost_us), k),
        )
        choices = {
            "full": cycle.used,
            "cost": cost_optimal_k(probs, cycle.used, held_out, cost, draft_cost_us),
            "oracle": oracle_k,
        }
        for threshold in thresholds:
            choices[f"threshold:{threshold:g}"] = threshold_k(probs, threshold)
        for name, k in choices.items():
            if k not in cost:
                # Unpriced width (e.g. k = 0 without a direct-step cost):
                # fall back to the smallest priced width that verifies >= k.
                priced = [w for w in sorted(cost) if w >= k and w <= cycle.used]
                k = priced[0] if priced else cycle.used
            result = results[name]
            result.tokens += min(cycle.accepted, k) + 1
            result.wall_us += truncated_cost(k, cycle.used, cost, draft_cost_us)
            result.verified += k
            result.cycles += 1
    return list(results.values())


def calibration_table(cycles: Sequence[Cycle], edges: Sequence[float]) -> list[dict]:
    eligible = eligible_cycles(cycles)
    max_position = max((c.used for c in eligible), default=1)
    calibration = Calibration(edges=edges, max_position=max_position)
    for cycle in eligible:
        calibration.observe(cycle)
    rows = []
    for (position, index), trials in sorted(calibration.trials.items()):
        accepts = calibration.accepts.get((position, index), 0)
        rows.append(
            {
                "position": position + 1,
                "bin": [edges[index], min(edges[index + 1], 1.0)],
                "trials": trials,
                "accept_rate": accepts / trials if trials else 0.0,
            }
        )
    return rows


def parse_cost_arg(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"cost trace not found: {value}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument(
        "--cost-trace",
        type=parse_cost_arg,
        action="append",
        required=True,
        help="Fixed-depth production-path trace; repeat for each depth.",
    )
    parser.add_argument(
        "--direct-step-us",
        type=float,
        default=None,
        help="Measured direct single-token step wall; enables k = 0 (no verify).",
    )
    parser.add_argument(
        "--draft-cost-us",
        type=float,
        default=None,
        help="Per-depth draft cost; default = median(draft_us / depth) of the calibration trace.",
    )
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    calibration_cycles = load_trace(args.calibration_trace)
    cost = build_cost_table([load_trace(path) for path in args.cost_trace])
    if args.direct_step_us is not None:
        if args.direct_step_us <= 0:
            parser.error("--direct-step-us must be > 0")
        cost[0] = float(args.direct_step_us)
    if args.draft_cost_us is not None:
        draft_cost_us = float(args.draft_cost_us)
    else:
        per_depth = [
            c.draft_us / c.depth
            for c in calibration_cycles
            if c.depth > 0 and c.draft_us > 0 and c.pure_mtp
        ]
        draft_cost_us = float(statistics.median(per_depth)) if per_depth else 0.0

    if args.draft_cost_us is not None and args.draft_cost_us < 0:
        parser.error("--draft-cost-us must be >= 0")
    try:
        results = simulate(
            calibration_cycles, cost, draft_cost_us, args.thresholds, DEFAULT_BIN_EDGES
        )
    except ValueError as error:
        parser.error(str(error))
    baseline = next(r for r in results if r.name == "full")
    if baseline.cycles == 0:
        parser.error("no eligible calibration cycle has a priced verify width")
    report = {
        "calibration_trace": str(args.calibration_trace),
        "cost_traces": [str(p) for p in args.cost_trace],
        "cost_us_by_verify_width": {str(k): v for k, v in sorted(cost.items())},
        "draft_cost_us_per_depth": draft_cost_us,
        "eligible_cycles": baseline.cycles,
        "policies": [
            {
                "name": r.name,
                "cycles": r.cycles,
                "tokens": r.tokens,
                "verified_drafts": r.verified,
                "tok_per_s": round(r.tok_per_s, 3),
                "vs_full": (
                    round(r.tok_per_s / baseline.tok_per_s, 4) if baseline.tok_per_s else None
                ),
            }
            for r in results
        ],
        "calibration": calibration_table(calibration_cycles, DEFAULT_BIN_EDGES),
    }
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
