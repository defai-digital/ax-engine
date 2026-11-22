"""Validate and summarize Flash Next's fixed-width resident stage profile."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

WIDTHS = (512, 512, 512, 511, 1) * 4


def checked_time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative measurement")
    return value


def audit(result):
    if result["state_and_final_logits_exact"] is not True:
        raise ValueError("Attribution changed the baseline result")
    rows = result["chunks"]
    if tuple(row["tokens"] for row in rows) != WIDTHS:
        raise ValueError("Physical forwards differ from the archived geometry")
    cursor = 0
    times = {"control_seconds": [], "traced_seconds": []}
    totals = {}
    cache_only_stateless = []
    full_width_control = []
    for index, row in enumerate(rows):
        if row["offset"] != cursor or row["cache_only"] is not (index < len(rows) - 1):
            raise ValueError(f"Unexpected forward boundary or logits selection at index {index}")
        cursor += row["tokens"]
        control = checked_time(row["wall_seconds"])
        traced = checked_time(row["traced_wall_seconds"])
        times["control_seconds"].append(control)
        times["traced_seconds"].append(traced)
        stage_total = 0.0
        for stage, cost in row["stages"].items():
            valid = checked_time(cost)
            totals[stage] = totals.get(stage, 0.0) + valid
            stage_total += valid
        terminal = checked_time(row["terminal_stateless_stages_seconds"])
        if (not control or not traced or not stage_total or terminal > stage_total
                or stage_total > traced):
            raise ValueError(f"Recorded stages exceed the forward at index {index}")
        if row["cache_only"]:
            cache_only_stateless.append(terminal)
        if row["tokens"] in (511, 512):
            full_width_control.append(control)
    if totals.keys() != result["traced_stages"].keys() or any(
            not math.isclose(total, checked_time(result["traced_stages"][stage]), abs_tol=1e-9)
            for stage, total in totals.items()):
        raise ValueError("Aggregate stages disagree with the forward records")
    for name, chunks in times.items():
        if sum(chunks) > checked_time(result[name]):
            raise ValueError(f"Forward times exceed {name}")
    measured = sum(cache_only_stateless)
    return {"geometry": WIDTHS, "largest_synchronized_stages_seconds":
            dict(sorted(totals.items(), key=lambda item: item[1], reverse=True)[:8]),
            "wide_forward_median_control_seconds": median(full_width_control),
            "cache_only_terminal_stateless_seconds": measured,
            "cache_only_terminal_stateless_share_of_traced_wall": measured / result["traced_seconds"],
            "control_seconds": result["control_seconds"], "traced_seconds": result["traced_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.receipt.read_text())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
