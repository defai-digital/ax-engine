"""Sweep decode repayment while keeping the observed Flash Next prefill cuts."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile

ROOT = Path(__file__).resolve().parent
DEBT = ROOT.parent / "flash-next-prefill-debt-replay-m5-20261122" / "replay.py"
ARCHIVE = (ROOT.parent / "flash-next-mxfp4-prefill-step-cost-m5-20261121"
           / "evidence.tar.xz")


def sweep(fixed, joint, own, model, *, include_grid=False):
    steps_available = model["GAPS"]
    percentile = model["percentile"]
    project = model["project"]
    require = model["require"]
    baseline_long_us = sum(fixed)
    report = {}
    for quantile in model["VARY_DECODE"]:
        pure = round(percentile(own, quantile))
        shared = round(percentile(joint, quantile))
        rows = []
        for share in range(101):
            short = project(fixed, shared, pure, share)
            # There is no repayment after the final slice. The ceiling
            # counts whole decode-only turns while the next slice waits.
            repayment_turns = sum((cost * share // 100 + pure - 1) // pure for cost in fixed[:-1])
            long_emission_step = len(fixed) + repayment_turns
            finishes = long_emission_step <= steps_available
            require(finishes == (short["remaining_prefill_slices"] == 0),
                    "First-token schedule and short-path replay disagree")
            rows.append({
                "repayment_percent": share,
                "short_runner_gap_p95_us": short["short_runner_gap_p95_us"],
                "short_runner_gap_max_us": short["short_runner_gap_max_us"],
                "short_runner_gaps_at_least_100ms": short["short_runner_gaps_at_least_100ms"],
                "unstarted_long_slices_at_short_completion": short["remaining_prefill_slices"],
                "long_first_token_added_runner_us_before_short_completion": (
                    repayment_turns * pure if finishes else None
                ),
            })
        baseline = rows[0]
        require(baseline["long_first_token_added_runner_us_before_short_completion"] == 0,
                "Zero-repayment replay does not match the observed order")
        completes = [row for row in rows if row["unstarted_long_slices_at_short_completion"] == 0]
        improvements = [row for row in rows if row["short_runner_gap_p95_us"]
                        < baseline["short_runner_gap_p95_us"]]
        report[f"p{round(100 * quantile)}"] = {
            "decode_only_runner_us": pure,
            "shared_decode_runner_us": shared,
            "zero_repayment": baseline,
            "largest_repayment_finishing_long_before_short_ends": completes[-1],
            "first_short_p95_improvement": improvements[0] if improvements else None,
            "short_p95_improvements_finishing_long_before_short_ends": sum(
                row["unstarted_long_slices_at_short_completion"] == 0 for row in improvements
            ),
        }
        if include_grid:
            report[f"p{round(100 * quantile)}"]["fixed_grid"] = rows
    return {"observed_runner_time_from_first_short_to_first_long_us": baseline_long_us,
            "decode_sensitivity": report}


def replay(*, include_grid=False):
    model = runpy.run_path(str(DEBT))
    require = model["require"]
    original = model["replay"](ARCHIVE)
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    require(original["input_archive_sha256"] == digest, "The existing archive is not bound")
    with tarfile.open(ARCHIVE) as archive:
        early = model["load_burst"](archive, "early-decoder")
        late = model["load_burst"](archive, "ending-decoder")
    fixed, joint, own, _ = model["costs"](early, late)
    short, long = early
    require(long["decode_step_ids"][0] == long["prefill"][-1]["step_id"]
            == short["decode_step_ids"][len(fixed)], "Long first-token step shifted")
    return {"scope": "Fixed-cost M5 scheduling projection with observed physical widths",
            "input_archive_sha256": digest,
            "fixed_step_widths": model["PHYSICAL_PLAN"],
            "grid_rows": 101 * len(model["VARY_DECODE"]),
            **sweep(fixed, joint, own, model, include_grid=include_grid)}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
