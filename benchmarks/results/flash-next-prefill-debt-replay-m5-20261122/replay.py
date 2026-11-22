"""Project fixed-slice decode repayment from the captured M5 runner steps."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import tarfile

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-mxfp4-prefill-step-cost-m5-20261121"
VARY_DECODE = (0.1, 0.5, 0.9)
SHARES = (0, 25, 50, 75, 100)
PHYSICAL_PLAN = (512, 512, 512, 511, 1) * 4
GAP_THRESHOLD_US = 100_000
GAPS = 95


def percentile(values, q):
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    index = int(position)
    weight = position - index
    return (sorted_values[index] * (1 - weight)
            + sorted_values[min(index + 1, len(sorted_values) - 1)] * weight)


def valid_time(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Runner time must be a positive integer")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_burst(archive, label):
    path = f"{label}-steps.json"
    result = archive.extractfile(path)
    if result is None:
        raise ValueError(f"Missing {path}")
    rows = json.load(result)
    require(len(rows) == 2, f"Expected two requests in {label}")
    short, long = sorted(rows, key=lambda row: row["cursor"])
    require((short["cursor"], long["cursor"]) == (512, 8192),
            f"Unexpected prompt cursors in {label}")
    require(all(row["initial_cursor"] == 0 and len(row["decode_step_ids"]) == GAPS + 1
                and len(set(row["decode_step_ids"])) == GAPS + 1 for row in rows),
            f"Unexpected decode events or restored prompt in {label}")
    return short, long


def step_index(trace):
    result = {row["step_id"]: row for row in trace["step_costs"]}
    require(len(result) == len(trace["step_costs"]), "Duplicate runner step IDs")
    return result


def costs(early, late):
    short, long = early
    slices = long["prefill"]
    require(tuple(row["width"] for row in slices) == PHYSICAL_PLAN,
            "Early burst does not preserve the physical prefill plan")
    short_steps, long_steps = step_index(short), step_index(long)
    first = valid_time(short_steps[short["decode_step_ids"][0]]["runner_time_us"])
    fixed = []
    for i, slice_ in enumerate(slices):
        step = slice_["step_id"]
        a, b = short_steps[step], long_steps[step]
        require(step == short["decode_step_ids"][i + 1]
                and slice_["split_end"] == b["split_end"] == (i // 5 + 1) * 2048
                and slice_["width"] == b["prefill_tokens"] and not a["prefill_tokens"]
                and a["emitted_tokens"] == 1
                and a["scheduled_requests"] == b["scheduled_requests"] == 2,
                f"Mixed runner step {step} does not match the prefill plan")
        time = valid_time(a["runner_time_us"])
        require(time == b["runner_time_us"], f"Mixed runner time disagrees at step {step}")
        fixed.append(time)
    joint = []
    for step in short["decode_step_ids"][len(fixed) + 1:]:
        a, b = short_steps[step], long_steps[step]
        require(not (a["prefill_tokens"] or b["prefill_tokens"])
                and a["emitted_tokens"] == b["emitted_tokens"] == 1
                and a["scheduled_requests"] == b["scheduled_requests"] == 2
                and a["runner_time_us"] == b["runner_time_us"],
                f"Expected consistent two-decoder data at step {step}")
        joint.append(valid_time(a["runner_time_us"]))
    late_short = late[0]
    late_steps = step_index(late_short)
    own = []
    for step in late_short["decode_step_ids"][1:]:
        row = late_steps[step]
        require(row["scheduled_requests"] == row["emitted_tokens"] == 1
                and row["prefill_tokens"] == 0,
                f"Expected decode-only data at late-request step {step}")
        own.append(valid_time(row["runner_time_us"]))
    return fixed, joint, own, first


def project(fixed, shared_decode_us, pure_decode_us, share):
    """Account only for runner-time gaps up to the short completion."""
    remaining = iter(fixed)
    head = next(remaining, None)
    debt_us = 0
    gaps = []
    consumed = 0
    for _ in range(GAPS):
        if head is None:
            spent = shared_decode_us
        elif debt_us:
            spent = pure_decode_us
            debt_us = max(0, debt_us - spent)
        else:
            spent = head
            consumed += 1
            debt_us = spent * share // 100
            head = next(remaining, None)
        gaps.append(spent)
    return {"mixed_slices_before_short_completion": consumed,
            "remaining_prefill_slices": len(fixed) - consumed,
            "short_runner_gap_max_us": max(gaps),
            "short_runner_gap_p95_us": percentile(gaps, 0.95),
            "short_runner_gaps_at_least_100ms": sum(gap >= GAP_THRESHOLD_US for gap in gaps),
            "runner_time_after_first_short_token_us": sum(gaps)}


def replay(path=PRIOR / "evidence.tar.xz"):
    checksums = {name: value for value, name in
                 (row.split() for row in (PRIOR / "SHA256SUMS").read_text().splitlines())}
    archive_hash = sha256(path.read_bytes()).hexdigest()
    require(archive_hash == checksums["evidence.tar.xz"], "Input archive checksum mismatch")
    with tarfile.open(path) as archive:
        early = load_burst(archive, "early-decoder")
        late = load_burst(archive, "ending-decoder")
        fixed, joint, own, first = costs(early, late)
    baseline = fixed + joint
    rows = {}
    for q in VARY_DECODE:
        single = round(percentile(own, q))
        double = round(percentile(joint, q))
        scenarios = {share: project(fixed, double, single, share) for share in SHARES}
        require(scenarios[0]["mixed_slices_before_short_completion"] == len(fixed),
                "Zero-repayment projection did not finish the physical prefill plan")
        rows[f"p{round(q * 100)}"] = {"one_decoder_us": single, "two_decoders_us": double,
                                      "scenarios": scenarios}
    return {"scope": "Fixed-cost projection from one measured M5 burst",
            "input_archive_sha256": archive_hash, "physical_plan": PHYSICAL_PLAN,
            "observed_early_short": {"gaps": len(baseline), "max_us": max(baseline),
                                      "p95_us": percentile(baseline, 0.95),
                                      "gaps_at_least_100ms": sum(x >= GAP_THRESHOLD_US for x in baseline)},
            "observed_first_short_token_runner_us": first,
            "observed_shared_slice_median_us": median(fixed), "decode_sensitivity": rows}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
