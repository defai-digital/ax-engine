"""Attribute observed short-token gaps to the ending engine step."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
import tarfile


HERE = Path(__file__).resolve().parent
SOURCE = "flash-next-natural-short-drain-m5-20261123"


def summarize(values):
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "minimum_ms": ordered[0],
        "median_ms": median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "maximum_ms": ordered[-1],
        "at_least_100_ms": sum(gap >= 100 for gap in ordered),
    }


def ending_work(previous, end, peer_prefill, peer_decode):
    if end != previous + 1:
        return "multiple_engine_steps"
    if end in peer_prefill:
        step = peer_prefill[end]
        assert step["scheduled"] == 2
        return f"shared_prefill_{step['width']}"
    if end in peer_decode:
        return "two_decoders"
    return "short_decode"


def check_arm(workload, trace, pairs):
    requests = {row["request_id"]: row for row in trace}
    rows = {row["prompt_id"]: row for row in workload["observations"] if row["phase"] == "measured"}
    assert len(requests) == len(trace) and len(rows) == 220
    bins = defaultdict(list)
    maxima = Counter()
    for short_id, long_id, cut in pairs:
        short, long = rows[short_id], rows[long_id]
        assert short["input_tokens"] == 512 and long["input_tokens"] == 8192
        assert len(short["output_token_ids"]) == len(long["output_token_ids"]) == 96
        short_trace, peer_trace = (requests[row["request_id"]] for row in (short, long))
        steps = short_trace["decode_step_ids"]
        assert all(a < b for a, b in zip(steps, steps[1:]))
        earlier = [row for row in peer_trace["prefill"] if row["step_id"] < steps[0]]
        assert earlier and earlier[-1]["cursor"] == cut
        peer_prefill = {row["step_id"]: row for row in peer_trace["prefill"]}
        peer_decode = set(peer_trace["decode_step_ids"])
        intervals = short["stream_step_interval_ms"]
        assert len(intervals) == len(steps) - 1 == 95
        classes = []
        for previous, end, gap in zip(steps[:-1], steps[1:], intervals, strict=True):
            assert math.isfinite(gap) and gap >= 0
            kind = ending_work(previous, end, peer_prefill, peer_decode)
            classes.append(kind)
            bins[kind].append(gap)
        largest = intervals.index(max(intervals))
        maxima[classes[largest]] += 1
    return {"matched_short_requests": len(pairs),
            "largest_gap_by_ending_work": dict(maxima),
            "gaps_by_ending_work": {key: summarize(value) for key, value in bins.items()}}


def archive_json(archive, name):
    source = archive.extractfile(name)
    assert source is not None
    with source:
        return json.load(source)


def replay(source):
    comparison = (source / "comparison.json").read_bytes()
    validated = subprocess.check_output([sys.executable, "-B", source / "replay.py"])
    assert comparison == validated
    annotated = json.loads(comparison)
    bundle = source / "evidence.tar.xz"
    arms = {}
    with tarfile.open(bundle) as archive:
        for order in ("first", "second"):
            pairs = annotated["comparison"][order]["matched_priority_turns"]
            assert pairs and len({item[0] for item in pairs}) == len(pairs)
            for arm in (f"mixed-{order}", f"drain-{order}"):
                path = f"observed/{arm}/{arm}"
                workload = archive_json(archive, path + ".json")
                trace = archive_json(archive, path + "-steps.json")
                arms[arm] = check_arm(workload, trace, pairs)
    return {
        "scope": "Client output gaps of matched short requests in the signed M5 natural-arrival capture",
        "qualification": False,
        "source_archive_sha256": sha256(bundle.read_bytes()).hexdigest(),
        "source_comparison_sha256": sha256(comparison).hexdigest(),
        "server_sha256": annotated["server_sha256"],
        "arms": arms,
    }


if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else HERE.parent / SOURCE
    print(json.dumps(replay(src), indent=2, sort_keys=True, allow_nan=False))
