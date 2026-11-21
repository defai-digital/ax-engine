"""Attribute 100 ms or longer gaps in the older stream to adjacent prefill steps."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import statistics


LENGTHS = (2048, 2049, 2050, 8192)
SLOW_GAP_SECONDS = 0.100


def output_steps(request):
    steps = [step for step in request["scheduled_work"] if step["delta_token_count"]]
    if len(steps) != 96 or any(step["delta_token_count"] != 1 for step in steps):
        raise ValueError("The fixed workload must publish 96 singleton output events")
    if any(a["step_id"] >= b["step_id"] for a, b in zip(steps, steps[1:])):
        raise ValueError("Output steps must increase")
    return steps


def observe_trial(trial):
    requests = sorted(trial["requests"], key=lambda request: output_steps(request)[0]["step_id"])
    lead, late = requests
    steps = output_steps(lead)
    late_steps = output_steps(late)
    if steps[0]["step_id"] == late_steps[0]["step_id"]:
        raise ValueError("The earlier decoder must be identifiable")
    prefill = {step["step_id"] for step in late["prefill_progress"]}
    gaps = []
    slow = shared = 0
    for a, b in zip(steps, steps[1:]):
        duration = b["received_seconds"] - a["received_seconds"]
        if duration < 0:
            raise ValueError("The client clock moved backwards")
        gaps.append(duration)
        if duration >= SLOW_GAP_SECONDS:
            slow += 1
            shared += a["step_id"] in prefill or b["step_id"] in prefill
    return {"slow_gaps": slow, "slow_adjacent_to_peer_prefill": shared,
            "maximum_gap_seconds": max(gaps),
            "p95_gap_seconds": statistics.quantiles(gaps, n=100, method="inclusive")[94]}


def main():
    root = Path(__file__).resolve().parent
    result = {}
    for length in LENGTHS:
        directory = root / f"sse-cut-{length}"
        audit = json.loads((directory / "audit.json").read_text())
        raw = gzip.decompress((directory / "result.json.gz").read_bytes())
        if sha256(raw).hexdigest() != audit["receipt_sha256"] or audit["prompt_tokens"] != length:
            raise ValueError("The raw scenario differs from its audited receipt")
        data = json.loads(raw)
        result[str(length)] = {
            arm["arm"]: [
                {"arrival": trial["arrival"], "repeat": trial["repeat"], **observe_trial(trial)}
                for trial in arm["observations"] if trial["arrival"] != "serial"
            ]
            for arm in data["arms"] if arm["concurrency"] == 2
        }
    print(json.dumps({"slow_gap_seconds": SLOW_GAP_SECONDS, "older_streams": result},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
