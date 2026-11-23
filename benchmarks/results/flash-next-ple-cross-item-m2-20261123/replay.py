"""Reconcile the PLE gather trace and attribute exact boundary consumption."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_bytes())


def summarize(payload):
    assert payload["schema"] == "ax-engine.flash-next.ple-boundary-probe.v1"
    assert payload["qualification"] is False
    assert (payload["logical_tokens"], payload["physical_tokens"]) == (2048, 512)
    runs = payload["runs"]
    assert len(runs) == 8
    boundary_starts = (2048, 4096, 6144)
    traces = []
    paired_prefill = []
    for index, row in enumerate(runs):
        case, arm = divmod(index, 2)
        candidate = bool((case + arm) % 2)
        assert (row["case"], row["arm"]) == (case, arm)
        assert row["cross_item"] is candidate
        assert 0 < len(row["output_tokens"]) <= 32
        assert row["stop_reason"] in ("eos_token", "max_output_tokens")
        trace = row["trace"]
        assert trace["submitted_boundaries"] == (list(boundary_starts) if candidate else [])
        boundaries = [event for event in trace["gathers"] if event["offset"] in boundary_starts]
        assert tuple(event["offset"] for event in boundaries) == boundary_starts
        assert all(event["tokens"] == 512 and event["wall_us"] >= 0 for event in boundaries)
        assert candidate or not any(event["prefetched"] for event in boundaries)
        if arm == 1:
            previous = runs[index-1]
            assert previous["output_tokens"] == row["output_tokens"]
            assert previous["stop_reason"] == row["stop_reason"]
            control, treatment = (previous, row) if candidate else (row, previous)
            before, after = control["prefill_seconds"], treatment["prefill_seconds"]
            assert before > 0 and after > 0
            paired_prefill.append({"control_seconds": before, "candidate_seconds": after,
                                   "candidate_change_percent": 100 * (after / before - 1)})
        traces.append((candidate, boundaries))
    timings = {}
    for enabled, label in ((False, "control"), (True, "candidate")):
        events = [event for candidate, group in traces if candidate == enabled for event in group]
        ready = [event for event in events if event["prefetched"]]
        timings[label] = {
            "boundary_events": len(events),
            "ready_matches": len(ready),
            "host_gather_median_us": median(event["wall_us"] for event in events),
            "ready_match_host_gather_median_us": median(event["wall_us"] for event in ready) if ready else None,
        }
    return {
        "case_pairs": 4,
        "matched_output_ids": sum(len(row["output_tokens"]) for row in runs if row["arm"] == 0),
        "boundary_gathers": timings,
        "paired_prefill": paired_prefill,
        "scope": "Feasibility on M2 Ultra, with pack and input reuse",
        "qualification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run = args.run
    assert (run / "status.txt").read_text().strip() == "0"
    assert (run / "pack-before.json").read_bytes() == (run / "pack-after.json").read_bytes()
    pack = read(run / "pack-before.json")
    assert pack["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert pack["verified_members"] == 49
    assert sha256((run / "mxfp4-inventory.json").read_bytes()).hexdigest() == pack["inventory_sha256"]
    proof = read(run / "ple-boundary-inputs-receipt.json")
    assert sha256(gzip.decompress((run / "source.patch.gz").read_bytes())).hexdigest() == read(run / "build.json")["patch_sha256"]
    inputs = gzip.decompress((run / "ple-boundary-inputs.json.gz").read_bytes())
    outputs = gzip.decompress((run / "ple-boundary-output.json.gz").read_bytes())
    assert sha256(inputs).hexdigest() == proof["inputs_sha256"]
    assert all(len(prompt) == 8192 for prompt in json.loads(inputs)) and len(json.loads(inputs)) == 4
    assert sha256(outputs).hexdigest() == (run / "output.sha256").read_text().split()[0]
    print(json.dumps(summarize(json.loads(outputs)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
