"""Recompute the completed overlap and the cancelled-stream telemetry gap."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
GUARDED = ROOT.parent / "flash-next-openai-stream-disconnect-m2-20261122" / "evidence.tar.xz"
REUSED = "ax_mlx_prefix_cache_reused_tokens"
SOURCE = "ax_mlx_prefix_cache_restore_source"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(archive, name):
    item = archive.getmember(name)
    require(item.isfile(), f"Expected regular archive member {name}")
    with archive.extractfile(item) as handle:
        return handle.read()


def overlap(pair):
    peer = [frame["step_id"] for frame in pair["peer"]["scheduled_work"] if frame["delta_token_count"]]
    target = pair["target"]["scheduled_work"][0]
    require(target["scheduled_requests"] == 2 and target["step_id"] in peer[1:-1]
            and target["cursor"] == 2048 and target["delta_token_count"] == 1,
            "The initial target token did not share active peer decode")
    return target


def exact(row):
    target = overlap(row)
    peer = row["peer"]
    times = [frame["received_seconds"] for frame in peer["scheduled_work"] if frame["delta_token_count"]]
    require(len(times) == len(peer["output_tokens"]) == 192 and all(b > a for a, b in zip(times, times[1:]))
            and target["received_seconds"] == row["target"]["ttft_seconds"]
            and 0 < target["received_seconds"] < row["wall_seconds"], "Invalid stream timing")
    return {"target_ttft_seconds": target["received_seconds"], "first_step_scheduled_tokens": target["scheduled_tokens"],
            "peer_maximum_gap_seconds": max(b-a for a, b in zip(times, times[1:])),
            "pair_wall_seconds": row["wall_seconds"]}


def replay(path):
    with tarfile.open(path) as archive, tarfile.open(GUARDED) as prior:
        build_raw = read(archive, "candidate-build.json")
        build = json.loads(build_raw)
        checksums = {name: digest for digest, name in map(str.split, read(archive, "inputs.sha256").decode().splitlines())}
        require(build_raw == read(prior, "candidate-build.json")
                and build["server_sha256"] == checksums["ax-engine-server-resident-mixed"]
                and sha256(read(archive, "run_live_prefix.py")).hexdigest() == checksums["run_live_prefix.py"],
                "The signed executable or diagnostic driver changed")
        require(read(archive, "status.txt") == b"1\n"
                and read(archive, "failure.txt") == b"ValueError: Unexpected cache route during overlap\n",
                "The incomplete trial's terminal status or diagnostic changed")
        partial = json.loads(read(archive, "partial.json"))
        require(partial["qualification"] is False and partial["build"] == build
                and partial["workload_sha256"] == checksums["mixed-corpus.jsonl"]
                and set(partial["arms"]) == {"cold-1"}, "Unexpected completed-arm roster")
        cold = partial["arms"]["cold-1"]
        warm = json.loads(read(archive, "warm-1-partial.json"))
        require(set(warm) == {"before", "seeds", "exact", "cancel"}, "Unexpected warm progress roster")
        left, right = cold["exact"], warm["exact"]
        for name, count in (("target", 96), ("peer", 192)):
            require(left[name]["prompt_id"] == right[name]["prompt_id"]
                    and left[name]["output_tokens"] == right[name]["output_tokens"]
                    and len(left[name]["output_tokens"]) == count, "Output trajectory changed")
        target = right["target"]
        require(any(event["request_id"] == target["request_id"]
                    and event["decisions"].get(REUSED) == 2048 and event["decisions"].get(SOURCE) == 2
                    for event in target["cache_events"]), "Missing request-owned L1 restore evidence")
        a, b = exact(left), exact(right)
        require((a["first_step_scheduled_tokens"], b["first_step_scheduled_tokens"]) == (2049, 2),
                "The warm first step did not skip prefix rows")
        require(overlap(warm["cancel"])["scheduled_tokens"] == 2
                and warm["cancel"]["target"]["cache_events"] == []
                and warm["cancel"]["target"]["first_output_tokens"] == cold["cancel"]["target"]["first_output_tokens"],
                "The cancelled stream no longer demonstrates the missing intermediate cache event")
    return {"qualification": False, "timing_eligible": False, "matched_complete_output_ids": 288,
            "cold_cpu_idle_percent": [cold["before"], cold["after"]], "warm_cpu_idle_before_percent": warm["before"],
            "cold": a, "warm": b, "cancelled_first_step_scheduled_tokens": 2,
            "cancelled_cache_events_observed": 0, "trial_exit_code": 1}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True, allow_nan=False))
