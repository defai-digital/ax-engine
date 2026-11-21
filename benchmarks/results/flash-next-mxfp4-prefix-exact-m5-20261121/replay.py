"""Check the archived M5 Flash Next exact-prefix control without a model."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile


HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
REUSED = "ax_mlx_prefix_cache_reused_tokens"
SOURCE = "ax_mlx_prefix_cache_restore_source"
STORES = "ax_mlx_prefix_cache_stores"
GAUGES = {
    "ax_engine_jobs_in_flight",
    "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued",
    "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cache_events(reply):
    for event in reply["cache_events"]:
        require(event["request_id"] == reply["request_id"], "foreign prefix event")
        if event["event"] == "step":
            require(
                any(step["step_id"] == event["step_id"] for step in reply["scheduled_work"]),
                "prefix event has no matching request step",
            )
    return [event["decisions"] for event in reply["cache_events"]]


def compare(receipt, sample_logs):
    require(receipt["status"] == "complete", "incomplete receipt")
    cold, warm = (receipt["arms"][name] for name in ("cold", "l1"))
    exact = warm["exact-replay"]
    require(
        any(event.get(SOURCE) == 2 and event.get(REUSED) == 2048 for event in cache_events(exact)),
        "exact replay did not restore 2,048 tokens from L1",
    )
    for name in ("seed-a", "seed-b"):
        require(any(event.get(STORES, 0) for event in cache_events(warm[name])), "missing L1 seed")
    for reply in cold["outputs"].values():
        require(
            not any(event.get(REUSED, 0) or event.get(STORES, 0) for event in cache_events(reply)),
            "cache activity in cold process",
        )

    def match(name, actual, count, output_field="output_tokens"):
        baseline = cold["outputs"][name]["output_tokens"]
        require(actual["prompt_id"] == name, f"wrong prompt in {name}")
        require(
            len(actual[output_field]) == count and actual[output_field] == baseline[:count],
            f"changed output in {name}",
        )
        return count

    matched = match("exact-replay", exact, 96)
    matched += match("replacement", warm["replacement"], 96)
    controls = (
        ("paired", "target", "survivor-a", 96, "output_tokens"),
        ("disconnect", "cancelled", "survivor-b", 1, "first_output_tokens"),
    )
    overlap = {}
    for arm, name, peer_name, count, output_field in controls:
        target, peer = (warm[arm][role] for role in ("target", "peer"))
        matched += match(name, target, count, output_field)
        matched += match(peer_name, peer, 192)
        require(
            not any(event.get(REUSED, 0) for event in cache_events(peer)),
            "peer inherited cache reuse",
        )
        ongoing = [step["step_id"] for step in peer["scheduled_work"] if step["delta_token_count"]]
        overlap[arm] = [
            step["step_id"]
            for step in target["prefill_progress"]
            if step["scheduled_requests"] == 2 and step["step_id"] in ongoing[1:-1]
        ]
        require(len(overlap[arm]) == 2, f"missing measured peer overlap for {arm}")
    for reply in (warm["paired"]["target"], warm["disconnect"]["target"], warm["replacement"]):
        require(not any(event.get(REUSED, 0) for event in cache_events(reply)), "extension hit L1")
        require(
            sum(step["advanced_prompt_tokens"] for step in reply["prefill_progress"]) == 2112,
            "extension did not prefill all prompt tokens",
        )
    require(warm["disconnect"]["target"]["disconnect_at"] == "output", "wrong disconnect point")
    require(warm["health_after"] == "ok", "unhealthy server after replacement")
    drains = [
        cold["drained"],
        warm["after_seed_a"],
        warm["exact_replay_drained"],
        warm["paired"]["drained"],
        warm["after_seed_b"],
        warm["disconnect"]["drained"],
        warm["replacement_drained"],
    ]
    for gauges in drains:
        require(gauges.keys() == GAUGES and all(value == 0 for value in gauges.values()), "not drained")
    arm_stats = {}
    for name, arm in (("cold", cold), ("l1", warm)):
        rows = sample_logs[name]
        require(bool(rows) and all(row["pressure"] == 1 for row in rows), "non-normal pressure")
        peak = max(row["footprint_bytes"] for row in rows)
        require(peak == arm["peak_footprint_bytes"], "peak footprint differs from receipt")
        arm_stats[name] = {"samples": len(rows), "peak_footprint_bytes": peak}
    return {
        "exact_cold_ttft_seconds": cold["outputs"]["exact-replay"]["ttft_seconds"],
        "exact_l1_ttft_seconds": exact["ttft_seconds"],
        "matched_output_ids": matched,
        "shared_prefill_steps": overlap,
        "drained_snapshots": len(drains),
        "memory_observations": arm_stats,
    }


def replay():
    source = json.loads((HERE / "source.json").read_bytes())
    for name, digest in source["artifacts_sha256"].items():
        require(sha256((HERE / name).read_bytes()).hexdigest() == digest, f"changed {name}")
    with tarfile.open(FROZEN, "r:xz") as frozen:
        for name, digest in source["frozen_inputs_sha256"].items():
            require(sha256(frozen.extractfile(name).read()).hexdigest() == digest, f"changed {name}")
        build = json.load(frozen.extractfile("build.json"))
    raw = gzip.decompress((HERE / "receipt.json.gz").read_bytes())
    require(sha256(raw).hexdigest() == source["original_receipt_sha256"], "changed raw receipt")
    receipt = json.loads(raw)
    require(receipt["binary_sha256"] == build["server_sha256"], "different executable")
    require(receipt["driver_sha256"] == source["artifacts_sha256"]["run_warm_prefix.py"], "different driver")
    require(
        receipt["inventory_sha256"] == source["frozen_inputs_sha256"]["mxfp4-inventory.json"],
        "different inventory",
    )
    for key in ("pack_revision", "manifest_sha256"):
        require(receipt[key] == source[key], f"different {key}")
    samples = {
        name: [json.loads(line) for line in (HERE / f"{name}-samples.jsonl").read_bytes().splitlines()]
        for name in ("cold", "l1")
    }
    return compare(receipt, samples)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("usage: python3 replay.py")
    print(json.dumps(replay(), indent=2))
