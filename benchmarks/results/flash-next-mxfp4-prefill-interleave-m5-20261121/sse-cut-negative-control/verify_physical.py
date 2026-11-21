"""Check the served Flash Next split experiment and summarize matched trials."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

PACK_REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
INVENTORY_SHA = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
BASE_PROMPT_SHA = "849f8af3077f89be0a49d9b86bd1032eed1513e54a53bac242285cd0d2076032"
HOLDOUT_SHA = {
    2049: "f7ae4c54aaf42c022f3d30bd1c7f8d31867030016f0846b17d45b977e68d5fca",
    2050: "99dbc25a0300b9869f7ef3b265707cd27d2b5b0b49a465c0e39cceda0499ee84",
    8192: "d89dcc24ebc47b66ecc40b48ab2675e1ac5585391d5b3ffbb82716211068a95f",
}
ARM_NAMES = ("solo-default", "pair-prefill-split-off", "pair-prefill-split-on")
ARRIVALS = ("serial", "simultaneous", "decode-arrival")
PHYSICAL_WIDTHS = [512, 512, 512, 511, 1]
SPLIT_ROUTE_PREFIX = "ax_scheduler_prefill_split_end."
COUNTERS = {
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}


def verify_partition(request, length, allow_splits):
    progress = request["prefill_progress"]
    widths = [step["advanced_prompt_tokens"] for step in progress]
    cursor = 0
    split_count = 0
    index = 0
    for logical_start in range(0, length, 2048):
        logical_width = min(2048, length - logical_start)
        if widths[index] == logical_width:
            group = [logical_width]
        else:
            assert allow_splits and logical_width == 2048, widths
            assert widths[index:index + len(PHYSICAL_WIDTHS)] == PHYSICAL_WIDTHS, widths
            group = PHYSICAL_WIDTHS
            split_count += 1
            # The recorded SDK suppresses route metadata on progress frames.
            # This diagnostic checks physical cursors and shared steps alone.
        for width in group:
            cursor += width
            assert progress[index]["cursor"] == cursor
            index += 1
    assert index == len(progress) and cursor == length
    assert all(step["cursor"] == length for step in request["scheduled_work"] if step["delta_token_count"] > 0)
    return split_count


def verify_route_overlap(trial):
    left, right = trial["requests"]
    for prefill, sibling in ((left, right), (right, left)):
        output_steps = [step["step_id"] for step in sibling["scheduled_work"]
                        if step["delta_token_count"] > 0]
        decoding_steps = set(output_steps[1:])
        if any(step["advanced_prompt_tokens"] == 512 and step["scheduled_requests"] == 2
               and step["step_id"] in decoding_steps for step in prefill["prefill_progress"]):
            return
    raise AssertionError("no physical prefill step published a sibling decode token")


def metrics(trial):
    gaps = []
    for request in trial["requests"]:
        clocks = [step["received_seconds"] for step in request["scheduled_work"] if step["delta_token_count"] > 0]
        gaps.extend(b - a for a, b in zip(clocks, clocks[1:]))
    assert gaps and min(gaps) >= 0
    return {
        "output_tokens_per_second": sum(len(r["output_tokens"]) for r in trial["requests"]) / trial["client_wall_seconds"],
        "max_gap_seconds": max(gaps),
        "p95_gap_seconds": statistics.quantiles(gaps, n=100, method="inclusive")[94],
        "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in trial["requests"]),
    }


def verify(data, server_sha256):
    assert data["pack_revision"] == PACK_REVISION and data["inventory_sha256"] == INVENTORY_SHA
    assert data["binary_sha256"] == server_sha256 and data["prompts_sha256"] == BASE_PROMPT_SHA
    length = data.get("holdout_prompt_tokens", 2048)
    if length > 2048:
        assert data["holdout_prompts_sha256"] == HOLDOUT_SHA[length]
    arms = {arm["arm"]: arm for arm in data["arms"]}
    assert tuple(arms) == ARM_NAMES
    solo = {(trial["repeat"], row["prompt_id"]): row["output_tokens"]
            for trial in arms[ARM_NAMES[0]]["observations"] for row in trial["requests"]}
    assert len(solo) == 4
    checked = 0
    split_requests = 0
    disconnects = 0
    reports = {}
    for name, arm in arms.items():
        enabled = name.endswith("-on")
        assert arm["server_overrides"]["AX_MLX_FLASH_NEXT_PREFILL_SPLIT"] == str(int(enabled))
        assert arm["concurrency"] == (1 if name == ARM_NAMES[0] else 2)
        assert arm["max_batch_tokens"] == (2048 if name == ARM_NAMES[0] else 2049)
        rows = arm["observations"]
        expected = {(arrival, repeat) for repeat in (0, 1)
                    for arrival in (("serial",) if name == ARM_NAMES[0] else ARRIVALS)}
        assert len(rows) == len(expected) and {(t["arrival"], t["repeat"]) for t in rows} == expected
        for trial in rows:
            assert len(trial["requests"]) == 2
            split = False
            for request in trial["requests"]:
                assert len(request["output_tokens"]) == 96
                assert request["output_tokens"] == solo[trial["repeat"], request["prompt_id"]]
                splits = verify_partition(request, length, enabled and trial["arrival"] != "serial")
                split_requests += bool(splits)
                split |= bool(splits)
                checked += 1
            if enabled and trial["arrival"] != "serial":
                assert split
                verify_route_overlap(trial)
        trials = arm["disconnect_trials"]
        assert len(trials) == (4 if enabled else 2 if name == ARM_NAMES[1] else 0)
        for trial in trials:
            assert trial["shared_steps"] and trial["survivor_matches"] and trial["replacement_matches"]
            assert trial["disconnected_prefix_matches"]
            for key in ("drained_after_disconnect", "drained_after_replacement"):
                assert trial[key].keys() == COUNTERS and all(value == 0 for value in trial[key].values())
            disconnected = trial["disconnected"]
            if disconnected["disconnect_at"] == "prefill":
                assert enabled and disconnected["first_output_tokens"] == []
                assert 0 < disconnected["prefill_progress"][-1]["cursor"] < length
            disconnects += 1
        assert arm["memory_samples"] and all(sample["pressure"] == 1 for sample in arm["memory_samples"])
        reports[name] = {
            "peak_footprint_bytes": max(sample["footprint_bytes"] for sample in arm["memory_samples"]),
            "trials": [{"arrival": t["arrival"], "repeat": t["repeat"], **metrics(t)} for t in rows],
        }
    return {
        "prompt_tokens": length, "output_arrays_matched": checked,
        "published_logical_end_verified": False,
        "split_requests": split_requests, "disconnect_reuse_trials": disconnects,
        "arms": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--server-sha256", required=True)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    summary = verify(json.loads(raw), args.server_sha256)
    summary["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(summary, indent=2, sort_keys=True))
