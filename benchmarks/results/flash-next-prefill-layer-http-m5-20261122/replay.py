"""Recompute native HTTP layer-yield observations and matched-arm comparisons."""

import gzip
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
BURST = ROOT.parent / "flash-next-mxfp4-prefill-step-cost-m5-20261121"
SOLO = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121"
LABELS = ("early-decoder", "ending-decoder")
EXPECTED_ARMS = ("split-1", "layers-1", "layers-2", "split-2")
COUNTERS = {
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def input_rows():
    frozen = {}
    with tarfile.open(BURST / "evidence.tar.xz") as archive:
        for label in LABELS:
            frozen[label] = [json.loads(line) for line in archive.extractfile(f"{label}.jsonl").read().splitlines()]
    with tarfile.open(SOLO / "evidence.tar.xz") as archive:
        raw = archive.extractfile("mixed-result/solo-1.json").read()
    available = {row["prompt_id"]: row for row in json.loads(raw)["observations"] if row["phase"] == "measured"}
    for rows in frozen.values():
        for row in rows:
            source = available[row["id"]]
            require(row["input_tokens"] == source["prompt_token_ids"], "Frozen prompt differs from solo")
            row["expected_ids"] = source["output_token_ids"][:row["max_output_tokens"]]
    return frozen


def workload(client, traces, cases, wired, label, candidate):
    rows = sorted(client["observations"], key=lambda row: len(row["prompt_token_ids"]))
    require(len(rows) == 2 and len(traces) == 2, "Two streams were required")
    for row, case in zip(rows, cases, strict=True):
        require(row["ok"] and row["phase"] == "measured" and row["prompt_id"] == case["id"]
                and row["prompt_token_ids"] == case["input_tokens"] and row["output_token_ids"] == case["expected_ids"]
                and len(row["output_token_ids"]) == case["max_output_tokens"]
                and row["scheduled_at_s"] == case["arrival_s"], "Workload or output changed")
    by_id = {row["request_id"]: row for row in traces}
    short, long = [by_id[row["request_id"]] for row in rows]
    require(short["cursor"] == 512 and long["cursor"] == 8192 and not short["partial"],
            "The prefill path or output stream identity changed")
    partial = long["partial"]
    require((candidate or not partial) and (label != "early-decoder" or not candidate or partial),
            "Layer treatment was absent or reached the control")
    completed = {row["cursor"] - row["width"]: row for row in long["prefill"]}
    progress = {}
    for row in partial:
        cursor = row["cursor"]
        last = progress.get(cursor, 0)
        require(row["previous"] == last and row["completed"] == last + 8 <= 40 and row["scheduled"] == 2
                and row["runner_time_us"] > 0 and row["step_id"] in short["decode_step_ids"][1:]
                and row["step_id"] < completed[cursor]["step_id"]
                and completed[cursor]["width"] in (511, 512)
                and row["logical_end"] == completed[cursor]["split_end"], "Invalid request-local progress")
        progress[cursor] = row["completed"]
    widths = [row["width"] for row in long["prefill"]]
    require(wired["layer_progress_steps"] == len(partial) and wired["long_widths"] == widths
            and wired["drained"] == dict.fromkeys(COUNTERS, 0), "Receipt disagrees with stream or drain")
    short_obs, long_obs = rows
    short_gaps = short_obs["stream_step_interval_ms"]
    require(len(short_gaps) == cases[0]["max_output_tokens"] - 1 and all(gap >= 0 for gap in short_gaps),
            "Short stream did not publish one event per token")
    elapsed = max(row["started_at_s"] + row["e2e_latency_ms"] / 1000 for row in rows)
    throughput = sum(len(row["output_token_ids"]) for row in rows) / elapsed
    require(isclose(throughput, client["summary"]["output_token_throughput_tok_s"], rel_tol=1e-9),
            "The summary throughput did not match the timed stream")
    short_complete = short_obs["started_at_s"] + short_obs["e2e_latency_ms"] / 1000
    long_first = long_obs["started_at_s"] + long_obs["ttft_ms"] / 1000
    return {"short_p95_gap_ms": percentile(short_gaps, 0.95), "short_max_gap_ms": max(short_gaps),
            "long_arrival_to_first_token_ms": (long_first - long_obs["scheduled_at_s"]) * 1000,
            "long_first_token_before_short_completion": long_first < short_complete,
            "output_tokens_per_second": throughput, "matched_output_ids": sum(len(row["output_token_ids"]) for row in rows),
            "layer_progress_steps": len(partial), "long_widths": widths,
            "layer_runner_p95_ms": percentile([row["runner_time_us"] / 1000 for row in partial], 0.95)
            if partial else None}


def main():
    evidence = gzip.decompress((ROOT / "evidence.json.gz").read_bytes())
    raw = json.loads(evidence)
    receipt = raw["receipt"]
    require(receipt["status"] == "complete" and receipt["qualification"] is False
            and raw["exit_status"] == 0 and tuple(receipt["arms"]) == EXPECTED_ARMS, "Incomplete HTTP run")
    require(receipt["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and receipt["manifest"].get("model-manifest.json") ==
            "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
            and receipt["build"]["profile"] == "release-server"
            and receipt["build"]["mlx"] == receipt["build"]["mlx_metal"] == "0.32.4",
            "Pack or software identity changed")
    cases = input_rows()
    all_arms = {}
    for name, arm in receipt["arms"].items():
        captures = raw["arms"][name]
        samples = captures["samples"]
        require(samples and len(samples) == arm["pressure_samples"]
                and all(row["pressure"] == 1 for row in samples), "Memory pressure changed")
        values = {}
        for label in LABELS:
            observation = captures["outcomes"][label]
            meta = arm["outcomes"][label]
            require(observation["original_client_sha256"] == meta["client_sha256"]
                    and digest(observation["trace"].encode()) == meta["trace_sha256"],
                    "The terminal receipt does not bind the raw capture")
            client = json.loads(observation["client"])
            require("base_url" not in client["target"], "A machine address entered the recorded inputs")
            values[label] = workload(client, json.loads(observation["trace"]), cases[label], meta,
                                     arm["layer_yield"], label)
        all_arms[name] = {"outcomes": values, "ended_cpu_idle_percent": arm["idle_after_percent"],
                         "initial_cpu_idle_percent": arm["quiet_before_percent"],
                         "timing_eligible": min(arm["quiet_before_percent"], arm["idle_after_percent"]) >= 90,
                         "peak_footprint_bytes": max(row["footprint_bytes"] for row in samples)}
    cancel = receipt["arms"]["layers-1"]["outcomes"]["cancellations"]
    require([row["canceled"] for row in cancel] == ["long", "short"], "Missing canceled request")
    require(all(row["drained"] == dict.fromkeys(COUNTERS, 0) and len(row["requests"]) == 2
                and sum(item["closed"] for item in row["requests"]) == 1
                and sum(item.get("matched_output_ids", 0) for item in row["requests"]) == 96
                and any(item["seen_layer_steps"] for item in row["requests"])
                for row in cancel), "Post-cancel cleanup or peer output changed")
    by_mode = {}
    for name, selected in (("physical", ("split-1", "split-2")), ("layers", ("layers-1", "layers-2"))):
        by_mode[name] = {label: {metric: statistics.median(all_arms[arm]["outcomes"][label][metric] for arm in selected)
                                for metric in ("short_p95_gap_ms", "short_max_gap_ms",
                                               "long_arrival_to_first_token_ms", "output_tokens_per_second")}
                         for label in LABELS}
    eligible = [pair for pair in (("split-1", "layers-1"), ("split-2", "layers-2"))
                if all(all_arms[arm]["timing_eligible"] for arm in pair)]
    print(json.dumps({"qualification": False, "arms": all_arms, "diagnostic_median_by_mode": by_mode,
                      "timing_eligible_pairs": eligible, "cancellations": cancel},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
