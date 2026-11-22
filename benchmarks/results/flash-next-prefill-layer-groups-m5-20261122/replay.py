"""Verify the layer-group sweep and recompute its serving metrics."""

import gzip
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "flash-next-prefill-layer-http-m5-20261122"
GROUPS = {"group-8": 8, "group-12": 12, "group-16": 16}
WIDTHS = [512, 512, 512, 511, 1] * 4
CASE = "early-decoder"

common = runpy.run_path(str(PREVIOUS / "replay.py"))
require = common["require"]
percentile = common["percentile"]


def digest(data):
    return sha256(data).hexdigest()


def verify_trace(records, requests, size, receipt):
    require(len(records) == 2, "Expected exactly two traces")
    traces = {row["request_id"]: row for row in records}
    short, long = (traces[row["request_id"]] for row in requests)
    require((short["cursor"], long["cursor"]) == (512, 8192) and not short["partial"],
            "Unexpected prefill cursor or short-stream suspension")
    widths = [row["width"] for row in long["prefill"]]
    require(widths == receipt["long_widths"] == WIDTHS, "Physical prefill widths changed")
    boundaries = {row["cursor"] - row["width"]: row for row in long["prefill"]}
    partial = long["partial"]
    require(len(partial) == receipt["layer_progress_steps"] > 0, "Layer progress is absent")
    progress = {}
    for row in partial:
        cursor = row["cursor"]
        boundary = boundaries[cursor]
        previous = progress.get(cursor, 0)
        require(row["previous"] == previous and row["completed"] == previous + size < 48
                and row["logical_end"] == boundary["split_end"] == ((cursor // 2048) + 1) * 2048
                and row["scheduled"] == 2 and row["runner_time_us"] > 0
                and row["step_id"] in short["decode_step_ids"][1:]
                and row["step_id"] < boundary["step_id"] and boundary["width"] in (511, 512),
                "Suspension advanced the wrong request, layer or physical boundary")
        progress[cursor] = row["completed"]
    require(receipt["drained"] == dict.fromkeys(common["COUNTERS"], 0), "Service did not drain")
    return partial


def recompute(client, traces, cases, size, receipt):
    require("base_url" not in client["target"] and "path" not in client["corpus"],
            "Unredacted client metadata")
    requests = sorted(client["observations"], key=lambda row: len(row["prompt_token_ids"]))
    require(len(requests) == len(cases) == 2, "Expected two completed requests")
    for request, case in zip(requests, cases, strict=True):
        require(request["ok"] and request["phase"] == "measured" and request["prompt_id"] == case["id"]
                and request["prompt_token_ids"] == case["input_tokens"]
                and request["output_token_ids"] == case["expected_ids"]
                and request["scheduled_at_s"] == case["arrival_s"], "Prompt, output or arrival changed")
        intervals = request["stream_step_interval_ms"]
        require(len(intervals) == len(case["expected_ids"]) - 1 and all(gap >= 0 for gap in intervals),
                "An output token is missing its timing record")
    partial = verify_trace(traces, requests, size, receipt)
    short, long = requests
    ids = sum(len(row["output_token_ids"]) for row in requests)
    finish = [row["started_at_s"] + row["e2e_latency_ms"] / 1000 for row in requests]
    first = long["started_at_s"] + long["ttft_ms"] / 1000
    throughput = ids / max(finish)
    require(isclose(throughput, client["summary"]["output_token_throughput_tok_s"], rel_tol=1e-9),
            "Client throughput does not match stream completion")
    gaps = short["stream_step_interval_ms"]
    return {
        "layer_group": size,
        "matched_output_ids": ids,
        "layer_progress_steps": len(partial),
        "layer_runner_p95_ms": percentile([row["runner_time_us"] / 1000 for row in partial], 0.95),
        "short_p95_gap_ms": percentile(gaps, 0.95),
        "short_max_gap_ms": max(gaps),
        "long_arrival_to_first_token_ms": (first - long["scheduled_at_s"]) * 1000,
        "long_first_token_before_short_completion": first < finish[0],
        "output_tokens_per_second": throughput,
    }


def main():
    raw = json.loads(gzip.decompress((ROOT / "evidence.json.gz").read_bytes()))
    receipt = raw["receipt"]
    require(receipt["status"] == "complete" and receipt["qualification"] is False
            and raw["exit_status"] == 0 and list(receipt["arms"]) == list(raw["arms"]) == list(GROUPS),
            "The sweep did not complete its three arms")
    build = receipt["build"]
    inputs = {name: checksum for checksum, name in
              (line.split() for line in (ROOT / "inputs.sha256").read_text().splitlines())}
    require(build["mlx"] == build["mlx_metal"] == "0.32.4" and build["rustc"].startswith("rustc 1.99.1 ")
            and build["profile"] == "release-server"
            and build["server_sha256"] == inputs["ax-engine-server-layer-http"]
            and receipt["inventory_sha256"] == inputs["mxfp4-inventory.json"], "Build or inventory changed")
    previous = json.loads(gzip.decompress((PREVIOUS / "evidence.json.gz").read_bytes()))["receipt"]
    require(receipt["pack_revision"] == previous["pack_revision"] and receipt["manifest"] == previous["manifest"],
            "Pack or delivered manifest differs from the preceding study")
    cases = common["input_rows"]()[CASE]
    results = {}
    for name, size in GROUPS.items():
        arm = receipt["arms"][name]
        captures = raw["arms"][name]
        require(arm["layer_group"] == size and list(arm["outcomes"]) == list(captures["outcomes"]) == [CASE],
                "Group size or workload changed")
        meta, capture = arm["outcomes"][CASE], captures["outcomes"][CASE]
        require(capture["original_client_sha256"] == meta["client_sha256"]
                and digest(capture["trace"].encode()) == meta["trace_sha256"],
                "Client or trace does not match the terminal receipt")
        samples = captures["samples"]
        require(samples and len(samples) == arm["pressure_samples"] and all(row["pressure"] == 1 for row in samples),
                "Sampled memory pressure exceeded normal")
        before, after = arm["quiet_before_percent"], arm["idle_after_percent"]
        results[name] = recompute(json.loads(capture["client"]), json.loads(capture["trace"]), cases, size, meta) | {
            "peak_footprint_bytes": max(row["footprint_bytes"] for row in samples),
            "cpu_idle_before_percent": before,
            "cpu_idle_after_percent": after,
            "timing_eligible": min(before, after) >= 90,
        }
    print(json.dumps({"qualification": False, "arms": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
