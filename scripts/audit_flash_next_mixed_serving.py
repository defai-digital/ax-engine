#!/usr/bin/env python3
"""Audit Flash Next mixed-load receipts and attribute streamed gaps to engine steps."""

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

if __package__:
    from .bench_ax_serving import load_corpus, percentile
else:
    from bench_ax_serving import load_corpus, percentile


ARMS = {
    "solo-1": (1, False), "pair-off-1": (2, False), "pair-on-1": (2, True),
    "pair-on-2": (2, True), "pair-off-2": (2, False), "solo-2": (1, False),
}
PACK = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
WIDTHS = [512, 512, 512, 511, 1]
PREFILL_ITEM = sum(WIDTHS)
METRICS = (
    "client_lane_wait_ms", "client_ttft_ms", "arrival_to_first_token_ms",
    "arrival_to_completion_ms", "per_request_gap_p95_ms", "per_request_gap_max_ms",
)
COUNTERS = {
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def audit_prefill(trace, prompt_tokens, split_allowed, label):
    require(trace["initial_cursor"] == 0 and trace["cursor"] == prompt_tokens, label)
    steps = trace["prefill"]
    widths = [step["width"] for step in steps]
    cursor = 0
    index = 0
    split_steps = set()
    while cursor < prompt_tokens:
        logical_width = min(PREFILL_ITEM, prompt_tokens - cursor)
        selected = [logical_width]
        marked = index < len(steps) and steps[index]["split_end"] is not None
        if marked:
            require(split_allowed and logical_width == PREFILL_ITEM, label)
            selected = WIDTHS
        end = cursor + logical_width
        require(widths[index:index + len(selected)] == selected, label)
        for step in steps[index:index + len(selected)]:
            cursor += step["width"]
            require(step["cursor"] == cursor and step["split_end"] == (end if marked else None), label)
            if marked and step["width"] == WIDTHS[0] and step["scheduled"] == 2:
                split_steps.add(step["step_id"])
        index += len(selected)
    require(index == len(steps), label)
    return split_steps


def streams(artifact, traces, prompts, lanes, split, budget, name, *, split_min_prompt_tokens=PREFILL_ITEM):
    measured = [row for row in artifact["observations"] if row["phase"] == "measured"]
    by_prompt = {row["prompt_id"]: row for row in measured}
    by_request = {trace["request_id"]: trace for trace in traces}
    require(len(by_prompt) == len(measured) and by_prompt.keys() == prompts.keys(),
            f"{name}: measured prompt coverage")
    require(len(by_request) == len(traces) == len(artifact["observations"])
            and by_request.keys() == {row["request_id"] for row in artifact["observations"]},
            f"{name}: trace identity")
    prefill_owners = defaultdict(set)
    decoder_owners = defaultdict(set)
    records = {}
    for prompt_id, row in by_prompt.items():
        prompt = prompts[prompt_id]
        request = row["request_id"]
        trace = by_request[request]
        label = f"{name}/{prompt_id}"
        require(row["ok"] is True and row["prompt_token_ids"] == prompt.input_tokens
                and row["scheduled_at_s"] == prompt.arrival_s and row["category"] == prompt.category
                and row["input_tokens"] == len(prompt.input_tokens) and row["output_tokens"] == budget
                and len(row["output_token_ids"]) == budget, f"{label}: measured input or output shape")
        wait = (row["started_at_s"] - row["scheduled_at_s"]) * 1000
        require(wait >= 0 and isclose(wait, row["queue_delay_ms"], abs_tol=1e-6)
                and 0 <= row["ttft_ms"] <= row["e2e_latency_ms"] < float("inf"),
                f"{label}: client timeline")
        steps = trace["decode_step_ids"]
        intervals = row["stream_step_interval_ms"]
        require(len(steps) == budget and len(intervals) == budget - 1
                and all(type(step) is int and step > 0 for step in steps)
                and all(a < b for a, b in zip(steps, steps[1:]))
                and all(isfinite(gap) and gap >= 0 for gap in intervals), f"{label}: gap alignment")
        prefill_ids = [step["step_id"] for step in trace["prefill"]]
        require(prefill_ids and all(a < b for a, b in zip(prefill_ids, prefill_ids[1:]))
                and 0 < prefill_ids[0] <= prefill_ids[-1] <= steps[0]
                and all(1 <= event["scheduled"] <= lanes for event in trace["prefill"]),
                f"{label}: prefill step ordering")
        split_allowed = split and row["input_tokens"] >= split_min_prompt_tokens
        shared_candidates = audit_prefill(trace, row["input_tokens"], split_allowed, f"{label}: prefill cuts")
        for event in trace["prefill"]:
            route = "split" if event["split_end"] is not None else "whole"
            prefill_owners[event["step_id"]].add((request, row["input_tokens"], route))
        for step in steps[1:]:
            decoder_owners[step].add(request)
        records[prompt_id] = (row, steps, shared_candidates)

    prefill_ids = sorted(prefill_owners)
    results = {}
    shared_splits = 0
    for prompt_id, (row, steps, candidates) in records.items():
        ident = row["request_id"]
        shared_splits += any(decoder_owners[step] - {ident} for step in candidates)
        gaps = row["stream_step_interval_ms"]
        peer_prefill = False
        long_gap_peers = Counter()
        for a, b, gap in zip(steps, steps[1:], gaps):
            # Work at the previous output step has already reached this stream.
            lo, hi = bisect_right(prefill_ids, a), bisect_right(prefill_ids, b)
            peers = {
                (length, route)
                for step in prefill_ids[lo:hi]
                for owner, length, route in prefill_owners[step]
                if owner != ident
            }
            if peers:
                peer_prefill = True
                if gap >= 100:
                    label = "+".join(f"{length}/{route}" for length, route in sorted(peers))
                    long_gap_peers[label] += 1
        results[prompt_id] = {
            "prompt_tokens": row["input_tokens"],
            "peer_prefill": peer_prefill,
            "gaps_at_least_100ms": sum(gap >= 100 for gap in gaps),
            "peer_prefill_gaps_at_least_100ms": long_gap_peers.total(),
            "long_gap_peer_profiles": dict(sorted(long_gap_peers.items())),
            "client_lane_wait_ms": row["queue_delay_ms"],
            "client_ttft_ms": row["ttft_ms"],
            "arrival_to_first_token_ms": row["queue_delay_ms"] + row["ttft_ms"],
            "arrival_to_completion_ms": row["queue_delay_ms"] + row["e2e_latency_ms"],
            "per_request_gap_p95_ms": percentile(gaps, 0.95),
            "per_request_gap_max_ms": max(gaps),
        }
    return results, {name: row["output_token_ids"] for name, row in by_prompt.items()}, shared_splits


def percentiles(values):
    return {f"p{q}": percentile(values, q / 100) for q in (50, 95, 100)}


def summarize(rows):
    peer_profiles = Counter()
    for row in rows:
        peer_profiles.update(row["long_gap_peer_profiles"])
    return {
        "requests": len(rows),
        "gaps_at_least_100ms": sum(row["gaps_at_least_100ms"] for row in rows),
        "peer_prefill_gaps_at_least_100ms": sum(row["peer_prefill_gaps_at_least_100ms"] for row in rows),
        "long_gap_peer_profiles": dict(sorted(peer_profiles.items())),
        **{name: percentiles([row[name] for row in rows]) for name in METRICS},
    }


def compare(current, baseline):
    """Keep the paired workload fixed even when peer-prefill exposure changes."""
    require(current.keys() == baseline.keys(), "comparison prompt inventory differs")
    groups = defaultdict(list)
    for name, row in current.items():
        original = baseline[name]
        require(row["prompt_tokens"] == original["prompt_tokens"], "comparison prompt length differs")
        groups[row["prompt_tokens"]].append({key: row[key] - original[key] for key in METRICS})
    return {
        str(length): {
            "requests": len(rows),
            "lower_max_gap_with_higher_gap_p95": sum(
                row["per_request_gap_max_ms"] < 0 < row["per_request_gap_p95_ms"] for row in rows
            ),
            "lower_max_gap_with_higher_client_ttft": sum(
                row["per_request_gap_max_ms"] < 0 < row["client_ttft_ms"] for row in rows
            ),
            **{key: percentiles([row[key] for row in rows]) for key in METRICS},
        }
        for length, rows in sorted(groups.items())
    }


def audit_arm(folder, name, metadata, workload, prompts, *, split_min_prompt_tokens=PREFILL_ITEM):
    lanes, split = ARMS[name]
    require((metadata["lanes"], metadata["split"]) == (lanes, split) and metadata["matched_outputs"] == len(prompts),
            f"{name}: arm metadata")
    require(metadata["drained"].keys() == COUNTERS and not any(metadata["drained"].values()),
            f"{name}: lifecycle drain")
    quiet = metadata["quiet"]
    require(90 <= quiet["min_idle_percent"] <= quiet["cpu_idle_percent"] <= 100, f"{name}: host idle")
    raw_traces = (folder / f"{name}-steps.json").read_bytes()
    require(digest(raw_traces) == metadata["step_trace_sha256"], f"{name}: step trace checksum")
    artifact = read_json(folder / f"{name}.json")
    require(artifact["corpus"]["sha256"] == workload["corpus_sha256"]
            and artifact["load"]["concurrency"] == lanes and artifact["load"]["corpus_arrivals"] is True
            and artifact["load"]["measured_requests"] == len(prompts)
            and artifact["target"]["input_kind"] == "tokens"
            and artifact["sampling"]["temperature"] == 0 and artifact["sampling"]["ignore_eos"] is True,
            f"{name}: client configuration")
    data, outputs, sharing = streams(
        artifact, json.loads(raw_traces), prompts, lanes, split, workload["reply_budget"], name,
        split_min_prompt_tokens=split_min_prompt_tokens,
    )
    require(sharing == metadata["split_activation"]["split_requests_with_route_and_shared_decode"],
            f"{name}: shared split activation")
    groups = defaultdict(list)
    for row in data.values():
        length = row["prompt_tokens"]
        groups[f"{length}/all"].append(row)
        groups[f"{length}/{'peer-prefill' if row['peer_prefill'] else 'other'}"].append(row)
    samples = [json.loads(line) for line in (folder / f"{name}-samples.jsonl").read_text().splitlines()]
    require(samples and {sample["pressure"] for sample in samples} == {1}
            and max(sample["footprint_bytes"] for sample in samples) == metadata["peak_footprint_bytes"],
            f"{name}: measured memory pressure or footprint")
    return {
        "shared_split_requests": sharing,
        "all": summarize(list(data.values())),
        "by_length_and_peer_prefill": {name: summarize(rows) for name, rows in sorted(groups.items())},
    }, outputs, data


def audit(root):
    folder = root / "mixed-result"
    receipt = read_json(folder / "receipt.json")
    build = read_json(root / "build.json")
    workload = read_json(root / "workload.json")
    corpus = root / "mixed-corpus.jsonl"
    require(receipt["status"] == "complete" and set(receipt["arms"]) == set(ARMS), "incomplete mixed campaign")
    require(receipt["binary_sha256"] == build["server_sha256"] and receipt["workload"] == workload
            and workload["pack_revision"] == PACK and digest(corpus.read_bytes()) == workload["corpus_sha256"]
            and digest((root / "run_mixed_serving.py").read_bytes()) == receipt["driver_sha256"]
            and digest((root / "capture_mixed_steps.py").read_bytes()) == receipt["capture_sha256"]
            and all(
                digest(source.read_bytes()) == workload["benchmark_sha256"]
                for source in (root / "bench_ax_serving.py", Path(__file__).with_name("bench_ax_serving.py"))
            ),
            "mixed campaign source binding")
    rows = load_corpus(corpus)
    prompts = {row.id: row for row in rows}
    require(len(rows) == len(prompts) == workload["requests"]
            and Counter(str(len(row.input_tokens)) for row in rows) == workload["groups"]
            and all(row.max_output_tokens == workload["reply_budget"] for row in rows),
            "frozen prompt inventory")
    arms = {}
    measurements = {}
    baseline = None
    for name in ARMS:
        arms[name], outputs, measurements[name] = audit_arm(folder, name, receipt["arms"][name], workload, prompts)
        if baseline is None:
            baseline = outputs
        require(outputs == baseline, f"{name}: measured output differs from solo baseline")
    require(receipt["split_activation_confirmed"] is True and all(
        (arm["shared_split_requests"] > 0) == ARMS[name][1] for name, arm in arms.items()
    ), "missing split/no-split treatment contrast")
    pairs = (("pair-off-1", "solo-1"), ("pair-on-1", "pair-off-1"),
             ("pair-on-2", "pair-off-2"), ("pair-off-2", "solo-2"))
    return {
        "binary_sha256": receipt["binary_sha256"], "corpus_sha256": workload["corpus_sha256"],
        "arms": arms,
        "paired_deltas_ms": {
            f"{current}_vs_{prior}": compare(measurements[current], measurements[prior])
            for current, prior in pairs
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="directory containing the sealed mixed serving files")
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign), indent=2, allow_nan=False))
