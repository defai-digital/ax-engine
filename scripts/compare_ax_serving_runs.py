#!/usr/bin/env python3
"""Compare repeated native serving runs collected with --record-token-ids."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any

from bench_ax_serving import SCHEMA_VERSION, output_identity


def native_ids(value: Any) -> bool:
    return isinstance(value, list) and all(type(token) is int and 0 <= token < 2**32 for token in value)


def read_run(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    run = json.loads(path.read_text())
    if run["schema_version"] != SCHEMA_VERSION or run["target"]["input_kind"] != "tokens":
        raise ValueError(f"{path}: expected a native serving artifact with token inputs")
    sampling = run["sampling"]
    if sampling["temperature"] != 0.0 or sampling["top_p"] != 1.0 or sampling["top_k"] != 0:
        raise ValueError(f"{path}: expected greedy sampling")
    requests = {}
    request_ids = set()
    for row in run["observations"]:
        if row["phase"] != "measured":
            continue
        prompt_id = row["prompt_id"]
        request_id = row.get("request_id")
        output = row.get("output_token_ids")
        prompt = row.get("prompt_token_ids")
        if (
            not row["ok"] or not native_ids(prompt) or not prompt or not native_ids(output)
            or len(prompt) != row["input_tokens"] or len(output) != row["output_tokens"]
            or type(request_id) is not int or request_id <= 0 or request_id in request_ids
        ):
            raise ValueError(f"{path}: incomplete native request {prompt_id}")
        if len(output) > row["max_output_tokens"] or (
            sampling["ignore_eos"] and len(output) != row["max_output_tokens"]
        ):
            raise ValueError(f"{path}: unexpected output length for {prompt_id}")
        if row.get("output_identity") != output_identity(output, []):
            raise ValueError(f"{path}: stream token identity differs from the terminal output for {prompt_id}")
        if prompt_id in requests:
            raise ValueError(f"{path}: repeated prompt {prompt_id} within one pass")
        requests[prompt_id] = row
        request_ids.add(request_id)
    if not requests or len(requests) != run["load"]["measured_requests"]:
        raise ValueError(f"{path}: incomplete measured workload")
    return run, requests


def workload(run: dict[str, Any], rows: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    return (
        run["corpus"]["sha256"], run["target"]["model_id"], run["sampling"],
        {key: (row["category"], row["prompt_token_ids"], row["max_output_tokens"])
         for key, row in rows.items()},
    )


def first_difference(reference: list[int], candidate: list[int]) -> dict[str, int | None] | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return {"index": index, "reference_token": left, "candidate_token": right}
    if len(reference) == len(candidate):
        return None
    index = min(len(reference), len(candidate))
    return {
        "index": index,
        "reference_token": reference[index] if index < len(reference) else None,
        "candidate_token": candidate[index] if index < len(candidate) else None,
    }


def decode_ms(row: dict[str, Any]) -> float | None:
    if row["output_tokens"] < 2 or row["client_tpot_ms"] is None:
        return None
    duration = row["client_tpot_ms"] * (row["output_tokens"] - 1)
    return duration if math.isfinite(duration) and duration > 0 else None


def compare(
    reference: list[dict[str, dict[str, Any]]], candidate: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    identical_decode_ratios = []
    paired_decode_ratios = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"prompts": 0, "exact": 0})
    for prompt_id, first in sorted(reference[0].items()):
        left = [repeat[prompt_id] for repeat in reference]
        right = [repeat[prompt_id] for repeat in candidate]
        expected = first["output_token_ids"]
        differences = [first_difference(expected, row["output_token_ids"]) for row in right]
        reference_stable = all(row["output_token_ids"] == expected for row in left)
        candidate_stable = all(row["output_token_ids"] == right[0]["output_token_ids"] for row in right)
        exact = reference_stable and all(diff is None for diff in differences)
        counts = by_category[first["category"]]
        counts["prompts"] += 1
        counts["exact"] += int(exact)
        left_times = [duration for row in left if (duration := decode_ms(row)) is not None]
        right_times = [duration for row in right if (duration := decode_ms(row)) is not None]
        ratio = None
        if len(left_times) == len(left) and len(right_times) == len(right):
            ratio = statistics.median(right_times) / statistics.median(left_times)
            paired_decode_ratios.append(ratio)
            if exact:
                identical_decode_ratios.append(ratio)
        rows.append({
            "prompt_id": prompt_id, "category": first["category"], "exact": exact,
            "reference_repeatable": reference_stable, "candidate_repeatable": candidate_stable,
            "candidate_differences_from_first_reference": differences,
            "reference_lengths": [row["output_tokens"] for row in left],
            "candidate_lengths": [row["output_tokens"] for row in right],
            "candidate_to_reference_decode_time_ratio": ratio,
        })
    return {
        "prompts": len(rows), "exact": sum(row["exact"] for row in rows), "by_category": by_category,
        "paired_decode_prompts": len(paired_decode_ratios),
        "paired_decode_time_ratio_median": statistics.median(paired_decode_ratios) if paired_decode_ratios else None,
        "identical_output_decode_prompts": len(identical_decode_ratios),
        "identical_output_decode_time_ratio_median": (
            statistics.median(identical_decode_ratios) if identical_decode_ratios else None
        ),
        "pairs": rows,
    }


def trace_requests(path: Path, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["request_id"]: row for row in rows.values()}
    rounds: dict[int, list[dict[str, int]]] = defaultdict(list)
    probes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as source:
        for line in source:
            _, marker, payload = line.partition("dflash2_round ")
            if not marker:
                _, marker, payload = line.partition("dflash2_margin ")
                if not marker:
                    continue
            fields = {key: int(value) for key, value in (word.split("=", 1) for word in payload.split())}
            request_id = fields["request_id"]
            if request_id not in by_id:
                continue
            if marker == "dflash2_round ":
                rounds[request_id].append(fields)
            else:
                position, token = fields["position"], fields["token"]
                emitted = by_id[request_id]["output_token_ids"]
                left, right = fields["left_id"], fields["right_id"]
                scores = [struct.unpack("!f", fields[f"{side}_bits"].to_bytes(4, "big"))[0]
                          for side in ("left", "right")]
                if (not 0 <= position < len(emitted) or token != emitted[position] or left == right
                    or token not in (left, right) or not all(map(math.isfinite, scores))):
                    raise ValueError(f"{path}: inconsistent verifier logit probe for request {request_id}")
                probes[request_id].append({
                    "index": position, "emitted": token, "rivals": [left, right], "logits": scores,
                    "left_minus_right": scores[0] - scores[1],
                })

    if not rounds:
        raise ValueError(f"{path}: no DFlash2 rounds for the measured request IDs")
    result = {}
    for request_id, row in by_id.items():
        records = rounds[request_id]
        previous_end = 0
        for item in records:
            start = item["generated"]
            end = start + item["emitted"]
            if (start < previous_end or end > row["output_tokens"] or item["emitted"] < 1
                or item["cohort_lanes"] < 1 or not 1 <= item["keep"] <= item["window"]
                or item["proposed"] != item["window"] - 1
                or not 0 <= item["accepted"] <= min(item["proposed"], item["emitted"])):
                raise ValueError(f"{path}: inconsistent DFlash2 trace for request {request_id}")
            previous_end = end
        emitted = sum(item["emitted"] for item in records)
        result[row["prompt_id"]] = {
            "request_id": request_id, "rounds": len(records),
            "cohort_rounds": sum(item["cohort_lanes"] > 1 for item in records),
            "drafts_proposed": sum(item["proposed"] for item in records),
            "drafts_committed": sum(item["accepted"] for item in records),
            "tokens_emitted_by_dflash2": emitted,
            "output_tokens_outside_traced_rounds": row["output_tokens"] - emitted,
            "prefix_recomputations": sum(item["recomputed"] for item in records),
            "round_trace": records,
            "logit_probes": probes[request_id],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=ARTIFACT")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--trace", action="append", default=[], metavar="ARTIFACT=SERVER_LOG")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        traces = dict(pair.split("=", 1) for pair in args.trace)
        if len(traces) != len(args.trace):
            raise ValueError("an artifact may have only one server trace")
        completed_traces = {}
        repeats = defaultdict(list)
        passes = defaultdict(list)
        seen = set()
        expected = None
        for arm in args.arm:
            name, filename = arm.split("=", 1)
            path = Path(filename).resolve()
            if not name or path in seen:
                raise ValueError(f"{arm}: expected a named arm with a distinct artifact")
            seen.add(path)
            run, rows = read_run(path)
            current = workload(run, rows)
            if expected is None:
                expected = current
            elif current != expected:
                raise ValueError(f"{filename}: workload differs from the first artifact")
            if filename in traces:
                completed_traces[filename] = trace_requests(Path(traces[filename]), rows)
            repeats[name].append(rows)
            passes[name].append({"artifact": filename, "load": run["load"], "summary": run["summary"]})
        if args.reference not in repeats or len(repeats) < 2 or set(traces) != set(completed_traces):
            raise ValueError("expected a reference and another arm, with every trace attached to an artifact")
        report = {
            "schema_version": "ax.serving_paired.v1", "reference": args.reference,
            "corpus_sha256": expected[0], "model_id": expected[1], "sampling": expected[2],
            "passes": passes, "diagnostic_traces": completed_traces,
            "comparisons": {name: compare(repeats[args.reference], runs) for name, runs in repeats.items()},
        }
    except (KeyError, OSError, OverflowError, TypeError, ValueError) as error:
        parser.error(str(error))
    result = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
