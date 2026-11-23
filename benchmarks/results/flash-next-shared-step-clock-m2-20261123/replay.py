"""Recheck exact shared-step identities and the nested clock intervals."""
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

from fixtures import fixtures

HERE = Path(__file__).resolve().parent
SOURCE = "flash-next-natural-short-drain-m5-20261123"
PREFILL = "ax_mlx_prefill_wall_us"
FORWARD = "ax_mlx_prefill_forward_wall_us"
SPLIT_END = "ax_scheduler_prefill_split_end"
VERIFIED = "ax_mlx_flash_next_mtp_verified_steps"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clocks(row):
    fields = [row[key] for key in ("runner_us", "core_us", "step_http_us")]
    assert all(type(value) is int for value in fields)
    assert 0 < fields[0] <= fields[1] <= fields[2]


def prefill_interval(mixed, history):
    final, previous = history[-1], history[-2]
    baseline = previous["decisions"]
    if previous["scheduled_requests"] != 2:
        # The preceding short-only prefill does not carry the long
        # request's retained counters from its separate bootstrap.
        assert len(history) == 2 and previous["short_after"] == 1
        assert previous["long_before"] == previous["long_after"]
        bootstrap = mixed.get("long_bootstrap")
        if bootstrap is None:
            return None, None
        clocks(bootstrap)
        assert bootstrap["step_id"] + 1 == previous["step_id"]
        assert bootstrap["scheduled_requests"] == 1
        baseline = {key: baseline[key] + bootstrap["decisions"][key] for key in (PREFILL, FORWARD)}
    else:
        assert previous["long_after"] > previous["long_before"]
        assert previous["short_after"] == previous["short_before"] + 1
    forward = final["decisions"][FORWARD] - baseline[FORWARD]
    prefill = final["decisions"][PREFILL] - baseline[PREFILL]
    assert 0 < forward <= prefill <= final["runner_us"]
    return forward, prefill


def check_case(case, row):
    assert row["short_id"] == case["short_id"]
    mixed, solo = row["mixed"], row["solo"]
    history = mixed["history"]
    assert history and all(next_row["step_id"] == prev["step_id"] + 1
                           for prev, next_row in zip(history, history[1:]))
    cursor = case["original_short_priority_long_cursor"]
    widths = [cursor]
    depth = 0
    for step in history:
        clocks(step)
        assert (step["long_before"], step["short_before"]) == (cursor, depth)
        width, emitted = step["long_after"] - cursor, step["short_after"] - depth
        assert width in (0, 1, 511, 512) and emitted in (0, 1) and width + emitted > 0
        cursor, depth = step["long_after"], step["short_after"]
        if width:
            widths.append(width)
        assert not step["decisions"].get(VERIFIED)
    final = history[-1]
    assert widths[:-1] == case["long_completed_widths"]
    assert final["long_before"] == case["long_prefill_cursor_before_step"]
    assert widths[-1] == case["observed_mixed_prefill_width"]
    assert final["decisions"][f"{SPLIT_END}.{mixed['long_request_id']}"] == case["observed_logical_end"]
    assert final["scheduled_requests"] == 2 and final["scheduled_tokens"] == widths[-1] + 1
    expected = [*case["short_emitted_ids_before_step"], case["short_token_at_shared_step"]]
    assert final["short_before"] + 1 == final["short_after"] == len(expected)
    assert mixed["short_request_id"] != mixed["long_request_id"]
    assert mixed["output"] == solo["output"] and row["same_host_solo_ids_match"] is True
    assert len(mixed["output"]) == len(expected)
    assert row["archived_ids_match"] == (mixed["output"] == expected)
    assert sorted(mixed["drained"]) == sorted(solo["drained"]) and len(mixed["drained"]) == 5
    assert not any(mixed["drained"].values()) and not any(solo["drained"].values())
    clocks(solo["step"])
    assert solo["step"]["scheduled_requests"] == solo["step"]["scheduled_tokens"] == 1
    forward_us, prefill_us = prefill_interval(mixed, history)
    return {
        "short_id": row["short_id"], "width": widths[-1], "output_tokens": len(expected),
        "archived_ids_match": row["archived_ids_match"],
        "archive_sse_gap_ms": case["source_output_gap_ms"],
        "prefill_forward_us": forward_us, "prefill_interval_us": prefill_us,
        "mixed_runner_us": final["runner_us"],
        "mixed_core_us": final["core_us"], "mixed_step_http_us": final["step_http_us"],
        "same_step_runner_outside_prefill_us": (
            None if prefill_us is None else final["runner_us"] - prefill_us
        ),
        "same_step_core_outside_runner_us": final["core_us"] - final["runner_us"],
        "solo_decode_runner_us": solo["step"]["runner_us"],
    }


def replay(folder, source):
    saved = json.loads((folder / "fixtures.json").read_bytes())
    assert saved == fixtures(source)
    receipt = json.loads((folder / "receipt.json").read_bytes())
    assert receipt["fixtures_sha256"] == sha(folder / "fixtures.json")
    assert receipt["probe_sha256"] == sha(folder / "probe.py")
    assert receipt["build"]["server_sha256"] == saved["server_sha256"]
    assert receipt["build"]["mlx"] == receipt["build"]["mlx_metal"] == "0.32.4"
    assert receipt["qualification"] is False
    assert receipt["storage"]["internal"] and receipt["storage"]["solid_state"]
    assert (folder / "status.txt").read_text().strip() == "0"
    pressure = [json.loads(line) for line in (folder / "pressure.jsonl").read_text().splitlines()]
    assert len(pressure) == receipt["pressure_samples"] > 0
    assert all(row["pressure"] == 1 and 0 < row["footprint_bytes"] <= 96 * 2**30 for row in pressure)
    cases = [check_case(case, row) for case, row in zip(saved["cases"], receipt["rows"], strict=True)]
    fractions = [row["prefill_forward_us"] / row["mixed_runner_us"] for row in cases
                 if row["prefill_forward_us"] is not None]
    return {
        "hardware": {key: receipt[key] for key in ("soc", "memory_bytes", "macos", "storage")},
        "samples": cases, "exact_same_host_tokens": sum(row["output_tokens"] for row in cases),
        "exact_archive_case_count": sum(row["archived_ids_match"] for row in cases),
        "prefill_attributed_cases": len(fractions),
        "mixed_median_forward_fraction": median(fractions),
        "qualification": False,
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / SOURCE
    print(json.dumps(replay(folder, source), indent=2, sort_keys=True, allow_nan=False))
