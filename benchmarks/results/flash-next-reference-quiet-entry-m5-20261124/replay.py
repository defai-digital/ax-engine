"""Verify the frozen ordinary reference's quiet-entry admission attempt."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
GIB = 1 << 30


def document(name):
    return json.loads((RAW / name).read_bytes())


def rows(name):
    return [json.loads(line) for line in (RAW / name).read_bytes().splitlines()]


def swap_used_mib(sample):
    return float(re.search(r"\bused = ([0-9.]+)M\b", sample["swap"])[1])


def replay():
    start = document("clean-preflight.json")
    preflight = document("reference-preflight.json")
    assert {key: value for key, value in start.items() if key != "capacity"} == preflight
    assert preflight["source_sha256"] == "b1c8f8e41923e9e2076b824b4e055bf458a1fce39761ff5a964658f541ce1897"
    assert preflight["signed_reference_sha256"] == "2eeeb8ab3b5f329726550b675ec61d57edeab6558a582582d75287983d4a19b7"
    assert preflight["macos"] == "27.2.1" and preflight["versions"]["mlx"] == "0.32.2"

    headroom = [start["capacity"], *rows("headroom.jsonl")]
    assert len(headroom) == 3
    for entry in headroom:
        assert entry["physical_bytes"] == 128 * GIB
        margin = entry["physical_bytes"] - entry["signed_peak_mlx_bytes"]
        assert margin == entry["observed_signed_margin_bytes"] == 6917952302
        assert 0 < entry["largest_other_process_rss_bytes"] == entry["largest_background_rss_bytes"] < margin
        assert entry["eligible_for_attempt"] is True and entry["pressure"] == 1
        assert 90 <= entry["cpu_idle_percent"] <= 100

    partial = document("heartbeat/reference/result.json")
    assert preflight["versions"] == partial["versions"] == partial["reference"]["versions"]
    assert preflight["language_sha256"] == partial["language_sha256"] == partial["reference"]["language_sha256"]
    assert partial["reference"]["commit"] == "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb"
    assert partial["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert partial["inventory_sha256"] == "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
    assert partial["hardware"]["os"] == preflight["macos"]
    assert partial["hardware"]["memory_bytes"] == 128 * GIB and partial["hardware"]["storage"]["internal"]
    assert partial["weights_materialized"] == 2326 and partial["weights_materialize_seconds"] > 0
    assert partial["phases"] == ["qa"] and partial["phase"] == "qa" and partial["qa"] == {"rows": []}
    assert partial["completed"] is partial["release_ready"] is partial["qualification"] is False
    assert partial["host_quiet"]["cpu_idle_percent"] >= 90

    outcome = document("outcome.json")
    status = document("heartbeat/status.json")
    assert (RAW / "status.txt").read_bytes() == b"1\n"
    assert outcome == {"phase": "heartbeat-stopped", "exit_code": 1}
    assert status == {
        "diagnostic_only": True, "frozen_source_manifest_sha256": preflight["source_sha256"],
        "qa_limit": 4, "wired_residency_override": False, "exit_code": -15,
        "stop_reason": "critical system memory pressure", "complete": False, "completed_cases": 0,
    }
    progress = rows("heartbeat/token-progress.jsonl")
    assert progress == [{"case": 1, "tokens": 0}]
    samples = rows("heartbeat/samples.jsonl")
    assert [sample["pressure"] for sample in samples] == [1] * 8 + [2, 4]
    final = samples[-1]
    assert final["token_progress_seen"]
    return {
        "scope": "Four-case independent reference admission on M5 Max 128 GiB",
        "macos": preflight["macos"],
        "ordinary_reference_started": False,
        "heartbeat_cases_completed": status["completed_cases"],
        "last_progress": progress[-1],
        "stop_reason": status["stop_reason"],
        "pressure_samples": dict(sorted(Counter(s["pressure"] for s in samples).items())),
        "startup_cpu_idle_percent": start["capacity"]["cpu_idle_percent"],
        "runner_cpu_idle_percent": partial["host_quiet"]["cpu_idle_percent"],
        "signed_margin_gib": round(headroom[0]["observed_signed_margin_bytes"] / GIB, 2),
        "startup_largest_other_rss_gib": round(headroom[0]["largest_background_rss_bytes"] / GIB, 2),
        "startup_compressor_gib": round(headroom[0]["compressor_bytes"] / GIB, 2),
        "weights_materialized": partial["weights_materialized"],
        "weights_materialize_seconds": partial["weights_materialize_seconds"],
        "final_largest_other_rss_gib": round(final["largest_other_process_rss_bytes"] / GIB, 2),
        "final_compressor_gib": round(final["compressor_bytes"] / GIB, 2),
        "final_swap_used_mib": swap_used_mib(final),
        "preflight_sha256": sha256((RAW / "reference-preflight.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
