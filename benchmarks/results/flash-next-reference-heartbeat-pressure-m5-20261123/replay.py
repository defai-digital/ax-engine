"""Replay the independent reference admission failure from the retained counters."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
MEMBERS = {
    "status.txt", "outcome.json", "heartbeat/status.json", "heartbeat/token-progress.jsonl",
    "heartbeat/samples.jsonl", "clean-preflight.json", "reference-preflight.json",
}
GIB = 1 << 30


def replay():
    with tarfile.open(ROOT / "evidence.tar.xz") as archive:
        assert len(archive.getmembers()) == len(MEMBERS) and set(archive.getnames()) == MEMBERS
        files = {}
        for name in MEMBERS:
            member = archive.getmember(name)
            assert member.isfile()
            with archive.extractfile(member) as stream:
                files[name] = stream.read()
    assert not any(key in data for data in files.values()
                   for key in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    clean = json.loads(files["clean-preflight.json"])
    preflight = json.loads(files["reference-preflight.json"])
    assert {key: value for key, value in clean.items() if key != "capacity"} == preflight
    assert preflight["macos"] == "27.2.1" and preflight["versions"]["mlx"] == "0.32.2"
    assert preflight["language_sha256"] == "30f059d6289451bde0f5ef816090ff946ebc857d1b557f83415e2ba867d76199"
    cap = clean["capacity"]
    assert cap["eligible_for_attempt"] is True and cap["pressure"] == 1
    assert cap["physical_bytes"] == 128 * GIB
    margin = cap["physical_bytes"] - cap["signed_peak_mlx_bytes"]
    assert margin == cap["observed_signed_margin_bytes"] >= cap["largest_background_rss_bytes"]
    state = json.loads(files["heartbeat/status.json"])
    outcome = json.loads(files["outcome.json"])
    tokens = [json.loads(row) for row in files["heartbeat/token-progress.jsonl"].splitlines()]
    samples = [json.loads(row) for row in files["heartbeat/samples.jsonl"].splitlines()]
    assert files["status.txt"] == b"1\n" and outcome == {"phase": "heartbeat-stopped", "exit_code": 1}
    assert state["diagnostic_only"] and state["qa_limit"] == 4 and not state["wired_residency_override"]
    assert state["stop_reason"] == "critical system memory pressure"
    assert state["complete"] is False and state["completed_cases"] == 3 and state["exit_code"] == -15
    assert tokens[-1] == {"case": 4, "tokens": 114}
    assert Counter(row["pressure"] for row in samples) == {1: 98, 2: 189, 4: 1}
    last = samples[-1]
    assert last["pressure"] == 4
    swap = re.search(r"\bused = ([0-9.]+)M\b", last["swap"])
    assert swap
    return {
        "scope": "Four-case independent reference admission",
        "macos": preflight["macos"], "complete": False, "ordinary_reference_started": False,
        "heartbeat_cases_completed": state["completed_cases"], "last_case": tokens[-1],
        "stop_reason": state["stop_reason"], "pressure_samples": dict(sorted(Counter(
            row["pressure"] for row in samples).items())),
        "preflight_background_gib": round(cap["largest_background_rss_bytes"] / GIB, 2),
        "signed_margin_gib": round(margin / GIB, 2),
        "final_largest_other_rss_gib": round(last["largest_other_process_rss_bytes"] / GIB, 2),
        "final_compressor_gib": round(last["compressor_bytes"] / GIB, 2),
        "final_swap_used_mib": float(swap[1]),
        "preflight_sha256": sha256(files["reference-preflight.json"]).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
