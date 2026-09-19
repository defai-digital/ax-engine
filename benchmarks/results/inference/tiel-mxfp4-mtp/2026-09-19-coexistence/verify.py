"""Verify bounded co-residency evidence without loading model weights."""
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
doc = json.loads((root / "results.json").read_text())
assert len(doc["runs"]) == 2
assert {r["order"] for r in doc["runs"]} == {"tiel", "cyber"}
golden = {}
for run in doc["runs"]:
    assert run["complete"] and run["extension_sha256"] == doc["extension_sha256"]
    assert run["hardware"] == {"chip": "Apple M5 Max", "memory_bytes": 128 * 1024**3}
    isolated = next(p for p in run["phases"] if p["name"] == "isolated")
    assert len(isolated["rows"]) == 3
    golden[run["order"]] = isolated["rows"][0]["tokens"]
count = 0
prior_end = 0
for run in doc["runs"]:
    assert prior_end <= run["started"] < run["ended"]
    prior_end = run["ended"]
    assert {"isolated", "sequential", "concurrent", "cancel-and-reuse", "survivor"} <= {
        p["name"] for p in run["phases"]
    }
    for phase in run["phases"]:
        expected_rows = {"isolated-warmup": 2, "isolated": 3, "co-resident-warmup": 4,
                         "sequential": 6, "concurrent": 6, "cancel-and-reuse": 2, "survivor": 3}
        if phase["name"] in expected_rows:
            assert len(phase["rows"]) == expected_rows[phase["name"]]
        elif phase["name"].startswith("released-"):
            assert len(phase["rows"]) == 2
        else:
            assert phase["name"].startswith("competing-") and len(phase["rows"]) in (0, 2, 4, 6)
        assert phase["before"]["wired_limit"] == phase["after"]["wired_limit"] == 0
        for row in phase["rows"]:
            assert row["tokens"] == golden[row["model"]] and len(row["tokens"]) == 128
            assert row["finish_reason"] == "max_output_tokens"
            assert row["mtp"]["ax_mtp_source_mtp_submitted_tokens"] > 0
            assert row["mtp"]["ax_mtp_drafted_depth2"] > 0
            events = row["emissions"]
            assert sum(n for _, n in events) == 128
            assert all(a[0] < b[0] for a, b in zip(events, events[1:]))
            first, first_n = events[0]
            last = events[-1][0]
            assert row["started"] < first < last <= row["ended"]
            assert math.isclose(row["ttft_s"], first-row["started"], rel_tol=1e-8)
            assert math.isclose(row["completion_tok_s"], 128/(last-row["started"]), rel_tol=1e-8)
            assert math.isclose(row["decode_tok_s"], (128-first_n)/(last-first), rel_tol=1e-8)
            count += 1
        if phase["name"] == "concurrent" or phase["name"].startswith(("competing-", "released-")):
            for rep in {r["rep"] for r in phase["rows"]}:
                pair = [r for r in phase["rows"] if r["rep"] == rep]
                assert len(pair) == 2
                assert max(r["started"] for r in pair) < min(r["ended"] for r in pair)
    for pressure in run["pressure"]:
        assert pressure["exit_code"] == 0
        released = pressure["released"]
        if not pressure["held_through_requests"]:
            assert released["reason"] in {"compressor_growth", "headroom", "swap_growth", "critical_pressure", "deadline"}
            assert pressure is run["pressure"][-1], "must not escalate after guard stop"
            continue
        assert pressure["ready"]["event"] == "ready" and released["reason"] == "released"
        committed = pressure["ready"]["snapshot"]
        samples = [committed] + [s for s in released["samples"] if s["time"] >= committed["time"]]
        target = pressure["requested_gib"] * 1024**3
        assert all(s["rss"] >= 0.9 * target and s["pressure"] < 4 for s in samples)
        assert released["after"]["rss"] < target * 0.1
        phase = next(p for p in run["phases"] if p["name"] == f"competing-{pressure['requested_gib']}gib")
        assert len(phase["rows"]) == 6
        assert max(r["ended"] for r in phase["rows"]) <= released["release_started"]
        assert any(r["started"] <= s["time"] <= r["ended"] for r in phase["rows"] for s in samples)
attempted = [p["requested_gib"] for r in doc["runs"] for p in r["pressure"]]
assert attempted and attempted == [16, 32, 48][:len(attempted)]
assert count == doc["completed_requests"]
if "--check-source" in sys.argv:
    for name, digest in doc["source_sha256"].items():
        assert hashlib.sha256((root.parents[4] / name).read_bytes()).hexdigest() == digest, name
qualified = [p["requested_gib"] for r in doc["runs"] for p in r["pressure"] if p["held_through_requests"]]
print(f"PASS: {count} completed requests, token parity and overlap; clean retained loads: {qualified} GiB")
