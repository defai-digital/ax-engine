"""Verify recorded PATH regression evidence without loading model weights."""
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
data = json.loads((root / "trials.json").read_text())
assert data["hardware"] == {"chip": "Apple M5 Max", "memory_bytes": 137438953472}
assert len(data["runs"]) == 8
assert len({run["name"] for run in data["runs"]}) == 8
for pack in data["pack_verification"]:
    assert not pack["detected_delta"]
    assert all(file["sha256"] == file["expected"] for file in pack["files"])
    assert all(norm["unchanged"] for norm in pack["norms"])
expected = {}
count = 0
previous_end = 0
for run in data["runs"]:
    assert run["exit_code"] == 0 and run["complete"]
    assert previous_end <= run["started"] < run["ended"]
    previous_end = run["ended"]
    model, arm, path = run["name"].split("-")
    assert model in {"tiel", "cyber"} and arm in {"baseline", "candidate"}
    assert path in {"normal", "restricted"}
    assert run["extension_sha256"] == data["extensions"][arm]
    assert (run["wired_limit_after_requests"] > 0) == (arm == "baseline" and path == "restricted")
    case = run["case"]
    tokens = data["inputs"][model]["token_ids"]
    assert case["prompt_sha256"] == hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()
    assert case["prompt_tokens"] == len(tokens)
    assert case["generation_tokens"] == data["inputs"][model]["generation_tokens"]
    assert len(case["trials"]) == 5 and len(case["warmup_trials"]) == 2
    for measured, rows in [(False, case["warmup_trials"]), (True, case["trials"])]:
        for row in rows:
            assert row["measured"] == measured
            assert row["finish_reason"] == "max_output_tokens"
            assert row["mtp"]["ax_mtp_requested"] == 1
            assert row["mtp"]["ax_mtp_source_mtp_submitted_tokens"] > 0
            assert row["mtp"]["ax_mtp_source_mtp_accepted_tokens"] > 0
            assert row["mtp"]["ax_mtp_drafted_depth2"] > 0
            assert row["output_tokens"] == expected.setdefault(model, row["output_tokens"])
            assert len(row["output_tokens"]) == case["generation_tokens"]
            events = row["emissions"]
            assert all(a[0] < b[0] for a, b in zip(events, events[1:]))
            assert sum(n for _, n in events) == len(row["output_tokens"])
            first, first_n = events[0]
            last = events[-1][0]
            assert row["first_batch_tokens"] == first_n
            assert 0 < first < last <= row["api_return_s"]
            for actual, wanted in [(row["ttft_s"], first), (row["completion_s"], last),
                                   (row["completion_tok_s"], len(row["output_tokens"]) / last),
                                   (row["decode_tok_s"], (len(row["output_tokens"]) - first_n) / (last - first))]:
                assert math.isclose(actual, wanted, rel_tol=1e-8)
            count += 1
if "--check-source" in sys.argv:
    repo = root.parents[4]
    for name, digest in data["source_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
print(f"PASS: {count} requests, token parity, timing boundaries, residency and sequential execution")
