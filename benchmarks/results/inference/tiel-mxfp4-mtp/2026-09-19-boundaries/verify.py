"""Verify recorded native boundary probes without model inference."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
doc = json.loads((root / "results.json").read_text())
assert doc["hardware"] == {"chip": "Apple M5 Max", "memory_bytes": 137438953472}
assert len(doc["runs"]) == 4
assert {r["name"] for r in doc["runs"]} == {
    f"{model}-{arm}" for model in ("tiel", "cyber") for arm in ("baseline", "candidate")
}
expected = {}
previous_end = 0
for run in doc["runs"]:
    model, arm = run["name"].split("-")
    assert run["exit_code"] == 0 and run["complete"]
    assert previous_end <= run["started"] < run["ended"]
    previous_end = run["ended"]
    assert run["extension_sha256"] == doc["extensions"][arm]
    assert run["prompts"] == expected.setdefault((model, "prompts"), run["prompts"])
    assert len(run["valid"]) == 20 and len(run["rejected"]) == 4
    assert {(r["prompt"], r["budget"], r["repeat"]) for r in run["valid"]} == {
        (kind, budget, repeat) for kind, budgets in
        [("coding", [1, 2, 3, 4, 5, 16]), ("single", [1, 2, 4, 5])]
        for budget in budgets for repeat in (0, 1)
    }
    for row in run["valid"]:
        assert len(row["tokens"]) == sum(row["batches"]) == row["budget"]
        assert all(n >= 0 for n in row["batches"])
        assert row["finish_reason"] == "max_output_tokens"
        key = (model, row["prompt"], row["budget"])
        assert row["tokens"] == expected.setdefault(key, row["tokens"])
        if row["budget"] == 16:
            assert row["mtp"]["ax_mtp_source_mtp_submitted_tokens"] > 0
            assert row["mtp"]["ax_mtp_drafted_depth2"] > 0
    assert {r["case"] for r in run["rejected"]} == {
        "zero_budget", "empty_prompt", "negative_budget", "overflow_budget"
    }
    for row in run["rejected"]:
        assert row["type"] == ("ValueError" if row["case"] in
                               {"zero_budget", "empty_prompt"} else "OverflowError")
        key = (model, row["case"], "error")
        assert row["message"] == expected.setdefault(key, row["message"])
        recovery = row["recovery"]
        assert recovery["tokens"] == expected[(model, "coding", 16)]
        assert len(recovery["tokens"]) == sum(recovery["batches"]) == 16
        assert recovery["finish_reason"] == "max_output_tokens"
if "--check-source" in sys.argv:
    for name, digest in doc["source_sha256"].items():
        assert hashlib.sha256((root.parents[4] / name).read_bytes()).hexdigest() == digest, name
print("PASS: 80 short requests, 16 rejections, 16 recoveries, exact token parity and MTP activity")
