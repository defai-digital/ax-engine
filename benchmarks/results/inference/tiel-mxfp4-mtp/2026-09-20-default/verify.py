#!/usr/bin/env python3
"""Verify all arms, bounds, outputs and lifecycle in the default-server artifact."""
import json
from pathlib import Path
import statistics
import re


def verify(rows):
    expected = {(m, b, a) for m in ("tiel", "cyber") for b in (0, 1) for a in ("auto", "off", "on")}
    assert len(rows) == len(expected)
    assert {(r["model"], r["block"], r["arm"]) for r in rows} == expected
    assert len({r["server_sha256"] for r in rows}) == 1
    outputs = {}
    measured = 0
    for row in rows:
        assert not row["forced_shutdown"] and not row.get("stop_guard")
        assert row["exit_code"] == 0
        events = "\n".join(row["route_events"])
        if row["arm"] == "auto":
            assert "permitted=true kv_pool_tokens=16384 prefill_chunk=2048" in events
            assert "expert streaming active:" not in events
        elif row["arm"] == "off":
            assert "evaluated bounded Tiel" not in events and "expert streaming active:" not in events
        else:
            assert "expert streaming active:" in events and "evaluated bounded Tiel" not in events
        assert row["samples"] and all(s["pressure"] == "1" for s in row["samples"])
        def swap_mib(value):
            match = re.search(r"used = ([0-9.]+)M", value)
            assert match
            return float(match.group(1))
        baseline = swap_mib(row["before"]["swap"])
        assert max(swap_mib(sample["swap"]) for sample in row["samples"]) - baseline <= 256
        assert len(row["trials"]) == (4 if row["arm"] == "on" else 7)
        assert [t["warmup"] for t in row["trials"] if t["case"] == "short"] == [True, False, False, False]
        if row["arm"] != "on":
            assert [t["warmup"] for t in row["trials"] if t["case"] == "long"] == [True, False, False]
        assert row["models"]["data"][0]["limit"]["context"] == 16384
        assert row["models"]["data"][0]["runtime"]["host"]["detected_soc"] == "Apple M4 Pro"
        minimum_span = row["startup_s"] + sum(t["wall_s"] + 3 for t in row["trials"]) - 2
        assert row["samples"][-1]["elapsed_s"] >= minimum_span
        assert all(b["elapsed_s"] - a["elapsed_s"] <= 5 for a, b in zip(row["samples"], row["samples"][1:]))
        for trial in row["trials"]:
            assert trial["done"] and not trial["cancelled"]
            assert trial["request"]["temperature"] == 0 and trial["request"]["seed"] == 0
            assert trial["request"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert trial["usage"]["completion_tokens"] == trial["request"]["max_tokens"]
            assert trial["ttft_s"] > 0 and trial["wall_s"] >= trial["ttft_s"]
            # Match the same request/cache phase across arms and process blocks.
            # Long first-request vs logical-prefix-warmup divergence exists in
            # both Auto and Off and is retained explicitly in the report.
            phase = trial["warmup"] if trial["case"] == "long" else None
            key = row["model"], trial["case"], phase
            identity = (trial["prompt_sha256"], trial["usage"]["prompt_tokens"], trial["text"])
            assert outputs.setdefault(key, identity) == identity, (key, row["arm"], "output mismatch")
            if trial["case"] == "long":
                assert 14500 <= trial["usage"]["prompt_tokens"] <= 16256
            assert "ax_engine_mlx_mtp_model_policy_active 0" in trial["metrics_after"]
            measured += not trial["warmup"]
        if row["arm"] != "on":
            assert row["cancel"]["cancelled"] and row["cancel"]["ttft_s"] is not None
            assert row["recovery"]["done"] and row["recovery"]["usage"]["completion_tokens"] == 16
            assert row["recovery"]["text"] == outputs[(row["model"], "short", None)][2]
    return measured


if __name__ == "__main__":
    rows = json.loads(Path(__file__).with_name("trials.json").read_text())
    print("Verified measured requests:", verify(rows))
    for model in ("tiel", "cyber"):
        for arm in ("auto", "off", "on"):
            for case in ("short", "long"):
                trials = [t for r in rows if r["model"] == model and r["arm"] == arm
                          for t in r["trials"] if t["case"] == case and not t["warmup"]]
                if trials:
                    print(model, arm, case, "n", len(trials), "TTFT_ms", round(statistics.median(t["ttft_s"] for t in trials)*1000,2),
                          "completion_tps", round(statistics.median(t["completion_tps"] for t in trials),2))
