"""Verify and summarize native AX idle-touch screening; standard library only."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent

def digest(value):
    return hashlib.sha256(value).hexdigest()

def verify(data):
    assert digest((HERE / "bench_idle_screen.py").read_bytes()) == data["harness_sha256"]
    peer = HERE.parent / "2026-09-20-peer" / "trials.json"
    assert digest(peer.read_bytes()) == data["prior_peer_trials_sha256"]
    assert set(data["hosts"]) == {"m2", "m3", "m4", "m5"}
    count = warmups = 0
    cells = {}
    for host, entry in data["hosts"].items():
        expected = {(m, b, a) for m in ("tiel", "cyber") for b in (0, 1)
                    for a in ("wired", "touch", "unwired")}
        assert len(entry["runs"]) == len(expected)
        assert {(r["model"], r["block"], r["arm"]) for r in entry["runs"]} == expected
        for model in ("tiel", "cyber"):
            orders = [[r["arm"] for r in entry["runs"] if r["model"] == model and r["block"] == b] for b in (0, 1)]
            assert orders[0] == orders[1][::-1]
        outputs = {}
        for run in entry["runs"]:
            result, model, arm = run["result"], run["model"], run["arm"]
            assert result["complete"] and result["engine"] == "ax"
            assert result["extension_sha256"] == data["extension_sha256"]
            assert result["mlx_version"] == "0.32.2"
            assert result["greedy"] and not result["prefix_cache"]
            assert result["depth_cap"] == 3 and result["conservative"] == 0
            assert result["stream_experts"] == "off" and result["cooldown_s"] == 3
            assert result["idle_touch"] == (arm == "touch")
            assert result["idle_touch_interval_s"] == (0.5 if arm == "touch" else None)
            assert result["environment"]["AX_MLX_WIRED_LIMIT_SCALE"] == ("0" if arm == "unwired" else "0.9")
            assert result["warmups"] == 2 and result["repetitions"] == 3
            workload = next(c for c in data["workloads"][model] if c["case"] == "python-lru")
            assert digest(json.dumps(workload["token_ids"], separators=(",", ":")).encode()) == workload["sha256"]
            assert len(result["cases"]) == 1
            case = result["cases"][0]
            assert case["prompt_sha256"] == workload["sha256"]
            assert case["prompt_tokens"] == len(workload["token_ids"])
            assert case["generation_tokens"] == workload["generation_tokens"] == 256
            assert len(case["trials"]) == 3 and len(case["warmup_trials"]) == 2
            for measured, rows in ((True, case["trials"]), (False, case["warmup_trials"])):
                for row in rows:
                    assert row["measured"] == measured
                    assert len(row["output_tokens"]) == 256
                    assert row["native"]["output_tokens"] == row["output_tokens"]
                    assert row["native"]["runtime"]["host"]["detected_soc"] == entry["chip"]
                    outputs.setdefault(model, set()).add(tuple(row["output_tokens"]))
                    assert row["idle_elapsed_s"] >= 3
                    assert (row["idle_touches"] > 0) == (arm == "touch")
                    ts = [t for t, n in row["emissions"]]
                    assert all(t > 0 for t in ts) and ts == sorted(ts)
                    assert sum(n for t, n in row["emissions"]) == 256
                    first, first_n = row["emissions"][0]
                    last = row["emissions"][-1][0]
                    assert last > first and first_n < 256
                    for key, value in {"ttft_s": first, "completion_s": last,
                        "completion_tok_s": 256 / last, "decode_s": last - first,
                        "decode_tok_s": (256-first_n)/(last-first)}.items():
                        assert math.isfinite(row[key]) and math.isclose(row[key], value, rel_tol=1e-9), key
                    c = row["native"]["route"]["crossover_decisions"]
                    assert c["ax_mtp_requested"] == c["ax_mtp_correctness_mode"] == 1
                    assert c["ax_mtp_draft_tokens"] > 0 and c["ax_mtp_drafted_depth2"] > 0
                    assert c["ax_mtp_direct_fallback_steps"] == c["ax_mtp_optimistic_steps"] == 0
                    assert c["core_prefix_reuse_disabled"] == c["ax_mlx_prefix_cache_blocked"] == 1
            cells.setdefault((host, model, arm), []).extend(case["trials"])
            count += 3
            warmups += 2
        assert all(len(variants) == 1 for variants in outputs.values()), host
    return cells, count, warmups

if __name__ == "__main__":
    cells, count, warmups = verify(json.loads((HERE / "trials.json").read_text()))
    for key, rows in sorted(cells.items()):
        print(" / ".join(key), "TTFT ms", round(1000*median(r["ttft_s"] for r in rows), 2),
              "decode tok/s", round(median(r["decode_tok_s"] for r in rows), 2))
    print(f"PASS: {count} measured + {warmups} warmups; matched phase accounting, active MTP, cold KV and exact AX output parity across arms.")
