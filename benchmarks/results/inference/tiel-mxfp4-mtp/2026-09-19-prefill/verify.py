"""Validate retained trial contracts and recompute medians (no model required)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(check_source=False):
    data = json.loads((HERE / "trials.json").read_text())
    workloads = {}
    for model, cases in data["workloads"].items():
        for case in cases:
            encoded = json.dumps(case["token_ids"], separators=(",", ":")).encode()
            assert digest(encoded) == case["sha256"]
            workloads[model, case["sha256"]] = case
    assert digest((HERE / "bench_native_peer.py").read_bytes()) == data["provenance"]["harness_sha256"]
    if check_source:
        source = HERE.parents[4] / "crates/ax-engine-mlx/src/generate.rs"
        assert digest(source.read_bytes()) == data["provenance"]["source_sha256"]
    before = data["preflight_models"]
    after = data["postflight_models"]
    assert len(before) == len(after) == 2
    for pre, post in zip(before, after):
        assert pre["model"] == post["model"]
        assert pre["files"] == post["files"]
        assert all(f["sha256"] == f["expected"] for f in post["files"])
        assert not post["detected_delta"]
        assert all(n["unchanged"] for n in post["norms"])

    runs = {run["name"]: run for run in data["runs"]}
    measured = warmups = 0
    print("run / case: completion tok/s, TTFT ms, decode tok/s (medians)")
    for run in runs.values():
        assert run["complete"]
        for case in run["cases"]:
            workload = workloads[run["model"], case["prompt_sha256"]]
            assert len(workload["token_ids"]) == case["prompt_tokens"]
            assert len(case["trials"]) == run["repetitions"]
            assert len(case["warmup_trials"]) == run["warmups"]
            for bucket in ("trials", "warmup_trials"):
                for row in case[bucket]:
                    assert row["measured"] == (bucket == "trials")
                    n = case["generation_tokens"]
                    assert len(row["output_tokens"]) == n
                    events = row["emissions"]
                    assert sum(count for _, count in events) == n
                    assert all(b[0] > a[0] for a, b in zip(events, events[1:]))
                    expected = {
                        "ttft_s": events[0][0], "completion_s": events[-1][0],
                        "completion_tok_s": n / events[-1][0],
                        "decode_s": events[-1][0] - events[0][0],
                        "decode_tok_s": (n - events[0][1]) / (events[-1][0] - events[0][0]),
                    }
                    for key, value in expected.items():
                        assert math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-9)
                    assert row["first_batch_tokens"] == events[0][1]
                    assert row["decode_tokens"] == n - events[0][1]
                    assert row["api_return_s"] >= row["completion_s"]
                    assert row["memory_after"]["peak"] >= row["memory_before"]["active"]
                    if run["engine"] == "ax":
                        c = row["counters"]
                        assert row["backend"] == "mlx"
                        if "-direct" in run["name"]:
                            assert c["ax_mtp_requested"] == c["ax_mtp_draft_tokens"] == 0
                        else:
                            assert c["ax_mtp_requested"] == 1 and c["ax_mtp_draft_tokens"] > 0
                            assert c["ax_mtp_correctness_mode"] == 1
                            assert c["ax_mtp_direct_fallback_steps"] == 0
                            assert c["ax_mtp_optimistic_steps"] == 0
                        assert c["ax_mtp_ngram_proposed_tokens"] == 0
                        assert c["core_prefix_reuse_disabled"] == 1
                        assert c["ax_mlx_prefix_cache_blocked"] == 1
            measured += len(case["trials"])
            warmups += len(case["warmup_trials"])
            med = [median(r[k] for r in case["trials"]) for k in
                   ("completion_tok_s", "ttft_s", "decode_tok_s")]
            print(f"{run['name']} / {case['case']}: {med[0]:.2f}, {1000*med[1]:.2f}, {med[2]:.2f}")

    for model in ("tiel", "cyber"):
        for kind in ("short", "long", "direct"):
            base = runs[f"final-{model}-baseline-{kind}"]
            candidate = runs[f"final-{model}-candidate-{kind}"]
            assert len(base["cases"]) == len(candidate["cases"])
            for a, b in zip(base["cases"], candidate["cases"]):
                assert a["prompt_sha256"] == b["prompt_sha256"]
                for bucket in ("trials", "warmup_trials"):
                    assert [r["output_tokens"] for r in a[bucket]] == [r["output_tokens"] for r in b[bucket]]
        for arm in ("baseline", "candidate", "sustained", "turbo"):
            run = runs[f"final-{model}-{arm}-short"]
            assert run["repetitions"] == 5 and run["warmups"] == 2
            assert run["cooldown_s"] == 3 and len(run["cases"]) == 3
        repeat = runs[f"final-{model}-candidate-repeat"]
        assert repeat["repetitions"] == 20 and repeat["warmups"] == 2
        reference = runs[f"final-{model}-candidate-short"]["cases"][0]["trials"][0]["output_tokens"]
        assert all(r["output_tokens"] == reference for r in repeat["cases"][0]["trials"])

    for log in data["phase_logs"]:
        chunks = []
        finals = 0
        for row in log["rows"]:
            if row["kind"] == "mtp_chunk":
                assert row["chunk_offset"] == sum(c["chunk_len"] for c in chunks)
                chunks.append(row)
            else:
                assert row["kind"] == "mtp_final"
                assert len(chunks) == row["chunks"]
                assert sum(c["chunk_len"] for c in chunks) == row["prompt_len"]
                assert row["retained_chunks"] == sum(c["phase"].startswith("retained") for c in chunks)
                assert chunks[-1]["phase"] == "retained_final"
                chunks = []
                finals += 1
        assert not chunks and finals > 0
        run = runs[log["run"]]
        assert finals == len(run["cases"]) * (run["repetitions"] + run["warmups"])
    for primary in data["primary_reference"]:
        assert primary["repetitions"] == 5 and primary["cooldown_s"] == 3
        for row in primary["results"]:
            assert row["method"] == "mlx_lm.benchmark"
            assert row["prompt_token_ids_sha256"] in {
                key[1] for key in workloads if key[0] == primary["model"]}
            assert row["warmup_repetitions_effective"] == 2
            assert len(row["trials"]) == 5
            for metric in ("prefill_tok_s", "decode_tok_s"):
                assert math.isclose(row[metric]["median"], median(t[metric] for t in row["trials"]))
    assert data["clippy_baseline_diagnostics_identical"]
    assert {x["name"] for x in data["software_checks"] if x["exit_code"]} == {"clippy-all"}
    print(f"PASS: {measured} measured + {warmups} warmup trials; AX output identity; phase geometry; immutable packs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-source", action="store_true")
    verify(parser.parse_args().check_source)
