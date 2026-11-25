"""Recompute the Qwen paired serving summary from portable receipts."""

import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
PACKS = {
    "27b": "b2c5354f779e430d0c1733143db848a72b71c16e",
    "flash": "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3",
}


def measured_rows(result):
    rows = [row for row in result["observations"] if row["phase"] == "measured"]
    assert len(rows) == result["summary"]["requests"] == len({r["prompt_id"] for r in rows})
    assert all(r["ok"] and len(r["output_token_ids"]) == r["max_output_tokens"] for r in rows)
    return {r["prompt_id"]: (r["prompt_token_ids"], r["output_token_ids"]) for r in rows}


def summarize(meta, result, host, model, source):
    assert meta["server_sha256"] == source["server_sha256"]
    assert meta["model_revision"] == PACKS[model] and meta["corpus_sha256"] == result["corpus"]["sha256"]
    assert meta["hardware"] == ("Apple M5 Max" if host == "m5" else "Apple M4 Pro")
    assert meta["settings"]["AX_ENGINE_PREFIX_REUSE_DISABLED"] == "1"
    assert result["sampling"]["temperature"] == 0 and result["sampling"]["ignore_eos"]
    assert result["load"]["concurrency"] == (1 if model == "flash" or meta["arm"] == "direct" else 4)
    assert meta["pressure_levels"] and max(meta["pressure_levels"]) < 4
    summary = result["summary"]
    routes = summary["route_decisions"]
    row = {
        "arm": meta["arm"],
        "cpu_idle_percent": [meta["cpu_idle_before"], meta["cpu_idle_after"]],
        "pressure_levels": sorted(set(meta["pressure_levels"])),
        "throughput_tok_s": round(summary["output_token_throughput_tok_s"], 6),
        "ttft_p95_ms": round(summary["ttft_ms"]["p95"], 3),
        "stream_gap_p95_ms": round(summary["stream_step_interval_ms"]["p95"], 3),
        "per_prompt_tpot_ms": {r["prompt_id"]: round(r["client_tpot_ms"], 3)
                               for r in result["observations"] if r["phase"] == "measured"},
    }
    if model == "flash":
        assert routes.get("ax_mlx_flash_next_mtp_step_errors", 0) == 0
        row["verifier_steps"] = routes["ax_mlx_flash_next_mtp_verified_steps"]
        row["drafted_by_depth"] = [routes.get(f"ax_mtp_drafted_depth{i}", 0) for i in range(3)]
        if meta["arm"] == "direct":
            assert row["verifier_steps"] == 0 and row["drafted_by_depth"] == [0, 0, 0]
        else:
            assert row["verifier_steps"] > 0 and row["drafted_by_depth"][0] > 0
        if meta["arm"] == "block2":
            assert row["drafted_by_depth"][1] > 0 and row["drafted_by_depth"][2] == 0
        if meta["arm"] == "block3":
            assert row["drafted_by_depth"][2] > 0
    else:
        row["cohort_lanes"] = routes.get("ax_mlx_dflash2_cohort_lanes", 0)
        row["cohort_rows"] = routes.get("ax_mlx_dflash2_cohort_rows", 0)
        if meta["arm"] == "direct":
            assert row["cohort_lanes"] == row["cohort_rows"] == 0
        else:
            assert row["cohort_lanes"] > 0
        if meta["arm"] == "cohort4":
            assert row["cohort_rows"] <= 4 * row["cohort_lanes"]
    if meta["arm"] == "replay-probe":
        rounds = meta["round_trace"]
        evaluated = [r for r in rounds if r["replay_eval_us"] > 0]
        assert evaluated and all(1 < r["keep"] < r["window"] and r["tape_layers"] == 0 for r in evaluated)
        row["replay_profile"] = {
            "all_traced_rounds": len(rounds),
            "evaluated_partial_rounds": len(evaluated),
            "graph_build_median_us": statistics.median(r["replay_graph_us"] for r in evaluated),
            "evaluation_median_us": statistics.median(r["replay_eval_us"] for r in evaluated),
            "evaluation_max_us": max(r["replay_eval_us"] for r in evaluated),
            "pooled_eval_us_per_round": round(sum(r["replay_eval_us"] for r in evaluated) / len(rounds), 1),
        }
    return row


def replay():
    source = json.loads((ROOT / "build.json").read_bytes())
    groups = {}
    with tarfile.open(ROOT / "evidence.tar.xz") as archive:
        paths = {m.name: m for m in archive.getmembers() if m.isfile()}
        for path in sorted(p for p in paths if p.endswith("/metadata.json")):
            host, model, case, _ = path.split("/")
            base = f"{host}/{model}/{case}"
            meta = json.load(archive.extractfile(paths[path]))
            result = json.load(archive.extractfile(paths[base + "/serving.json"]))
            groups.setdefault(f"{host}/{model}", []).append(
                (summarize(meta, result, host, model, source), measured_rows(result))
            )
    assert set(groups) == {"m5/flash", "m5/27b", "m4/27b"}
    summary = {}
    for group, runs in groups.items():
        baseline_name = "canonical" if group.endswith("/flash") else "incumbent"
        baseline = next((tokens for row, tokens in runs if row["arm"] == baseline_name), runs[0][1])
        incumbents = iter(tokens for row, tokens in runs if row["arm"] == "incumbent")
        arms = {}
        for row, tokens in runs:
            assert tokens.keys() == baseline.keys()
            assert all(tokens[k][0] == baseline[k][0] for k in tokens)
            comparisons = {}
            if group == "m5/flash":
                comparisons["block3"] = next(items for arm, items in runs if arm["arm"] == "block3")
            if row["arm"] == "cohort4":
                comparisons["paired-incumbent"] = next(incumbents)
            if row["arm"] == "replay-probe":
                comparisons["replay-control"] = next(
                    items for arm, items in runs if arm["arm"] == "replay-control")
            row["matches_control"] = sum(tokens[k][1] == baseline[k][1] for k in tokens)
            row["additional_matches"] = {
                name: sum(tokens[k][1] == control[k][1] for k in tokens)
                for name, control in comparisons.items()
            }
            row["requests"] = len(tokens)
            row["match_control_arm"] = baseline_name
            arms.setdefault(row["arm"], []).append(row)
        summary[group] = arms
    assert {arm: len(rows) for arm, rows in summary["m5/flash"].items()} == {
        "block3": 2, "block2": 2, "canonical": 1, "direct": 1,
    }
    assert {arm: len(rows) for arm, rows in summary["m5/27b"].items()} == {
        "incumbent": 2, "cohort4": 2, "replay-probe": 1, "replay-control": 1, "direct": 1,
    }
    assert {arm: len(rows) for arm, rows in summary["m4/27b"].items()} == {"incumbent": 1, "cohort4": 1}
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
