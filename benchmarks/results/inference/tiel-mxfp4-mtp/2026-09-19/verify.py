#!/usr/bin/env python3
"""Regrade the retained diagnostic trials; optionally bind current source."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    data = json.loads((here / "trials.json").read_text())
    assert data["claim_status"] == "diagnostic_only"
    assert len(data["sessions"]) == 8
    if args.check_source:
        root = here.parents[4]
        for name, expected in data["build"]["source_files_sha256"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

    cells = {}
    for session in data["sessions"]:
        assert session["host"]["chip"] == "Apple M5 Max"
        assert session["host"]["memory_gb"] == 128
        assert session["repetitions"] == 5 and session["warmup_repetitions"] == 2
        assert session["prefix_cache_mode"] == "disabled_for_cold_prefill_benchmark"
        assert session["mtp_policy"] == "required" and not session["ngram_stacking"]
        for row in session["rows"]:
            assert row["sampler_settings"] == "greedy"
            assert row["memory_source"] == "server_process_rss_after_stream"
            prompt = row["prompt_artifact"]["token_ids"]
            assert len(prompt) == row["prompt_tokens"]
            assert token_hash(prompt) == row["prompt_token_ids_sha256"]
            assert len(row["trials"]) == 5
            assert len({tuple(t["output_token_ids"]) for t in row["trials"]}) == 1
            for trial in row["trials"]:
                assert trial["output_tokens"] == row["generation_tokens"]
                assert len(trial["output_token_ids"]) == trial["output_tokens"]
                assert trial["decode_s"] > 0
                assert math.isclose(
                    trial["decode_tok_s"], trial["output_tokens"] / trial["decode_s"],
                    rel_tol=1e-6,
                )
                telemetry = trial["ngram_acceleration_telemetry"]
                assert telemetry["ax_mtp_correctness_mode"] == 1
                assert telemetry["ax_mtp_proposal_law"] == 1
                assert telemetry["ax_mtp_draft_tokens"] > 0
                assert telemetry["ax_mtp_accepted_tokens"] > 0
                for key in ("direct_fallback_steps", "optimistic_steps", "auto_optimistic_steps",
                            "draft_source_ngram_tokens", "correctness_mode_conflicts",
                            "proposal_law_conflicts"):
                    assert telemetry["ax_mtp_" + key] == 0, key
                assert telemetry["ax_mtp_conservative_depth_code"] == int(session["conservative"])
                active = session["conservative"] and session["model"] == "tiel" and session["workload"] == "random"
                decisions = telemetry["ax_mtp_conservative_depth_decisions"]
                assert (decisions > 0) if active else (decisions == 0)
            key = (session["model"], session["workload"], row["prompt_tokens"])
            cells.setdefault(key, {})[session["conservative"]] = row

    assert len(cells) == 6
    print("model workload prompt off_tok_s on_tok_s change output_ids_equal decisions")
    for (model, workload, length), pair in sorted(cells.items()):
        off, on = pair[False], pair[True]
        assert off["prompt_artifact"]["token_ids"] == on["prompt_artifact"]["token_ids"]
        assert off["sampler_settings"] == on["sampler_settings"]
        assert off["generation_tokens"] == on["generation_tokens"]
        equal = [t["output_token_ids"] for t in off["trials"]] == [t["output_token_ids"] for t in on["trials"]]
        if (model, workload) != ("tiel", "random"):
            assert equal, (model, workload)
        before = median(t["decode_tok_s"] for t in off["trials"])
        after = median(t["decode_tok_s"] for t in on["trials"])
        decisions = sum(t["ngram_acceleration_telemetry"]["ax_mtp_conservative_depth_decisions"] for t in on["trials"])
        print(f"{model} {workload} {length} {before:.2f} {after:.2f} {(after / before - 1) * 100:+.2f}% {equal} {decisions}")

    for reference in data["primary_reference"]:
        row = reference["results"][0]
        assert row["prompt_token_ids_sha256"] == cells[(reference["model"], "random", 128)][False]["prompt_token_ids_sha256"]
        assert len(row["trials"]) == 3
    print("Evidence checks passed. This is not certification or a quality evaluation.")


if __name__ == "__main__":
    main()
