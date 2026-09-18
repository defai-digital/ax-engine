#!/usr/bin/env python3
"""Check retained first-split evidence without loading a model or using a network."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_regression(publication):
    final = read("regression/final.json")
    validation = read("source-validation.json")
    require(digest((ROOT / "regression/final.patch").read_bytes()) == final["patch_sha256"] == validation["runtime_patch_sha256"], "frozen runtime patch")
    require(final["source_sha256"] == validation["frozen_runtime_source_sha256"], "frozen source files")
    for label, code, marker in [
        ("before", 101, "left: 3\n right: 0"),
        ("skip-before", 101, "forced replay must not reuse batched logits"),
        ("final-focused", 0, "3 passed; 0 failed"),
        ("final-linear-mtp", 0, "13 passed; 0 failed"),
        ("final-ngram-tests", 0, "50 passed; 0 failed"),
    ]:
        receipt = read(f"regression/{label}.json")
        artifact = publication["artifacts"][f"regression/{label}.log"]
        require(receipt["exit_code"] == code, f"regression exit: {label}")
        require(receipt["log_sha256"] == artifact["raw_source_sha256"], f"original regression log hash: {label}")
        require(marker in (ROOT / f"regression/{label}.log").read_text(), f"regression evidence: {label}")
        if label.startswith("final-"):
            require(receipt["source_sha256"] == final["source_sha256"], f"final test source: {label}")
    broad = validation["broad_validation"]
    require(broad["complete"] and broad["source_before"] == broad["source_after"], "completed stable-source validation")
    require(all(broad["source_after"][k] == v for k, v in final["source_sha256"].items()), "integrated runtime identity")
    checks = broad["checks"]
    require(len(checks) == 11, "eleven broad validation outcomes")
    require([(x["check"], x["exit_code"]) for x in checks if x["exit_code"]] == [("strict-clippy", 101)], "preserved strict Clippy failure")
    require(not broad["all_commands_exit_zero"] and not broad["strict_clippy_passed"], "no false green Clippy claim")
    for check in checks:
        artifact = publication["artifacts"][check["log_file"]]
        require(check["raw_log_sha256"] == artifact["raw_source_sha256"], "original validation log hash")
        require(check["published_log_sha256"] == artifact["published_sha256"], "sanitized validation log hash")
    require("209 passed, 26 skipped, 140 subtests passed" in (ROOT / "regression/validation/python.log").read_text(), "Python validation count")
    reviews = validation["advisory_reviews"]
    require(len(reviews) == 5 and sum(x["timed_out"] for x in reviews) == 4, "advisory completion/timeouts")
    require(not validation["default_qwen_split_fixed_claim"] and not validation["release_ready_claim"], "regression scope")


def main():
    publication = read("publication.json")
    for name, receipt in publication["artifacts"].items():
        data = (ROOT / name).read_bytes()
        require(digest(data) == receipt["published_sha256"], f"published hash: {name}")
        require(len(data) == receipt["bytes"], f"published length: {name}")
    case = read("case.json")
    request = read("request.json")
    require(request == case["baseline_request"], "retained request convention")
    require(request["input_tokens"] == case["prompt"], "full prompt token identity")
    require(len(case["prompt"]) == 409, "prompt length")
    require(case["divergence_index"] == 116, "split index")
    require(case["absolute_input_offset"] == 524, "absolute position")
    direct = case["baseline_direct_output"]
    mtp = case["baseline_mtp_output"]
    require(direct[:116] == mtp[:116] == case["common_generated_prefix"], "prefix")
    require((direct[116], mtp[116], direct[117]) == (279, 6397, 40278), "split IDs")
    runs = read("runs.json")
    require([r["name"] for r in runs] == ["clean-direct", "clean-mtp", "observer-mtp"], "arms")
    for run in runs:
        response_bytes = (ROOT / run["response_file"]).read_bytes()
        require(digest(response_bytes) == run["response_body_sha256"], "response body hash")
        require(digest((ROOT / "request.json").read_bytes()) == run["request_body_sha256"], "request body hash")
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        require(digest(canonical) == run["request_canonical_sha256"], "canonical request")
        response = json.loads(response_bytes)
        expected = direct if run["mode"] == "direct" else mtp
        require(len(response["output_tokens"]) == 192, "192 outputs")
        require(response["output_tokens"] == expected, f"full fidelity: {run['name']}")
        counters = response["route"]["crossover_decisions"]
        require(response["runtime"]["selected_backend"] == "mlx", "native MLX route")
        if run["mode"] == "direct":
            require(all(counters[key] == 0 for key in ["ax_mtp_requested", "ax_mtp_draft_tokens", "ax_mtp_verify_tokens"]), "direct route counters")
        else:
            require(counters["ax_mtp_requested"] == 1 and counters["ax_mtp_draft_tokens"] == 190 and counters["ax_mtp_verify_tokens"] == 260, "active MTP route counters")
        require(run["http_status"] == 200 and not run["watchdog_expired"], "request completed")
        require(run["cleanup"]["returncode"] == 0, "owned server cleanup")
    events = [json.loads(line.split("first_split_probe ", 1)[1])
              for line in (ROOT / "observer.log").read_text().splitlines()
              if "first_split_probe " in line]
    require(len(events) == 19, "probe event count")

    def event(kind):
        matches = [e for e in events if e["event"] == kind]
        require(len(matches) == 1, f"unique {kind}")
        return matches[0]

    window = event("window")
    require((window["offset"], window["row"], window["position"], window["accept_count"]) == (522, 2, 524, 2), "admitted window")
    require(window["inputs"] == [279, 1906, 314, 279], "actual inputs")
    require(window["lazy_inputs_materialized"], "materialized inputs")
    require(window["generated_tokens"] == direct[:114], "generated window prefix")
    require(window["inputs"][:3] == direct[113:116], "all selected inputs")
    require(event("prefix_validated")["every_selected_input_checked"], "prefix admission")
    require(event("same_scope_replay_valid")["full_logits_exact"], "faithful replay")
    arrays = {e["label"]: e for e in events if e["event"] == "array"}
    live = arrays["live_target"]["metadata"]
    require(live == arrays["same_scope_target_replay"]["metadata"], "full replay metadata")
    require(live["shape"] == [4, 248320] and live["elements"] == 993280 and live["finite"], "full live logits")
    comparisons = {e["label"]: e for e in events if e["event"] == "logit_comparison"}
    replay = comparisons["live_vs_same_scope_full_logits"]
    require((replay["elements"], replay["unequal"], replay["max_abs"]) == (993280, 0, 0.0), "full replay equality")
    before, after = event("cache_before"), event("cache_after")
    for name, position in [("snapshot", 522), ("source", 522), ("live_verify", 526)]:
        require(before[name] == after[name], f"immutable cache view: {name}")
        require(before[name]["seq_len"] == position, f"cache position: {name}")
        leaves = before[name]["logical_arrays"]
        require(len(leaves) == 128 and all(x["array"]["finite"] for x in leaves), "128 finite cache leaves")
    rows = {e["label"]: e for e in events if e["event"] == "row"}
    require(rows["ordinary_batch"]["top8"][0] == [6397, 21.875], "ordinary batch maximum")
    require(rows["ordinary_singleton"]["top8"][:3] == [[279, 21.75], [2849, 21.75], [6397, 21.75]], "singleton triple tie")
    require(not any(arrays["ordinary_batch"][key] for key in ["exact", "target", "relaxed", "qmm_guard", "whole_trace"]), "ordinary batch scopes")
    emission = event("emission")
    require(emission["generated_count_before"] == 114 and emission["result"] == [1906, 314, 6397], "actual output emission")
    require(not event("valid_window")["production_return_replaced"], "production return preserved")
    analysis = read("observer-analysis.json")
    require(analysis["window"] == window and analysis["comparisons"] == comparisons, "analysis matches events")
    require(analysis["arrays"] == arrays and analysis["selected_rows"] == rows, "analysis rows match")
    require(not analysis["direct_cache_observed"], "no direct cache claim")
    provenance = read("observer-provenance.json")
    require(len(provenance["model_files_freshly_rehashed"]) == 22, "22 model files")
    require(provenance["model_files_freshly_rehashed"] == provenance["manifest"]["model_files"], "fresh model hashes")
    require(not provenance["runtime_arithmetic_overrides"], "no arithmetic overrides")
    require(provenance["manifest"]["source_commit"] == "d121f107999e3b8627f60455a79bdd0dad7ada6c" and not provenance["manifest"]["dirty"], "clean installed source")
    require(provenance["observer_build"]["source_base"] == "0f7f2d2d0ae6e66a80c7d6f746fbb058efd2a951", "observer source base")
    for run in runs:
        expected_binary = provenance["observer_build"]["binary_sha256"] if run["name"] == "observer-mtp" else provenance["manifest"]["server_sha256"]
        require(run["binary_sha256"] == expected_binary, "executed server identity")
    require(digest((ROOT / "observer.patch").read_bytes()) == provenance["observer_build"]["patch_sha256"], "exact observer patch")
    for library in ["libmlx.dylib", "libjaccl.dylib"]:
        paths = analysis["loaded_native_paths"][library]
        require(len(paths) == 1 and paths[0].endswith("site-packages/ax_engine/.dylibs/" + library), "loaded installed library")
        require(paths[0] in (ROOT / "observer.log").read_text(), "loader trace")
    old, fixed, repeat = [read(name + ".json") for name in ["oracle-old", "oracle-direct-prefill", "oracle-old-repeat"]]
    require([x["exit_code"] for x in [old, fixed, repeat]] == [1, 0, 1], "oracle reversal exits")
    for receipt, name in [(old, "oracle-old"), (fixed, "oracle-direct-prefill"), (repeat, "oracle-old-repeat")]:
        require(digest((ROOT / (name + ".log")).read_bytes()) == receipt["log_sha256"], "oracle log hash")
        require(receipt["model_files"] == provenance["model_files_freshly_rehashed"], "oracle model parity")
        require(receipt["linked_native_files"] == provenance["installed_native_files"], "oracle native parity")
        require(receipt["case_sha256"] == provenance["case_sha256"], "oracle case parity")
    require((ROOT / "oracle-old.log").read_bytes() == (ROOT / "oracle-old-repeat.log").read_bytes(), "old log reversal")
    require("index 109: expected 11870, got 8240" in (ROOT / "oracle-old.log").read_text(), "old expected failure")
    fixed_log = (ROOT / "oracle-direct-prefill.log").read_text()
    require("replayed_generated_tokens=115 next_prediction_index=116" in fixed_log, "corrected prefix")
    require("primary=314 draft=279 singleton=[279,2849] batched=[279, 40278]" in fixed_log, "second-step fidelity limit")
    require(digest((ROOT / "oracle-direct-prefill.patch").read_bytes()) == fixed["build"]["patch_sha256"], "exact oracle patch")
    verify_regression(publication)
    print("PASS: published hashes; 3x192 output fidelity; 993280 recorded exact logits; 128x3 immutable cache witnesses; oracle failure/correction/reversal and output117 limit")
    print("PASS: frozen forced-replay regression; meaningful negative runs; stable-source 11-check validation with strict-Clippy failure retained; advisory timeouts retained")


if __name__ == "__main__":
    main()
