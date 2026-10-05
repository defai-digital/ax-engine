#!/usr/bin/env python3
"""Evaluate the Flash Next direct-default release gates from recorded evidence.

Reads the result files produced by the delivery, QA, lifecycle, generate-matrix
and reference harnesses plus the frozen thresholds in
`flash_next_release_thresholds.json`, and reports one pass/fail line per gate.
Nothing here runs a model. The verdict covers only the stated release scope:
direct decoding is the default, model MTP stays an explicit experimental
opt-in, and MTP-S, MTP-P and MTP-D remain `not_assessed`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
THRESHOLDS = SCRIPTS / "flash_next_release_thresholds.json"
GIB = 1024**3
AX_ARMS = ("disabled", "default")
OUTPUT_TOKENS = 128
LONG_ID = "long_context_record_mass"
RESULT_FILES = {"delivery": "delivery/result.json", "qa": "qa/result.json",
                "lifecycle": "lifecycle/result.json", "matrix": "generate-matrix/result.json",
                "reference": "reference/result.json"}


def gate(name: str, passed: bool, detail: str, **data: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail, **data}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def gate_delivery(delivery: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec, target = th["delivery"], th["target"]
    verification = delivery.get("verification") or {}
    support = delivery.get("native_support") or {}
    problems = []
    if not delivery.get("completed"):
        problems.append("delivery did not complete")
    if not delivery.get("fresh_cache"):
        problems.append("cache was not empty at the start")
    if (delivery.get("download") or {}).get("exit_code") != 0:
        problems.append("downloader did not exit 0")
    if delivery.get("revision") != target["pack_revision"]:
        problems.append("revision is not the pinned one")
    if verification.get("members_verified") != target["published_members"] \
            or verification.get("bytes_verified") != target["published_bytes"] \
            or not verification.get("all_sizes_and_sha256_match"):
        problems.append("members, bytes or hashes do not match the published pack")
    if not verification.get("native_manifest_runtime_ready"):
        problems.append("generated native manifest is not runtime ready")
    if sorted(support.get("modes", [])) != sorted(spec["functional_controls_on_delivered_copy"]) \
            or not support.get("passed"):
        problems.append("functional controls did not pass in every required mode")
    return [gate("delivery", not problems, "; ".join(problems) or
                 f"{verification.get('members_verified')} members, "
                 f"{verification.get('bytes_verified')} bytes verified from an empty cache",
                 download_seconds=(delivery.get("download") or {}).get("elapsed_seconds"))]


def gate_readiness(delivery: dict[str, Any], runs: list[tuple[str, float]],
                   th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["readiness"]
    delivered = (delivery.get("native_support") or {}).get("readiness_seconds") or {}
    first_mode = next(iter(delivered), None)
    problems = []
    if first_mode is None:
        problems.append("no first-load readiness recorded after delivery")
    for mode, seconds in delivered.items():
        limit = spec["first_load_after_delivery_seconds_max"] if mode == first_mode \
            else spec["warm_seconds_max"]
        if seconds > limit:
            problems.append(f"delivered {mode} readiness {seconds}s exceeds {limit}s")
    for label, seconds in runs:
        if seconds > spec["warm_seconds_max"]:
            problems.append(f"{label} readiness {seconds}s exceeds {spec['warm_seconds_max']}s")
    worst = max([s for _, s in runs] or [0])
    return [gate("readiness", not problems, "; ".join(problems) or
                 f"first load after delivery {delivered.get(first_mode)}s; "
                 f"staged-pack worst {worst}s", delivered=delivered)]


def gate_memory(qa: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    limit = th["memory"]["peak_server_rss_gib_max"]
    peaks = {mode: qa["summary"]["modes"][mode]["peak_rss_kib"] * 1024 / GIB for mode in AX_ARMS}
    over = {mode: round(peak, 2) for mode, peak in peaks.items() if peak > limit}
    return [gate("memory", not over, f"peak server RSS {over} exceeds {limit} GiB" if over else
                 f"peak server RSS {{{', '.join(f'{m}: {p:.1f}' for m, p in peaks.items())}}} GiB "
                 f"within {limit} GiB", peaks_gib={m: round(p, 3) for m, p in peaks.items()})]


def gate_long_lookup(qa: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    limit = th["long_lookup"]["wall_seconds_max"]
    seconds = {mode: qa["summary"]["modes"][mode].get("long_context_elapsed_seconds") for mode in AX_ARMS}
    bad = {mode: s for mode, s in seconds.items() if s is None or s > limit}
    return [gate("long_lookup", not bad, f"{bad} outside {limit}s" if bad else
                 f"{th['long_lookup']['prompt_tokens']}-token lookup took {seconds} s (limit {limit} s)",
                 seconds=seconds)]


def gate_quality(qa: dict[str, Any], reference: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    modes = qa["summary"]["modes"]
    ref_qa = reference.get("qa") or {}
    problems = []
    if not reference.get("completed") or len(ref_qa.get("rows", [])) != 105:
        problems.append("reference QA is incomplete")
    ref_failures = set(ref_qa.get("failures", []))
    ax_only: dict[str, list[str]] = {}
    for mode in AX_ARMS:
        summary = modes.get(mode, {})
        if summary.get("completed") != 105 or not summary.get("normal_stops") or summary.get("route_errors"):
            problems.append(f"{mode}: incomplete cohort, abnormal stop or route error")
        extra = sorted(set(summary.get("quality_failures", [])) - ref_failures)
        ax_only[mode] = extra
        if len(extra) > th["quality"]["ax_only_failures_max"]:
            problems.append(f"{mode}: failures the reference passes: {extra}")
    differing = qa["summary"]["text_identity_vs_disabled"]["default"]["differing_text_ids"]
    if differing:
        problems.append(f"default text differs from direct on {differing}")
    return [gate("quality", not problems, "; ".join(problems) or
                 "105/105 complete in both arms, default text equals direct text, every AX failure "
                 f"also fails on the reference ({sorted(ref_failures)})",
                 ax_failures={m: modes[m]["quality_failures"] for m in AX_ARMS},
                 reference_failures=sorted(ref_failures), ax_only_failures=ax_only)]


def gate_lifecycle(lifecycle: dict[str, Any], th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["lifecycle"]
    by_mode = {run["mode"]: run for run in lifecycle.get("runs", [])}
    problems = []
    for mode in spec["modes_required"]:
        run = by_mode.get(mode)
        if run is None or len(run["actions"]) != spec["actions_per_mode"]:
            problems.append(f"{mode}: missing or incomplete actions")
            continue
        if any(any(value != 0 for value in action["drained"].values()) for action in run["actions"]):
            problems.append(f"{mode}: counters did not drain")
        if run.get("exit_code_after_stop") != 0:
            problems.append(f"{mode}: server did not exit cleanly")
    if not lifecycle.get("lifecycle_passed"):
        problems.append("lifecycle result is not marked passed")
    return [gate("lifecycle", not problems, "; ".join(problems) or
                 f"{spec['actions_per_mode']} actions in each of {spec['modes_required']} with drained counters")]


def cell_stats(rows: list[dict[str, Any]], tokens: Any) -> dict[str, Any]:
    """Medians over measured runs of one cell; every measured run must be full length."""
    measured = [r for r in rows if not r["warmup"]]
    full = [r for r in measured if tokens(r) == OUTPUT_TOKENS]
    rates, ttfts = [], []
    for row in measured:
        timing = row["timing"] if "timing" in row else row
        if timing.get("decode_seconds") and timing.get("decode_tokens"):
            rates.append(timing["decode_tokens"] / timing["decode_seconds"])
        ttfts.append(timing["ttft_seconds"])
    return {"measured": len(measured), "full_length": len(full),
            "decode_tok_s": statistics.median(rates) if rates else None,
            "ttft_s": statistics.median(ttfts) if ttfts else None}


def gate_reference_relative(matrix: dict[str, Any], reference: dict[str, Any],
                            th: dict[str, Any]) -> list[dict[str, Any]]:
    spec = th["reference_relative"]
    ref_cells = (reference.get("matrix") or {}).get("cells") or []
    lengths = sorted({c["length"] for c in ref_cells})
    problems, table = [], {}
    if not reference.get("completed") or not ref_cells:
        problems.append("reference matrix is incomplete")
    for length in lengths:
        ref = cell_stats([c for c in ref_cells if c["length"] == length], lambda r: r["generated_tokens"])
        if ref["full_length"] != ref["measured"]:
            problems.append(f"{length}: reference has short measured outputs")
        for arm in AX_ARMS:
            rows = [c for c in matrix["runs"][arm]["cells"] if c["length"] == length]
            ax = cell_stats(rows, lambda r: len(r["output_tokens"]))
            if ax["full_length"] != ax["measured"]:
                problems.append(f"{length}/{arm}: AX has short measured outputs")
            if not (ax["decode_tok_s"] and ref["decode_tok_s"] and ax["ttft_s"] and ref["ttft_s"]):
                problems.append(f"{length}/{arm}: missing timing")
                continue
            decode_ratio = ax["decode_tok_s"] / ref["decode_tok_s"]
            ttft_ratio = ax["ttft_s"] / ref["ttft_s"]
            table[f"{length}/{arm}"] = {"decode_ratio": round(decode_ratio, 4),
                                        "ttft_ratio": round(ttft_ratio, 4),
                                        "ax_decode_tok_s": round(ax["decode_tok_s"], 3),
                                        "reference_decode_tok_s": round(ref["decode_tok_s"], 3),
                                        "ax_ttft_s": round(ax["ttft_s"], 3),
                                        "reference_ttft_s": round(ref["ttft_s"], 3)}
            if decode_ratio < spec["decode_ratio_min"]:
                problems.append(f"{length}/{arm}: decode ratio {decode_ratio:.3f} < {spec['decode_ratio_min']}")
            if ttft_ratio > spec["ttft_ratio_max"]:
                problems.append(f"{length}/{arm}: TTFT ratio {ttft_ratio:.3f} > {spec['ttft_ratio_max']}")
    return [gate("reference_relative", not problems and bool(table), "; ".join(problems) or
                 f"all {len(table)} AX-arm cells within decode >= {spec['decode_ratio_min']} and "
                 f"TTFT <= {spec['ttft_ratio_max']} of the reference", cells=table)]


def gate_mtp_policy(delivery: dict[str, Any], qa: dict[str, Any], lifecycle: dict[str, Any],
                    matrix: dict[str, Any], th: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    problems = []
    support = ((delivery.get("native_support") or {}).get("modes")) or []
    if "default" not in support:
        problems.append("default mode missing from the delivered-copy controls")
    verified = {"qa": qa["summary"]["modes"]["default"]["verified_steps"],
                "matrix": sum(t["verified_steps"] for t in matrix["runs"]["default"]["trajectories"])
                + sum(c["verified_steps"] for c in matrix["runs"]["default"]["cells"])}
    for action_run in lifecycle.get("runs", []):
        if action_run["mode"] == "default":
            verified["lifecycle"] = action_run.get("final_verified_steps")
    for where, steps in verified.items():
        if steps != th["mtp"]["default_verified_steps"]:
            problems.append(f"default mode verified {steps} MTP steps in {where}")
    if contract.get("mtp_certification") != th["mtp"]["gate_state"]:
        problems.append("MTP-S/P/D are not all not_assessed in the qualification contract")
    return [gate("mtp_policy", not problems, "; ".join(problems) or
                 "default mode never activated the drafter in any harness; required MTP stays an "
                 "explicit opt-in and MTP-S/P/D remain not_assessed", default_verified_steps=verified)]


def gate_integrity(results: dict[str, dict[str, Any]], th: dict[str, Any]) -> list[dict[str, Any]]:
    problems = []
    servers = {name: r.get("server_sha256") for name, r in results.items() if "server_sha256" in r}
    if len(set(servers.values())) != 1:
        problems.append(f"server binaries differ across results: {servers}")
    inventories = {name: r.get("inventory_sha256") for name, r in results.items() if "inventory_sha256" in r}
    if len(set(inventories.values())) != 1:
        problems.append(f"inventories differ across results: {inventories}")
    for name, result in results.items():
        hardware = result.get("hardware") or {}
        if hardware.get("form_factor") != "Mac Studio" or "Ultra" not in hardware.get("soc", "") \
                or hardware.get("memory_bytes", 0) < 192 * GIB:
            problems.append(f"{name}: not the Mac Studio Ultra 192 GiB target")
        if result.get("qualification") or result.get("release_ready"):
            problems.append(f"{name}: a harness result claims qualification")
    if results["reference"].get("versions", {}).get("mlx-vlm") != "0.7.0rc0":
        problems.append("reference runtime is not the pinned MLX-VLM")
    return [gate("integrity", not problems, "; ".join(problems) or
                 "one server binary and one pack inventory across all results; hardware is the target SKU")]


def evaluate(results: dict[str, dict[str, Any]], th: dict[str, Any],
             contract: dict[str, Any]) -> dict[str, Any]:
    qa, lifecycle, matrix = results["qa"], results["lifecycle"], results["matrix"]
    reference, delivery = results["reference"], results["delivery"]
    runs = [(f"qa/{mode}", qa["summary"]["modes"][mode]["ready_seconds"]) for mode in AX_ARMS]
    runs += [(f"lifecycle/{run['mode']}", run["ready_seconds"]) for run in lifecycle.get("runs", [])
             if run["mode"] in AX_ARMS]
    runs += [(f"matrix/{mode}", matrix["runs"][mode]["ready_seconds"]) for mode in AX_ARMS]
    gates = [*gate_delivery(delivery, th), *gate_integrity(results, th), *gate_quality(qa, reference, th),
             *gate_lifecycle(lifecycle, th), *gate_memory(qa, th), *gate_readiness(delivery, runs, th),
             *gate_long_lookup(qa, th), *gate_reference_relative(matrix, reference, th),
             *gate_mtp_policy(delivery, qa, lifecycle, matrix, th, contract)]
    return {"gates": gates, "release_ready": all(g["passed"] for g in gates)}


def source_state() -> dict[str, Any]:
    root = SCRIPTS.parent
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                          "--untracked-files=no"], text=True).strip())
    return {"commit": commit, "tracked_tree_dirty": dirty}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the verdict JSON here")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    args = parser.parse_args()
    import qualify_qwen38_flash_next as qualify
    th = json.loads(args.thresholds.read_text())
    paths = {name: args.evidence_dir / relative for name, relative in RESULT_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing evidence: {missing}")
    results = {name: json.loads(path.read_text()) for name, path in paths.items()}
    verdict = evaluate(results, th, qualify.contract())
    verdict.update(
        schema="ax-engine.flash-next.release-verdict.v1", release_scope=th["release_scope"],
        pack_revision=th["target"]["pack_revision"], thresholds_sha256=sha256_file(args.thresholds),
        evidence_sha256={name: sha256_file(path) for name, path in paths.items()},
        mtp_certification=qualify.contract()["mtp_certification"], source=source_state(),
        disclosure=th["disclosure"])
    for item in verdict["gates"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['gate']}: {item['detail']}")
    print("release_ready:", verdict["release_ready"])
    if args.output:
        args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    return 0 if verdict["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
