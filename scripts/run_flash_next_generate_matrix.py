#!/usr/bin/env python3
"""Compare Flash Next direct and required-MTP generation on the server path.

Two phases per mode (MTP disabled and required), both through the native
`/v1/generate/stream` endpoint with greedy decoding:

* trajectory identity: 16 chat-templated natural-language prompts, 128 output
  tokens each; the output token IDs must match between modes position by
  position, and any divergence is recorded with its index.
* fixed workload: 512, 2048 and 8192 input tokens, 128 output tokens, two
  warmups and three measured runs per cell. Each run uses a different token
  offset into the same long document so a prefix-cache hit cannot skip prefill;
  both modes use the same offsets, so runs pair across modes.

The pack and server binary are hash-bound before and after. Dry-run is
weight-free. This records server-path token identity and timings for one
source/pack/SKU combination; it is not MTP-S/P/D qualification and it does not
include a reference-runtime baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time
from typing import Any
import urllib.request

try:
    import check_flash_next_native_support as native
except ImportError:
    from scripts import check_flash_next_native_support as native

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa"
MODES = ("disabled", "required")
TOTAL_BLOCKS = 4096
OUTPUT_TOKENS = 128
LENGTHS = (512, 2048, 8192)
WARMUPS = 2
MEASURED = 3
OFFSET_STRIDE = 7
READY_TIMEOUT = 900
REQUEST_TIMEOUT = 3600
LONG_ID = "long_context_record_mass"
PROMPTS = (
    "Explain how a hash table handles collisions, with an example.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe the water cycle for a ten-year-old.",
    "Write a Python function that merges two sorted lists, and explain its complexity.",
    "Summarize the causes of the French Revolution.",
    "Give a step-by-step recipe for making sourdough bread.",
    "Explain the difference between TCP and UDP.",
    "Write a product description for a reusable water bottle.",
    "What are the main differences between renting and buying a home?",
    "Explain how photosynthesis works.",
    "Write a polite email asking a colleague to review a document by Friday.",
    "Describe three strategies for managing stress at work.",
    "Explain what a mutex is and when to use one.",
    "Write a haiku sequence about autumn, then explain the imagery.",
    "Compare supervised and unsupervised learning.",
    "Describe how a bill becomes law in a parliamentary system.",
)


def post_json(base: str, path: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def tokenize(base: str, text: str, add_special: bool = False) -> list[int]:
    return post_json(base, "/v1/tokenize", {"content": text, "add_special": add_special})["tokens"]


def chat_prompt_ids(base: str, user: str) -> list[int]:
    templated = post_json(base, "/v1/apply-template", {
        "messages": [{"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return tokenize(base, templated)


def read_generate_stream(lines, clock=time.monotonic) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a /v1/generate/stream body line by line and time its token events.

    The server leaves `performance` timings at zero on this path, so TTFT and
    the decode window are measured on the client (loopback, so overhead is small
    but not zero). Returns the terminal response and the client timing record.
    """
    t0 = clock()
    event, response, token_events = None, None, []
    for raw in lines:
        line = raw.decode().rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if event == "error":
                raise ValueError(f"generate stream error: {data}")
            if event == "step":
                count = len(json.loads(data).get("delta_tokens") or [])
                if count:
                    token_events.append((clock(), count))
            elif event == "response":
                response = json.loads(data)["response"]
    if response is None:
        raise ValueError("generate stream ended without a response event")
    timing: dict[str, Any] = {"token_events": len(token_events)}
    if token_events:
        timing["ttft_seconds"] = round(token_events[0][0] - t0, 6)
        timing["first_event_tokens"] = token_events[0][1]
        timing["decode_tokens"] = sum(count for _, count in token_events[1:])
        timing["decode_seconds"] = (round(token_events[-1][0] - token_events[0][0], 6)
                                    if len(token_events) > 1 else None)
    timing["total_seconds"] = round(clock() - t0, 6)
    return response, timing


def generate(base: str, ids: list[int], max_tokens: int = OUTPUT_TOKENS) -> dict[str, Any]:
    body = {"input_tokens": ids, "max_output_tokens": max_tokens, "sampling": {"temperature": 0}}
    req = urllib.request.Request(base + "/v1/generate/stream", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as stream:
        response, timing = read_generate_stream(stream)
    response["client_timing"] = timing
    return response


def slim(response: dict[str, Any], before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    perf = response.get("performance", {})
    return {"output_tokens": response["output_tokens"], "prompt_tokens": len(response["prompt_tokens"]),
            "finish_reason": response.get("finish_reason"), "status": response.get("status"),
            "timing": response["client_timing"], "mtp": perf.get("mtp", {}),
            "verified_steps": after.get(native.VERIFIED, 0.0) - before.get(native.VERIFIED, 0.0),
            "route_errors": sum(after.get(n, 0.0) - before.get(n, 0.0) for n in native.ERRORS)}


def workload_ids(long_ids: list[int], length: int, run: int) -> list[int]:
    offset = run * OFFSET_STRIDE
    if offset + length > len(long_ids):
        raise ValueError("long document is too short for the requested workload")
    return long_ids[offset: offset + length]


def decode_rate(timing: dict[str, Any]) -> float | None:
    """Tokens per second after the first token event, from client timestamps."""
    seconds, count = timing.get("decode_seconds"), timing.get("decode_tokens")
    return count / seconds if seconds and count else None


def run_mode(server: Path, root: Path, output: Path, mode: str, prompts: tuple[str, ...] = PROMPTS,
             lengths: tuple[int, ...] = LENGTHS) -> dict[str, Any]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = output / f"server-{mode}.log"
    command = [*native.server_command(server, root, mode, port), "--total-blocks", str(TOTAL_BLOCKS)]
    started = time.monotonic()
    result: dict[str, Any] = {"mode": mode, "trajectories": [], "cells": []}
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=native.server_env(),
                                   start_new_session=True)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"owned {mode} server exited with {process.returncode}")
                try:
                    native.request(base, "/v1/models")
                    break
                except (OSError, ValueError):
                    if time.monotonic() - started >= READY_TIMEOUT:
                        raise TimeoutError("native server readiness timeout") from None
                    time.sleep(1)
            result["ready_seconds"] = round(time.monotonic() - started, 3)
            items = json.loads((FROZEN / "full-qa-items.json").read_text())
            long_text = next(i for i in items if i["id"] == LONG_ID)["user"]
            long_ids = tokenize(base, long_text)
            result["long_document_tokens"] = len(long_ids)
            result["prompt_id_hashes"] = {}
            for index, prompt in enumerate(prompts):
                ids = chat_prompt_ids(base, prompt)
                result["prompt_id_hashes"][f"traj-{index:02d}"] = ids_sha(ids)
                before = native.metrics(native.request(base, "/metrics").decode())
                response = generate(base, ids)
                after = native.metrics(native.request(base, "/metrics").decode())
                row = {"index": index, "prompt": prompt, **slim(response, before, after)}
                result["trajectories"].append(row)
                print(f"{mode} traj {index + 1}/{len(prompts)}: {len(row['output_tokens'])} tok, "
                      f"{row['verified_steps']:.0f} verified", flush=True)
            for length in lengths:
                for run in range(WARMUPS + MEASURED):
                    ids = workload_ids(long_ids, length, run)
                    result["prompt_id_hashes"][f"cell-{length}-{run}"] = ids_sha(ids)
                    before = native.metrics(native.request(base, "/metrics").decode())
                    response = generate(base, ids)
                    after = native.metrics(native.request(base, "/metrics").decode())
                    row = {"length": length, "run": run, "warmup": run < WARMUPS,
                           **slim(response, before, after)}
                    result["cells"].append(row)
                    rate = decode_rate(row["timing"])
                    print(f"{mode} {length} run {run}{' (warmup)' if row['warmup'] else ''}: "
                          f"ttft {row['timing'].get('ttft_seconds', 0):.2f}s "
                          f"decode {rate if rate is None else round(rate, 2)} tok/s "
                          f"{row['verified_steps']:.0f} verified", flush=True)
        finally:
            native.stop_owned(process)
    native.scrub_log(log_path)
    return result


def summarize(runs: dict[str, dict[str, Any]],
              lengths: tuple[int, ...] = LENGTHS) -> dict[str, Any]:
    """Pure aggregation: token identity, paired challenger/direct decode ratios, route errors.

    `disabled` is the direct baseline; the single other mode is the challenger
    (`required` MTP, keyed `mtp_*`, or the product `default`, keyed `default_*`).
    """
    challengers = [mode for mode in runs if mode != "disabled"]
    if "disabled" not in runs or len(challengers) != 1:
        raise ValueError("summary needs the disabled baseline and exactly one other mode")
    challenger = challengers[0]
    prefix = "mtp" if challenger == "required" else challenger
    direct, other = runs["disabled"], runs[challenger]
    summary: dict[str, Any] = {"challenger_mode": challenger,
                               "prompt_ids_equal_across_modes":
                               direct["prompt_id_hashes"] == other["prompt_id_hashes"]}
    identical, divergences = 0, []
    for d, m in zip(direct["trajectories"], other["trajectories"], strict=True):
        a, b = d["output_tokens"], m["output_tokens"]
        if a == b:
            identical += 1
        else:
            index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            divergences.append({"prompt": d["prompt"], "first_difference_index": index,
                                "direct_length": len(a), f"{prefix}_length": len(b)})
    summary["trajectories"] = {"compared": len(direct["trajectories"]), "identical": identical,
                               "divergences": divergences,
                               f"{prefix}_verified_steps":
                               sum(t["verified_steps"] for t in other["trajectories"]),
                               "direct_verified_steps":
                               sum(t["verified_steps"] for t in direct["trajectories"])}
    cells: dict[str, Any] = {}
    cell_identical = cell_total = 0
    for length in lengths:
        d_runs = [c for c in direct["cells"] if c["length"] == length and not c["warmup"]]
        m_runs = [c for c in other["cells"] if c["length"] == length and not c["warmup"]]
        ratios, d_rates, m_rates = [], [], []
        for d, m in zip(d_runs, m_runs, strict=True):
            cell_total += 1
            cell_identical += d["output_tokens"] == m["output_tokens"]
            rd, rm = decode_rate(d["timing"]), decode_rate(m["timing"])
            d_rates.append(rd)
            m_rates.append(rm)
            if rd and rm:
                ratios.append(rm / rd)
        cells[str(length)] = {
            "measured_runs": len(d_runs),
            "direct_decode_tok_s_median": statistics.median([r for r in d_rates if r]) if any(d_rates) else None,
            f"{prefix}_decode_tok_s_median": statistics.median([r for r in m_rates if r]) if any(m_rates) else None,
            f"{prefix}_over_direct_decode_ratio_median": statistics.median(ratios) if ratios else None,
            f"{prefix}_over_direct_decode_ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "direct_ttft_s_median": statistics.median(c["timing"]["ttft_seconds"] for c in d_runs),
            f"{prefix}_ttft_s_median": statistics.median(c["timing"]["ttft_seconds"] for c in m_runs),
            f"{prefix}_draft_tokens": sum(c["mtp"].get("draft_tokens", 0) for c in m_runs),
            f"{prefix}_accepted_tokens": sum(c["mtp"].get("accepted_tokens", 0) for c in m_runs),
            f"{prefix}_verified_steps": sum(c["verified_steps"] for c in m_runs)}
    summary["cells"] = cells
    summary["cell_token_identity"] = {"identical": cell_identical, "compared": cell_total}
    summary["route_errors"] = sum(r["route_errors"] for run in runs.values()
                                  for r in (*run["trajectories"], *run["cells"]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-prompts", type=int, help="smoke runs: first N trajectory prompts")
    parser.add_argument("--lengths", type=int, nargs="+", help="smoke runs: input lengths to measure")
    parser.add_argument("--modes", nargs="+", choices=native.SERVER_MODES, default=list(MODES),
                        help="the disabled baseline plus one challenger: required MTP or default")
    args = parser.parse_args()
    if len(args.modes) != 2 or "disabled" not in args.modes:
        parser.error("--modes must be the disabled baseline plus exactly one challenger")
    prompts = PROMPTS[: args.limit_prompts] if args.limit_prompts else PROMPTS
    lengths = tuple(args.lengths) if args.lengths else LENGTHS
    contract = {"repo_id": native.PRIMARY_REPO, "revision": native.PACK_REVISION,
                "modes": list(args.modes), "endpoint": "/v1/generate/stream", "temperature": 0,
                "output_tokens": OUTPUT_TOKENS, "trajectory_prompts": len(prompts),
                "lengths": list(lengths), "warmups": WARMUPS, "measured": MEASURED,
                "offset_stride": OFFSET_STRIDE, "total_blocks": TOTAL_BLOCKS,
                "expert_stream": "auto", "generic_ngram_acceleration":
                "product default (on) in default mode, disabled in the other modes",
                "reference_baseline": "none", "qualification": False, "release_ready": False,
                "scope": "server-path token identity and timings; MTP-S/P/D not assessed"}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return 0
    if not all((args.model_dir, args.inventory, args.server_bin, args.output)):
        parser.error("live run requires model-dir, inventory, server-bin and output")
    if platform.system() != "Darwin":
        parser.error("live run requires macOS on the Ultra-class 192 GiB+ target")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if "Ultra" not in chip or memory < 192 * 1024**3:
        parser.error("live run requires the Ultra-class 192 GiB+ Flash Next target")
    storage = native.storage_info(args.model_dir)
    server_sha, inventory_sha = native.sha256(args.server_bin), native.sha256(args.inventory)
    inventory = json.loads(args.inventory.read_text())
    manifest_before = native.validate_inventory(args.model_dir, inventory)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {mode: run_mode(args.server_bin.resolve(), args.model_dir.resolve(), args.output, mode,
                           prompts, lengths) for mode in args.modes}
    manifest_after = native.validate_inventory(args.model_dir, inventory)
    if manifest_before and manifest_before != manifest_after:
        raise ValueError("native manifest changed during the run")
    if server_sha != native.sha256(args.server_bin) or inventory_sha != native.sha256(args.inventory):
        raise ValueError("server binary or staging receipt changed during the run")
    (args.output / "raw-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    summary = summarize(runs, lengths)
    evidence = {**contract, "server_sha256": server_sha, "inventory_sha256": inventory_sha,
                "hardware": {"form_factor": native.form_factor(), "soc": chip, "memory_bytes": memory,
                             "storage": {"declared": inventory["storage"], **storage},
                             "os": platform.mac_ver()[0]},
                "summary": summary, "runs": runs}
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
