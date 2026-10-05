#!/usr/bin/env python3
"""Replay the frozen Flash Next installed-QA cohort against the pinned pack.

Runs the 105 frozen items (104 short closed-answer cases and one 29,774-token
lookup) once with MTP disabled and once with MTP required, grades them with
the frozen checkers, and records peak server RSS, readiness time and request
latency. The pack and server binary are hash-bound before and after.

Dry-run is weight-free. Live execution owns only its spawned server processes.
This is functional/quality evidence for one source/pack/SKU combination, not
MTP-S/P/D, performance or release qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
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
SHORT_BUDGET = 256
LONG_BUDGET = 64
LONG_ID = "long_context_record_mass"
REQUEST_TIMEOUT = 1800
READY_TIMEOUT = 900
# The default KV pool (16 tokens x 1024 blocks) caps context at 16,384 tokens,
# below the 29,774-token lookup; the original cohort ran with 4,096 blocks.
TOTAL_BLOCKS = 4096
MODEL_ID = "qwen3.8-flash-next:mxfp4"

# Runs in the frozen directory so the original checkers grade the answers.
GRADER = """
import json, sys
import closed_checks
payload = json.load(sys.stdin)
items = {item['id']: item for item in payload['items']}
out = {}
for item_id, text in payload['texts'].items():
    result = closed_checks.evaluate_checker(text, items[item_id])
    out[item_id] = {'passed': bool(result.passed), 'detail': result.detail}
json.dump(out, sys.stdout)
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_items(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text())
    ids = [item["id"] for item in items]
    if len(items) != 105 or len(set(ids)) != 105 or ids[-1] != LONG_ID:
        raise ValueError("frozen cohort must be 105 unique items ending with the long lookup")
    return items


def budget_for(item: dict[str, Any]) -> int:
    return LONG_BUDGET if item["id"] == LONG_ID else SHORT_BUDGET


def request_body(item: dict[str, Any]) -> dict[str, Any]:
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user", "content": item["user"]})
    return {"model": MODEL_ID, "messages": messages, "temperature": 0,
            "max_tokens": budget_for(item), "chat_template_kwargs": {"enable_thinking": False}}


def grade(items: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, dict[str, Any]]:
    payload = json.dumps({"items": items, "texts": texts})
    done = subprocess.run([sys.executable, "-B", "-c", GRADER], cwd=FROZEN, input=payload,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


class RssPeak(threading.Thread):
    """Sample resident memory of one process; the peak is evidence, not a limit."""

    def __init__(self, pid: int, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak_kib = pid, interval, 0
        self._halt = threading.Event()

    def sample(self) -> None:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True)
        if out.stdout.strip().isdigit():
            self.peak_kib = max(self.peak_kib, int(out.stdout.strip()))

    def run(self) -> None:
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(self.interval)

    def finish(self) -> int:
        self._halt.set()
        self.join(timeout=10)
        self.sample()
        return self.peak_kib


def post_chat(base: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read())


def run_mode(server: Path, root: Path, output: Path, mode: str, items: list[dict[str, Any]],
             resume: bool) -> dict[str, Any]:
    mode_dir = output / mode
    mode_dir.mkdir(parents=True, exist_ok=resume)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = mode_dir / "server.log"
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with log_path.open("ab") as log:
        command = [*native.server_command(server, root, mode, port),
                   "--total-blocks", str(TOTAL_BLOCKS)]
        process = subprocess.Popen(command, stdout=log, stderr=log, env=native.server_env(),
                                   start_new_session=True)
        rss = RssPeak(process.pid)
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
            ready_seconds = time.monotonic() - started
            rss.start()
            for index, item in enumerate(items):
                row_path = mode_dir / f"{index:03d}-{item['id']}.json"
                if resume and row_path.exists():
                    rows.append(json.loads(row_path.read_text()))
                    continue
                before = native.metrics(native.request(base, "/metrics").decode())
                t0 = time.monotonic()
                response = post_chat(base, request_body(item))
                elapsed = time.monotonic() - t0
                after = native.metrics(native.request(base, "/metrics").decode())
                choice = response["choices"][0]
                delta = {name: after.get(name, 0.0) - before.get(name, 0.0)
                         for name in (native.VERIFIED, *native.ERRORS)}
                row = {"index": index, "id": item["id"], "budget": budget_for(item),
                       "elapsed_seconds": round(elapsed, 3), "finish_reason": choice["finish_reason"],
                       "text": choice["message"].get("content") or "",
                       "usage": response["usage"], "activation_delta": delta}
                row_path.write_text(json.dumps(row, indent=2) + "\n")
                rows.append(row)
                print(f"{mode} {index + 1}/{len(items)} {item['id']}: "
                      f"{row['usage']['completion_tokens']} tok, {elapsed:.1f}s, "
                      f"{delta[native.VERIFIED]:.0f} verified", flush=True)
        finally:
            peak = rss.finish() if rss.is_alive() else rss.peak_kib
            native.stop_owned(process)
    native.scrub_log(log_path)
    return {"mode": mode, "ready_seconds": round(ready_seconds, 3), "peak_rss_kib": peak,
            "rows": rows}


def summarize(items: list[dict[str, Any]], runs: dict[str, dict[str, Any]],
              grades: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Pure aggregation so it can be tested without a model."""
    summary: dict[str, Any] = {"requests": 0, "modes": {}}
    for mode, run in runs.items():
        rows = run["rows"]
        failures = sorted(i for i, g in grades[mode].items() if not g["passed"])
        route_errors = sum(abs(r["activation_delta"].get(name, 0.0))
                           for r in rows for name in native.ERRORS)
        summary["requests"] += len(rows)
        summary["modes"][mode] = {
            "completed": len(rows),
            "passed": len(rows) - len(failures),
            "quality_failures": failures,
            "normal_stops": all(r["finish_reason"] == "stop" for r in rows),
            "verified_steps": sum(r["activation_delta"].get(native.VERIFIED, 0.0) for r in rows),
            "route_errors": route_errors,
            "ready_seconds": run["ready_seconds"],
            "peak_rss_kib": run["peak_rss_kib"],
            "total_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in rows), 3),
            "long_context_elapsed_seconds": next(
                (r["elapsed_seconds"] for r in rows if r["id"] == LONG_ID), None),
        }
    if "disabled" in runs and len(runs) > 1:
        direct = {r["id"]: r["text"] for r in runs["disabled"]["rows"]}
        identity = {}
        for mode, run in runs.items():
            if mode == "disabled":
                continue
            other = {r["id"]: r["text"] for r in run["rows"]}
            identity[mode] = {
                "matching_text_pairs": sum(1 for k in direct if other.get(k) == direct[k]),
                "differing_text_ids": sorted(k for k in direct if other.get(k) != direct[k])}
        summary["text_identity_vs_disabled"] = identity
        if set(runs) == set(MODES):
            summary["matching_text_pairs"] = identity["required"]["matching_text_pairs"]
            summary["differing_text_ids"] = identity["required"]["differing_text_ids"]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--items", type=Path, default=FROZEN / "full-qa-items.json")
    parser.add_argument("--limit", type=int, help="run only the first N items (smoke runs)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run with an unchanged contract")
    parser.add_argument("--modes", nargs="+", choices=native.SERVER_MODES, default=list(MODES),
                        help="control modes; `default` runs the server with no policy flags")
    args = parser.parse_args()
    if "disabled" not in args.modes or len(set(args.modes)) != len(args.modes):
        parser.error("--modes must list each mode once and include the disabled baseline")
    items = load_items(args.items)
    contract = {"repo_id": native.PRIMARY_REPO, "revision": native.PACK_REVISION,
                "items": len(items), "items_sha256": sha256_bytes(args.items.read_bytes()),
                "modes": list(args.modes), "short_budget": SHORT_BUDGET, "long_budget": LONG_BUDGET,
                "temperature": 0, "thinking": "disabled", "expert_stream": "auto",
                "total_blocks": TOTAL_BLOCKS, "block_size_tokens": 16,
                "request_timeout_seconds": REQUEST_TIMEOUT, "grader": "frozen closed_checks",
                "qualification": False, "release_ready": False,
                "scope": "functional and closed-answer QA; MTP-S/P/D and performance not assessed"}
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
    if args.limit is not None:
        items = items[: args.limit]
        contract["items"] = len(items)
        contract["smoke_limit"] = args.limit
    storage = native.storage_info(args.model_dir)
    server_sha, inventory_sha = native.sha256(args.server_bin), native.sha256(args.inventory)
    inventory = json.loads(args.inventory.read_text())
    manifest_before = native.validate_inventory(args.model_dir, inventory)
    contract.update(server_sha256=server_sha, inventory_sha256=inventory_sha)
    contract_path = args.output / "contract.json"
    if args.resume:
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("resume requires an unchanged contract")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    runs = {mode: run_mode(args.server_bin.resolve(), args.model_dir.resolve(), args.output, mode,
                           items, args.resume) for mode in args.modes}
    manifest_after = native.validate_inventory(args.model_dir, inventory)
    if manifest_before and manifest_before != manifest_after:
        raise ValueError("native manifest changed during the run")
    if server_sha != native.sha256(args.server_bin) or inventory_sha != native.sha256(args.inventory):
        raise ValueError("server binary or staging receipt changed during the run")
    grades = {mode: grade(items, {r["id"]: r["text"] for r in run["rows"]})
              for mode, run in runs.items()}
    summary = summarize(items, runs, grades)
    evidence = {**contract, "hardware": {
        "form_factor": native.form_factor(), "soc": chip, "memory_bytes": memory,
        "storage": {"declared": inventory["storage"], **storage}, "os": platform.mac_ver()[0]},
        "summary": summary, "grades": grades}
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
