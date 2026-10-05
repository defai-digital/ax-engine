#!/usr/bin/env python3
"""Run the Flash Next API lifecycle controls against the pinned pack.

Seven actions per mode (MTP disabled and required), 14 in total, on the chat
completions route: baseline, full SSE, one- and two-token budgets, a stop
sequence, an unfinished-stream client disconnect, and an identical recovery.
After every action the server's in-flight counters must drain to zero. The
pack and server binary are hash-bound before and after.

Dry-run is weight-free. Live execution owns only its spawned server processes.
This is lifecycle evidence for one source/pack/SKU combination, not MTP-S/P/D,
performance or release qualification.
"""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any
import urllib.request

try:
    import check_flash_next_native_support as native
except ImportError:
    from scripts import check_flash_next_native_support as native

MODES = ("disabled", "required")
MODEL_ID = "qwen3.8-flash-next:mxfp4"
TOTAL_BLOCKS = 4096
READY_TIMEOUT = 900
REQUEST_TIMEOUT = 600
DRAIN_TIMEOUT = 120
COUNTERS = ("ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
            "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
            "ax_engine_generation_buffered_stream_events")
BASELINE_PROMPT = "Count from 1 to 12, separated by commas. Output only the numbers."
LONG_PROMPT = "Write the integers from 1 to 400 separated by commas. Output only the numbers."
ACTIONS = ("baseline", "sse", "budget_1", "budget_2", "stop", "disconnect", "recovery")


def chat_body(prompt: str, max_tokens: int, **extra: Any) -> dict[str, Any]:
    return {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}


def parse_sse(raw: bytes) -> dict[str, Any]:
    """Reassemble a chat SSE stream; require a finish chunk, usage and [DONE]."""
    text, finishes, usage, done = [], [], None, False
    for line in raw.decode().splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        event = json.loads(data)
        if "error" in event:
            raise ValueError(f"stream error event: {event['error']}")
        for choice in event.get("choices", []):
            text.append(choice.get("delta", {}).get("content") or "")
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
        if event.get("usage"):
            usage = event["usage"]
    if not done or len(finishes) != 1 or usage is None:
        raise ValueError("stream lacks a single finish chunk, usage or [DONE]")
    return {"text": "".join(text), "finish_reason": finishes[0], "usage": usage}


def counters(base: str) -> dict[str, float]:
    parsed = native.metrics(native.request(base, "/metrics").decode())
    missing = [name for name in COUNTERS if name not in parsed]
    if missing:
        raise ValueError(f"missing lifecycle counters: {missing}")
    return {name: parsed[name] for name in COUNTERS}


def wait_drained(base: str, timeout: float = DRAIN_TIMEOUT) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    while True:
        current = counters(base)
        if all(value == 0 for value in current.values()):
            return current
        if time.monotonic() >= deadline:
            raise TimeoutError(f"lifecycle counters did not drain: {current}")
        time.sleep(0.5)


def post(base: str, body: dict[str, Any]) -> bytes:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def complete(base: str, body: dict[str, Any]) -> dict[str, Any]:
    response = json.loads(post(base, body))
    choice = response["choices"][0]
    return {"text": choice["message"].get("content") or "", "finish_reason": choice["finish_reason"],
            "usage": response["usage"]}


def disconnect_midstream(base: str) -> dict[str, Any]:
    """Open a long stream, confirm the server tracks it, then drop the socket."""
    host, port = base.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=REQUEST_TIMEOUT)
    body = json.dumps(chat_body(LONG_PROMPT, 512, stream=True))
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    first = b""
    for _ in range(50):  # chunked framing puts size lines before the first data line
        first = response.fp.readline()
        if first.startswith(b"data:") or not first:
            break
    if not first.startswith(b"data:"):
        raise ValueError("stream produced no data event before the disconnect")
    seen_active = 0.0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and seen_active < 1:
        seen_active = counters(base)["ax_engine_generation_active_streams"]
        time.sleep(0.2)
    conn.sock.shutdown(socket.SHUT_RDWR)
    conn.close()
    if seen_active < 1:
        raise ValueError("server never reported the open stream as active")
    return {"first_event_received": True, "active_streams_while_open": seen_active}


def run_actions(base: str, mode: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baseline: dict[str, Any] = {}

    def record(name: str, detail: dict[str, Any]) -> None:
        drained = wait_drained(base)
        results.append({"mode": mode, "action": name, **detail, "drained": drained})
        print(f"{mode}/{name}: ok", flush=True)

    before = native.metrics(native.request(base, "/metrics").decode())
    baseline.update(complete(base, chat_body(BASELINE_PROMPT, 48)))
    if baseline["finish_reason"] != "stop" or not baseline["text"].strip():
        raise ValueError("baseline did not complete normally")
    after = native.metrics(native.request(base, "/metrics").decode())
    activation = native.check_activation(before, after, mode == "required",
                                         baseline["usage"]["completion_tokens"], 48)
    record("baseline", {**baseline, "activation_delta": activation})

    streamed = parse_sse(post(base, chat_body(BASELINE_PROMPT, 48, stream=True,
                                              stream_options={"include_usage": True})))
    if streamed["text"] != baseline["text"] or streamed["finish_reason"] != "stop":
        raise ValueError("SSE stream differs from the baseline completion")
    record("sse", streamed)

    for tokens in (1, 2):
        out = complete(base, chat_body(BASELINE_PROMPT, tokens))
        if out["usage"]["completion_tokens"] != tokens or out["finish_reason"] != "length":
            raise ValueError(f"{tokens}-token budget was not honored: {out}")
        record(f"budget_{tokens}", out)

    out = complete(base, chat_body(BASELINE_PROMPT, 48, stop=["4"]))
    if out["finish_reason"] != "stop" or "4" in out["text"] \
            or not baseline["text"].startswith(out["text"].rstrip()):
        raise ValueError(f"stop sequence was not honored: {out}")
    record("stop", out)

    record("disconnect", disconnect_midstream(base))

    again = complete(base, chat_body(BASELINE_PROMPT, 48))
    if again["text"] != baseline["text"] or again["usage"] != baseline["usage"]:
        raise ValueError("recovery request differs from the baseline")
    record("recovery", again)
    return results


def run_mode(server: Path, root: Path, output: Path, mode: str) -> dict[str, Any]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = output / f"server-{mode}.log"
    started = time.monotonic()
    command = [*native.server_command(server, root, mode, port), "--total-blocks", str(TOTAL_BLOCKS)]
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
            ready = round(time.monotonic() - started, 3)
            actions = run_actions(base, mode)
            alive = process.poll() is None
        finally:
            native.stop_owned(process)
    native.scrub_log(log_path)
    if not alive:
        raise RuntimeError(f"{mode} server died during the lifecycle controls")
    return {"mode": mode, "ready_seconds": ready, "actions": actions,
            "exit_code_after_stop": process.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=native.SERVER_MODES, default=list(MODES),
                        help="control modes; `default` runs the server with no policy flags")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must list each mode once")
    contract = {"repo_id": native.PRIMARY_REPO, "revision": native.PACK_REVISION,
                "modes": list(args.modes), "actions": list(ACTIONS), "counters": list(COUNTERS),
                "total_blocks": TOTAL_BLOCKS, "expert_stream": "auto", "qualification": False,
                "release_ready": False,
                "scope": "API lifecycle; MTP-S/P/D, quality and performance not assessed"}
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
    runs = [run_mode(args.server_bin.resolve(), args.model_dir.resolve(), args.output, mode)
            for mode in args.modes]
    manifest_after = native.validate_inventory(args.model_dir, inventory)
    if manifest_before and manifest_before != manifest_after:
        raise ValueError("native manifest changed during the run")
    if server_sha != native.sha256(args.server_bin) or inventory_sha != native.sha256(args.inventory):
        raise ValueError("server binary or staging receipt changed during the run")
    evidence = {**contract, "server_sha256": server_sha, "inventory_sha256": inventory_sha,
                "hardware": {"form_factor": native.form_factor(), "soc": chip,
                             "memory_bytes": memory,
                             "storage": {"declared": inventory["storage"], **storage},
                             "os": platform.mac_ver()[0]},
                "lifecycle_passed": True, "runs": runs}
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Lifecycle controls passed; qualification and MTP-S/P/D remain unassessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
