#!/usr/bin/env python3
"""Bounded default-server residency acceptance; no private host configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import threading
import subprocess
import time
import urllib.error
import urllib.request


def snapshot():
    return {name: subprocess.check_output(cmd, text=True).strip() for name, cmd in {
        "pressure": ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        "swap": ["sysctl", "-n", "vm.swapusage"],
        "vm_stat": ["vm_stat"],
    }.items()}


def metrics(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    prefixes = ("ax_engine_memory_mlx_", "ax_engine_mtp_", "ax_engine_mlx_prefix_cache_", "ax_engine_step_prefix_", "ax_engine_mlx_prefill_", "ax_engine_mlx_mtp_")
    return [line for line in lines if line.startswith(prefixes)]


def stream(port, prompt, count, cancel=False):
    body = {"model": "local", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": count, "temperature": 0, "top_p": 1, "top_k": 0,
            "seed": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    frames, text, first, usage, done = [], [], None, None, False
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            elapsed = time.perf_counter() - start
            data = raw[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            frame = json.loads(data)
            if "error" in frame:
                raise RuntimeError(frame["error"])
            frames.append({"elapsed_s": elapsed, "payload": frame})
            usage = frame.get("usage") or usage
            for choice in frame.get("choices", []):
                delta = choice.get("delta", {})
                emitted = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if emitted:
                    first = elapsed if first is None else first
                    text.append(emitted)
            if cancel and first is not None:
                break
    wall = time.perf_counter() - start
    result = {"request": body, "frames": frames, "ttft_s": first, "wall_s": wall,
              "usage": usage, "done": done, "cancelled": cancel,
              "text": "".join(text), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    if not cancel:
        assert done and usage and usage["completion_tokens"] == count, result
        assert first is not None
        result["completion_tps"] = count / wall
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31947)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AX_", "MTPLX_"))}
    env["RUST_LOG"] = "info"
    rows = []
    for model in ("tiel", "cyber"):
        pack = "AX-" + ("Cyber-" if model == "cyber" else "") + "Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP"
        for block, arms in enumerate((("auto", "off", "on"), ("on", "off", "auto"))):
            for arm in arms:
                name = f"{model}-{block}-{arm}"
                cmd = [str(args.server), "--model-id", "local", "--mlx",
                       "--mlx-model-artifacts-dir", str(args.model_root / pack),
                       "--host", "127.0.0.1", "--port", str(args.port)]
                if arm != "auto":
                    cmd += ["--stream-experts", arm]
                row = {"model": model, "block": block, "arm": arm,
                       "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
                       "before": snapshot(), "trials": []}
                def swap_mib(value):
                    match = re.search(r"used = ([0-9.]+)M", value)
                    if not match:
                        raise ValueError("unrecognized swap probe")
                    return float(match.group(1))

                stop_monitor = threading.Event()
                row["samples"] = []
                baseline_swap = swap_mib(row["before"]["swap"])
                print("Starting", name, flush=True)
                with (args.output / (name + ".log")).open("w") as log:
                    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    def monitor():
                        while not stop_monitor.is_set():
                            sample = snapshot()
                            sample["elapsed_s"] = time.monotonic() - monitor_started
                            row["samples"].append(sample)
                            if sample["pressure"] != "1" or swap_mib(sample["swap"]) - baseline_swap > 256:
                                row["stop_guard"] = "pressure or swap growth"
                                if proc.poll() is None:
                                    os.killpg(proc.pid, signal.SIGTERM)
                                return
                            stop_monitor.wait(1)

                    monitor_started = time.monotonic()
                    monitor_thread = threading.Thread(target=monitor, daemon=True)
                    monitor_thread.start()
                    try:
                        started = time.monotonic()
                        while True:
                            if proc.poll() is not None:
                                raise RuntimeError(f"server exited {proc.returncode}")
                            try:
                                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as r:
                                    row["models"] = json.load(r)
                                break
                            except (OSError, urllib.error.URLError):
                                if time.monotonic() - started > 600:
                                    raise TimeoutError("startup timeout")
                                time.sleep(1)
                        row["startup_s"] = time.monotonic() - started
                        short = "Implement a Python LRU cache with get and put in O(1). Explain the invariant and include full code and unit tests."
                        cases = [("short", short, 16, i == 0) for i in range(4)]
                        if arm != "on":
                            long = "Review this numbered data before implementing the requested cache.\n" + "\n".join(f"item {i}: alpha beta gamma delta" for i in range(1350)) + "\n" + short
                            cases += [("long", long, 128, i == 0) for i in range(3)]
                        for case, prompt, count, warmup in cases:
                            before = snapshot()
                            if before["pressure"] != "1":
                                raise RuntimeError("memory pressure stop guard")
                            time.sleep(3)
                            before_metrics = metrics(args.port)
                            result = stream(args.port, prompt, count)
                            result["metrics_before"] = before_metrics
                            result["metrics_after"] = metrics(args.port)
                            result.update(case=case, warmup=warmup, before=before, after=snapshot())
                            row["trials"].append(result)
                            (args.output / (name + ".json")).write_text(json.dumps(row, indent=2) + "\n")
                        if arm != "on":
                            row["cancel"] = stream(args.port, "Write a detailed Python tutorial with many examples.", 4096, cancel=True)
                            time.sleep(2)
                            row["recovery"] = stream(args.port, short, 16)
                        row["after"] = snapshot()
                    finally:
                        stop_monitor.set()
                        monitor_thread.join(timeout=5)
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=30)
                            row["forced_shutdown"] = False
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=10)
                            row["forced_shutdown"] = True
                        row["exit_code"] = proc.returncode
                        (args.output / (name + ".json")).write_text(json.dumps(row, indent=2) + "\n")
                rows.append({"name": name, "exit_code": row["exit_code"], "forced_shutdown": row["forced_shutdown"]})
                (args.output / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
                print("Done", name, flush=True)
                time.sleep(3)


if __name__ == "__main__":
    main()
