#!/usr/bin/env python3
"""Measure exact-pack, single-request streaming HTTP throughput on Apple Silicon."""

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import threading
import time

from tokenizers import Tokenizer

PACK_PINS = {
    "dense": "b2c5354f779e430d0c1733143db848a72b71c16e",
    "flash": "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3",
}
REPOSITORIES = {
    "dense": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
    "flash": "AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP",
}
PROMPT_IDS = ("code-shards", "math-trains", "prose-library")
REPEATS = 2
CAP = 64
WARMUP = "short-answer"
HOST = "127.0.0.1"
STARTUP_TIMEOUT = 360
REQUEST_TIMEOUT = 120
MLX_RUN = Path.home() / ".venvs/ax-engine/bin/mlx_lm.server"


class Failure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def pressure():
    return int(subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
    ))


def available_port():
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def command(stage, lane, pack, port, scratch):
    listen = ["--host", HOST, "--port", str(port)]
    homebrew = Path("/opt/homebrew/bin")
    key = "dense" if pack.name == PACK_PINS["dense"] else "flash"
    model_id = "peer-27b" if key == "dense" else "peer-flash-next"
    match lane:
        case "ax-engine":
            return [stage / "ax-engine-server", *listen,
                    "--mlx", "--model-id", model_id, "--mlx-model-artifacts-dir", pack,
                    "--max-concurrent-requests", "1", "--mlx-mtp-policy", "disabled",
                    "--disable-ngram-acceleration", "--stream-experts", "off"]
        case "omlx":
            models = scratch / "models"
            models.mkdir(exist_ok=True)
            (models / model_id).unlink(missing_ok=True)

            (models / model_id).symlink_to(pack, target_is_directory=True)
            return [homebrew / "omlx", "serve", *listen, "--model-dir", models,
                    "--max-concurrent-requests", "1", "--memory-guard", "safe",
                    "--no-cache", "--no-hf-cache", "--base-path", scratch / "omlx", "--headless"]
        case "mtplx":
            return [homebrew / "mtplx", "serve", *listen, "--model", pack, "--profile", "stable",
                    "--no-auth", "--generation-mode", "ar", "--no-mtp", "--ngram-prewarm", "off",
                    "--scheduler-mode", "serial", "--max-active-requests", "1"]
        case "tensorfold":
            builds = sorted(stage.glob("tensorfold/*/bin/tensorfold-native"))
            if len(builds) != 1:
                raise Failure("setup_failed", "Expected one unpacked TensorFold macOS executable.")
            return [builds[0], "serve", pack, *listen, "--name", model_id, "--context", "2048",
                    "--max-inflight", "1", "--no-drafts", "--no-update-check"]
        case "mlxcel":
            return [stage / "mlxcel/mlxcel-server", *listen, "--model", pack, "--offline"]
        case "mlx-lm":
            return [MLX_RUN, *listen, "--model", pack,
                    "--decode-concurrency", "1", "--prompt-concurrency", "1",
                    "--prompt-cache-size", "0"]
        case "ds4":
            # ds4 expects one GGUF file. Give its loader the actual first weight shard.
            index = json.loads((pack / "model.safetensors.index.json").read_text())
            shard = sorted(set(index["weight_map"].values()))[0]
            return [stage / "ds4-server", *listen, "--model", pack / shard, "--ctx", "2048"]
        case "splash":
            return [homebrew / "splash", "serve", *listen, "--model", REPOSITORIES[key],
                    "--model-dir", pack, "--draft-model", stage / "splash-drafts" / key,
                    "--language-only", "--offline", "--no-webui", "--max-context", "2048"]
    raise ValueError(lane)


def wait_for_model(proc, port, stopped):
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if stopped.is_set():
            raise Failure("pressure_stop", "The host entered critical memory pressure.")
        if proc.poll() is not None:
            raise Failure("startup_failed", "The server exited before model discovery.")
        connection = http.client.HTTPConnection(HOST, port, timeout=2)
        try:
            connection.request("GET", "/v1/models")
            with connection.getresponse() as response:
                if response.status == 200:
                    models = json.load(response).get("data", [])
                    if len(models) == 1 and models[0].get("id"):
                        return models[0]["id"]
        except (OSError, http.client.HTTPException, ValueError):
            pass
        finally:
            connection.close()
        time.sleep(0.5)
    raise Failure("startup_timeout", "No single model became discoverable within 360 seconds.")


def complete(port, model, prompt, tokenizer):
    body = {
        "model": model, "prompt": prompt["input_text"], "max_tokens": CAP,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    connection = http.client.HTTPConnection(HOST, port, timeout=REQUEST_TIMEOUT)
    start = time.monotonic()
    first = None
    fragments = []
    usage = None
    reason = None
    done = False
    try:
        connection.request("POST", "/v1/completions", json.dumps(body),
                     headers={"Content-Type": "application/json"})
        with connection.getresponse() as response:
            if response.status != 200:
                detail = response.read(3000).decode("utf-8", errors="replace")
                raise Failure("http_error", f"HTTP {response.status}: {detail}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise Failure("stream_error", "The response did not use SSE.")
            for line in response:
                if time.monotonic() - start > REQUEST_TIMEOUT:
                    raise Failure("request_timeout", "Streaming exceeded 120 seconds.")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise Failure("stream_error", str(event["error"]))
                if (event.get("usage") or {}).get("completion_tokens") is not None:
                    usage = event["usage"]["completion_tokens"]
                for choice in event.get("choices", []):
                    fragment = choice.get("text")
                    if fragment:
                        if first is None:
                            first = time.monotonic()
                        fragments.append(fragment)
                    if choice.get("finish_reason"):
                        reason = choice["finish_reason"]
    finally:
        connection.close()
    wall = time.monotonic() - start
    if first is None or not (done or reason in ("stop", "length")):
        raise Failure("stream_error", "No complete nonempty text stream was received.")
    output = "".join(fragments)
    count = len(tokenizer.encode(output, add_special_tokens=False).ids)
    if count <= 0 or (usage is not None and not 0 < usage <= CAP):
        raise Failure("stream_error", f"Invalid completion count (text={count}, usage={usage}).")
    return {
        "prompt": prompt["id"], "text_tokens": count, "reported_tokens": usage,
        "wall_seconds": round(wall, 6), "ttft_seconds": round(first - start, 6),
        "finish_reason": reason, "done": done,
        "text_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def public_error(detail, log, paths, status):
    for path in paths:
        detail = detail.replace(str(path), "<local>")
    # Keep a single diagnostic line. Remove machine-specific paths and endpoint addresses.
    interesting = re.compile(
        r"gguf|unsupported|not supported|unknown model|failed|exception|error|"
        r"missing|not found|no such|requires|not compatible|invalid|not implemented|"
        r"cannot load|BadAffine|quantized weight shape|runtime kernels read|"
        r"memory budget|too large",
        re.I,
    )
    lines = log.splitlines()
    matches = [line.strip() for line in lines if interesting.search(line)]
    if matches:
        detail += " " + " ".join(dict.fromkeys(matches[-4:]))
    detail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", detail)
    detail = re.sub(r"/(?:private|Users|Volumes|System|opt|var|tmp)/[^\s\"',<>;)}\]]+", "<local>", detail)
    detail = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "<address>", detail)
    detail = re.sub(r"\b[\w-]+\.(?:local|lan)\b", "<host>", detail, flags=re.I)
    detail = re.sub(r"\s+", " ", detail).strip()[:1200]
    unsupported = re.search(
        r"not a GGUF|GGUF only|BadAffine|quantized weight shape inconsistency|"
        r"runtime kernels read MLX affine|missing biases|"
        r"(?:unsupported|unknown|not supported|does not support).{0,55}"
        r"(?:architecture|model type|quantization|mxfp4|qwen4_exp)|"
        r"(?:architecture|model type|quantization|mxfp4|qwen4_exp).{0,55}"
        r"(?:unsupported|unknown|not supported|not implemented)",
        detail, re.I,
    )
    if unsupported and status not in ("request_timeout", "pressure_stop"):
        status = "unsupported_pack"
    return status, detail


def guard_process(proc, finished, stopped):
    while not finished.wait(1):
        if pressure() >= 4:
            stopped.set()
            proc.terminate()
            return


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def trial(stage, lane, pack, tokenizer, prompts):
    key = "dense" if pack.name == PACK_PINS["dense"] else "flash"
    scratch = stage / "logs" / f"{key}-{lane}"
    scratch.mkdir(parents=True, exist_ok=True)
    log_path = scratch / "server.log"
    port = available_port()
    before = pressure()
    result = {"model": key, "engine": lane, "pressure_before": before, "rows": []}
    proc = None
    guard = None
    finished = threading.Event()
    stopped = threading.Event()
    phase = "setup"
    try:
        if before >= 4:
            raise Failure("pressure_stop", "The host was already at critical memory pressure.")
        args = command(stage, lane, pack, port, scratch)
        env = {
            key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG")
            if key in os.environ
        }
        env.update({
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(scratch / "cache"), "XDG_CONFIG_HOME": str(scratch / "config"),
        })
        if lane == "mlx-lm":
            hf_cache = scratch / "hf-cache"
            hf_cache.mkdir(exist_ok=True)
            env["HF_HUB_CACHE"] = str(hf_cache)
        if lane == "splash":
            splash_home = scratch / "home"
            splash_home.mkdir(exist_ok=True)
            env["HOME"] = str(splash_home)
        if lane == "ax-engine":
            env["DYLD_LIBRARY_PATH"] = os.environ["AX_BENCH_MLX_LIB"]
        with log_path.open("wb") as log:
            proc = subprocess.Popen(args, cwd=args[0].parent, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        guard = threading.Thread(target=guard_process, args=(proc, finished, stopped), daemon=True)
        guard.start()
        phase = "discovery"
        start = time.monotonic()
        model = wait_for_model(proc, port, stopped)
        result["discovery_seconds"] = round(time.monotonic() - start, 6)
        phase = "warmup"
        result["warmup"] = complete(port, model, prompts[WARMUP], tokenizer)
        phase = "measured"
        for repeat in range(REPEATS):
            for prompt_id in PROMPT_IDS:
                row = complete(port, model, prompts[prompt_id], tokenizer)
                result["rows"].append({"repeat": repeat + 1, **row})
        wall = sum(row["wall_seconds"] for row in result["rows"])
        total = sum(row["text_tokens"] for row in result["rows"])
        result.update(status="passed", tok_s=round(total / wall, 4), text_tokens=total)
        ttft = sorted(row["ttft_seconds"] for row in result["rows"])
        result["ttft_p95_seconds"] = ttft[math.ceil(0.95 * len(ttft)) - 1]
    except (Failure, OSError, http.client.HTTPException, ValueError, KeyError) as exc:
        if stopped.is_set():
            status = "pressure_stop"
        elif isinstance(exc, Failure):
            status = exc.status
        elif isinstance(exc, TimeoutError):
            status = "request_timeout"
        elif proc is None:
            status = "setup_failed"
        else:
            status = "request_failed"
        result.update(
            status=status, detail=str(exc), tok_s=0, phase=phase,
            process_exit=proc.poll() if proc is not None else None,
        )
    finally:
        finished.set()
        if guard is not None:
            guard.join()
        if proc is not None:
            stop_process(proc)
    if result["status"] != "passed":
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        result["status"], result["detail"] = public_error(
            result["detail"], log, (stage, pack, Path.home()), result["status"]
        )
    result["pressure_after"] = pressure()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=PACK_PINS, default=list(PACK_PINS))
    lanes = ("ax-engine", "omlx", "mtplx", "tensorfold", "mlxcel", "mlx-lm", "ds4", "splash")
    parser.add_argument("--engines", nargs="+", choices=lanes, default=list(lanes))
    args = parser.parse_args()
    stage = args.stage.resolve()
    corpus = args.corpus.read_bytes()
    prompts = {row["id"]: row for row in map(json.loads, corpus.splitlines())}
    print(json.dumps({
        "kind": "environment", "os": platform.mac_ver()[0],
        "arch": platform.machine(),
        "unified_memory_bytes": int(subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True
        )),
        "packs": PACK_PINS, "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "prompt_ids": PROMPT_IDS, "warmup_id": WARMUP, "output_cap": CAP, "repeats": REPEATS,
    }), flush=True)
    for key in args.models:
        pack = Path((stage / f"{key}.path").read_text().strip())
        if pack.name != PACK_PINS[key] or not (pack / "config.json").is_file():
            raise RuntimeError(f"Missing pinned {key} snapshot.")
        tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
        for lane in args.engines:
            print(json.dumps(trial(stage, lane, pack, tokenizer, prompts)), flush=True)


if __name__ == "__main__":
    main()
