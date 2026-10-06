#!/usr/bin/env python3
"""Run pack-bound Flash Next direct/required-MTP functional HTTP controls.

Dry-run is weight-free. Live execution owns only its spawned server processes.
This verifies functional execution, not model quality, speed or MTP-S/P/D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import socket
import subprocess
import time
from typing import Any
import urllib.request

try:
    from qualify_qwen38_flash_next import PACK_REVISION, PRIMARY_REPO, _mtp_metadata
except ImportError:
    from scripts.qualify_qwen38_flash_next import PACK_REVISION, PRIMARY_REPO, _mtp_metadata

VERIFIED = "ax_engine_flash_next_mtp_verified_steps_total"
ERRORS = (
    "ax_engine_flash_next_mtp_attach_failed_total",
    "ax_engine_flash_next_mtp_prefill_absorb_failures_total",
    "ax_engine_flash_next_mtp_step_errors_total",
)
CASES = (
    ("count", "Count from 1 to 12, separated by commas. Output only the numbers.", 48),
    ("repeat", "Count from 1 to 12, separated by commas. Output only the numbers.", 48),
    ("extend", "Count from 1 to 12, separated by commas. Output only the numbers. End with a period.", 48),
    ("json", 'Return a JSON object with keys "color" and "fruit", values "red" and "apple".', 48),
    ("stream", "Count from 1 to 12, separated by commas. Output only the numbers.", 48),
    ("after_stream", "Count from 1 to 12, separated by commas. Output only the numbers.", 48),
    ("one_token", "Write a sentence about a green apple.", 1),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inventory(root: Path, inventory: dict[str, Any],
                       link_root: Path | None = None) -> dict[str, str]:
    """Bind every pack member by size and SHA-256.

    A Hugging Face snapshot holds symlinks into a sibling blob store; pass that
    cache as `link_root` so links may resolve there but nowhere else.
    """
    if inventory.get("repo_id") != PRIMARY_REPO or inventory.get("revision") != PACK_REVISION:
        raise ValueError("inventory must bind the current immutable Flash Next MXFP4 MTP pack")
    members = inventory.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("inventory needs a nonempty member list")
    if not isinstance(inventory.get("storage"), str) or not inventory["storage"].strip():
        raise ValueError("inventory must record the storage medium")
    names = set()
    for member in members:
        name = member["name"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in names:
            raise ValueError("inventory contains an unsafe or duplicate path")
        names.add(name)
        path = root / relative
        resolved = path.resolve()
        if not (resolved.is_relative_to(root.resolve())
                or (link_root is not None and resolved.is_relative_to(link_root.resolve()))):
            raise ValueError("inventory member escapes the model directory")
        if path.stat().st_size != member["size"] or sha256(path) != member["sha256"]:
            raise ValueError(f"pack inventory mismatch: {name}")
        if name.endswith(".safetensors") and member.get("publisher_sha256") != member["sha256"]:
            raise ValueError(f"missing or mismatched publisher payload binding: {name}")
    required = {"config.json", "model.safetensors.index.json", "axquant_manifest.json",
                "mtplx_runtime.json", "mtp.safetensors"}
    if not required <= names:
        raise ValueError("inventory is missing required runtime files")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards <= names:
        raise ValueError("inventory omits indexed model shards")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    allowed_local = {"model-manifest.json", "ax-engine-pack-inventory.json", ".ax-engine-download.json"}
    if actual - names - allowed_local:
        raise ValueError(f"unbound model-directory files: {sorted(actual - names - allowed_local)}")
    _mtp_metadata(root)
    # The server may derive this from the published config and tensor headers.
    # A pre-existing manifest is part of the execution input, so bind it too.
    manifest = root / "model-manifest.json"
    return {"model-manifest.json": sha256(manifest)} if manifest.exists() else {}


# Mac Studio hardware identifiers (2022 M1 Ultra/Max, 2023 M2 Ultra/Max).
STUDIO_MODELS = frozenset({"Mac13,1", "Mac13,2", "Mac14,13", "Mac14,14"})


def form_factor() -> str:
    """Name the form factor only when the hardware model identifier proves it."""
    model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    return "Mac Studio" if model in STUDIO_MODELS else f"unrecognized ({model})"


def storage_info(root: Path) -> dict[str, Any]:
    device = subprocess.check_output(["df", "-P", str(root)], text=True).splitlines()[-1].split()[0]
    info = plistlib.loads(subprocess.check_output(["diskutil", "info", "-plist", device]))
    if info.get("Internal") is not True or info.get("SolidState") is not True:
        raise ValueError("functional controls require model files on the internal SSD")
    return {"internal": True, "solid_state": True, "bus_protocol": info.get("BusProtocol")}


def cpu_idle_percent() -> float:
    """System-wide idle CPU over a 3-second window (second `top` sample)."""
    out = subprocess.run(["top", "-l", "2", "-s", "3", "-n", "0"], capture_output=True, text=True,
                         check=True).stdout
    samples = re.findall(r"CPU usage:.*?([0-9.]+)% idle", out)
    if not samples:
        raise ValueError("top reported no CPU usage line")
    return float(samples[-1])


def wait_for_quiet(min_idle: float = 90.0, timeout: float = 900.0, sample=cpu_idle_percent,
                   sleep=time.sleep, clock=time.monotonic) -> dict[str, float]:
    """Block until system-wide idle CPU is at least `min_idle` percent.

    Timing and memory evidence is only meaningful on an otherwise idle host; the
    returned record is stored with the result so a reader can check it. The idle
    share (not the load average) is used because a desktop session keeps the
    load average near 3 on this 24-core host while the CPUs stay about 93% idle.
    """
    started = clock()
    while True:
        idle = sample()
        if idle >= min_idle:
            return {"cpu_idle_percent": round(idle, 2), "min_idle_percent": min_idle,
                    "waited_seconds": round(clock() - started, 1)}
        if clock() - started >= timeout:
            raise TimeoutError(f"host stayed busy: {idle:.1f}% idle < {min_idle}%")
        sleep(10)


def metrics(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and "{" not in parts[0]:
            result[parts[0]] = float(parts[1])
    return result


def check_activation(before: dict[str, float], after: dict[str, float], required: bool,
                     completion_tokens: int, budget: int) -> dict[str, float]:
    for name in (VERIFIED, *ERRORS):
        if name not in before or name not in after:
            raise ValueError(f"missing required route metric: {name}")
    delta = {name: after[name] - before[name] for name in (VERIFIED, *ERRORS)}
    if any(delta[name] != 0 for name in ERRORS):
        raise ValueError(f"Flash Next MTP route errors: {delta}")
    if completion_tokens <= 0 or completion_tokens > budget:
        raise ValueError("completion token count is outside its request budget")
    if required and budget > 1 and (completion_tokens <= 1 or delta[VERIFIED] <= 0):
        raise ValueError("required-MTP request completed without verified MTP steps")
    if not required and delta[VERIFIED] != 0:
        raise ValueError("disabled-MTP or default-policy control activated the drafter")
    return delta


def request(base: str, path: str, payload: Any = None) -> bytes:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read()


def parse_stream(raw: bytes) -> dict[str, Any]:
    content = []
    finish = None
    usage = None
    done = False
    for line in raw.decode().splitlines():
        if not line.startswith("data: "):
            continue
        value = line.removeprefix("data: ")
        if value == "[DONE]":
            done = True
            continue
        event = json.loads(value)
        if "error" in event:
            raise ValueError(f"stream returned an error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            finish = choice.get("finish_reason") or finish
    if not done or finish is None or usage is None:
        raise ValueError("stream is missing its finish, usage or DONE frame")
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(content)},
                         "finish_reason": finish}], "usage": usage}


def check_answer(name: str, response: dict[str, Any]) -> None:
    text = response["choices"][0]["message"]["content"].strip()
    if name == "one_token":
        return
    if name == "json":
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        if json.loads(text) != {"color": "red", "fruit": "apple"}:
            raise ValueError("JSON control returned incorrect content")
    else:
        expected = ", ".join(str(number) for number in range(1, 13))
        if name == "extend":
            expected += "."
        if re.sub(r",\s*", ", ", text) != expected:
            raise ValueError(f"count control returned incorrect content: {text!r}")


SERVER_MODES = ("disabled", "required", "default")


def server_command(server: Path, root: Path, mode: str, port: int) -> list[str]:
    """Server argv for one control mode.

    `disabled` and `required` pin the MTP policy and turn generic n-gram
    acceleration off to isolate the drafter. `default` passes none of those
    flags, so it exercises exactly what an operator gets from the defaults
    (auto MTP policy, generic n-gram acceleration, Auto expert paging).
    """
    if mode not in SERVER_MODES:
        raise ValueError(f"unknown control mode {mode!r}; expected one of {SERVER_MODES}")
    base = [str(server), "--mlx", "--model-id", "qwen3.8-flash-next:mxfp4",
            "--mlx-model-artifacts-dir", str(root)]
    tail = ["--host", "127.0.0.1", "--port", str(port)]
    if mode == "default":
        return [*base, *tail]
    return [*base, "--mlx-mtp-policy", mode, "--disable-ngram-acceleration",
            "--stream-experts", "auto", *tail]


def server_env(explicit: dict[str, str] | None = None) -> dict[str, str]:
    # Inherited overrides must not silently replace arithmetic, paging,
    # prefix-reuse or the stated MTP control. A caller that deliberately selects
    # an override passes it in `explicit` and records it in its own contract.
    return {**{name: value for name, value in os.environ.items()
               if not name.startswith(("AX_", "MLX_"))}, "RUST_LOG": "info", **(explicit or {})}


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def scrub_log(path: Path) -> None:
    if path.exists():
        text = path.read_text(errors="replace").replace("127.0.0.1", "[loopback]")
        path.write_text(text.replace(str(Path.home()), "[home]"))


def check_blocked_required(server: Path, root: Path, output: Path) -> dict[str, Any]:
    log_path = output / "server-required-kill-switch.log"
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(server_command(server, root, "required", 0),
                                       stdout=log, stderr=log,
                                       env={**server_env(), "AX_NO_SPEC": "1"},
                                       start_new_session=True)
            try:
                code = process.wait(timeout=900)
            finally:
                stop_owned(process)
        text = log_path.read_text(errors="replace")
        if code == 0 or not any(error in text for error in (
                "no usable drafter was activated", "MlxMtpRequiredButUnavailable")):
            raise ValueError("required MTP plus AX_NO_SPEC did not fail with its activation error")
        return {"AX_NO_SPEC": "1", "mtp_policy": "required", "exit_code": code,
                "activation_error_observed": True}
    finally:
        scrub_log(log_path)


def _run_mode(server: Path, root: Path, output: Path, mode: str,
              ready: dict[str, float] | None = None) -> list[dict[str, Any]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = server_env()
    command = server_command(server, root, mode, port)
    raw_log = output / f"server-{mode}.log"
    rows = []
    with raw_log.open("wb") as log:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            deadline = started + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"owned {mode} server exited with {process.returncode}")
                try:
                    request(base, "/v1/models")
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native server readiness timeout")
                    time.sleep(1)
            if ready is not None:
                ready[mode] = round(time.monotonic() - started, 3)
            for name, prompt, budget in CASES:
                before = metrics(request(base, "/metrics").decode())
                body = {"model": "qwen3.8-flash-next:mxfp4", "messages": [
                    {"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": budget, "chat_template_kwargs": {"enable_thinking": False}}
                if name == "stream":
                    body.update(stream=True, stream_options={"include_usage": True})
                raw = request(base, "/v1/chat/completions", body)
                response = parse_stream(raw) if name == "stream" else json.loads(raw)
                if name == "stream":
                    (output / f"{mode}-stream.sse").write_bytes(raw)
                check_answer(name, response)
                after = metrics(request(base, "/metrics").decode())
                count = response["usage"]["completion_tokens"]
                delta = check_activation(before, after, mode == "required", count, budget)
                rows.append({"name": name, "request": body, "response": response,
                             "metrics_before": before, "metrics_after": after,
                             "activation_delta": delta})
                (output / f"{mode}-{name}.json").write_text(json.dumps(rows[-1], indent=2) + "\n")
                print(f"{mode}/{name}: {count} tokens, {delta[VERIFIED]} verified steps", flush=True)
        finally:
            stop_owned(process)
    return rows


def run_mode(server: Path, root: Path, output: Path, mode: str,
             ready: dict[str, float] | None = None) -> list[dict[str, Any]]:
    try:
        return _run_mode(server, root, output, mode, ready)
    finally:
        scrub_log(output / f"server-{mode}.log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--link-root", type=Path,
                        help="cache root that pack symlinks may resolve into (Hugging Face snapshots)")
    parser.add_argument("--modes", nargs="+", choices=SERVER_MODES, default=["disabled", "required"],
                        help="control modes; `default` runs the server with no policy flags")
    args = parser.parse_args()
    if "disabled" not in args.modes or len(set(args.modes)) != len(args.modes):
        parser.error("--modes must list each mode once and include the disabled baseline")
    contract = {"repo_id": PRIMARY_REPO, "revision": PACK_REVISION, "cases": [c[0] for c in CASES],
                "modes": list(args.modes), "expert_stream": "auto",
                "qualification": False, "release_ready": False,
                "negative_control": "required plus AX_NO_SPEC must reject activation"
                if "required" in args.modes else "none: required mode not requested",
                "scope": "functional HTTP execution; MTP-S/P/D and performance not assessed"}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return 0
    if not all((args.model_dir, args.inventory, args.server_bin, args.output)):
        parser.error("live controls require model-dir, inventory, server-bin and output")
    if platform.system() != "Darwin":
        parser.error("live controls require macOS on the Ultra-class 192 GiB+ target")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if "Ultra" not in chip or memory < 192 * 1024**3:
        parser.error("live controls require the Ultra-class 192 GiB+ Flash Next target")
    storage = storage_info(args.model_dir)
    executable_before = sha256(args.server_bin)
    receipt_before = sha256(args.inventory)
    inventory = json.loads(args.inventory.read_text())
    derived_before = validate_inventory(args.model_dir, inventory, args.link_root)
    args.output.mkdir(parents=True, exist_ok=False)
    negative = (check_blocked_required(args.server_bin.resolve(), args.model_dir.resolve(), args.output)
                if "required" in args.modes else None)
    manifest = args.model_dir / "model-manifest.json"
    derived_ready = {"model-manifest.json": sha256(manifest)}
    if derived_before and derived_before != derived_ready:
        raise ValueError("native manifest changed during activation control")
    readiness: dict[str, float] = {}
    results = {mode: run_mode(args.server_bin.resolve(), args.model_dir.resolve(), args.output, mode,
                              readiness) for mode in contract["modes"]}
    derived_after = validate_inventory(args.model_dir, inventory, args.link_root)
    if derived_ready != derived_after:
        raise ValueError("native manifest changed during functional controls")
    if executable_before != sha256(args.server_bin) or receipt_before != sha256(args.inventory):
        raise ValueError("server binary or staging receipt changed during controls")
    comparisons = {
        mode: [{"name": direct["name"], "text_equal":
                direct["response"]["choices"] == other["response"]["choices"]}
               for direct, other in zip(results["disabled"], results[mode], strict=True)]
        for mode in args.modes if mode != "disabled"}
    evidence = {**contract, "functional_controls_passed": True, "negative_control_result": negative,
                "hardware": {
        "form_factor": form_factor(), "soc": chip, "memory_bytes": memory,
        "storage": {"declared": inventory["storage"], **storage}, "os": platform.mac_ver()[0]},
        "server_sha256": executable_before, "inventory_sha256": receipt_before,
        "native_manifest_before": derived_before, "native_manifest_ready": derived_ready,
        "native_manifest_after": derived_after,
        "readiness_seconds": readiness, "page_cache": "not controlled",
        "comparisons": comparisons, "results": results}
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Functional controls passed; qualification and MTP-S/P/D remain unassessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
