"""Verify native startup policy for corrupt artifacts and unavailable MTP heads."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from run import ENV, REVISION, CRITICAL_PRESSURE, fresh_clone, matrix, native, sample_host


def connect(port):
    with socket.socket() as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


def boot(args, root, fault, policy, prompt, expected):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [
        str(args.server), "--mlx", "--model-id", "qwen3.8-flash-next:mxfp4",
        "--mlx-model-artifacts-dir", str(root), "--mlx-mtp-policy", policy,
        "--stream-experts", "off", "--disable-ngram-acceleration", "--max-concurrent-requests", "1",
        "--total-blocks", "2048", "--host", "127.0.0.1", "--port", str(port),
    ]
    variables = ENV | {"AX_FLASH_NEXT_RESIDENT_LOAD_LAB": "lazy-cpu",
                       "AX_FLASH_NEXT_SIDECAR_LOAD_LAB": "descriptor-lazy"}
    log_path = args.output / f"{fault}-{policy}.log"
    deadline = time.monotonic() + 300
    serving_expected = fault == "head-attachment" and policy == "disabled"
    saw_listener = False
    matched = None
    with log_path.open("x") as log:
        child = subprocess.Popen(command, env=native.server_env(variables), stdout=log, stderr=log)
        try:
            while child.poll() is None and time.monotonic() < deadline:
                if sample_host(child.pid)["pressure"] in CRITICAL_PRESSURE:
                    raise RuntimeError("Critical pressure in the incomplete-sidecar control")
                if connect(port):
                    saw_listener = True
                    break
                time.sleep(1)
            if not serving_expected:
                if saw_listener or child.poll() in (None, 0):
                    raise RuntimeError(f"{fault}/{policy} did not fail before serving")
            else:
                if not saw_listener:
                    raise RuntimeError("Disabled-MTP startup did not recover with the direct route")
                result = matrix.generate(base, prompt, max_tokens=32, ignore_eos=True)
                mtp = result["performance"]["mtp"]
                matched = result["output_tokens"] == expected and result["prompt_tokens"] == prompt
                assert matched and not any(mtp[key] for key in ("available", "requested", "active"))
                assert json.loads(native.request(base, "/health"))["status"] == "ok"
        finally:
            native.stop_owned(child)
    logs = log_path.read_text()
    attachment_failures = logs.count("sidecar failed to attach")
    if fault == "artifact-integrity":
        assert "InvalidManifest" in logs and "exceeds file length" in logs, logs[-1200:]
        assert attachment_failures == 0
        failure_boundary = "manifest"
    else:
        assert attachment_failures == 1, logs[-1200:]
        assert "mtp_norm_layout must be raw_hf_delta or mlx_multiplier" in logs, logs[-1200:]
        if policy == "required":
            assert "Error: Engine(MlxMtpRequiredButUnavailable)" in logs, logs[-1200:]
        failure_boundary = "head attachment"
    with socket.socket() as released:
        released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        released.bind(("127.0.0.1", port))
    return {"listener_opened": saw_listener, "returncode": child.returncode,
            "direct_tokens_match": matched, "attachment_failures_logged": attachment_failures,
            "failure_boundary": failure_boundary, "rebound_port": True}


def change_clone(root, fault, sidecar_sha256):
    sidecar = root / "mtp.safetensors"
    if fault == "artifact-integrity":
        with sidecar.open("r+b") as fixture:
            fixture.truncate(sidecar.stat().st_size - 2)
        return
    assert sidecar_sha256 == native.sha256(sidecar)
    runtime = root / "mtplx_runtime.json"
    config = json.loads(runtime.read_text())
    assert config["mtp_norm_layout"] in ("raw_hf_delta", "mlx_multiplier")
    config["mtp_norm_layout"] = "invalid_test_layout"
    runtime.write_text(json.dumps(config) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "pack", "inventory", "scratch-dir", "positive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    positive = json.loads(args.positive.read_text())
    assert positive["status"] == "complete" and positive["pack_revision"] == REVISION
    assert positive["build"]["server_sha256"] == native.sha256(args.server)
    inventory = json.loads(args.inventory.read_text())
    members = [row["name"] for row in inventory["members"]]
    assert len(members) == 49 and "mtp.safetensors" in members and inventory["revision"] == REVISION
    case = positive["arms"]["disabled"]["descriptor-lazy"]["cases"][0]
    prompt = matrix.workload_ids(matrix.pack_vocab_size(args.pack), 512, 0)
    assert case["prompt_hash"] == matrix.ids_sha(prompt)
    args.output.mkdir()
    record = {"status":"starting", "pack_revision":REVISION,
              "server_sha256":positive["build"]["server_sha256"],
              "fault_runner_sha256":native.sha256(Path(__file__)), "arms":{}}
    sidecar, = (entry for entry in inventory["members"] if entry["name"] == "mtp.safetensors")
    try:
        for fault in ("artifact-integrity", "head-attachment"):
            arms = record["arms"][fault] = {}
            for policy in ("required", "disabled"):
                with tempfile.TemporaryDirectory(dir=args.scratch_dir) as directory:
                    root = Path(directory)
                    for member in [*members, "model-manifest.json"]:
                        fresh_clone(args.pack / member, root / member)
                    change_clone(root, fault, sidecar["sha256"])
                    arms[policy] = boot(args, root, fault, policy, prompt, case["output_tokens"])
                    print(f"{fault}/{policy} passed", flush=True)
        record["status"] = "complete"
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
