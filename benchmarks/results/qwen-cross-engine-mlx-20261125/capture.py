"""Measure bounded fixed-token Qwen serving arms using the stock serving client."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ARMS = {
    "incumbent": {},
    "direct": {},
    "cohort4": {"AX_MLX_DFLASH2_COHORT_MAX_ROWS": "4"},
    "replay-probe": {"AX_MLX_DFLASH2_WINDOW": "4", "AX_MLX_DFLASH2_ROUND_TRACE": "1",
                     "AX_MLX_DFLASH2_PROFILE_REPLAY": "1", "AX_MLX_MTP_LINEAR_TAPE_CAPTURE": "0",
                     "AX_MLX_MTP_SKIP_PREFIX_CHECKPOINT": "1", "AX_MLX_MTP_LINEAR_PROJECTED_REPLAY": "1",
                     "AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP": "0", "AX_MLX_MTP_LINEAR_LAYER_COMPILE": "0",
                     "AX_MLX_MTP_WHOLE_VERIFY_COMPILE": "0", "AX_MLX_MTP_REUSE_PROCESSED_GDN": "1",
                     "AX_MLX_MTP_RELAXED_TARGET_VERIFY": "1", "AX_MLX_MTP_FUSED_GDN_VERIFY": "1",
                     "AX_MLX_MTP_LAZY_ADOPT_STATE": "1"},
    "canonical": {"AX_FLASH_NEXT_MTP_VERIFIER": "canonical"},
    "block3": {"AX_FLASH_NEXT_MTP_VERIFIER": "block"},
    "block2": {"AX_FLASH_NEXT_MTP_VERIFIER": "block", "AX_FLASH_NEXT_MTP_BLOCK_DRAFTS": "2"},
}
ARMS["replay-control"] = {**ARMS["replay-probe"], "AX_MLX_DFLASH2_PROFILE_REPLAY": "0"}


def invoke(*command):
    return subprocess.check_output(command, text=True).strip()


def idle():
    return float(re.findall(r"([0-9.]+)% idle", invoke("top", "-l", "2", "-s", "3", "-n", "0"))[-1])


def sha(path):
    from hashlib import file_digest
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


@contextmanager
def pressure_watch(server, samples):
    finished = threading.Event()

    def watch():
        while not finished.is_set() and server.poll() is None:
            level = int(invoke("sysctl", "-n", "kern.memorystatus_vm_pressure_level"))
            samples.append(level)
            if level == 4:
                server.terminate()
                break
            finished.wait(1)

    worker = threading.Thread(target=watch)
    worker.start()
    try:
        yield
    finally:
        finished.set()
        worker.join()


def run(args, arm, index, base):
    flash = args.drafter is None
    direct = arm == "direct"
    concurrency = 1 if flash or direct else 4
    folder = args.output / f"{index:02d}-{arm}"
    folder.mkdir()
    with socket.socket() as port_check:
        if port_check.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("benchmark port already in use")
    model_id = "qwen3.8-flash-next:axq" if flash else "qwen3.8-27b:axq"
    settings = {"AX_MLX_PREFIX_CACHE_DISK_DISABLED": "1", "AX_ENGINE_PREFIX_REUSE_DISABLED": "1",
                "AX_MLX_TREE_VERIFY": "0",
                "AX_MLX_DFLASH2": "0" if flash or direct else "1", **ARMS[arm]}
    if args.drafter and not direct:
        settings.update(AX_MLX_DFLASH2_DIR=str(args.drafter), AX_MLX_DFLASH2_COHORT="1",
                        AX_MLX_MTP_RELAXED_TARGET_VERIFY="1", AX_MLX_MTP_LINEAR_PROJECTED_REPLAY="1")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("AX_", "MLX_"))}
    policy = "disabled" if direct else ("required" if flash else "auto")
    command = [str(args.server), "--mlx", "--model-id", model_id,
               "--mlx-model-artifacts-dir", str(args.pack), "--host", "127.0.0.1", "--api-key", "",
               "--port", str(args.port), "--max-concurrent-requests", str(concurrency),
               "--mlx-mtp-policy", policy]
    if flash or direct:
        command.append("--disable-ngram-acceleration")
    if flash:
        command.extend(("--stream-experts", "off", "--total-blocks", "2048"))
    samples = []
    idle_before = idle()
    started = time.monotonic()
    with (folder / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env | settings)
        try:
            with pressure_watch(server, samples):
                while time.monotonic() - started < 360 and server.poll() is None:
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as response:
                            if json.load(response)["status"] == "ok":
                                break
                    except OSError:
                        pass
                    time.sleep(0.3)
                else:
                    raise RuntimeError(f"{arm}: server failed to become ready")
                ready = time.monotonic() - started
                count = len(args.corpus.read_text().splitlines())
                subprocess.run([sys.executable, str(Path(__file__).with_name("bench_ax_serving.py")),
                                "--base-url", base, "--model-id", model_id, "--corpus", str(args.corpus),
                                "--input-kind", "tokens", "--record-token-ids", "--ignore-eos",
                                "--requests", str(count), "--concurrency", str(concurrency),
                                "--warmup-requests", "2", "--output", str(folder / "serving.json")],
                               check=True, timeout=360)
                idle_after = idle()
        finally:
            stop(server)
    result = json.loads((folder / "serving.json").read_bytes())
    if result["summary"]["error_requests"] or not samples or max(samples) == 4:
        raise RuntimeError(f"{arm}: request or pressure failure")
    routes = result["summary"]["route_decisions"]
    if not flash and not direct and routes.get("ax_mlx_dflash2_cohort_lanes", 0) == 0:
        raise RuntimeError(f"{arm}: no DFlash2 cohort ran")
    result["target"]["base_url"] = "<loopback>"
    result["corpus"]["path"] = args.corpus.name
    (folder / "serving.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    trace = []
    for line in (folder / "server.log").read_text(errors="replace").splitlines():
        if line.startswith("dflash2_round "):
            trace.append({k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)})
    metadata = {"arm": arm, "server_sha256": sha(args.server), "corpus_sha256": sha(args.corpus),
                "model_revision": args.pack.name, "hardware": invoke("sysctl", "-n", "machdep.cpu.brand_string"),
                "memory_bytes": int(invoke("sysctl", "-n", "hw.memsize")), "macos": platform.mac_ver()[0],
                "cpu_idle_before": idle_before, "cpu_idle_after": idle_after, "ready_seconds": ready,
                "pressure_levels": samples, "settings": {k: v for k, v in settings.items() if k != "AX_MLX_DFLASH2_DIR"},
                "round_trace": trace}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(arm, result["summary"]["output_token_throughput_tok_s"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "corpus", "pack", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--drafter", type=Path)
    parser.add_argument("--port", type=int, default=31647)
    parser.add_argument("arms", nargs="+", choices=ARMS)
    args = parser.parse_args()
    args.output.mkdir()
    for index, arm in enumerate(args.arms):
        run(args, arm, index, f"http://127.0.0.1:{args.port}")


if __name__ == "__main__":
    main()
