"""Measure the Flash Next PLE pool through the native HTTP server."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from string import hexdigits
import json
from math import isfinite
from pathlib import Path
import platform
from statistics import median
import subprocess
from threading import Event
from unittest.mock import patch

import check_flash_next_native_support as native
import run_flash_next_generate_matrix as matrix

REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
CONTEXT = 32768
POOL_THREADS = 31
PRIMARY_ARMS = (False, True, True, False)
SMOKE_ARMS = (False, True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def sysctl(name):
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()


def pressure():
    return int(sysctl("kern.memorystatus_vm_pressure_level"))


def sample_until(done, samples):
    while not done.wait(1):
        samples.append(pressure())


def validate(result, prompts, lengths):
    require(result["mode"] == "default" and result["verifier"] == "canonical",
            "Unexpected product default serving route")
    trajectories = result["trajectories"]
    cells = result["cells"]
    require([r["index"] for r in trajectories] == list(range(len(prompts)))
            and [r["prompt"] for r in trajectories] == list(prompts),
            "The supplied trajectory prompts changed")
    expected = [(length, index, index < matrix.WARMUPS) for length in lengths
                for index in range(matrix.WARMUPS + matrix.MEASURED)]
    require([(r["length"], r["run"], r["warmup"]) for r in cells] == expected,
            "The fixed timing cells are incomplete")
    expected_hashes = {f"traj-{i:02d}" for i in range(len(prompts))}
    expected_hashes.update(f"cell-{length}-{index}" for length, index, _ in expected)
    require(result["prompt_id_hashes"].keys() == expected_hashes
            and all(isinstance(x, str) and len(x) == 64 and set(x) <= set(hexdigits)
                    for x in result["prompt_id_hashes"].values()),
            "The native prompt digest binding is incomplete")
    require(all(r["status"] == "finished" and r["verified_steps"] == r["route_errors"] == 0
                and 0 < len(r["output_tokens"]) <= matrix.OUTPUT_TOKENS for r in (*trajectories, *cells))
            and all(len(r["output_tokens"]) == matrix.OUTPUT_TOKENS for r in cells),
            "A request changed length, used MTP or encountered a route error")
    return {
        "prompt_id_hashes": result["prompt_id_hashes"],
        "outputs": [[r[k] for k in ("prompt_tokens", "output_tokens", "finish_reason")]
                    for r in (*trajectories, *cells)],
    }


def metrics(result, lengths):
    output = {}
    for length in lengths:
        rows = [r for r in result["cells"] if r["length"] == length and not r["warmup"]]
        ttft = [r["timing"]["ttft_seconds"] for r in rows]
        rates = [matrix.decode_rate(r["timing"]) for r in rows]
        require(len(rows) == matrix.MEASURED
                and all(x is not None and isfinite(x) and x > 0 for x in (*ttft, *rates)),
                "A measured native timing cell is invalid")
        output[str(length)] = {
            "scored_requests": len(rows), "ttft_median_s": median(ttft),
            "decode_median_tok_s": median(rates),
        }
    return output


def run_arm(args, label, pooled, prompts, lengths):
    before = native.wait_for_quiet()["cpu_idle_percent"] if args.scope == "m5" else None
    sampled = [pressure()]
    require(sampled[0] == 1, "The benchmark requires normal memory pressure at startup")
    workers = []
    advertised = []
    stop_owned = native.stop_owned
    request = native.request

    def observe_shutdown(process):
        try:
            if process.poll() is None:
                workers.append(int(subprocess.check_output([args.observer, str(process.pid)],
                                                           text=True, timeout=10)))
        finally:
            stop_owned(process)

    def observe_models(base, path, *request_args, **request_kwargs):
        raw = request(base, path, *request_args, **request_kwargs)
        if path == "/v1/models":
            advertised.extend(model["context_length"] for model in json.loads(raw)["data"])
        return raw

    done = Event()
    with ThreadPoolExecutor(max_workers=1) as monitor:
        future = monitor.submit(sample_until, done, sampled)
        try:
            with (patch.object(native, "stop_owned", observe_shutdown),
                  patch.object(native, "request", observe_models),
                  patch.object(matrix, "TOTAL_BLOCKS", CONTEXT // 16)):
                result = matrix.run_mode(
                    args.server, args.pack, args.output, "default", prompts, lengths, label=label,
                    extra_env={"AX_FLASH_NEXT_NGRAM_READ_POOL": "1" if pooled else "0",
                               "AX_FLASH_NEXT_NGRAM_READ_THREADS": "32"},
                )
        finally:
            done.set()
            future.result()
    sampled.append(pressure())
    after = native.cpu_idle_percent() if args.scope == "m5" else None
    require(workers == [POOL_THREADS if pooled else 0], f"{label}: actual PLE worker count changed")
    require(advertised == [CONTEXT], f"{label}: the server did not advertise the one-slot 32K profile")
    require(all(sample == 1 for sample in sampled), f"{label}: memory pressure exceeded normal")
    identity = validate(result, prompts, lengths)
    raw = args.output / f"{label}.json"
    raw.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return identity, {
        "label": label, "pooled": pooled, "pool_workers": workers[0], "context_tokens": advertised[0],
        "ready_s": result["ready_seconds"], "raw_sha256": digest(raw), "pressure_samples": len(sampled),
        "cpu_idle_percent": [before, after],
        "timing_eligible": args.scope == "m5" and all(90 <= value <= 100 for value in (before, after)),
        "cells": metrics(result, lengths),
    }


def run(args):
    build = json.loads(args.build.read_bytes())
    require(digest(args.server) == build["server_sha256"]
            and digest(args.observer) == build["observer_sha256"], "Native build or observer identity changed")
    require(digest(args.inventory) == INVENTORY, "Pinned pack inventory changed")
    inventory = json.loads(args.inventory.read_bytes())
    expected_manifest = {"model-manifest.json": MANIFEST}
    require(native.validate_inventory(args.pack, inventory) == expected_manifest,
            "The frozen delivered manifest changed")
    if args.scope == "m5":
        native.storage_info(args.pack)
    memory = int(sysctl("hw.memsize"))
    soc = sysctl("machdep.cpu.brand_string")
    expected = ("Apple M5 Max", 128) if args.scope == "m5" else ("Apple M2 Ultra", 192)
    require((soc, memory // 2**30) == expected, "This is not the requested hardware tier")
    args.output.mkdir(exist_ok=False)
    prompts = matrix.PROMPTS if args.scope == "m5" else matrix.PROMPTS[:1]
    lengths = matrix.LENGTHS if args.scope == "m5" else (512,)
    schedule = ((("initial", False),) + tuple((f"arm-{i}", flag) for i, flag in enumerate(PRIMARY_ARMS, 1))
                if args.scope == "m5" else tuple((f"arm-{i}", flag) for i, flag in enumerate(SMOKE_ARMS, 1)))
    first = None
    arms = []
    for label, flag in schedule:
        identity, arm = run_arm(args, label, flag, prompts, lengths)
        if first is None:
            first = identity
        require(identity == first, f"{label}: prompt, generated tokens or stop state differs")
        arms.append(arm)
        print(f"{label} verified with {arm['pool_workers']} pool workers.", flush=True)
    require(native.validate_inventory(args.pack, inventory) == expected_manifest,
            "The checkpoint changed during serving")
    record = {"scope": "M5 serving comparison" if args.scope == "m5" else "M2 functional serving",
              "qualification": False, "hardware": {"soc": soc, "memory_bytes": memory,
                                                     "os": platform.mac_ver()[0]},
              "pack_revision": REVISION, "manifest": expected_manifest, "inventory_sha256": INVENTORY,
              "build": build, "matching_responses_per_arm": len(first["outputs"]), "arms": arms}
    (args.output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("server", "observer", "pack", "inventory", "build", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scope", choices=("m5", "functional"), required=True)
    run(parser.parse_args())
