"""Compare one and 32 native PLE read threads on exactly the same rows."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import platform
import subprocess

import check_flash_next_native_support as native

MARKER = "PLE_READ_PROBE "
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
ARMS = (1, 32, 32, 1)


def pressure():
    return int(subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True))


def run(args):
    binary, inventory, workload = args.binary.resolve(), args.inventory.resolve(), args.workload.resolve()
    build = json.loads(args.build.read_bytes())
    if native.sha256(binary) != build["binary_sha256"]:
        raise ValueError("The staged Rust test executable changed")
    data = json.loads(inventory.read_bytes())
    if native.sha256(inventory) != "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6":
        raise ValueError("The pinned checkpoint inventory changed")
    root = args.pack.resolve()
    observed_manifest = native.validate_inventory(root, data)
    if observed_manifest != {"model-manifest.json": MANIFEST}:
        raise ValueError("The pinned native manifest changed")
    record = {"scope": args.scope, "qualification": False, "pack_revision": data["revision"],
              "build": build, "manifest": observed_manifest, "inventory_sha256": native.sha256(inventory),
              "soc": subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                             text=True).strip(),
              "memory_bytes": int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                                                          text=True)),
              "os": platform.mac_ver()[0], "arms": []}
    argv = [str(binary), "weights::qwen4_exp::ple_reader_probe::flash_next_real_ple_reader_dispatch",
            "--exact", "--ignored", "--nocapture"]
    if args.scope == "m5-diagnostic":
        record["pre_cpu_idle_percent"] = native.wait_for_quiet()["cpu_idle_percent"]
    central = None
    for index, threads in enumerate(ARMS, 1):
        before = pressure()
        if before != 1:
            raise ValueError("Normal memory pressure is required")
        env = dict(os.environ, AX_FLASH_NEXT_REAL_PACK=str(root), AX_FLASH_NEXT_PLE_WORKLOAD=str(workload),
                   AX_FLASH_NEXT_NGRAM_READ_THREADS=str(threads))
        child = subprocess.run(argv, text=True, capture_output=True, check=False, env=env, timeout=180)
        raw = child.stdout + child.stderr
        (args.output.parent / f"arm-{index}.txt").write_text(raw)
        rows = [json.loads(line[len(MARKER):]) for line in child.stdout.splitlines() if line.startswith(MARKER)]
        if child.returncode != 0 or len(rows) != 1 or "1 passed; 0 failed" not in child.stdout:
            raise RuntimeError(f"Native stage arm {index} exited with {child.returncode}")
        measured, = rows
        after = pressure()
        if (measured["reader_threads"] != threads or after != 1
                or measured["workload_sha256"] != native.sha256(workload)):
            raise ValueError("Native stage arm did not exercise its bound inputs under normal pressure")
        identity = [{key: val for key, val in stage.items() if key != "elapsed_ns"}
                    for stage in measured["stages"]]
        if central is None:
            central = identity
        if central != identity:
            raise ValueError("Staged bytes, reading count or address order differ between thread counts")
        record["arms"].append(measured | {"pressure_before": before, "pressure_after": after})
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print("Completed native PLE arm", index, "with", threads, "configured read threads.", flush=True)
    if args.scope == "m5-diagnostic":
        record["post_cpu_idle_percent"] = native.cpu_idle_percent()
    record["completed"] = True
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("binary", "workload", "inventory", "pack", "build", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--scope", choices=("functional", "m5-diagnostic"), required=True)
    run(parser.parse_args())
