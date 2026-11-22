"""Run two real Flash Next cold native sessions in one shared context."""
from argparse import ArgumentParser
import json
from pathlib import Path
import platform
import subprocess
import time

import check_flash_next_native_support as native

TEST = "runner::flash_next_tests::flash_next_real_concurrent_cold_runners_share_weights"
CONFIRMED = "Two cold Flash Next runners share weights and match distinct solo trajectories at the same request ID."


def query(name):
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = Path("ax-flash-shared-load-test").resolve()
    build = json.loads(Path("test-build.json").read_bytes())
    inventory = json.loads(Path("inventory.json").read_bytes())
    if native.sha256(executable) != build["binary_sha256"] or len(inventory["members"]) != 49:
        raise ValueError("The test build or package inventory changed")
    files = native.validate_inventory(args.pack, inventory)
    if files["model-manifest.json"] != "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef":
        raise ValueError("The real-model loader received a different manifest")
    args.output.mkdir()
    output = args.output / "test-output.txt"
    samples = []
    began = time.monotonic()
    clean_env = native.server_env({"AX_FLASH_NEXT_REAL_PACK": str(args.pack)})
    try:
        with output.open("x") as log:
            process = subprocess.Popen([executable, "--ignored", "--exact", "--nocapture", TEST],
                                       env=clean_env, stdout=log, stderr=log, start_new_session=True)
            try:
                while process.poll() is None:
                    level = int(query("kern.memorystatus_vm_pressure_level"))
                    samples.append(level)
                    if level != 1:
                        raise ValueError(f"Shared startup reached memory pressure level {level}")
                    if time.monotonic() - began > 1200:
                        raise TimeoutError("Shared startup and generation exceeded the test timeout")
                    time.sleep(3)
            finally:
                native.stop_owned(process)
        text = output.read_text(errors="replace")
        if process.returncode != 0 or CONFIRMED not in text or "1 passed; 0 failed" not in text:
            raise ValueError("Native shared weight reuse or token identity failed")
    finally:
        log = output.read_text(errors="replace").replace(str(args.pack), "[model]").replace(str(Path.home()), "[home]")
        output.write_text(log)

    if native.validate_inventory(args.pack, inventory) != files:
        raise ValueError("The publisher artifacts changed during the test")

    receipt = {
        "scope": "native functional session isolation with shared cold weights",
        "qualification": False, "pack_revision": inventory["revision"], "manifest": files,
        "inventory_sha256": native.sha256(Path("inventory.json")),
        "build": build,
        "hardware": {"soc": query("machdep.cpu.brand_string"), "memory_bytes": int(query("hw.memsize")),
                     "os_version": platform.mac_ver()[0]},
        "test": TEST,
        "weight_allocations": 1, "concurrent_runners": 2, "solo_runners": 2,
        "shared_request_id": 1, "matched_solo_ids_per_runner": 8,
        "pressure_samples": samples, "result": process.returncode,
        "test_output_sha256": native.sha256(output),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
