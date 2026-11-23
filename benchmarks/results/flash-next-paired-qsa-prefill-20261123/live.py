"""Run and verify the signed native paired QSA prefill test."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Thread

import check_flash_next_native_support as native
from run_qwen38_reference_serving import monitor_memory

ROOT = Path(__file__).resolve().parent
TEST = "model::qwen4_exp::profiling::flash_next_paired_gather_prefill_exactness_and_timing"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "link-root", "out"):
        cli.add_argument(f"--{name}", type=Path, required=name != "link-root")
    args = cli.parse_args()
    inventory = json.loads((ROOT / "mxfp4-inventory.json").read_bytes())
    assert len(inventory["members"]) == 49 and version("mlx") == version("mlx-metal") == "0.32.4"
    before = native.validate_inventory(args.pack, inventory, args.link_root)
    build = json.loads((ROOT / "build.json").read_bytes())
    assert before["model-manifest.json"] == build["manifest_sha256"]
    binary = ROOT / "ax-flash-paired-qsa-test"
    assert native.sha256(binary) == build["binary_sha256"]
    assert native.sha256(ROOT / "prompt.json") == build["prompt_sha256"]
    assert native.sha256(ROOT / "source.patch") == build["patch_sha256"]
    args.out.mkdir()
    idle = [native.wait_for_quiet()["cpu_idle_percent"]]
    environment = os.environ | {
        "AX_MLX_NATIVE_CONFIRM": "1",
        "AX_FLASH_NEXT_PROFILE_PACK": str(args.pack),
        "AX_FLASH_NEXT_PROMPT_IDS_FILE": str(ROOT / "prompt.json"),
        "AX_FLASH_NEXT_PROFILE_OUTPUT": str(args.out / "result.json"),
    }
    with (args.out / "native.log").open("w") as log:
        child = subprocess.Popen([binary, "--ignored", "--exact", TEST, "--nocapture"],
                                 env=environment, stdout=log, stderr=log)
        stop = Event()
        monitor = Thread(target=monitor_memory, args=(args.out, "native", child, stop), daemon=True)
        monitor.start()
        try:
            code = child.wait(600)
            assert code == 0, code
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(15)
            stop.set()
            monitor.join()
    idle.append(native.cpu_idle_percent())
    assert "1 passed" in (args.out / "native.log").read_text()
    result = json.loads((args.out / "result.json").read_bytes())
    assert result["final_state_and_logits_exact"] and not result["qualification"]
    assert [row["paired"] for row in result["arms"]] == [True, False, False, True]
    assert native.validate_inventory(args.pack, inventory, args.link_root) == before
    pressure = [json.loads(line) for line in (args.out / "native-samples.jsonl").read_text().splitlines()]
    assert pressure and all(row["pressure"] == 1 for row in pressure)
    receipt = {
        "qualification": False, "build": build, "manifest": before,
        "cpu_idle_percent": idle, "storage": native.storage_info(args.pack),
        "soc": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
        "pressure_samples": len(pressure),
        "result_sha256": native.sha256(args.out / "result.json"),
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
