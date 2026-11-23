"""Validate the same-build Flash Next mapped/automatic startup crossover."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from runpy import run_path
import tarfile
from tempfile import TemporaryDirectory

ARMS = ("mapped-1", "auto-1", "auto-2", "mapped-2")
AUTO_FLAG = "AX_MLX_FLASH_NEXT_MTP_SIDECAR_PREFETCH"
PREFETCH_BYTES = re.compile(rb"Flash Next MTP sidecar prefetch completed bytes=5214301696\b")
PINNED_MANIFEST = {"model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"}
PINNED_REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
PRIOR = "flash-next-main-lifecycle-m5-20261123"
TOOLS = "flash-next-probe-followup-lifecycle-m5-20261122"
STAGED_CODE = ("check_flash_next_native_support.py", "compare_startup_arm.py",
               "qualify_qwen38_flash_next.py", "run_flash_next_lifecycle.py",
               "run_guarded.sh", "run_owner.sh")
RAW_INPUTS = ("build.json", "mxfp4-inventory.json", "source.patch")
ARCHIVE_INPUTS = (*RAW_INPUTS, *STAGED_CODE, "inputs.sha256", "capture.json")


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare(capture_root, results_root):
    helpers = run_path(str(results_root / TOOLS / "replay.py"))
    stable = helpers["stable_action"]
    gates = helpers["gates"]
    thresholds = read(gates.THRESHOLDS)
    bound = thresholds["readiness"]["warm_seconds_max"]
    idle_floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    previous = read(results_root / PRIOR / "lifecycle.json")
    prior_default, = (run for run in previous["runs"] if run["mode"] == "default")
    previous_actions = list(map(stable, prior_default["actions"]))
    hardware = dict(previous["hardware"], storage=dict(previous["hardware"]["storage"]))
    del hardware["storage"]["declared"]

    build = read(capture_root / "build.json")
    capture = read(capture_root / "capture.json")
    inventory_path = capture_root / "mxfp4-inventory.json"
    inventory = read(inventory_path)
    patch = capture_root / "source.patch"
    input_lines = (capture_root / "inputs.sha256").read_text().splitlines()
    inputs = dict(line.split()[::-1] for line in input_lines)
    names = {*RAW_INPUTS, *STAGED_CODE, "ax-engine-server"}
    need(len(input_lines) == len(inputs) and set(inputs) == names
         and all(inputs[name] == digest(capture_root / name) for name in (*RAW_INPUTS, *STAGED_CODE)),
         "The sealed staged input roster or checksum changed")
    need(capture["run_status"] == 0 and tuple(capture["order"]) == ARMS
         and capture["codesign_verified"] is build["codesign_verified"] is True
         and build["profile"] == "release-server"
         and build["server_sha256"] == capture["server_sha256"] == inputs["ax-engine-server"]
         and build["source_base"] == read(results_root / PRIOR / "build.json")["commit"]
         and build["patch_sha256"] == inputs["source.patch"] == digest(patch)
         and inputs["build.json"] == digest(capture_root / "build.json")
         and build["mlx"] == build["mlx_metal"] == "0.32.4"
         and build["cargo"].startswith("cargo 1.99.1 ")
         and build["rustc"].startswith("rustc 1.99.1 "),
         "The crossover must bind one verified current-source executable and patch")
    need(inventory["revision"] == PINNED_REVISION == previous["revision"]
         and len(inventory["members"]) == 49
         and inputs["mxfp4-inventory.json"] == digest(inventory_path) == previous["inventory_sha256"],
         "The tested publisher pack has changed")
    need(capture["hardware"] == hardware and capture["final_pressure"] == 1,
         "Expected the same M5 SKU, macOS version and normal final pressure")

    result = {}
    for arm in ARMS:
        arm_root = capture_root / "arms" / arm
        row = read(arm_root / "result.json")
        automatic = arm.startswith("auto-")
        need(row["arm"] == arm and row["automatic_prefetch_observed"] is automatic
             and row["override"] == ({} if automatic else {AUTO_FLAG: "0"}),
             f"Unexpected reader selection in {arm}")
        need(row["manifest_before"] == row["manifest_after"] == PINNED_MANIFEST
             and row["publisher_members"] == 49
             and row["server_sha256"] == build["server_sha256"]
             and row["inventory_sha256"] == inputs["mxfp4-inventory.json"],
             f"Pack or executable changed within {arm}")
        admission = capture["prior_reference_admission"][arm]
        need(admission == {"pressure": 1, "eligible_for_attempt": False,
                           "reason": "background-rss-exceeds-signed-margin"}
             or admission == {"previous_reference_terminal": True},
             f"Independent reference had priority in {arm}")
        run = row["run"]
        need(run["mode"] == "default" and run["final_verified_steps"] == 0
             and run["exit_code_after_stop"] == 0
             and list(map(stable, run["actions"])) == previous_actions,
             f"Stable lifecycle changed in {arm}")
        log = (arm_root / "server-default.log").read_bytes()
        need(b"decision=resident" in log and b"sidecar prefetch failed" not in log
             and bool(PREFETCH_BYTES.search(log)) is automatic,
             f"Startup log contradicts the tested path in {arm}")
        ready = row["ready_seconds"]
        need(type(ready) in (int, float) and isfinite(ready) and 0 < ready == run["ready_seconds"],
             f"Malformed process-to-HTTP readiness in {arm}")
        idle = [row["cpu_idle_before"], row["cpu_idle_after"]]
        need(all(type(sample) in (int, float) and isfinite(sample) and 0 <= sample <= 100 for sample in idle),
             f"Invalid host samples in {arm}")
        timing = helpers["measure"](helpers["extract"](log)["timestamps"], ready, automatic)
        result[arm] = {"startup": timing, "within_staged_readiness_bound": ready <= bound,
                       "cpu_idle_percent": idle, "timing_eligible": min(idle) >= idle_floor,
                       "matching_actions": len(run["actions"])}
    pairs = []
    for mapped, auto in ((ARMS[0], ARMS[1]), (ARMS[3], ARMS[2])):
        old, new = result[mapped], result[auto]
        eligible = old["timing_eligible"] and new["timing_eligible"]
        change = (new["startup"]["http_ready_seconds"] - old["startup"]["http_ready_seconds"])
        pairs.append({"mapped": mapped, "automatic": auto, "timing_eligible": eligible,
                      "automatic_minus_mapped_seconds": round(change, 3) if eligible else None})
    return {"server_sha256": build["server_sha256"], "source_base": build["source_base"],
            "patch_sha256": build["patch_sha256"], "arms": result, "paired_timing": pairs,
            "automatic_readiness_passed": all(result[arm]["timing_eligible"]
                and result[arm]["within_staged_readiness_bound"] for arm in ("auto-1", "auto-2")),
            "crossover_timing_eligible": all(pair["timing_eligible"] for pair in pairs),
            "release_ready": False}


def replay():
    root = Path(__file__).resolve().parent
    arm_paths = tuple(f"arms/{arm}/{name}" for arm in ARMS
                      for name in ("result.json", "server-default.log"))
    expected = {*ARCHIVE_INPUTS, *arm_paths}
    with TemporaryDirectory(prefix="flash-next-reader-replay-") as work:
        capture_root = Path(work)
        with tarfile.open(root / "evidence.tar.xz") as archive:
            rows = archive.getmembers()
            need(len(rows) == len(expected) and {row.name for row in rows} == expected
                 and all(row.isfile() for row in rows), "Unexpected evidence archive roster")
            for row in rows:
                output = capture_root / row.name
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(row) as stream:
                    output.write_bytes(stream.read())
        return compare(capture_root, root.parent)


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
