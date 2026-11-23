"""Replay native Flash Next lifecycle and bind it to fresh M5 delivery."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
DELIVERY = RESULTS / "flash-next-auto-reader-fresh-m5-20261122" / "evidence.tar.xz"
PREVIOUS = RESULTS / "flash-next-current-lifecycle-m5-20261122" / "lifecycle.json"
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_release_gates as gates

sys.path.insert(0, str(RESULTS / "flash-next-sidecar-fresh-delivery-pair-m5-20261122"))
from startup_timeline import extract, measure


def need(condition, error):
    if not condition:
        raise ValueError(error)


def stable_action(row):
    value = dict(row)
    if "usage" in value and "ax_engine" in value["usage"]:
        usage = dict(value["usage"])
        usage["ax_engine"] = {key: val for key, val in usage["ax_engine"].items()
                              if key not in ("prefill_time_us", "decode_time_us")}
        value["usage"] = usage
    return value


def replay(path=ROOT / "lifecycle.json"):
    raw = path.read_bytes()
    observed = json.loads(raw)
    previous_raw = PREVIOUS.read_bytes()
    earlier = {case["mode"]: case for case in json.loads(previous_raw)["runs"]}
    with tarfile.open(DELIVERY) as archive:
        build = json.load(archive.extractfile("raw/build.json"))
        delivered = json.load(archive.extractfile("raw/delivery/result.json"))
        controls = archive.extractfile("raw/delivery/native-support/result.json").read()
        need(not gates.delivery_control_problems(delivered, controls), "Mismatched raw delivery support")
    need(build["codesign_verified"] is True
         and observed["server_sha256"] == delivered["server_sha256"] == build["server_sha256"]
         and all(observed[key] == delivered[key] for key in ("inventory_sha256", "revision", "hardware")),
         "Lifecycle receipt changed build, pack, or host identity")
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    modes = thresholds["lifecycle"]["modes_required"]
    need(observed["modes"] == modes == [run["mode"] for run in observed["runs"]]
         and observed["qualification"] is False and observed["release_ready"] is False,
         "Unexpected lifecycle execution profile")
    gate, = gates.gate_lifecycle(observed, thresholds)
    need(gate["passed"], gate["detail"])
    mode_results = {}
    for run in observed["runs"]:
        mode = run["mode"]
        rows = run["actions"]
        ready = run["ready_seconds"]
        need(isfinite(ready) and ready > 0
             and list(map(stable_action, rows)) == list(map(stable_action, earlier[mode]["actions"])),
             f"Changed HTTP lifecycle behavior in {mode}")
        mode_results[mode] = {"actions_matching": len(rows), "ready_seconds": ready,
                              "all_five_generation_counters_drained": True}
    idle = [observed["host_quiet"]["cpu_idle_percent"], observed["host_idle_after"]]
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    with tarfile.open(ROOT / "startup-logs.tar.xz") as archive:
        names = [f"server-{mode}.log" for mode in modes]
        need(archive.getnames() == names and all(row.isfile() for row in archive),
             "Startup log roster changed")
        intervals = {}
        for mode, name in zip(modes, names, strict=True):
            events = extract(archive.extractfile(name).read())
            intervals[mode] = {**events, **measure(events["timestamps"],
                                                    mode_results[mode]["ready_seconds"], True)}
    return {"scope": "Current-build native lifecycle on M5", "release_ready": False,
            "runtime_commit": build["commit"], "server_sha256": observed["server_sha256"],
            "lifecycle_gate": gate, "runs": mode_results, "startup_intervals": intervals,
            "timing_cpu_idle_percent": idle, "timing_eligible": all(floor <= sample <= 100 for sample in idle),
            "receipt_sha256": sha256(raw).hexdigest(), "prior_lifecycle_sha256": sha256(previous_raw).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
