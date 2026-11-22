"""Evaluate the current-native M5 lifecycle phase on the production build."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122"
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_release_gates as gates  # noqa: E402

MODES = ("disabled", "default", "required")


def replay(root=ROOT):
    raw = (root / "lifecycle.json").read_bytes()
    case = json.loads(raw)
    previous = json.loads((DELIVERY / "comparison.json").read_bytes())
    with tarfile.open(DELIVERY / "evidence.tar.xz") as archive:
        binary_receipt = archive.extractfile("build.json").read()
        inventory_receipt = archive.extractfile("native-support.json").read()
    assert sha256(binary_receipt).hexdigest() == previous["build_sha256"]
    assert sha256(inventory_receipt).hexdigest() == previous["native_support_sha256"]
    build, delivered = map(json.loads, (binary_receipt, inventory_receipt))
    assert case["server_sha256"] == delivered["server_sha256"] == build["binaries_sha256"]["ax-engine-server"]
    for field in ("inventory_sha256", "revision", "hardware"):
        assert case[field] == delivered[field]
    thresholds_raw = gates.THRESHOLDS.read_bytes()
    assert sha256(thresholds_raw).hexdigest() == previous["thresholds_sha256"]
    thresholds = json.loads(thresholds_raw)
    assert case["modes"] == list(MODES) and tuple(run["mode"] for run in case["runs"]) == MODES
    assert case["lifecycle_passed"] is True
    assert case["qualification"] is False and case["release_ready"] is False
    official, = gates.gate_lifecycle(case, thresholds)
    assert official["passed"]
    observed = {}
    for run in case["runs"]:
        assert tuple(action["action"] for action in run["actions"]) == gates.LIFECYCLE_ACTIONS
        ready = run["ready_seconds"]
        assert isfinite(ready) and ready > 0 and run["exit_code_after_stop"] == 0
        for action in run["actions"]:
            assert action["mode"] == run["mode"]
            assert gates.LIFECYCLE_COUNTERS <= set(action["drained"])
            assert all(type(value) in (float, int) and value == 0 for value in action["drained"].values())
        observed[run["mode"]] = {"actions_passed": len(run["actions"]), "ready_seconds": ready}
    samples = [case["host_quiet"]["cpu_idle_percent"], case["host_idle_after"]]
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    assert all(floor <= value <= 100 for value in samples)
    return {"scope": "Fresh production binary lifecycle phase", "release_ready": False,
            "lifecycle_gate": official, "matched_server_sha256": case["server_sha256"],
            "receipt_sha256": sha256(raw).hexdigest(),
            "timing_cpu_idle_percent": samples, "modes": observed}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
