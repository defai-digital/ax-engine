"""Check the M5 isolated sidecar reader against the original native controls."""
from gzip import decompress
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_native_support as native  # noqa: E402

DELIVERY = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122" / "evidence.tar.xz"
M2 = ROOT.parent / "flash-next-owned-positional-native-sidecar-m2-20261122" / "evidence.tar.xz"
ORDER = (("required", "mmap"), ("required", "pread"),
         ("default", "pread"), ("default", "mmap"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def archived(archive, key):
    with tarfile.open(archive) as source:
        item = source.extractfile(key)
        if item is None:
            raise ValueError(f"Missing {key} in {archive.name}")
        return item.read()


def signature(row):
    return row["name"], row["response"]["choices"], row["activation_delta"]


def replay(receipt):
    require(receipt["completed"] is True and receipt["qualification"] is False,
            "Missing completed diagnostic status")
    stock_raw = archived(DELIVERY, "native-support.json")
    stock = json.loads(stock_raw)
    require(sha256(stock_raw).hexdigest() == receipt["reference_control_sha256"]
            and stock["functional_controls_passed"] is True,
            "Changed production control")
    require(stock["hardware"] == receipt["hardware"] and native.is_target_hardware(stock["hardware"])
            and stock["native_manifest_ready"] == receipt["native_manifest_sha256"]
            and stock["inventory_sha256"] == receipt["inventory_sha256"]
            and stock["revision"] == receipt["revision"], "Changed pack or hardware")
    m2 = json.loads(archived(M2, "receipt.json"))
    require(receipt["build"] == m2["build"] and m2["verified_steps"] == 107
            and m2["kill_switch"]["activation_error_observed"] is True,
            "Sidecar source does not match the tested build")
    require(tuple(map(tuple, receipt["order"])) == ORDER, "Changed arm order")
    data = {}
    for arm, (policy, reader) in zip(receipt["arms"], ORDER, strict=True):
        label = f"{policy}-{reader}"
        require((arm["label"], arm["policy"], arm["reader"]) == (label, policy, reader),
                f"Invalid reader or policy for {label}")
        require([signature(row) for row in arm["cases"]]
                == [signature(row) for row in stock["results"][policy]],
                f"Exact native output or activation changed for {label}")
        verified = 0
        for row in arm["cases"]:
            native.check_answer(row["name"], row["response"])
            count = row["response"]["usage"]["completion_tokens"]
            activity = native.check_activation(row["metrics_before"], row["metrics_after"],
                                               policy == "required", count, row["request"]["max_tokens"])
            require(activity == row["activation_delta"], f"Invalid activations for {label}")
            verified += activity[native.VERIFIED]
        require(verified == (107 if policy == "required" else 0), f"Unexpected verifier steps for {label}")
        require(arm["initial_sidecar_file_pages"] == {"selected_pages": 318256, "resident_pages": 0}
                and arm["pressure_after"] == 1, f"Page or pressure control failed for {label}")
        times = {key: arm[key] for key in ("ready_seconds", "resident_seconds", "sidecar_seconds")}
        require(all(math.isfinite(t) and t > 0 for t in times.values())
                and times["ready_seconds"] > times["resident_seconds"] + times["sidecar_seconds"],
                f"Invalid startup timings for {label}")
        idle = [arm["cpu_idle_before_percent"], arm["cpu_idle_after_percent"]]
        require(all(0 <= value <= 100 for value in idle), f"Invalid host idle for {label}")
        data[label] = {**times, "timing_cpu_idle_percent": idle, "controls": len(arm["cases"]),
                       "verified_steps": verified}
    timing = all(min(arm["timing_cpu_idle_percent"]) >= 90 for arm in data.values())
    require(timing is receipt["timing_eligible"], "Timing eligibility disagrees with samples")
    return {"scope": "Isolated MTP sidecar reader diagnostic", "qualification": False,
            "timing_eligible": timing, "arms": data}


if __name__ == "__main__":
    require((ROOT / "phase-status.txt").read_bytes() == b"m5-sidecar-positional 0\n",
            "Sidecar observer did not complete")
    raw = decompress((ROOT / "receipt.json.gz").read_bytes())
    require(sha256(raw).hexdigest() == (ROOT / "receipt.sha256").read_text().strip(),
            "Sidecar receipt hash changed")
    print(json.dumps(replay(json.loads(raw)), indent=2, sort_keys=True, allow_nan=False))
