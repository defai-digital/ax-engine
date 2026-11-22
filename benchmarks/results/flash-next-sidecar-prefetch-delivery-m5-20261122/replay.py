"""Replay the M5 delivered-copy sidecar pre-read diagnostic."""
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
FRESH = ROOT.parent / "flash-next-fresh-delivery-38c90b27a-m5-20261122"
M2 = ROOT.parent / "flash-next-native-sidecar-prefetch-m2-20261122"
MODES = ("disabled", "required", "default")
LIMITS = dict(zip(MODES, (300, 120, 120), strict=True))
BYTES = 5_214_301_696


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, name):
    with tarfile.open(path) as data:
        member = data.getmember(name)
        require(member.isfile(), f"Invalid evidence member {name}")
        with data.extractfile(member) as source:
            return source.read()


def stock_replay():
    spec = importlib.util.spec_from_file_location("stock_delivery_replay", FRESH / "replay.py")
    require(spec and spec.loader, "Missing the frozen baseline replay")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(bundle=ROOT / "evidence.tar.xz"):
    def load(file):
        return read(bundle, file)

    checksums = {name: sha for sha, name in map(str.split, load("delivery-inputs.sha256").decode().splitlines())}
    build_raw = load("build.json")
    build = json.loads(build_raw)
    earlier = read(M2 / "evidence.tar.xz", "build.json")
    source = read(M2 / "evidence.tar.xz", "candidate.patch")
    require(build_raw == earlier and build["codesign_verified"] is True
            and digest(build_raw) == checksums["build.json"]
            and digest(source) == build["runtime_diff_sha256"]
            and build["server_sha256"] == checksums["ax-engine-server-prefetch"]
            and build["driver_sha256"] == checksums["run_prefetch.py"]
            and all(digest(load(name)) == checksums[name] for name in (
                "verify_prefetch_delivered.py", "mxfp4-inventory.json"))
            and all(digest(read(M2 / "evidence.tar.xz", name)) == checksums[name]
                    for name in ("check_flash_next_native_support.py", "run_prefetch.py")),
            "Candidate executable, exact patch or driver lineage changed")
    stock_raw = read(FRESH / "evidence.tar.xz", "native-support.json")
    current_raw = load("native-result.json")
    stock, current, sampled = (json.loads(raw) for raw in (stock_raw, current_raw, load("native-comparison.json")))
    require(load("status.txt") == b"0\n" and sampled["build"] == build
            and sampled["stock_result_sha256"] == digest(stock_raw) == checksums["stock-native.json"]
            and sampled["native_result_sha256"] == digest(current_raw)
            and sampled["hardware"] == current["hardware"] == stock["hardware"]
            and current["server_sha256"] == build["server_sha256"]
            and sampled["revision"] == current["revision"] == stock["revision"]
            and stock["inventory_sha256"] == current["inventory_sha256"] == checksums["mxfp4-inventory.json"]
            and all(stock[f"native_manifest_{state}"] == current[f"native_manifest_{state}"]
                    for state in ("before", "ready", "after")),
            "Primary host, delivered copy or native result binding changed")
    original = stock_replay()
    activations = original.check_cases(current, stock, list(MODES))
    require(activations == {"disabled": 0, "required": 107, "default": 0}, "Mode activation changed")
    logs = {mode: load(f"server-{mode}.log").decode() for mode in MODES}
    kill = load("server-required-kill-switch.log").decode()
    require("prefetch completed" not in kill and "prefetch failed" not in kill
            and any(word in kill for word in ("no usable drafter was activated", "MlxMtpRequiredButUnavailable")),
            "Required kill-switch protection changed")
    phases = {}
    for mode, log in logs.items():
        require("decision=resident" in log and "Flash Next MTP sidecar prefetch failed" not in log
                and "Flash Next resident positional read lab" not in log,
                "The expected mapped resident path changed")
        reads = re.findall(r"Flash Next MTP sidecar prefetch completed bytes=(\d+) elapsed_ms=([\d.]+)", log)
        require(len(reads) == 1 and int(reads[0][0]) == BYTES and 0 < float(reads[0][1]) < float("inf"),
                "The admitted pre-read extent changed")
        values = {"sidecar_prefetch": float(reads[0][1]) / 1000}
        for phase in ("resident_tensors", "mtp_sidecar"):
            samples = re.findall(fr'phase="{phase}" elapsed_us=(\d+)', log)
            require(len(samples) == 1 and int(samples[0]) > 0, f"Ambiguous {mode} {phase} measurement")
            values[phase] = int(samples[0]) / 1e6
        phases[mode] = values
    candidate = current["readiness_seconds"]
    historical = stock["readiness_seconds"]
    host = sampled["host_samples_by_mode"]
    require(set(candidate) == set(historical) == set(host) == set(MODES)
            and all(isfinite(value) and value > 0 for ready in (candidate, historical) for value in ready.values()),
            "The three startup modes have incomplete timings")
    require(all(len(sample["cpu_idle_percent"]) == len(sample["pressure"]) == 2
                and all(isfinite(x) and 0 <= x <= 100 for x in sample["cpu_idle_percent"])
                for sample in host.values()), "Host timing observations are incomplete")
    eligible = [mode for mode in MODES if host[mode]["pressure"] == [1, 1]
                and min(host[mode]["cpu_idle_percent"]) >= 90]
    within = all(candidate[mode] <= LIMITS[mode] for mode in MODES)
    require(sampled["phase_seconds"] == phases and sampled["readiness_seconds"] == candidate
            and sampled["limits_seconds"] == LIMITS and sampled["readiness_band_met"] is within
            and sampled["matched_functional_controls"] == sum(len(rows) for rows in current["results"].values()) == 21
            and sampled["verified_required_steps"] == activations["required"]
            and sampled["timing_eligible_modes"] == eligible, "Published startup summary changed")
    return {"qualification": False, "historical_control_commit": "38c90b27a", "build": build,
            "hardware": current["hardware"], "revision": current["revision"], "matched_functional_controls": 21,
            "verified_steps": activations, "historical_readiness_seconds": historical,
            "candidate_readiness_seconds": candidate, "limits_seconds": LIMITS, "readiness_band_met": within,
            "timing_eligible_modes": eligible, "host_samples_by_mode": host, "phase_seconds": phases}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
