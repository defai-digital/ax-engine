"""Replay the isolated sidecar candidate on the original M5 delivery."""
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import runpy
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
STOCK = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122" / "evidence.tar.xz"
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_native_support as native  # noqa: E402

NATIVE_SHA256 = "b583e1b09325eaab74fa4670278aa17693712d914292aaa273f6fd675b38f7b9"
STOCK_SHA256 = "73f4e899a9ec1eea4dd1e0eec8a00c35dd309f663385fc6414333a4b3296b78a"
TRACE_SHA256 = "2167372314e6d5674f5c353021d10abfd810bc04b5c97f39f6e12f2e23204796"
RUNNER_SHA256 = "7546005c63035dcb40f75d01967e5f320403f1a709e63df4e787859d50081c1c"


def replay(observed, stock, traces, build):
    runner = runpy.run_path(str(ROOT / "verify_delivered.py"))
    require, signature, limits = (runner[name] for name in ("require", "signature", "LIMITS"))
    require(observed["revision"] == stock["revision"] == native.PACK_REVISION
            and observed["inventory_sha256"] == stock["inventory_sha256"], "Publisher input changed")
    require(all(observed[field] == stock["native_manifest_ready"] for field in (
        "native_manifest_before", "native_manifest_ready", "native_manifest_after")),
        "Delivered manifest changed")
    require(observed["server_sha256"] == build["server_sha256"]
            and build["commit"] == "838b331a31a9c18bdc9b10c2ac6083c0877c0fad"
            and observed["hardware"] == stock["hardware"]
            and native.is_target_hardware(observed["hardware"]), "Wrong build or hardware")
    require(observed["functional_controls_passed"] is True and observed["qualification"] is False
            and observed["modes"] == stock["modes"] == list(limits), "Unexpected control contract")
    negative = observed["negative_control_result"]
    require(negative["activation_error_observed"] is True and negative["exit_code"] != 0
            and any("MlxMtpRequiredButUnavailable" in line or "no usable drafter was activated" in line
                    for line in traces["required-kill-switch"]), "Missing MTP kill-switch result")
    phases = {}
    verified = 0
    for policy in limits:
        rows = observed["results"][policy]
        require(signature(rows) == signature(stock["results"][policy])
                and len(rows) == len(native.CASES), f"Functional observations changed for {policy}")
        steps = sum(row["activation_delta"][native.VERIFIED] for row in rows)
        require(steps == (107 if policy == "required" else 0), f"Unexpected MTP activity for {policy}")
        verified += steps
        trace = "\n".join(traces[policy])
        require(trace.count("Flash Next MTP sidecar positional read lab") == 1
                and "Flash Next resident positional read lab" not in trace,
                f"Unexpected reader for {policy}")
        phases[policy] = {}
        for phase in ("resident_tensors", "mtp_sidecar"):
            values = re.findall(fr'phase="{phase}" elapsed_us=(\d+)', trace)
            require(len(values) == 1 and int(values[0]) > 0, f"Missing timing for {policy} {phase}")
            phases[policy][phase] = int(values[0]) / 1e6
    ready = observed["readiness_seconds"]
    require(list(ready) == list(limits) and all(math.isfinite(seconds)
            and seconds > sum(phases[policy].values()) for policy, seconds in ready.items()),
            "Invalid HTTP readiness times")
    return {"hardware": observed["hardware"], "revision": observed["revision"], "build": build,
            "native_result_sha256": NATIVE_SHA256, "stock_result_sha256": STOCK_SHA256,
            "matched_functional_controls": sum(len(rows) for rows in observed["results"].values()),
            "verified_required_steps": verified, "readiness_seconds": ready, "limits_seconds": limits,
            "readiness_band_met": all(seconds <= limits[policy] for policy, seconds in ready.items()),
            "phase_seconds": phases}


if __name__ == "__main__":
    observed = gzip.decompress((ROOT / "native-result.json.gz").read_bytes())
    with tarfile.open(STOCK) as archive:
        stock = archive.extractfile("native-support.json").read()
    traces = (ROOT / "startup-traces.json").read_bytes()
    bound_inputs = ((observed, NATIVE_SHA256), (stock, STOCK_SHA256), (traces, TRACE_SHA256),
                    ((ROOT / "verify_delivered.py").read_bytes(), RUNNER_SHA256))
    if any(sha256(payload).hexdigest() != digest for payload, digest in bound_inputs):
        raise ValueError("Original candidate, stock, trace or runner bytes changed")
    comparison = json.loads((ROOT / "comparison.json").read_bytes())
    result = replay(json.loads(observed), json.loads(stock), json.loads(traces), comparison["build"])
    if result != comparison or (ROOT / "status.txt").read_text() != "0\n":
        raise ValueError("Candidate terminal receipt and raw observations disagree")
    print(json.dumps(result, indent=2, allow_nan=False))
