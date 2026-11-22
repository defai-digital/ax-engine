"""Replay the instrumented Flash Next startup with its manifest rejection."""
import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "flash-next-mxfp4-startup-phases-m5-20261121"))
from capture_phases import read_log  # noqa: E402

MODES = ("disabled", "required", "default")
BINARY_SHA = "932ff3f0fc69cef65ee0aa2fba90c095aa23e3be242a1c024921b2d58ee025c7"
INPUTS_SHA = "bf167dd6dea380d8636c3175ea0dff6756fe475bb84e3cb4d1bd6d3e69ae5011"
PRIOR_SHA = "13d16180526c1c753d6049b5fc0277fb4dd8d7d8f47444fdf50361ddf4bc9031"
FROZEN_MANIFEST_SHA = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
OBSERVED_MANIFEST_SHA = "5bb769d05077134e0cf3c612061e35bb9930c3904dbc2cb30ede02d27afc23b5"
VERIFIED = "ax_engine_flash_next_mtp_verified_steps_total"


def read(archive, name):
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"Missing evidence {name}")
    return stream.read()


def choices(receipt):
    return {mode: [(row["name"], row["response"]["choices"], row["activation_delta"])
                   for row in receipt["results"][mode]] for mode in MODES}


def startup(log):
    phases, shards = read_log(log)
    selection = [[row["name"], row["tensors"]] for row in shards]
    assert len(selection) == 15 and sum(count for _, count in selection) == 1942
    copies = sum(row["read_us"] for row in shards)
    evaluation = sum(row["eval_us"] for row in shards)
    assert copies + evaluation + phases["resident_selection"] <= phases["resident_tensors"]
    assert "read-ahead lab" not in log
    return selection, {
        "resident_tensors_seconds": phases["resident_tensors"] / 1e6,
        "resident_shard_copy_allocation_seconds": copies / 1e6,
        "resident_shard_sync_eval_seconds": evaluation / 1e6,
        "mtp_sidecar_seconds": phases["mtp_sidecar"] / 1e6,
        "trunk_admission_seconds": phases["trunk_admission"] / 1e6,
        "slowest_shard": max(shards, key=lambda row: row["read_us"]),
    }


def replay(path=ROOT / "evidence.tar.xz"):
    with tarfile.open(path) as archive:
        return audit(archive)


def audit(archive):
    assert read(archive, "phase-status.txt") == b"delivery-trace 1\n"
    assert read(archive, "terminal-error.txt") == b"ValueError: Frozen output, activation, or manifest changed\n"
    assert sha256(read(archive, "inputs.sha256")).hexdigest() == INPUTS_SHA
    prior_bytes = gzip.decompress(read(archive, "prior-native-support.json.gz"))
    prior = json.loads(prior_bytes)
    assert sha256(prior_bytes).hexdigest() == PRIOR_SHA
    assert prior["readiness_seconds"]["default"] == 157.292
    observed_bytes = read(archive, "native-support/result.json")
    observed = json.loads(observed_bytes)
    assert observed["server_sha256"] == BINARY_SHA
    assert observed["inventory_sha256"] == prior["inventory_sha256"]
    assert observed["native_manifest_before"] == observed["native_manifest_ready"] == observed["native_manifest_after"]
    frozen = read(archive, "frozen-manifest.json")
    actual = read(archive, "observed-manifest.json")
    assert sha256(frozen).hexdigest() == prior["native_manifest_ready"]["model-manifest.json"] == FROZEN_MANIFEST_SHA
    assert sha256(actual).hexdigest() == observed["native_manifest_ready"]["model-manifest.json"] == OBSERVED_MANIFEST_SHA
    original, other = json.loads(frozen), json.loads(actual)
    manifest_fields_changed = [key for key in sorted(original.keys() | other.keys())
                               if original.get(key) != other.get(key)]
    assert manifest_fields_changed == ["dropped_tensors"]
    assert observed["functional_controls_passed"] is True and observed["negative_control_result"]["activation_error_observed"] is True
    assert observed["qualification"] is False and observed["release_ready"] is False
    assert observed["modes"] == prior["modes"] == list(MODES)
    assert tuple(observed["results"]) == tuple(observed["readiness_seconds"]) == MODES
    assert choices(observed) == choices(prior) and all(len(observed["results"][mode]) == 7 for mode in MODES)
    assert observed["hardware"] == prior["hardware"]
    preflight = json.loads(read(archive, "preflight.json"))
    assert preflight["initial"]["pressure"] == 1
    assert 90 <= preflight["quiet"]["cpu_idle_percent"] <= 100
    sampled = [json.loads(line) for line in read(archive, "samples.jsonl").splitlines()]
    assert sampled
    times = [row["monotonic_seconds"] for row in sampled]
    assert all(isfinite(x) for x in times) and all(a < b for a, b in zip(times, times[1:]))
    assert all(row["page_size"] == 16384 for row in sampled)

    starts = {}
    sources = json.loads(read(archive, "portable-log-sources.json"))
    resident_selection = None
    for mode in MODES:
        name = f"native-support/server-{mode}.log"
        raw_log = read(archive, name)
        assert sha256(raw_log).hexdigest() == sources[name]["portable_sha256"]
        selection, row = startup(raw_log.decode())
        if resident_selection is None:
            resident_selection = selection
        assert selection == resident_selection
        ready = observed["readiness_seconds"][mode]
        assert isfinite(ready) and 0 < row["resident_tensors_seconds"] <= ready < 900
        verification = sum(case["activation_delta"][VERIFIED] for case in observed["results"][mode])
        assert verification == (107 if mode == "required" else 0)
        starts[mode] = {**row, "http_ready_seconds": ready, "verified_steps": verification}
    name = "native-support/server-required-kill-switch.log"
    kill_switch = read(archive, name)
    assert sha256(kill_switch).hexdigest() == sources[name]["portable_sha256"]
    assert b"no usable drafter was activated" in kill_switch or b"MlxMtpRequiredButUnavailable" in kill_switch
    return {"scope": "Instrumented startup with a different derived manifest", "qualification": False,
            "matching_http_controls": 21, "terminal_status": "manifest_mismatch",
            "manifest_fields_changed": manifest_fields_changed,
            "frozen_manifest_sha256": FROZEN_MANIFEST_SHA, "observed_manifest_sha256": OBSERVED_MANIFEST_SHA,
            "reference_readiness_seconds": prior["readiness_seconds"],
            "server_sha256": BINARY_SHA, "trace_result_sha256": sha256(observed_bytes).hexdigest(),
            "cpu_idle_percent_before": preflight["quiet"]["cpu_idle_percent"],
            "pressure_levels": sorted({row["pressure"] for row in sampled}), "pressure_samples": len(sampled),
            "starts": starts}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
