"""Check the M5 startup repeat against its frozen delivery manifest and controls."""
import gzip
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import isfinite
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
FAILED_TRACE = ROOT.parent / "flash-next-delivered-startup-phases-m5-20261122"
INPUTS_SHA = "04236473a967d841494af33f8f2c07065666287c46e86401ff2bf8c9126e058b"
RELEASE_LIMIT_SECONDS = 120


def load_startup_trace():
    spec = spec_from_file_location("flash_next_startup_trace", FAILED_TRACE / "replay.py")
    if spec is None or spec.loader is None:
        raise ValueError("Missing startup trace verifier")
    verifier = module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def replay(path=ROOT / "evidence.tar.xz"):
    startup = load_startup_trace()
    with tarfile.open(path) as archive:
        return audit(startup, archive)


def audit(startup, archive):
    def read(name):
        return startup.read(archive, name)

    assert read("phase-status.txt") == b"delivery-trace 0\n"
    assert sha256(read("inputs.sha256")).hexdigest() == INPUTS_SHA
    prior_raw = gzip.decompress(read("prior-native-support.json.gz"))
    result_raw = read("native-support/result.json")
    manifest = sha256(read("frozen-manifest.json")).hexdigest()
    prior, result, terminal = map(json.loads, (prior_raw, result_raw, read("diagnostic.json")))
    assert sha256(prior_raw).hexdigest() == terminal["original_control_sha256"] == startup.PRIOR_SHA
    assert sha256(result_raw).hexdigest() == terminal["result_sha256"]
    assert prior["readiness_seconds"]["default"] == 157.292
    assert result["server_sha256"] == terminal["server_sha256"] == startup.BINARY_SHA
    assert result["inventory_sha256"] == prior["inventory_sha256"]
    assert result["hardware"] == prior["hardware"]
    for stage in ("before", "ready", "after"):
        assert result[f"native_manifest_{stage}"] == prior[f"native_manifest_{stage}"] == {"model-manifest.json": manifest}
    assert manifest == startup.FROZEN_MANIFEST_SHA
    assert result["functional_controls_passed"] is True and terminal["negative_control_passed"] is True
    assert result["negative_control_result"]["activation_error_observed"] is True
    assert result["qualification"] is False and result["release_ready"] is False and terminal["qualification"] is False
    assert result["modes"] == prior["modes"] == list(startup.MODES)
    assert tuple(terminal["readiness_seconds"]) == tuple(result["results"]) == startup.MODES
    assert terminal["readiness_seconds"] == result["readiness_seconds"]
    assert startup.choices(result) == startup.choices(prior)
    assert all(len(result["results"][mode]) == 7 for mode in startup.MODES) and terminal["cases"] == 21

    preflight = json.loads(read("preflight.json"))
    assert preflight["initial"]["pressure"] == 1
    assert 90 <= preflight["quiet"]["cpu_idle_percent"] == terminal["pre_cpu_idle_percent"] <= 100
    assert 0 <= terminal["post_cpu_idle_percent"] <= 100
    sampled = [json.loads(line) for line in read("samples.jsonl").splitlines()]
    assert sampled and len(sampled) == terminal["pressure_samples"]
    stamps = [row["monotonic_seconds"] for row in sampled]
    assert all(isfinite(stamp) for stamp in stamps) and all(a < b for a, b in zip(stamps, stamps[1:]))
    assert sorted({row["pressure"] for row in sampled}) == terminal["pressure_levels"]
    assert all(row["page_size"] == 16384 for row in sampled)

    starts = {}
    sources = json.loads(read("portable-log-sources.json"))
    prior_selection = None
    for mode in startup.MODES:
        name = f"native-support/server-{mode}.log"
        payload = read(name)
        assert sha256(payload).hexdigest() == sources[name]["portable_sha256"]
        selection, phases = startup.startup(payload.decode())
        if prior_selection is None:
            prior_selection = selection
        assert selection == prior_selection
        ready = terminal["readiness_seconds"][mode]
        assert isfinite(ready) and 0 < phases["resident_tensors_seconds"] <= ready < 900
        steps = sum(row["activation_delta"][startup.VERIFIED] for row in result["results"][mode])
        assert steps == (107 if mode == "required" else 0)
        starts[mode] = {**phases, "http_ready_seconds": ready, "verified_steps": steps}
    kill_switch = "native-support/server-required-kill-switch.log"
    raw_kill = read(kill_switch)
    assert sha256(raw_kill).hexdigest() == sources[kill_switch]["portable_sha256"]
    assert b"no usable drafter was activated" in raw_kill or b"MlxMtpRequiredButUnavailable" in raw_kill
    assert terminal["within_existing_120_second_threshold"] == (starts["default"]["http_ready_seconds"] <= RELEASE_LIMIT_SECONDS)
    return {
        "scope": terminal["scope"], "qualification": False, "release_ready": False,
        "manifest_sha256": manifest, "server_sha256": startup.BINARY_SHA,
        "trace_result_sha256": terminal["result_sha256"], "matching_http_controls": terminal["cases"],
        "frozen_delivery_readiness_seconds": prior["readiness_seconds"],
        "existing_threshold_seconds": RELEASE_LIMIT_SECONDS,
        "observed_default_within_threshold": terminal["within_existing_120_second_threshold"],
        "cpu_idle_percent_before": terminal["pre_cpu_idle_percent"],
        "cpu_idle_percent_after": terminal["post_cpu_idle_percent"],
        "pressure_levels": terminal["pressure_levels"], "pressure_samples": len(sampled),
        "starts": starts,
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
