"""Reproduce delivered-copy readiness using the pinned native gate functions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[2] / "scripts"
COMMIT = "4bd15681fc7e6eec0c88f62a1a12ec9261cfcce4"
INPUTS_SHA = "f43e0cb314bc52833e88248734ae5b1ecf3e32aa10d6dc32cc1fc3f847dd48a0"


def read(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing delivery evidence {name}")
    return member.read()


def audit(archive, scripts):
    sys.path.insert(0, str(scripts))
    import check_flash_next_native_support as native
    import check_flash_next_release_gates as gate

    manifest_raw = read(archive, "inputs.sha256")
    assert sha256(manifest_raw).hexdigest() == INPUTS_SHA
    bindings = {name: digest for digest, name in
                (line.split() for line in manifest_raw.decode().splitlines())}
    for name in ("check_flash_next_release_gates.py", "check_flash_next_native_support.py",
                 "qualify_qwen38_flash_next.py", "flash_next_release_thresholds.json"):
        assert sha256((scripts / name).read_bytes()).hexdigest() == bindings[f"scripts/{name}"]
    provenance = json.loads(read(archive, "source-hashes.json"))
    for name, digests in provenance.items():
        assert sha256(read(archive, name)).hexdigest() == digests["portable_sha256"]
    delivery_raw = read(archive, "delivery.json")
    native_raw = read(archive, "native-support.json")
    build_raw = read(archive, "build.json")
    delivery, observed, build = map(json.loads, (delivery_raw, native_raw, build_raw))
    assert build["commit"] == COMMIT
    assert build["rustc"].startswith("rustc 1.99.1 ") and build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert build["profiles"] == {"ax-engine": "release", "ax-engine-server": "release-server"}
    assert sha256(build_raw).hexdigest() == bindings["build.json"]
    assert all(digest == bindings[name] for name, digest in build["binaries_sha256"].items())
    raw_thresholds = (scripts / "flash_next_release_thresholds.json").read_bytes()
    thresholds = json.loads(raw_thresholds)
    modes = tuple(thresholds["delivery"]["functional_controls_on_delivered_copy"])
    assert tuple(observed["modes"]) == tuple(observed["results"]) == modes
    assert delivery["revision"] == observed["revision"] == native.PACK_REVISION
    assert delivery["fresh_cache"] is True and delivery["completed"] is True
    assert native.is_target_hardware(delivery["hardware"]) and delivery["hardware"] == observed["hardware"]
    assert delivery["server_sha256"] == observed["server_sha256"] == build["binaries_sha256"]["ax-engine-server"]
    assert delivery["download"]["ax_engine_sha256"] == build["binaries_sha256"]["ax-engine"]
    assert observed["functional_controls_passed"] is True and observed["negative_control_result"]["activation_error_observed"] is True
    terminal = delivery["download"]["terminal"]
    events = [json.loads(line) for line in read(archive, "download.ndjson").splitlines()]
    assert events[-1] == terminal and len(events) > 1
    expected_bytes = thresholds["target"]["published_bytes"]
    assert all(event["event"] == "progress" for event in events[:-1])
    phase_only = [event for event in events[:-1] if event["bytes_total"] is None]
    assert [event["phase"] for event in phase_only] == ["inventory", "validate", "ready"]
    assert all(event["bytes_done"] is None for event in phase_only)
    transfers = [event for event in events[:-1] if event["bytes_total"] is not None]
    assert transfers and all(event["bytes_total"] == expected_bytes for event in transfers)
    progress = [event["bytes_done"] for event in transfers]
    assert 0 <= progress[0] <= progress[-1] == expected_bytes
    assert all(a <= b for a, b in zip(progress, progress[1:]))
    assert terminal["schema_version"] == "ax.download_model.v1" and terminal["status"] == "ready"
    for key in ("revision", "resolved_revision"):
        assert terminal[key] == native.PACK_REVISION
    assert terminal["repo_id"] == native.PRIMARY_REPO and terminal["errors"] == []
    assert terminal["manifest_present"] is True and terminal["config_present"] is True
    manifest = {"model-manifest.json": delivery["verification"]["native_manifest_sha256"]}
    assert observed["native_manifest_before"] == observed["native_manifest_ready"] == observed["native_manifest_after"] == manifest
    for mode in modes:
        cases = observed["results"][mode]
        assert [row["name"] for row in cases] == [case[0] for case in native.CASES]
        for case in cases:
            native.check_answer(case["name"], case["response"])
            numbers = case["response"]["usage"]["completion_tokens"]
            computed = native.check_activation(case["metrics_before"], case["metrics_after"],
                                               mode == "required", numbers, case["request"]["max_tokens"])
            assert computed == case["activation_delta"]
    assert not gate.delivery_control_problems(delivery, native_raw)
    readiness, = gate.gate_readiness(delivery, [], thresholds)
    delivery_gate, = gate.gate_delivery(delivery, thresholds)
    return {"scope": "Fresh native delivery and delivered-copy readiness only",
            "release_ready": False, "downloaded_bytes": delivery["verification"]["bytes_verified"],
            "delivery": delivery_gate, "delivered_copy_readiness": readiness,
            "hardware": delivery["hardware"], "download_seconds": delivery["download"]["elapsed_seconds"],
            "readiness_seconds": observed["readiness_seconds"], "matching_native_manifest": manifest,
            "functional_controls": sum(len(observed["results"][mode]) for mode in modes),
            "delivery_sha256": sha256(delivery_raw).hexdigest(),
            "native_support_sha256": sha256(native_raw).hexdigest(),
            "build_sha256": sha256(build_raw).hexdigest(),
            "thresholds_sha256": sha256(raw_thresholds).hexdigest()}


def replay(path=ROOT / "evidence.tar.xz", scripts=SCRIPTS):
    with tarfile.open(path) as archive:
        return audit(archive, scripts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive), indent=2, sort_keys=True, allow_nan=False))
