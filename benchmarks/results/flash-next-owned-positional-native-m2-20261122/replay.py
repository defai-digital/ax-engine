"""Check the full-pack M2 positional-reader outcome against known native tokens."""
import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import isfinite
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-native-loader-m2-20261121" / "result.json"
PHASE_READER = ROOT.parent / "flash-next-mxfp4-startup-phases-m5-20261121" / "capture_phases.py"
PRIOR_SHA = "dbfa5493530693bd904969ca7425d68e5eedfe349b01496b11da2c37062b05a2"
INPUT_SHA = "45db1dd68b830bf0641e583ae18f9d3045fc8ef20c14d63197aff8d05d99ded7"
BUILD_COMMIT = "8729b80c2697efa0db6f78f0a15b5a6d2292deca"
HARDWARE = "Mac Studio M2 Ultra 192 GB, external SSD"
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"


def read(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing receipt member {name}")
    return member.read()


def paths(cases):
    return [(row["prompt_hash"], row["output_tokens"]) for row in cases]


def load_reader():
    spec = spec_from_file_location("owned_positional_phase_reader", PHASE_READER)
    if spec is None or spec.loader is None:
        raise ValueError("Missing established startup phase reader")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_log


def replay(path=ROOT / "evidence.tar.xz"):
    reader = load_reader()
    prior_raw = PRIOR.read_bytes()
    assert sha256(prior_raw).hexdigest() == PRIOR_SHA
    prior = json.loads(prior_raw)
    assert prior["pack_revision"] == REVISION and prior["hardware"] == HARDWARE
    expected_cases = paths(prior["arms"]["name-order"]["cases"])
    assert [len(ids) for _, ids in expected_cases] == [32, 32, 32]
    with tarfile.open(path) as archive:
        assert read(archive, "phase-status.txt") == b"native-positional-smoke 0\n"
        captured_inputs = read(archive, "inputs.sha256")
        assert sha256(captured_inputs).hexdigest() == INPUT_SHA
        hashes = {name: digest for digest, name in
                  (row.split() for row in captured_inputs.decode().splitlines())}
        selection_raw = read(archive, "expected-resident-selection.json")
        assert sha256(selection_raw).hexdigest() == hashes["expected-resident-selection.json"]
        selection = json.loads(selection_raw)
        assert len(selection) == 15 and sum(n for _, n in selection) == 1942
        build_raw = read(archive, "server-build-receipt.json")
        assert sha256(build_raw).hexdigest() == hashes["server-build-receipt.json"]
        receipt_raw = read(archive, "receipt.json")
        receipt = json.loads(receipt_raw)
        build = json.loads(build_raw)
        assert receipt["build"] == build and build["commit"] == BUILD_COMMIT
        assert build["profile"] == "release-server" and build["mlx"] == build["mlx_metal"] == "0.32.4"
        assert build["server_sha256"] == hashes["ax-engine-server-positional"]
        assert receipt["pack_revision"] == REVISION and receipt["hardware"] == HARDWARE
        assert receipt["prior_output_receipt_sha256"] == PRIOR_SHA == hashes["prior-native-result.json"]
        assert receipt["status"] == "complete" and receipt["matched_output_ids"] == 96
        assert tuple(receipt["arms"]) == ("mmap", "pread")
        logs = json.loads(read(archive, "logs.json"))
        rows = {}
        for mode, arm in receipt["arms"].items():
            assert (arm["shard_count"], arm["tensor_count"]) == (15, 1942)
            assert paths(arm["cases"]) == expected_cases
            assert all(row["verified_steps"] == row["route_errors"] == 0
                       and row["status"] == "finished" and row["finish_reason"] == "max_output_tokens"
                       for row in arm["cases"])
            name = f"{mode}.log"
            log = read(archive, name)
            assert sha256(log).hexdigest() == logs[name]["portable_sha256"]
            assert log.count(b"Flash Next resident positional read lab") == int(mode == "pread")
            phases, shards = reader(log.decode())
            assert [[row["name"], row["tensors"]] for row in shards] == selection
            assert arm["resident_seconds"] == phases["resident_tensors"] / 1e6
            shard_read = sum(row["read_us"] for row in shards)
            assert shard_read <= phases["resident_tensors"]
            ready = arm["ready_seconds"]
            assert isfinite(ready) and ready > arm["resident_seconds"] > 0
            samples = [json.loads(line) for line in read(archive, f"{mode}-samples.jsonl").splitlines()]
            assert samples and all(row["pressure"] == 1 and row["elapsed_seconds"] >= 0 for row in samples)
            assert all(x["elapsed_seconds"] < y["elapsed_seconds"] for x, y in zip(samples, samples[1:]))
            rows[mode] = {"ready_seconds": ready, "resident_seconds": arm["resident_seconds"],
                          "shard_read_allocation_seconds": shard_read / 1e6,
                          "sidecar_seconds": phases["mtp_sidecar"] / 1e6,
                          "pressure_samples": len(samples),
                          "peak_sampled_rss_bytes": max(row["reference_rss_bytes"] for row in samples),
                          "slowest_shards": sorted(shards, key=lambda row: row["read_us"], reverse=True)[:3]}
    return {"scope": "Full-pack native output smoke on M2 with uncontrolled file residency",
            "native_receipt_sha256": sha256(receipt_raw).hexdigest(), "matched_output_ids": 96,
            "arms": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive), indent=2, sort_keys=True, allow_nan=False))
