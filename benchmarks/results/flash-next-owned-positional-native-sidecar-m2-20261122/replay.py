"""Replay the full native required-MTP sidecar controls."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import check_flash_next_native_support as native  # noqa: E402

sys.path.insert(0, str(ROOT.parent / "flash-next-mxfp4-startup-phases-m5-20261121"))
from capture_phases import read_log  # noqa: E402

INPUT_SHA = "c580eb1fd184cf3dabe841373abf36057c3661a6bc2d521e21eee1aa83202444"
BUILD_COMMIT = "8204ff9246a18f8771999dcf05ad14a1b1cb2e56"


def read(archive, name):
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"Missing evidence {name}")
    return member.read()


def replay(path=ROOT / "evidence.tar.xz"):
    with tarfile.open(path) as archive:
        assert read(archive, "phase-status.txt") == b"native-sidecar-smoke 0\n"
        inputs = read(archive, "inputs.sha256")
        assert sha256(inputs).hexdigest() == INPUT_SHA
        hashes = {key: value for value, key in (line.split() for line in inputs.decode().splitlines())}
        native_script = (ROOT.parents[2] / "scripts" / "check_flash_next_native_support.py").read_bytes()
        assert sha256(native_script).hexdigest() == hashes["check_flash_next_native_support.py"]
        raw = read(archive, "receipt.json")
        receipt = json.loads(raw)
        assert receipt["completed"] is True and receipt["qualification"] is False
        assert receipt["hardware"] == "Mac Studio M2 Ultra 192 GB, external SSD"
        assert receipt["revision"] == native.PACK_REVISION
        assert receipt["inventory_sha256"] == hashes["inventory.json"]
        build_raw = read(archive, "server-build-receipt.json")
        assert sha256(build_raw).hexdigest() == hashes["server-build-receipt.json"]
        build = json.loads(build_raw)
        assert build["commit"] == BUILD_COMMIT and receipt["build"] == build
        assert build["server_sha256"] == hashes["ax-engine-server-positional-sidecar"]
        assert build["profile"] == "release-server" and build["mlx"] == build["mlx_metal"] == "0.32.4"
        assert receipt["kill_switch"]["activation_error_observed"] is True
        assert tuple(receipt["arms"]) == ("mmap", "pread")
        row_counts = {}
        baseline = None
        base_shards = None
        logs = json.loads(read(archive, "logs.json"))
        for mode, info in receipt["arms"].items():
            data = info["cases"]
            assert [row["name"] for row in data] == [item[0] for item in native.CASES]
            signatures = []
            for row in data:
                response = row["response"]
                native.check_answer(row["name"], response)
                check = native.check_activation(row["metrics_before"], row["metrics_after"], True,
                                                 response["usage"]["completion_tokens"],
                                                 row["request"]["max_tokens"])
                assert check == row["activation_delta"]
                signatures.append((row["name"], response["choices"], check))
            if baseline is None:
                baseline = signatures
            assert signatures == baseline
            steps = sum(row["activation_delta"][native.VERIFIED] for row in data)
            assert steps == receipt["verified_steps"] == 107
            assert info["pressure_before"] == info["pressure_after"] == 1
            name = f"{mode}.log"
            log = read(archive, name)
            assert sha256(log).hexdigest() == logs[name]["portable_sha256"]
            assert b"Flash Next resident positional read lab" not in log
            assert log.count(b"Flash Next MTP sidecar positional read lab") == int(mode == "pread")
            phases, shards = read_log(log.decode())
            selection = [(row["name"], row["tensors"]) for row in shards]
            if base_shards is None:
                base_shards = selection
            assert selection == base_shards and len(selection) == 15 and sum(n for _, n in selection) == 1942
            assert info["sidecar_seconds"] == phases["mtp_sidecar"] / 1e6
            ready = info["ready_seconds"]
            resident = phases["resident_tensors"] / 1e6
            assert isfinite(ready) and ready > resident > 0
            row_counts[mode] = {"ready_seconds": ready, "resident_seconds": resident,
                                "sidecar_seconds": info["sidecar_seconds"], "verified_steps": steps}
        name = "kill-switch.log"
        kill = read(archive, name)
        assert sha256(kill).hexdigest() == logs[name]["portable_sha256"]
        assert b"no usable drafter was activated" in kill or b"MlxMtpRequiredButUnavailable" in kill
    return {"scope": receipt["scope"], "qualification": False, "matching_required_cases": 7,
            "kill_switch_rejected_required": True, "receipt_sha256": sha256(raw).hexdigest(), "arms": row_counts}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
