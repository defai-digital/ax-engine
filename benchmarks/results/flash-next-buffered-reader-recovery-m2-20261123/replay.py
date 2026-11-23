"""Replay the signed aligned-buffered failure and same-port restart."""

from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import tarfile

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "flash-next-natural-document-prefix-http-m2-20261123"
INJECTOR = ROOT.parent / "flash-next-native-loader-failure-m2-20261121" / "shard_read_fault.c"
PRIOR_SHA256 = "db1563861203784640453460a450c75e4dc43ad8f5e52b93bbcdadf64a872b58"
READ_ERROR = "aligned tensor payload read failed or ended early"
READER = "Flash Next resident reader aligned_nocache=false reader=AlignedBuffered"
SHARD_1 = "model-00001-of-00025.safetensors"
SHARD_12 = "model-00012-of-00025.safetensors"
MARKER = "ax_fixture selected shard pread returned EIO"
COUNTERS = {
    "ax_engine_jobs_in_flight", "ax_engine_generation_jobs_pending",
    "ax_engine_generation_commands_queued", "ax_engine_generation_active_streams",
    "ax_engine_generation_buffered_stream_events",
}
MEMBERS = {
    "build.json", "source.patch", "inputs.sha256", "run_read_recovery.py",
    "run_owner.sh", "status.txt", "owner.log",
    "observed/fault.log", "observed/recovery.log", "observed/receipt.json",
}


def digest(data):
    return sha256(data).hexdigest()


def replay():
    prior = PRIOR / "evidence.tar.xz"
    assert digest(prior.read_bytes()) == PRIOR_SHA256
    read_member = run_path(str(PRIOR / "replay.py"))["read_member"]
    with (tarfile.open(prior) as baseline, tarfile.open(ROOT / "evidence.tar.xz") as archive):
        cold = json.loads(read_member(baseline, "cold/observed/receipt.json"))
        data = {name: read_member(archive, name) for name in MEMBERS}
        assert set(archive.getnames()) == MEMBERS and len(archive.getnames()) == len(MEMBERS)
    receipt = json.loads(data["observed/receipt.json"])
    build = json.loads(data["build.json"])
    manifest = dict(line.split()[::-1] for line in data["inputs.sha256"].decode().splitlines())
    for key in ("build.json", "source.patch", "run_read_recovery.py", "run_owner.sh"):
        assert digest(data[key]) == manifest[key]
    assert digest(INJECTOR.read_bytes()) == manifest["shard_read_fault.c"]
    assert receipt["shim_sha256"] == manifest["shard_read_fault.dylib"]
    assert receipt["build"] == build and build["codesign_verified"] is True
    assert build["server_sha256"] == manifest["ax-engine-server"]
    assert build["patch_sha256"] == manifest["source.patch"]
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert build["rustc"].startswith("rustc 1.99.1 ")
    assert data["status.txt"] == b"0\n" and receipt["qualification"] is False
    assert receipt["manifest"] == cold["manifest"] and receipt["publisher_members_rechecked"] == 49
    assert receipt["hardware"] == {"form_factor": "Mac Studio", "soc": "Apple M2 Ultra",
                                   "memory_bytes": 206158430208, "os": "27.2.1",
                                   "storage": "internal startup SSD"}
    assert receipt["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    fault, recovery = receipt["fault"], receipt["recovery"]
    assert fault["exit_code"] == 1 and not fault["http_listener_observed"] and fault["port_reusable"]
    assert fault["completed_shard_before_fault"] == SHARD_1 and fault["failed_shard"] == SHARD_12
    assert fault["reader"] == recovery["reader"] == "AlignedBuffered" and fault["read_error"] == READ_ERROR
    assert recovery["context_length"] == 32768 and recovery["health_after"] == "ok"
    assert recovery["reused_faulted_port"] and recovery["drained"] == dict.fromkeys(COUNTERS, 0)
    result = recovery["result"]
    assert result["prompt_tokens"] == 512 and result["status"] == "finished"
    assert result["verified_steps"] == result["route_errors"] == 0
    assert result["output_tokens"] == cold["outputs"]["first"]["peer_ids"][:128]
    samples = [*fault["pressure_samples"], *recovery["pressure_samples"]]
    assert samples and all(sample == 1 for sample in samples)
    for name in ("observed/fault.log", "observed/recovery.log", "owner.log"):
        assert not any(part in data[name] for part in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    fault_log, restarted = (data[f"observed/{name}.log"].decode() for name in ("fault", "recovery"))
    assert READER in fault_log and READER in restarted
    assert READ_ERROR in fault_log and MARKER in fault_log and MARKER not in restarted
    completed = next(line for line in fault_log.splitlines() if "Flash Next resident shard" in line and SHARD_1 in line)
    assert fault_log.index(completed) < fault_log.index(MARKER)
    assert "preview listening" not in fault_log and 'phase="resident_tensors"' not in fault_log
    assert 'phase="resident_tensors"' in restarted
    return {"scope": receipt["scope"], "server_sha256": build["server_sha256"],
            "fault_exit_code": fault["exit_code"], "verified_recovery_token_ids": len(result["output_tokens"]),
            "port_reused": recovery["reused_faulted_port"], "normal_pressure_samples": len(samples),
            "publisher_members_rechecked": receipt["publisher_members_rechecked"], "qualification": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
