"""Verify the CLI rejection and its contrast with the previous startup phases."""
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "flash-next-delivery-bound-startup-m5-20261122/evidence.tar.xz"
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
PAYLOAD = ("Flash Next resident shard", 'phase="resident_tensors"', 'phase="mtp_sidecar"',
           "MTP admission at runner construction", "wired GPU residency for this session")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((ROOT / "result.json").read_bytes())
    build = json.loads((ROOT / "build.json").read_bytes())
    log = (ROOT / "server.txt").read_text()
    inventory = json.loads(INVENTORY.read_bytes())
    require(result["inventory_sha256"] == digest(INVENTORY) and result["pack_revision"] == inventory["revision"]
            and result["inventory_verified_members"] == len(inventory["members"]) == 49,
            "The pinned publisher inventory changed")
    require(result["server_sha256"] == build["server_sha256"] and build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["check_sha256"] == digest(ROOT / "check.py"), "The native binary or driver binding changed")
    require(set(build["changed_source_sha256"]) == {
                "crates/ax-engine-mlx/src/fastpath.rs", "crates/ax-engine-sdk/src/session/native.rs",
            } and all(re.fullmatch(r"[0-9a-f]{64}", checksum)
                      for checksum in build["changed_source_sha256"].values()),
            "The source binding is incomplete")
    require(result["qualification"] is False and result["policy"] == "required"
            and result["environment"] == {"AX_NO_SPEC": "1"} and result["exit_code"] == 1
            and result["activation_error"] == "MlxMtpRequiredButUnavailable"
            and f"Error: Engine({result['activation_error']})" in log, "The expected activation error did not occur")
    require(result["payload_startup_markers"] == {marker: log.count(marker) for marker in PAYLOAD}
            == dict.fromkeys(PAYLOAD, 0), "The rejection initialized inference or loaded its payloads")
    with tarfile.open(PREVIOUS) as archive:
        old_log = archive.extractfile("native-support/server-required-kill-switch.log").read().decode()
    require(result["activation_error"] in old_log, "The prior M5 case used a different error")
    elapsed = dict(re.findall(r'Flash Next startup phase phase="([a-z_]+)" elapsed_us=(\d+)', old_log))
    print(json.dumps({
        "previous_m5_resident_shards": old_log.count(PAYLOAD[0]),
        "previous_m5_resident_seconds": int(elapsed["resident_tensors"]) / 1e6,
        "previous_m5_sidecar_seconds": int(elapsed["mtp_sidecar"]) / 1e6,
        "new_m2_resident_shards": log.count(PAYLOAD[0]),
        "new_m2_process_seconds": result["process_elapsed_seconds"],
        "same_activation_error": True,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
