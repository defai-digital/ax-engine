"""Replay the signed guarded-reader delivery and its default integration gates."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
CANDIDATE = "d2220c26f3cfccf2583c583c6b264e22b9459f88"
HARNESS = "eb4922144b461e75bbd77fd8b477bab459e2689b"
SERVER = "c1a7c3f9dfb81613fc370758a0eece5f6105683d53baab7ea4bc3a371cb81de2"
PRIOR_SHA = "d509792298bfc2c815ea61bbfc8f429a0df00253c8a09365d8cb0a01e326685f"
TRANSFER_SHA = "69e89be57f5a70f19d294d2f069993bf94f3c01a919667dffb6e342d16a37a93"
MODES = {"disabled": 32768, "required": 16384, "default": 32768}
PROVENANCE = "source-sha256.json"
MEMBERS = {PROVENANCE, "status.txt", "build.json", "inputs.sha256", "harness-commit.txt", "delivery/result.json",
           "delivery/download.ndjson", "delivery/native-support/result.json", "software.json",
           *(f"delivery/native-support/server-{mode}.log" for mode in (*MODES, "required-kill-switch"))}


def checksum(data):
    return sha256(data).hexdigest()


def get_member(archive, name):
    entry = archive.getmember(name)
    assert entry.isfile()
    with archive.extractfile(entry) as stream:
        return stream.read()


def replay(archive_path, repo):
    bench = repo / "benchmarks/results"
    prior = bench / "flash-next-probe-followup-fresh-m5-20261123/evidence.tar.xz"
    transfer = bench / "flash-next-fresh-delivery-38c90b27a-m5-20261122/replay.py"
    assert checksum(prior.read_bytes()) == PRIOR_SHA
    assert checksum(transfer.read_bytes()) == TRANSFER_SHA
    with tarfile.open(archive_path) as archive, tarfile.open(prior) as previous:
        names = archive.getnames()
        assert len(names) == len(MEMBERS) and set(names) == MEMBERS
        data = {name: get_member(archive, name) for name in names}
        oracle = json.loads(get_member(previous, "raw/delivery/native-support/result.json"))
    assert data["status.txt"] == b"0\n" and data["harness-commit.txt"].decode().strip() == HARNESS
    assert all(token not in value for value in data.values()
               for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    provenance = json.loads(data[PROVENANCE])
    assert set(provenance) == {"delivery/result.json", "delivery/download.ndjson"}
    assert all(re.fullmatch(r"[a-f0-9]{64}", row["original"]) and row["portable"] == checksum(data[name])
               for name, row in provenance.items())
    entries = [line.split() for line in data["inputs.sha256"].decode().splitlines()]
    hashes = {name: digest for digest, name in entries}
    assert len(hashes) == len(entries) and hashes["build.json"] == checksum(data["build.json"])
    for name in ("check_flash_next_native_support.py", "check_flash_next_release_gates.py",
                 "qualify_qwen38_flash_next.py", "flash_next_release_thresholds.json", "run_flash_next_delivery.py"):
        path = f"scripts/{name}"
        assert checksum((repo / path).read_bytes()) == hashes[path]
    sys.path.insert(0, str(repo / "scripts"))
    import check_flash_next_release_gates as gates

    spec = spec_from_file_location("frozen_flash_next_transfer", transfer)
    assert spec is not None and spec.loader is not None
    canonical = module_from_spec(spec)
    spec.loader.exec_module(canonical)
    build, delivery, native, software = (json.loads(data[name]) for name in (
        "build.json", "delivery/result.json", "delivery/native-support/result.json", "software.json",
    ))
    assert build["server_sha256"] == delivery["server_sha256"] == hashes["ax-engine-server"] == SERVER
    assert build["inventory_sha256"] == delivery["inventory_sha256"] == hashes["mxfp4-inventory.json"]
    assert delivery["download"]["ax_engine_sha256"] == hashes["ax-engine"]
    assert build["profile"] == "release-server" and build["codesign_verified"] is True
    assert build["mlx"] == build["mlx_metal"] == "0.32.4" and build["rustc"].startswith("rustc 1.99.1 ")
    assert build["patch_sha256"] == software["patch_sha256"] == hashes["source.patch"]
    assert software["candidate_commit"] == CANDIDATE and software["integration_base"] == "16ed46747"
    assert all(software["checks"].values()) and len(software["checks"]) == 9
    assert delivery["hardware"] == native["hardware"]
    hardware = delivery["hardware"]
    assert hardware["form_factor"] == "MacBook Pro" and hardware["soc"] == "Apple M5 Max"
    assert hardware["memory_bytes"] == 128 * (1 << 30) and hardware["os"] == "27.2.1"
    assert hardware["storage"]["internal"] is True
    assert delivery["metadata_preflight_passed"] is True and not delivery["resumed"]
    assert delivery["previous_attempts"] == [] and delivery["fresh_cache"] is True
    assert delivery["qualification"] is False and delivery["release_ready"] is False
    assert not gates.delivery_control_problems(delivery, data["delivery/native-support/result.json"])
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    assert all(row["passed"] for row in gates.gate_delivery(delivery, thresholds)
               + gates.gate_readiness(delivery, [], thresholds))
    manifest = {"model-manifest.json": delivery["verification"]["native_manifest_sha256"]}
    assert all(native[f"native_manifest_{stage}"] == manifest for stage in ("before", "ready", "after"))
    for mode, limit in MODES.items():
        log = re.sub(r"\x1b\[[0-9;]*m", "", data[f"delivery/native-support/server-{mode}.log"].decode())
        assert "decision=resident" in log
        assert re.search(r"resolved the Flash Next M5 Max 128 GB serving profile[^\n]*context_tokens="
                         + str(limit) + r"\b[^\n]*effective_sequence_cap=Some\(1\)", log)
    steps = canonical.check_cases(native, oracle, list(MODES))
    assert steps == {"disabled": 0, "required": 107, "default": 0}
    payload_bytes = thresholds["target"]["published_bytes"]
    samples = canonical.check_transfer(data["delivery/download.ndjson"], delivery, payload_bytes)
    return {"candidate_commit": CANDIDATE, "server_sha256": SERVER, "publisher_bytes": payload_bytes,
            "publisher_members": delivery["verification"]["members_verified"],
            "download_seconds": delivery["download"]["elapsed_seconds"], "download_progress_counters": samples,
            "readiness_seconds": native["readiness_seconds"], "context_tokens": MODES,
            "verified_steps": steps, "stable_response_controls": sum(map(len, native["results"].values())),
            "software": software, "release_ready": False}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    parser.add_argument("--repo", type=Path, default=ROOT.parents[2])
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.repo.resolve()), sort_keys=True, indent=2, allow_nan=False))
