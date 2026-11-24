"""Verify the cold-delivery and HTTP controls for the URL-patched native build."""

from hashlib import sha256
import json
from pathlib import Path
import re
from runpy import run_path
import tarfile
import tomllib

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
REPO = HERE.parents[2]
SOURCE = "1e8ea48455909b65172d435d9d230cb2a5f93d34"
SERVER = "d79b31a147cf811c9ac87039853286bb95d23ffbb46f55e16f723c40d8698d35"
PRIOR = "19f1406406ee302d83ac03bc69e6768ff58236cc3b935d95b2f955895a203a76"
TRANSFER = "69e89be57f5a70f19d294d2f069993bf94f3c01a919667dffb6e342d16a37a93"
MODES = {"disabled": 32768, "required": 16384, "default": 32768}
CONTROLS = "delivery/native-support/result.json"
DELIVERY = "delivery/result.json"
DOWNLOAD = "delivery/download.ndjson"
LOGS = {f"delivery/native-support/server-{mode}.log" for mode in MODES}
MEMBERS = {"build.json", "Cargo.lock", "inventory.json", "inputs.sha256", "split-inputs.sha256",
           "run_split_native.sh", "run_split_native.py", "control-preflight.jsonl",
           "source-sha256.json", DELIVERY, DOWNLOAD, CONTROLS, *LOGS}


def digest(blob):
    return sha256(blob).hexdigest()


def member(archive, name):
    entry = archive.getmember(name)
    assert entry.isfile()
    with archive.extractfile(entry) as stream:
        return stream.read()


def manifest(blob):
    pairs = [line.split() for line in blob.decode().splitlines()]
    values = {name: sha for sha, name in pairs}
    assert len(values) == len(pairs)
    return values


def replay():
    transfer = BENCHMARKS / "flash-next-fresh-delivery-38c90b27a-m5-20261122/replay.py"
    prior = BENCHMARKS / "flash-next-consolidated-native-m5-20261124/evidence.tar.xz"
    assert digest(transfer.read_bytes()) == TRANSFER and digest(prior.read_bytes()) == PRIOR
    helpers = run_path(str(transfer))
    gates = helpers["gates"]
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    with tarfile.open(HERE / "evidence.tar.xz") as archive, tarfile.open(prior) as previous:
        assert len(archive.getmembers()) == len(MEMBERS) and set(archive.getnames()) == MEMBERS
        raw = {name: member(archive, name) for name in MEMBERS}
        oracle = json.loads(member(previous, CONTROLS))
    assert all(token not in blob for blob in raw.values()
               for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    portable = json.loads(raw["source-sha256.json"])
    assert portable.keys() <= {DELIVERY, DOWNLOAD, *LOGS}
    assert all(digest(raw[name]) == row["portable"] and re.fullmatch(r"[0-9a-f]{64}", row["original"])
               for name, row in portable.items())
    entries = manifest(raw["inputs.sha256"])
    for name in ("build.json", "Cargo.lock", "inventory.json"):
        assert entries[f"./{name}"] == digest(raw[name])
    for name in ("check_flash_next_native_support.py", "check_flash_next_release_gates.py",
                 "qualify_qwen38_flash_next.py", "flash_next_release_thresholds.json", "run_flash_next_delivery.py"):
        path = f"scripts/{name}"
        assert digest((REPO / path).read_bytes()) == entries[f"./{path}"]
    split = manifest(raw["split-inputs.sha256"])
    assert split == {name: digest(raw[name]) for name in ("run_split_native.sh", "run_split_native.py")}
    build, delivery, control = (json.loads(raw[name]) for name in ("build.json", DELIVERY, CONTROLS))
    assert build["commit"] == SOURCE and build["codesign_verified"] and build["profile"] == "release-server"
    assert build["rustc"].startswith("rustc 1.99.1 ") and build["mlx"] == build["mlx_metal"] == "0.32.4"
    lock = tomllib.loads(raw["Cargo.lock"].decode())
    assert [row["version"] for row in lock["package"] if row["name"] == "url"] == [build["url_version"]] == ["2.5.9"]
    assert build["cargo_lock_sha256"] == digest(raw["Cargo.lock"])
    assert build["server_sha256"] == delivery["server_sha256"] == control["server_sha256"] == entries["./ax-engine-server"] == SERVER
    assert build["cli_sha256"] == delivery["download"]["ax_engine_sha256"] == entries["./ax-engine"]
    assert delivery["inventory_sha256"] == control["inventory_sha256"] == digest(raw["inventory.json"])
    assert delivery["hardware"] == control["hardware"]
    hardware = delivery["hardware"]
    assert hardware["form_factor"] == "MacBook Pro" and hardware["soc"] == "Apple M5 Max"
    assert hardware["memory_bytes"] == 128 * 2**30 and hardware["os"] == "27.2.1" and hardware["storage"]["internal"]
    assert delivery["completed"] and delivery["fresh_cache"] and not delivery["resumed"] and delivery["previous_attempts"] == []
    assert delivery["metadata_preflight_passed"] and not gates.delivery_control_problems(delivery, raw[CONTROLS])
    delivery_gate, = gates.gate_delivery(delivery, thresholds)
    readiness, = gates.gate_readiness(delivery, [], thresholds)
    assert delivery_gate["passed"] and readiness["passed"]
    pack_manifest = {"model-manifest.json": delivery["verification"]["native_manifest_sha256"]}
    assert all(control[f"native_manifest_{stage}"] == pack_manifest for stage in ("before", "ready", "after"))
    for mode, limit in MODES.items():
        log = re.sub(r"\x1b\[[0-9;]*m", "", raw[f"delivery/native-support/server-{mode}.log"].decode())
        assert "decision=resident" in log
        assert re.search(r"resolved the Flash Next M5 Max 128 GB serving profile[^\n]*context_tokens="
                         + str(limit) + r"\b[^\n]*effective_sequence_cap=Some\(1\)", log)
    steps = helpers["check_cases"](control, oracle, list(MODES))
    assert steps == {"disabled": 0, "required": 107, "default": 0}
    preflight = [json.loads(row)["cpu_idle_percent"] for row in raw["control-preflight.jsonl"].splitlines()]
    assert len(preflight)>1 and all(0 <= idle < 90 for idle in preflight[:-1]) and 90 <= preflight[-1] <= 100
    progress = helpers["check_transfer"](raw[DOWNLOAD], delivery, thresholds["target"]["published_bytes"])
    return {"source": SOURCE, "server_sha256": SERVER, "url_version": build["url_version"],
            "bytes_verified": delivery["verification"]["bytes_verified"],
            "members_verified": delivery["verification"]["members_verified"],
            "download_seconds": delivery["download"]["elapsed_seconds"], "progress_rows": progress,
            "delivery_gate_passed": delivery_gate["passed"], "delivered_readiness_within_limits": readiness["passed"],
            "control_entry_cpu_idle_percent": preflight[-1], "readiness_seconds": control["readiness_seconds"],
            "context_tokens": MODES, "verified_steps": steps,
            "stable_response_controls": sum(map(len, control["results"].values())), "release_ready": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
