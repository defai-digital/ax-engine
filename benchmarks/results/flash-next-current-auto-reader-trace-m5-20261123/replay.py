"""Replay Flash Next startup phase and exact resident-shard timing evidence."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from runpy import run_path
import tarfile
from tempfile import TemporaryDirectory

ANSI = re.compile(rb"\x1b\[[0-9;]*m")
PHASE = re.compile(rb'Flash Next startup phase phase="([a-z_]+)" elapsed_us=(\d+)')
SHARD = re.compile(
    rb'shard=Some\("(model-\d+-of-00025\.safetensors)"\) tensor_count=(\d+) read_us=(\d+) eval_us=(\d+)'
)
EXPECTED_PHASES = ("trunk_admission", "resident_selection", "resident_tensors",
                   "trunk_weight_contract", "trunk_assembly", "mtp_sidecar", "vision_tower")
VM_KEYS = {"pageins", "pageouts", "compressions", "decompressions", "swapins", "swapouts"}
RAW_INPUTS = ("build.json", "mxfp4-inventory.json", "source.patch")
STAGED_CODE = ("check_flash_next_native_support.py", "compare_startup_arm.py",
               "qualify_qwen38_flash_next.py", "run_flash_next_lifecycle.py",
               "run_guarded.sh", "run_owner.sh")
CAPTURE_FILES = (*RAW_INPUTS, *STAGED_CODE, "trace.json", "capture.json",
                 "server-default.log", "inputs.sha256")
MANIFEST = {"model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def parse_timings(raw, geometry):
    log = ANSI.sub(b"", raw)
    rows = [(name.decode(), int(elapsed)) for name, elapsed in PHASE.findall(log)]
    phases = dict(rows)
    need(tuple(name for name, _ in rows) == EXPECTED_PHASES,
         "Startup trace phases are missing, repeated or misordered")
    shards = [{"name": name.decode(), "tensors": int(count),
               "read_and_allocate_us": int(read_us), "sync_eval_us": int(eval_us)}
              for name, count, read_us, eval_us in SHARD.findall(log)]
    names = [row["name"] for row in shards]
    need(len(names) == len(set(names)) == len(geometry) == 15 and set(names) == geometry.keys(),
         "Resident shard roster changed")
    need(sum(row["tensors"] for row in shards) == 1942
         and sum(row["selected_bytes"] for row in geometry.values()) == 68_516_576_280
         and all(row["tensors"] == geometry[row["name"]]["selected_tensors"]
                 and row["read_and_allocate_us"] > 0 for row in shards),
         "The traced unpacked resident selection changed relative to the frozen geometry")
    read_us = sum(row["read_and_allocate_us"] for row in shards)
    eval_us = sum(row["sync_eval_us"] for row in shards)
    residual_us = phases["resident_tensors"] - phases["resident_selection"] - read_us - eval_us
    need(residual_us >= 0, "Nested timers exceed the resident stage")
    return phases, shards, {"read_and_allocate_us": read_us, "sync_eval_us": eval_us,
                            "unattributed_us": residual_us}


def compare(capture_root, results_root, crossover):
    trace = read(capture_root / "trace.json")
    build = read(capture_root / "build.json")
    capture = read(capture_root / "capture.json")
    input_lines = (capture_root / "inputs.sha256").read_text().splitlines()
    inputs = dict(line.split()[::-1] for line in input_lines)
    need(len(input_lines) == len(inputs) and set(inputs) == {*RAW_INPUTS, *STAGED_CODE, "ax-engine-server"}
         and all(inputs[name] == digest(capture_root / name) for name in (*RAW_INPUTS, *STAGED_CODE))
         and inputs["ax-engine-server"] == build["server_sha256"],
         "The sealed candidate launch inputs changed")
    previous = read(results_root / "flash-next-main-lifecycle-m5-20261123" / "lifecycle.json")
    prior_default, = (row for row in previous["runs"] if row["mode"] == "default")
    helpers = run_path(str(results_root / "flash-next-probe-followup-lifecycle-m5-20261122" / "replay.py"))
    stable = helpers["stable_action"]
    need(trace["arm"] == "traced-auto"
         and trace["override"] == {"RUST_LOG": "info,ax_engine_mlx::flash_next_startup=debug"}
         and capture["status"] == 0 and capture["codesign_verified"] is build["codesign_verified"] is True
         and trace["server_sha256"] == build["server_sha256"] == crossover["server_sha256"]
         and build["source_base"] == crossover["source_base"]
         and build["patch_sha256"] == crossover["patch_sha256"] == digest(capture_root / "source.patch"),
         "Trace and retained crossover must use the same candidate binary")
    need(capture["form_factor"] == previous["hardware"]["form_factor"]
         and capture["soc"] == previous["hardware"]["soc"]
         and capture["memory_gib"] * 1024**3 == previous["hardware"]["memory_bytes"]
         and capture["os"] == previous["hardware"]["os"] and capture["final_pressure"] == 1,
         "Trace must use the same M5 SKU and macOS release with normal final pressure")
    need(trace["inventory_sha256"] == previous["inventory_sha256"]
         == digest(capture_root / "mxfp4-inventory.json")
         and trace["publisher_members"] == 49
         and trace["manifest_before"] == trace["manifest_after"] == MANIFEST,
         "Trace used a different publisher pack")
    run = trace["run"]
    need(run["mode"] == "default" and run["final_verified_steps"] == 0
         and run["exit_code_after_stop"] == 0
         and list(map(stable, run["actions"])) == list(map(stable, prior_default["actions"])),
         "Traced lifecycle changed relative to the uninstrumented default")
    raw = (capture_root / "server-default.log").read_bytes()
    need(b"decision=resident" in raw and trace["automatic_prefetch_observed"] is True
         and b"Flash Next MTP sidecar prefetch completed bytes=5214301696" in raw,
         "Expected the resident route and pinned automatic read extent")
    geometry_path = results_root / "flash-next-trunk-mapping-geometry-20261122" / "geometry.json"
    phases, shards, parts = parse_timings(raw, read(geometry_path)["files"])
    ready = trace["ready_seconds"]
    need(phases == trace["startup_phases_microseconds"] and isfinite(ready)
         and run["ready_seconds"] == ready and ready * 10**6 >= phases["resident_tensors"] > 0,
         "Archived startup phase data contradicts the raw trace")
    snapshots = trace["vm_counters"]
    before, after = snapshots["before"], snapshots["after"]
    page_size = before["page_size"]
    keys = VM_KEYS | {"monotonic", "page_size"}
    need(before.keys() == after.keys() == keys and page_size == after["page_size"] > 0
         and all(after[key] >= before[key] >= 0 for key in VM_KEYS)
         and after["monotonic"] > before["monotonic"],
         "Invalid aggregate before/after VM counters")
    return {"scope": "Instrumented retained-pack startup and seven API lifecycle controls",
            "server_sha256": trace["server_sha256"], "binary_bound_to_crossover": True,
            "ready_seconds": ready, "matched_lifecycle_actions": len(run["actions"]),
            "geometry_sha256": digest(geometry_path),
            "resident_fraction_of_ready": round(phases["resident_tensors"] / (ready * 10**6), 4),
            "startup_phases_us": phases, "resident_timer_breakdown": parts, "shards": shards,
            "system_vm_delta_bytes": {key: (after[key] - before[key]) * page_size for key in VM_KEYS},
            "endpoint_cpu_idle_percent": [trace["cpu_idle_before"], trace["cpu_idle_after"]],
            "qualification": False, "release_ready": False}


def replay():
    root = Path(__file__).resolve().parent
    prior = root.parent / "flash-next-current-auto-reader-crossover-m5-20261123" / "comparison.json"
    with TemporaryDirectory(prefix="flash-next-startup-trace-replay-") as work:
        capture = Path(work)
        with tarfile.open(root / "evidence.tar.xz") as archive:
            entries = archive.getmembers()
            need(len(entries) == len(CAPTURE_FILES) and {entry.name for entry in entries} == set(CAPTURE_FILES)
                 and all(entry.isfile() for entry in entries), "Unexpected phase trace archive roster")
            for entry in entries:
                with archive.extractfile(entry) as stream:
                    (capture / entry.name).write_bytes(stream.read())
        return compare(capture, root.parent, read(prior))


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
