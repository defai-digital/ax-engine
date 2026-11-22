"""Replay the observed Flash Next stepwise hold telemetry and cancellation."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"
CANDIDATE = "14235e84b01456cfcb40ba6fcb717ae4798fa8f7"


def require(condition, why):
    if not condition:
        raise ValueError(why)


def archive_files(path):
    with tarfile.open(path) as archive:
        members = {}
        for info in archive:
            name = Path(info.name)
            require(info.isfile() and len(name.parts) == 1 and ".." not in name.parts
                    and info.name not in members, "Unexpected evidence member")
            with archive.extractfile(info) as stream:
                members[info.name] = stream.read()
        return members


def load_early():
    spec = importlib.util.spec_from_file_location("tail_input_replay", EARLY / "replay.py")
    require(spec and spec.loader, "Missing frozen input verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_result(contents, oracle, baseline):
    trace, plan, build = (json.loads(contents[name]) for name in
                         ("prior-trace.json", "prior-engine-decisions.json", "prior-build.json"))
    require(build == baseline and contents["prior-status.txt"] == b"1\n"
            and plan["native_trace_sha256"] == sha256(contents["prior-trace.json"]).hexdigest(),
            "The earlier failure does not bind the original software or captured trace")
    require(trace["short_at_trigger"]["output_len"] == 84
            and trace["short_at_trigger"]["output_tokens"] == oracle[0]["output_token_ids"][:84]
            and trace["long_submit"]["prompt_tokens"] == oracle[1]["prompt_token_ids"]
            and trace["long_submit"]["processed_prompt_tokens"] == 0
            and trace["capacity_rejection"] == {"status": 429, "error": "concurrency_limit"},
            "The earlier attempt used a different scheduler admission envelope")
    rows = trace["path_to_hold"]
    reported = [row["step"]["step_id"] for row in rows]
    decisions = plan["bounded_scheduler_steps"]
    require(len(rows) == 8 and all(after == before + 1 for before, after in zip(reported, reported[1:]))
            and set(decisions) == {str(i) for i in reported}, "The eight original scheduler decisions changed")
    for index, row in enumerate(rows):
        step = row["step"]
        internal = decisions[str(step["step_id"])]
        last = index == 7
        require(row["short_output_len"] == 85 + index
                and row["long_cursor"] == (512 if last else 0)
                and step["scheduled_requests"] == internal["scheduled"] == (2 if last else 1)
                and step["scheduled_tokens"] == (513 if last else 1) and step["hold"] == 0
                and internal["deferred"] == (0 if last else 1)
                and internal["memory_blocked"] == 0 and internal["admitted"] == int(index == 0),
                "The bounded scheduler/HTTP evidence disagrees with the recorded marker omission")
    return {"runtime_commit": build["runtime_commit"], "newly_admitted_long_requests": 1,
            "singleton_decode_steps_without_reported_marker": len(rows) - 1,
            "first_long_prefill_tokens": rows[-1]["long_cursor"]}


def replay(root=ROOT):
    archive = archive_files(root / "evidence.tar.xz")
    early = load_early()
    with early.inputs(EARLY) as (input_driver, files):
        cases, baseline, inventory, oracle_receipt, oracle, _ = input_driver.inputs()
        before = prior_result(archive, oracle, baseline)
        build = json.loads(archive["candidate-build.json"])
        hashes = {name: digest for digest, name in map(str.split, archive["inputs.sha256"].decode().splitlines())}
        require(archive["status.txt"] == b"0\n"
                and build["runtime_commit"] == CANDIDATE
                and build["runtime_parent"] == baseline["runtime_commit"]
                and build["codesign_verified"] is True
                and all(build[key] == baseline[key]
                        for key in ("profile", "rustc", "cargo", "mlx", "mlx_metal", "libmlx_sha256"))
                and hashes["ax-engine-server-fixed"] == build["server_sha256"],
                "Fixed binary or runtime provenance changed")
        require(hashes["input-stage.sha256"] == sha256(archive["input-stage.sha256"]).hexdigest()
                and archive["input-stage.sha256"] == (root / "input-stage.sha256").read_bytes(),
                "Frozen candidate inputs changed")
        source = (root / "run_held_cancel.py").read_bytes()
        require(hashes["run_held_cancel.py"] == sha256(source).hexdigest()
                and hashes["candidate-build.json"] == sha256(archive["candidate-build.json"]).hexdigest(),
                "Measured driver or build record changed")
        (files / "run_held_cancel.py").write_bytes(source)
        import run_held_cancel as driver

        receipt = json.loads(archive["receipt.json"])
        require(receipt["scope"] == "M2 Ultra 192 GB stepwise HTTP held prefill cancellation"
                and receipt["qualification"] is False and receipt["build"] == build
                and receipt["manifest"] == oracle_receipt["manifest"]
                and receipt["pack_revision"] == inventory["revision"]
                and receipt["inventory_sha256"] == oracle_receipt["inventory_sha256"]
                and receipt["oracle_sha256"] == oracle_receipt["solo"]["data_sha256"]
                and receipt["driver_sha256"] == hashes["run_held_cancel.py"],
                "Native request replay does not use the measured binary, inventory or oracle")
        samples = archive["pressure.jsonl"]
        pressure = [json.loads(line)["pressure"] for line in samples.splitlines()]
        raw = archive["raw.json"]
        require(receipt["raw_sha256"] == sha256(raw).hexdigest()
                and receipt["pressure_sha256"] == sha256(samples).hexdigest()
                and set(pressure) == {1} and len(pressure) == receipt["pressure_samples"],
                "Raw result or normally pressured samples changed")
        after = driver.verify(json.loads(raw), cases, oracle)
        require(after == receipt["comparison"], "The live held-cancellation result differs from replay")
    software = json.loads(archive["software.json"])
    require(software["workspace_status"] == 0 and software["rust_failed"] == 0
            and software["rust_passed"] == 4791 and software["rust_ignored"] == 107 and software["python_passed"] == 345
            and software["python_skipped"] == 7, "Pinned software gates did not pass")
    return {
        "scope": receipt["scope"], "qualification": False, "commit": build["runtime_commit"],
        "prior": before, "fixed": after, "normal_pressure_samples": len(pressure), "software": software,
    }


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT),
                     indent=2, sort_keys=True, allow_nan=False))
