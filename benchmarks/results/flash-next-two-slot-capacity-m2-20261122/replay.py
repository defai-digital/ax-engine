"""Replay the Flash Next two-slot saturation and reuse control."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"


def sums(path):
    return dict(line.split()[::-1] for line in path.read_text().splitlines())


def replay():
    spec = importlib.util.spec_from_file_location("prior_run", FROZEN / "replay.py")
    if spec is None or spec.loader is None:
        raise ValueError("Missing frozen native prompt verifier")
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    need, digest, read = prior.require, prior.digest, prior.read
    hashes = sums(ROOT / "inputs.sha256")
    for name in ("run_capacity.py", "run_m2.sh", "build.json"):
        need(hashes.pop(name) == digest(ROOT / name), f"Capacity stage input changed: {name}")
    build = read(ROOT / "build.json")
    need(hashes.pop("ax-engine-server") == build["server_sha256"]
         and build["profile"] == "release-server" and build["commit"].startswith("38c90b27a"),
         "The active product version was not tested")
    old = sums(FROZEN / "stage.sha256")
    for name in ("ax-engine-server-tail", "run_early.py", "run_m2.sh", "build.json", "oracle.json", "oracle-receipt.json"):
        old.pop(name)
    need(old == hashes, "The pinned benchmark drivers or frozen prompts changed")

    with prior.inputs(FROZEN) as (baseline_driver, temp):
        (temp / "run_capacity.py").write_bytes((ROOT / "run_capacity.py").read_bytes())
        import run_capacity as capacity

        cases, _, inventory, prior_receipt, _, _ = baseline_driver.inputs()
        names = ("receipt.json", "status.txt", "solo.json", "burst-1.json", "burst-2.json", "capacity-samples.jsonl")
        prior.unpack(ROOT / "evidence.tar.xz", temp, {name: name for name in names})
        receipt = read(temp / "receipt.json")
        need((temp / "status.txt").read_text() == "0\n"
             and receipt["scope"] == "M2 Ultra 192 GB functional native HTTP two-slot capacity"
             and receipt["qualification"] is False and receipt["build"] == build
             and receipt["pack_revision"] == inventory["revision"] and receipt["manifest"] == prior_receipt["manifest"]
             and build["inventory_sha256"] == old["mxfp4-inventory.json"] == prior_receipt["inventory_sha256"],
             "The run did not bind the current binary and pinned pack")
        samples_path = temp / "capacity-samples.jsonl"
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        need(digest(samples_path) == receipt["pressure_sha256"] and samples
             and len(samples) == receipt["normal_pressure_samples"]
             and all(row["pressure"] == 1 for row in samples), "Memory pressure evidence changed")
        baseline_file = temp / "solo.json"
        need(digest(baseline_file) == receipt["baseline_sha256"], "The same-process serial controls changed")
        baseline = read(baseline_file)
        for oracle, case in zip(baseline, cases, strict=True):
            row, trace = oracle["request"], oracle["trace"]
            need(row["ok"] and row["prompt_token_ids"] == case.input_tokens
                 and len(row["output_token_ids"]) == 96 and trace["initial_cursor"] == 0
                 and trace["cursor"] == len(case.input_tokens)
                 and all(step["scheduled"] == 1 for step in trace["prefill"]),
                 "The serial baseline did not run unshared")

        result = {}
        need([row["name"] for row in receipt["arms"]] == ["burst-1.json", "burst-2.json"],
             "Missing repeated native capacity trials")
        for arm in receipt["arms"]:
            path = temp / arm["name"]
            need(digest(path) == arm["sha256"], "The recorded native HTTP trace changed")
            raw = read(path)
            verdict = capacity.verify(raw, cases, baseline)
            need(all(arm[key] == value for key, value in verdict.items()), "Native capacity summary changed")
            long = raw["observations"][1]["trace"]
            replacement = raw["observations"][2]["trace"]
            long_work = set(long["decode_step_ids"]) | {s["step_id"] for s in long["prefill"]}
            reuse_overlap = [s["step_id"] for s in replacement["prefill"]
                             if s["scheduled"] == 2 and s["step_id"] in long_work]
            need(reuse_overlap, "The replacement did not share a scheduler step with the live long request")
            result[arm["name"]] = verdict | {"replacement_overlap_steps": reuse_overlap}

        return {"scope": receipt["scope"], "qualification": False, "runtime_commit": build["commit"],
                "normal_pressure_samples": len(samples), "bursts": result}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
