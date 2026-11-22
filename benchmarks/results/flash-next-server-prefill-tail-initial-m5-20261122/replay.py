"""Replay the primary-tier four-arm server-side decode-tail comparison."""

import ast
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"
ORIGINAL = ROOT.parent / "flash-next-server-prefill-tail-m5-20261122"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_contents(path):
    with tarfile.open(path) as archive:
        names = {}
        for item in archive:
            relative = Path(item.name)
            require(item.isfile() and not relative.is_absolute() and ".." not in relative.parts
                    and item.name not in names, "Unexpected evidence member")
            with archive.extractfile(item) as stream:
                names[item.name] = stream.read()
        return names


def read_sha256(data):
    return {name: digest for digest, name in (line.split() for line in data.decode().splitlines())}


def load_early():
    spec = importlib.util.spec_from_file_location("tail_input_replay", EARLY / "replay.py")
    require(spec and spec.loader, "Missing sealed driver verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_arm(tail, raw, samples, summary, cases, expected, enabled):
    result = tail.verify(raw, enabled)
    rows = raw["observations"]
    require(all(observed["ok"] and observed["prompt_token_ids"] == prompt.input_tokens
                and observed["output_token_ids"] == expected[prompt.id]
                for observed, prompt in zip(rows, cases, strict=True)),
            "A native arm differs from the frozen M5 solo trajectory")
    require(all(raw[k] == v for k, v in result.items())
            and raw["drained"] == dict.fromkeys(tail.mixed.COUNTERS, 0),
            "The native tail measurement or five-counter drain changed")
    pressure = Counter(row["pressure"] for row in samples)
    require(pressure, "A published request arm has no memory observations")
    peak = max((row["footprint_bytes"] or 0 for row in samples), default=0)
    require(peak > 0, "A published request arm has no measured process footprint")
    result.update(pressure_samples=dict(sorted(pressure.items())),
                  peak_process_footprint_bytes=peak, matching_output_ids=192)
    if summary is not None:
        idle = (summary["cpu_idle_before_percent"], summary["cpu_idle_after_percent"])
        require(pressure.keys() == {1} and len(samples) == summary["sample_count"]
                and summary["scheduler_tail_deferral"] is enabled
                and all(isfinite(value) and 0 <= value <= 100 for value in idle),
                "Completed arm release, pressure or idle receipt changed")
        result["cpu_idle_percent"] = idle
        result["timing_eligible"] = min(idle) >= 90
    else:
        result["timing_eligible"] = False
    return result


def replay(root=ROOT):
    contents = archive_contents(root / "evidence.tar.xz")
    previous_inputs = read_sha256((ORIGINAL / "inputs.sha256").read_bytes())
    current_inputs = read_sha256(contents["inputs.sha256"])
    distinct_build_files = {"ax-engine-server-tail", "build.json", "run_server_tail.py"}
    require(current_inputs.keys() == previous_inputs.keys()
            and {name for name in current_inputs if current_inputs[name] != previous_inputs[name]} == distinct_build_files,
            "The published prompts, shared runners or original launcher changed")
    for name in ("build.json", "run_server_tail.py"):
        require(sha256(contents[name]).hexdigest() == current_inputs[name], f"Measured input changed: {name}")
    measured_build = json.loads(contents["build.json"])
    require(measured_build["runtime_commit"] == "fce5959de69dec8d499673ce401b9c18ecbbe93a"
            and measured_build["runtime_tree"] == "14970f06bf617302af4071e89c85d81e3c7c34d1"
            and measured_build["server_sha256"] == current_inputs["ax-engine-server-tail"]
            and measured_build["driver_sha256"] == current_inputs["run_server_tail.py"],
            "The receipt did not bind the older measured native build")
    orchestration = read_sha256(contents["orchestration.sha256"])
    require(set(orchestration) == {"inputs.sha256", "await_tail.sh", "run_tail_serialized.sh"},
            "Missing dispatch provenance")
    for name, digest in orchestration.items():
        require(sha256(contents[name]).hexdigest() == digest, "Dispatch inputs changed")
    dispatch = json.loads(contents["dispatch-capacity.json"])["capacity"]
    require(dispatch["pressure"] == 1 and dispatch["eligible_for_attempt"] is False
            and dispatch["reason"] == "background-rss-exceeds-signed-margin",
            "The ordinary reference dispatch eligibility changed")
    early = load_early()
    with early.inputs(EARLY) as (driver, files):
        cases, later_build, inventory, oracle, _, _ = driver.inputs()
        import run_server_tail as tail

        for field in ("profile", "rustc", "cargo", "mlx", "mlx_metal", "libmlx_sha256"):
            require(measured_build[field] == later_build[field], f"Toolchain changed: {field}")
        def verified_functions(source):
            names = {"capture_tail", "verify", "burst"}
            return {node.name: ast.dump(node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name in names}
        measured_functions = verified_functions(contents["run_server_tail.py"])
        require(set(measured_functions) == {"capture_tail", "verify", "burst"}
                and measured_functions == verified_functions((files / "run_server_tail.py").read_bytes()),
                "The older driver's request or route verification differs")
        receipt = json.loads(contents["receipt.json"])
        require(receipt["scope"] == "M5 ordered serving comparison"
                and receipt["qualification"] is False and receipt["build"] == measured_build
                and receipt["pack_revision"] == inventory["revision"]
                and receipt["inventory_sha256"] == oracle["inventory_sha256"]
                and receipt["manifest"] == oracle["manifest"],
                "The measured model, build, or release envelope changed")
        _, _, expected = tail.layer.files()["ending-decoder"]
        arm_names = tuple(name for name, _ in tail.ARMS)
        complete = tuple(receipt["arms"])
        require(complete == arm_names[:len(complete)], "Counterbalanced process order changed")
        status = contents["status.txt"].decode().strip()
        is_complete = status == "0"
        require((is_complete and receipt["status"] == "complete" and complete == arm_names)
                or (status == "1" and receipt["status"] == "failed" and complete != arm_names),
                "The terminal retry receipt does not match its completed arms")
        results = {}
        for label, enabled in tail.ARMS:
            data, sample = f"{label}.json", f"{label}-samples.jsonl"
            if data not in contents:
                require(label not in complete, "Missing measured arm file")
                continue
            item = receipt["arms"].get(label)
            if item is not None:
                require(sha256(contents[data]).hexdigest() == item["data_sha256"],
                        "Measured arm digest changed")
            observations = [json.loads(line) for line in contents[sample].splitlines()]
            results[label] = verify_arm(tail, json.loads(contents[data]), observations,
                                        item, cases, expected, enabled)
        require(results and set(complete) <= set(results), "No replayable native observations")
        timing_pairs = {}
        for index in (1, 2):
            names = f"physical-{index}", f"held-{index}"
            if all(label in results and results[label]["timing_eligible"] for label in names):
                physical, treatment = (results[name] for name in names)
                timing_pairs[str(index)] = {
                    metric: [physical[metric], treatment[metric]] for metric in
                    ("short_tail_max_gap_ms", "long_trigger_ttft_ms", "burst_output_tokens_per_second")
                }
        return {
            "scope": receipt["scope"], "qualification": False, "runtime_commit": measured_build["runtime_commit"],
            "terminal_status": int(status), "matched_output_ids": sum(r["matching_output_ids"] for r in results.values()),
            "arms": results, "timing_eligible_pairs": timing_pairs,
        }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    print(json.dumps(replay(path), indent=2, sort_keys=True, allow_nan=False))
