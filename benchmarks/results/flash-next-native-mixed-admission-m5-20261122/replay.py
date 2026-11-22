"""Recompute the guarded M5 dual-large queue and short/long controls."""
from hashlib import sha256
import importlib.util
import json
from math import isfinite
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
EARLY = ROOT.parent / "flash-next-server-prefill-early-m2-20261122"
CANCEL = ROOT.parent / "flash-next-stepwise-held-cancel-m2-20261122"
SUSTAINED = ROOT.parent / "flash-next-sustained-tail-deferral-m5-20261122"
CORPUS = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
COMMIT = "8cb06414cc4dc3bf5d3b2de24a3f6182435fa3ca"
TREE = "4826a57dc7eb5f9121225edae6981ad824c37e01"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read_archive(archive_path, requested=None):
    results = {}
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if requested is not None and member.name not in requested:
                continue
            require(member.isfile() and Path(member.name).name == member.name
                    and member.name not in results, "Unexpected evidence path")
            with archive.extractfile(member) as source:
                results[member.name] = source.read()
    return results


def hashes(data):
    pairs = [line.split() for line in data.decode().splitlines()]
    result = {name: checksum for checksum, name in pairs}
    require(len(result) == len(pairs), "Duplicate input")
    return result


def load_early():
    spec = importlib.util.spec_from_file_location("resident_mixed_inputs", EARLY / "replay.py")
    require(spec and spec.loader, "Missing frozen workload controls")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_queue(raw, cases, expected, counters):
    a, b = (reply["request_id"] for reply in raw["submissions"])
    require(a != b and [reply["prompt_tokens"] for reply in raw["submissions"]] == [
            case.input_tokens for case in cases], "The two accepted prompts or IDs changed")
    require(len(raw["frames"]) == 2, "The observed waiting sequence has a different length")
    steps = []
    for cursor, entry in zip((2048, 4096), raw["frames"], strict=True):
        step = entry["step"]
        active, waiting = entry["active"], entry["waiting"]
        require(step["scheduled_requests"] == 1 and step["scheduled_tokens"] == 2048
                and step["hold"] == 0 and active["request_id"] == a
                and active["processed_prompt_tokens"] == cursor
                and waiting["request_id"] == b and waiting["state"] == "waiting"
                and waiting["processed_prompt_tokens"] == waiting["output_len"] == 0,
                "The second large prompt acquired execution before release")
        steps.append(step["step_id"])
    cancel, release, promoted = (raw[k] for k in ("cancel", "release", "promoted"))
    terminal = raw["terminal"]
    require(cancel["request_id"] == a and cancel["state"] == "cancelled"
            and release["step_id"] > steps[-1] > steps[0]
            and release["scheduled_requests"] == 1 and release["scheduled_tokens"] == 2048
            and promoted["request_id"] == terminal["request_id"] == b
            and promoted["processed_prompt_tokens"] == 2048 and promoted["state"] != "waiting"
            and terminal["state"] == "finished" and terminal["processed_prompt_tokens"] == 8192
            and terminal["output_tokens"] == expected[cases[1].id]
            and raw["drained"] == dict.fromkeys(counters, 0),
            "Release, promotion, terminal output or resource drainage changed")
    return {"waiting_steps": steps, "promotion_step": release["step_id"], "matching_generated_ids": 96}


def replay():
    data = read_archive(ROOT / "evidence.tar.xz")
    prior = read_archive(SUSTAINED / "evidence.tar.xz", {"sustained-inputs.sha256", "run_sustained_tail.py"})
    corpus = read_archive(CORPUS, {"mixed-corpus.jsonl"})["mixed-corpus.jsonl"]
    before = hashes(prior["sustained-inputs.sha256"])
    measured = hashes(data["admission-inputs.sha256"])
    shared = before.keys() & measured.keys() - {"run_m5.sh", "await_tail.sh"}
    require(len(shared) >= 18 and all(before[name] == measured[name] for name in shared)
            and all(digest(data[name]) == measured[name] for name in (
                "candidate-build.json", "verify_mixed_admission.py", "run_m5.sh",
                "await_tail.sh", "run_tail_serialized.sh"))
            and digest(corpus) == measured["mixed-corpus.jsonl"]
            and digest((EARLY/"run_early.py").read_bytes()) == measured["run_early.py"]
            and digest((CANCEL/"run_held_cancel.py").read_bytes()) == measured["run_held_cancel.py"]
            and digest(prior["run_sustained_tail.py"]) == measured["run_sustained_tail.py"],
            "The paired workload or installed driver changed")
    build = json.loads(data["candidate-build.json"])
    require(build["runtime_commit"] == COMMIT and build["runtime_tree"] == TREE
            and build["codesign_verified"] and build["profile"] == "release-server"
            and build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["rustc"].startswith("rustc 1.99.1") and build["cargo"].startswith("cargo 1.99.1")
            and build["server_sha256"] == measured["ax-engine-server-resident-mixed"]
            and build["driver_sha256"] == measured["verify_mixed_admission.py"], "Exact executable changed")
    r = json.loads(data["receipt.json"])
    capacity = json.loads(data["dispatch-capacity.json"])["capacity"]
    samples = [json.loads(row) for row in data["resident-mixed-admission-samples.jsonl"].splitlines()]
    require(data["status.txt"] == b"0\n" and r["qualification"] is False and r["build"] == build
            and capacity["pressure"] == 1 and capacity["eligible_for_attempt"] is False
            and samples and {row["pressure"] for row in samples} == {1}
            and len(samples) == r["normal_pressure_samples"] and len(r["cpu_idle_percent"]) == 2
            and all(isfinite(x) and 0 <= x <= 100 for x in r["cpu_idle_percent"]),
            "Completion, dispatch, or measured host envelope changed")
    red = json.loads(data["c7-native-failure.json"])
    require(red["build"]["runtime_commit"] == "c7f82e3009376f79777eff7fab9af6b4b679204c"
            and red["build"]["codesign_verified"] is True
            and red["configured_live_sequences"] == 2 and red["priced_live_sequence_cap"] is None
            and red["exit_status"] == 1 and red["normal_pressure_samples"] > 0,
            "The first native run does not reproduce the uncapped memory plan")
    before = red["frame"]
    require(before["waiting"]["state"] == "runnable"
            and before["waiting"]["processed_prompt_tokens"] == before["waiting"]["output_len"] == 0
            and before["active"]["processed_prompt_tokens"] == 2048,
            "Missing original zero-progress early admission")
    early = load_early()
    with early.inputs(EARLY) as (_, folder):
        files = {"run_held_cancel.py": (CANCEL/"run_held_cancel.py").read_bytes(),
                 "verify_mixed_admission.py": data["verify_mixed_admission.py"],
                 "run_sustained_tail.py": prior["run_sustained_tail.py"], "mixed-corpus.jsonl": corpus}
        for name, content in files.items():
            (folder / name).write_bytes(content)
        import verify_mixed_admission as driver
        from run_server_tail import verify as verify_hold

        pair, oracle, large, expected = driver.inputs()
        require([before[k]["prompt_tokens"] for k in ("active", "waiting")] ==
                [case.input_tokens for case in large], "The repro and retry use different large prompts")
        queue = verify_queue(r["queue"], large, expected, driver.mixed.COUNTERS)
        held = r["existing_held_path"]
        result = verify_hold(held, True)
        require(all(held[k] == v for k, v in result.items())
                and all(row["prompt_token_ids"] == case.input_tokens
                        and row["output_token_ids"] == oracle[case.id]
                        for row, case in zip(held["observations"], pair, strict=True))
                and held["drained"] == dict.fromkeys(driver.mixed.COUNTERS, 0),
                "The retained supported pair changed trajectory or drainage")
    return {"scope": r["scope"], "qualification": False, "build": build, "queue": queue,
            "held": result, "matching_generated_ids": queue["matching_generated_ids"] + 192,
            "normal_pressure_samples": len(samples), "cpu_idle_percent": r["cpu_idle_percent"]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
