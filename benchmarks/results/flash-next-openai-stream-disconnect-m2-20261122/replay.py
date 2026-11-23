"""Replay the signed guarded-build OpenAI SSE capacity and recovery control."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
JOBS = "ax_engine_jobs_in_flight"
STREAMS = "ax_engine_generation_active_streams"
WAITING = "ax_engine_step_waiting_requests"
STEP = "ax_engine_steps_total"
SCHEDULED = "ax_engine_step_scheduled_requests"
COUNTERS = (
    JOBS, "ax_engine_generation_jobs_pending", "ax_engine_generation_commands_queued",
    STREAMS, "ax_engine_generation_buffered_stream_events",
)
CORPUS = "2c0c205d803207df6316dbd0076eef97451a55c80c0413fdc6b3bee5ffef7e9a"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def require(valid, condition):
    if not valid:
        raise ValueError(condition)


def member(tar, name):
    record = tar.getmember(name)
    require(record.isfile(), "Evidence member is not a regular file")
    return tar.extractfile(record).read()


def digest(value):
    return sha256(value).hexdigest()


def stable(stream):
    usage = stream["usage"]
    require(stream["text"] and stream["finish_reason"] in ("stop", "length")
            and set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
            and all(type(x) is int and x > 0 for x in usage.values())
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "Invalid OpenAI terminal response")
    return {k: stream[k] for k in ("text", "finish_reason", "usage")}


def replay(bundle, repo):
    bench = repo / "benchmarks" / "results"
    guard = bench / "flash-next-native-mixed-admission-m5-20261122" / "comparison.json"
    frozen = bench / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
    with tarfile.open(bundle) as stream, tarfile.open(frozen) as earlier:
        report = json.loads(member(stream, "receipt.json"))
        build_raw = member(stream, "candidate-build.json")
        build = json.loads(build_raw)
        inputs = {name: sha for sha, name in map(str.split, member(stream,"inputs.sha256").decode().splitlines())}
        prior = json.loads(guard.read_bytes())["build"]
        red = json.loads(member(stream, "native-gauge-red.json"))
        test = json.loads(member(stream, "deterministic-regression.json"))
        patch = digest(member(stream, "gauge-fix.patch"))
        old_inputs = {name: sha for sha, name in map(
            str.split, member(stream, "original-inputs.sha256").decode().splitlines())}
        variant = ("ax-engine-server-resident-mixed", "candidate-build.json")
        require(member(stream, "status.txt") == b"0\n" and build == report["build"]
                and set(inputs) == old_inputs.keys() | {"gauge-fix.patch"}
                and all(inputs[name] == sha for name, sha in old_inputs.items() if name not in variant)
                and build["prior_server_sha256"] == prior["server_sha256"] == old_inputs[variant[0]]
                and build["server_sha256"] == inputs[variant[0]] != build["prior_server_sha256"]
                and {key: value for key, value in build.items() if key in prior and key != "server_sha256"}
                    == {key: value for key, value in prior.items() if key != "server_sha256"}
                and patch == build["stream_gauge_patch_sha256"] == inputs["gauge-fix.patch"]
                    == test["guarded_source_patch_sha256"]
                and test["old_gauge"] == 2 and test["expected_gauge"] == 1
                and test["focused_replay_passed"] is True and test["service_tests_passed"] == 34
                and all(digest(member(stream, name)) == inputs[name] for name in (
                    "candidate-build.json", "mxfp4-inventory.json", "check_openai_concurrency.py")),
                "Original executable, gauge-only patch or M2 retry driver changed")
        old_state = red["final_polled_gauges"]
        require(red["terminal_status"] == 1 and red["build"] == prior
                and old_state[JOBS] == 1 and old_state[STREAMS] == 2
                and old_state[STEP] > red["observed"]["observed_queue"][STEP]
                and red["normal_pressure_samples"] == 99,
                "The original active-stream gauge discrepancy is missing")
        corpus = member(earlier, "mixed-corpus.jsonl")
        require(digest(corpus) == inputs["mixed-corpus.jsonl"] == CORPUS, "Frozen request inputs changed")
        large = [row for row in map(json.loads, corpus.splitlines()) if len(row["input_tokens"]) == 8192]
        require(report["frozen_case_ids"] == [large[i]["id"] for i in (0, 2)]
                and report["manifest"] == {"model-manifest.json": MANIFEST} and report["qualification"] is False,
                "The original two-large case or publisher manifest changed")
        samples = list(map(json.loads, member(stream,"openai-samples.jsonl").splitlines()))
        require(len(samples) == report["normal_pressure_samples"] > 0 and {row["pressure"] for row in samples} == {1},
                "Functional memory-pressure observations changed")
    raw = report["result"]
    original, queued, before, after = [raw[k] for k in (
        "initial","observed_queue","before_disconnect","after_disconnect")]
    require(original[JOBS] == original[STREAMS] == 1
            and queued[JOBS] == queued[STREAMS] == before[JOBS] == before[STREAMS] == 2
            and queued[WAITING] == queued[SCHEDULED] == 1 and queued[STEP] > original[STEP]
            and after[JOBS] == after[STREAMS] == 1
            and raw["overload"] == {"status":429,"code":"concurrency_limit"}
            and raw["disconnected"]["closed_after_first_content_event"] is True
            and raw["drained"] == dict.fromkeys(COUNTERS,0),
            "Shared capacity, waiting, disconnect, or resource drainage changed")
    baseline = report["baseline"]
    require(stable(raw["survivor"]) == stable(baseline["completion"])
            and stable(raw["replacement"]) == stable(baseline["chat"])
            and raw["survivor"]["usage"] == {"prompt_tokens":8192,"completion_tokens":96,"total_tokens":8288}
            and len({raw[name]["request_id"] for name in ("disconnected", "survivor", "replacement")}) == 3,
            "OpenAI streaming reply, usage, or request identity changed")
    return {"build":build,"qualification":False,"observed_waiting":queued[WAITING],
            "old_stream_gauge_after_disconnect":old_state[STREAMS],
            "corrected_stream_gauge_after_disconnect":after[STREAMS],
            "overload":raw["overload"],"normal_pressure_samples":len(samples),
            "survivor_prompt_tokens":8192,"survivor_completion_tokens":96,
            "identical_solo_response_text_and_usage":2,"all_work_counters_drained":True}


if __name__ == "__main__":
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--repo",type=Path,default=ROOT.parents[2])
    cli.add_argument("--bundle",type=Path,default=ROOT/"evidence.tar.xz")
    args=cli.parse_args()
    print(json.dumps(replay(args.bundle,args.repo),indent=2,sort_keys=True,allow_nan=False))
