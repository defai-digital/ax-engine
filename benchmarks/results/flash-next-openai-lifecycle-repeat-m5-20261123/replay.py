"""Replay the eight-round guarded OpenAI lifecycle control on M5 Max."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-openai-stream-disconnect-m2-20261122" / "evidence.tar.xz"
MIXED = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
BIN = "ax-engine-server-resident-mixed"
CORPUS = "mixed-corpus.jsonl"
DRIVER = (
    "bench_ax_serving.py", "candidate-build.json", "capture_mixed_steps.py",
    "check_flash_next_native_support.py", "check_openai_concurrency.py", "gauge-fix.patch",
    "mxfp4-inventory.json", "qualify_qwen38_flash_next.py", "replay_openai_repeat.py",
    "run_flash_next_prefill_split_probe.py", "run_openai_m5.sh", "run_openai_repeat.py",
    "run_qwen38_reference_serving.py", "run_selective_mixed.py", "split_step_route.py",
)
ROUNDS = 8
CAPTURE = (
    "baseline.json", "idle.json", "openai-samples.jsonl", "openai.log", "receipt.json",
    *(f"round-{index}.json" for index in range(1, ROUNDS + 1)),
    *(f"progress-{index}.json" for index in range(1, ROUNDS + 1)),
)


def require(valid, message):
    if not valid:
        raise ValueError(message)


def read(archive, name):
    item = archive.getmember(name)
    require(item.isfile(), f"Expected a regular archive member: {name}")
    with archive.extractfile(item) as handle:
        return handle.read()


def manifest(raw):
    lines = raw.decode().splitlines()
    values = {name: digest for digest, name in map(str.split, lines)}
    require(len(values) == len(lines), "Duplicate archive checksum")
    return values


def idle_observations(raw, receipt):
    samples = list(map(json.loads, raw.splitlines()))
    counters = receipt["idle"]["before"]
    require(len(samples) > 1 and all(row["health"] == "ok" and row["work"] == counters
                                    for row in samples), "Idle samples contain active work or unhealthy status")
    times = [row["elapsed_seconds"] for row in samples]
    duration = times[-1] - times[0]
    require(28.5 <= duration <= receipt["idle"]["elapsed_seconds"] + 0.5
            and all(0 < right - left < 2 for left, right in zip(times, times[1:]))
            and len({row["steps"] for row in samples}) == 1,
            "The cumulative step counter changed or the idle interval was not observed")
    return {"samples": len(samples), "observed_seconds": duration, "engine_steps_added": 0}


def replay(path):
    with (tarfile.open(path) as archive, tarfile.open(M2) as prior,
          tarfile.open(MIXED) as corpus, TemporaryDirectory(prefix="flash-next-openai-replay-") as directory):
        static = {name: read(archive, f"driver/{name}") for name in DRIVER}
        raw = {name: read(archive, f"raw/observed/{name}") for name in CAPTURE}
        terminal = {"raw/status.txt", "raw/idle-observer-status.txt", "raw/idle-engine-samples.jsonl",
                    "observer/observe_idle.py", "observer/sha256.txt", "raw/inputs.sha256"}
        roster = {f"driver/{name}" for name in DRIVER} | {f"raw/observed/{name}" for name in CAPTURE} | terminal
        require(set(archive.getnames()) == roster and len(archive.getnames()) == len(roster)
                and read(archive, "raw/status.txt") == read(archive, "raw/idle-observer-status.txt") == b"0\n",
                "Unexpected or incomplete OpenAI evidence archive")
        checksums = manifest(read(archive, "raw/inputs.sha256"))
        inherited = manifest(read(prior, "inputs.sha256"))
        static[CORPUS] = read(corpus, CORPUS)
        require(checksums.keys() == static.keys() | {BIN}
                and all(sha256(data).hexdigest() == checksums[name] for name, data in static.items())
                and all(checksums[name] == inherited[name] for name in checksums.keys() & inherited.keys())
                and static["candidate-build.json"] == read(prior, "candidate-build.json")
                and checksums[BIN] == json.loads(static["candidate-build.json"])["server_sha256"],
                "Frozen input, observation harness or signed executable changed")
        observer = read(archive, "observer/observe_idle.py")
        require(sha256(observer).hexdigest() == read(archive, "observer/sha256.txt").decode().strip(),
                "Idle observer source changed")
        receipt = json.loads(raw["receipt.json"])
        chat = receipt["baseline"]["chat"]
        expected = json.loads(read(prior, "receipt.json"))["baseline"]["chat"]
        require((chat["usage"], chat["finish_reason"]) == (expected["usage"], expected["finish_reason"])
                and b"decision=resident" in raw["openai.log"], "Short chat or resident execution changed")
        require(len({receipt["baseline"][case]["text"] for case in receipt["case_ids"]}) == 2,
                "The large solo replies cannot distinguish alternating survivor roles")
        observations = idle_observations(read(archive, "raw/idle-engine-samples.jsonl"), receipt)
        stage = Path(directory)
        capture = stage / "observed"
        capture.mkdir()
        for name, body in static.items():
            (stage / name).write_bytes(body)
        for name, body in raw.items():
            (capture / name).write_bytes(body)
        replayed = subprocess.check_output([sys.executable, "-B", stage / "replay_openai_repeat.py", capture])
    return {**json.loads(replayed), "idle_observations": observations}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True, allow_nan=False))
