"""Replay immediate and deferred Flash Next dispatch with trigger-relative timing."""
import hashlib
import json
from math import isfinite
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
TIMING_FLOOR = 90


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(path, files):
    with tarfile.open(path) as archive:
        for member in archive:
            relative = Path(member.name)
            target = files / relative
            require(member.isfile() and not relative.is_absolute() and ".." not in relative.parts
                    and not target.exists(), "Unexpected evidence member")
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(archive.extractfile(member).read())


def summarize(files):
    sys.path.insert(0, str(files))
    from run_late_deferral import ARMS, TRIGGER_TOKEN, layer, mixed, verify_deferral

    hashes = {name: checksum for checksum, name in
              (line.split() for line in (files / "inputs.sha256").read_text().splitlines())}
    for name, checksum in hashes.items():
        if name != "ax-engine-server-layer-http":
            require(digest(files / name) == checksum, f"Changed driver input: {name}")
    observed = files / "observed"
    receipt = json.loads((observed / "receipt.json").read_bytes())
    build = json.loads((files / "build.json").read_bytes())
    require((observed / "status.txt").read_text() == "0\n" and receipt["status"] == "complete",
            "The campaign did not complete")
    require(receipt["build"] == build
            and build["server_sha256"] == hashes["ax-engine-server-layer-http"]
            and receipt["inventory_sha256"] == hashes["mxfp4-inventory.json"]
            and receipt["manifest"]["model-manifest.json"] == MANIFEST
            and receipt["pack_revision"] == REVISION and receipt["qualification"] is False
            and list(receipt["arms"]) == [label for label, _ in ARMS],
            "The capture does not bind to the tested binary and pack")
    _, cases, expected = layer.files()["ending-decoder"]
    arms = {}
    for label, deferred in ARMS:
        row = receipt["arms"][label]
        path = observed / f"{label}.json"
        require(row["data_sha256"] == digest(path), f"Changed request trace: {label}")
        raw = json.loads(path.read_bytes())
        verified = verify_deferral(raw, deferred)
        require(all(raw[key] == value for key, value in verified.items())
                and row["deferral"] is deferred and raw["drained"] == dict.fromkeys(mixed.COUNTERS, 0),
                f"Dispatch summary or drained counters differ: {label}")
        for request, case in zip(raw["observations"], cases, strict=True):
            require(request["ok"] and request["prompt_token_ids"] == case.input_tokens
                    and request["output_token_ids"] == expected[case.id],
                    f"Prompt or greedy token output changed: {label}")
        samples = [json.loads(line) for line in
                   (observed / f"{label}-samples.jsonl").read_text().splitlines()]
        require(len(samples) == row["sample_count"] and {s["pressure"] for s in samples} == {1},
                f"Unexpected sample count or pressure: {label}")
        short, long = raw["observations"]
        gaps = short["stream_step_interval_ms"][TRIGGER_TOKEN - 1:]
        require(len(gaps) == 96 - TRIGGER_TOKEN and all(isfinite(gap) and gap >= 0 for gap in gaps)
                and verified["long_offered_ttft_ms"] > 0 and verified["long_dispatch_wait_ms"] >= 0,
                "Invalid short intervals or trigger-relative long latency")
        start = short["started_at_s"]
        ends = [obs["started_at_s"] + obs["e2e_latency_ms"] / 1000 for obs in (short, long)]
        require(all(isfinite(end) and end > start for end in ends), "Invalid burst completion time")
        burst_seconds = max(ends) - start
        outputs = sum(len(obs["output_token_ids"]) for obs in (short, long))
        before, after = row["cpu_idle_before_percent"], row["cpu_idle_after_percent"]
        arms[label] = {
            "timing_eligible": min(before, after) >= TIMING_FLOOR and max(before, after) <= 100,
            "idle_before_percent": before, "idle_after_percent": after,
            "exact_output_ids": outputs,
            "burst_seconds": burst_seconds,
            "burst_output_tokens_per_second": outputs / burst_seconds,
            "short_tail_service_ms": sum(gaps),
            "observed_normal_pressure_samples": len(samples),
            "peak_footprint_bytes": max(s["footprint_bytes"] for s in samples),
        } | verified

    pairs = []
    for index in (1, 2):
        immediate, deferred = arms[f"physical-{index}"], arms[f"deferred-{index}"]
        pairs.append({
            "order": index,
            "timing_eligible": immediate["timing_eligible"] and deferred["timing_eligible"],
            "short_tail_max_gap_change_percent":
                100 * (deferred["short_tail_max_gap_ms"] / immediate["short_tail_max_gap_ms"] - 1),
            "short_tail_service_change_ms": deferred["short_tail_service_ms"] - immediate["short_tail_service_ms"],
            "burst_output_rate_change_percent":
                100 * (deferred["burst_output_tokens_per_second"] / immediate["burst_output_tokens_per_second"] - 1),
            "long_extra_dispatch_wait_ms": deferred["long_dispatch_wait_ms"] - immediate["long_dispatch_wait_ms"],
            "long_offered_ttft_change_percent":
                100 * (deferred["long_offered_ttft_ms"] / immediate["long_offered_ttft_ms"] - 1),
        })
    return {"qualification": False, "arms": arms, "pairs": pairs}


def replay(root):
    with TemporaryDirectory() as temp:
        files = Path(temp)
        unpack(root / "evidence.tar.xz", files)
        return summarize(files)


if __name__ == "__main__":
    result = replay(ROOT)
    final = ROOT / "comparison.json"
    if final.exists():
        require(result == json.loads(final.read_bytes()), "Published comparison differs")
    print(json.dumps(result, indent=2, allow_nan=False))
