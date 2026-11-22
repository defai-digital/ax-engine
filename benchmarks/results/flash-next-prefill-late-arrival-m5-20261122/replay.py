"""Recompute late-arrival Flash Next overlap and verify captured stream identity."""
import hashlib
import json
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
    from run_late import ARMS, TRIGGER_TOKEN, bench, layer, verify_late_trace

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
            and receipt["pack_revision"] == REVISION,
            "The capture does not bind to the tested binary and pack")
    _, cases, expected = layer.files()["ending-decoder"]
    arms = {}
    for label, candidate in ARMS:
        row = receipt["arms"][label]
        path = observed / f"{label}.json"
        require(row["data_sha256"] == digest(path), f"Changed request trace: {label}")
        raw = json.loads(path.read_bytes())
        verified = verify_late_trace(raw, candidate)
        require(all(raw[key] == value for key, value in verified.items())
                and row["layer_progress_steps"] == verified["layer_progress_steps"],
                f"Overlap summary differs: {label}")
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
        require(len(gaps) == 96 - TRIGGER_TOKEN and all(gap >= 0 for gap in gaps),
                "Incomplete short post-trigger intervals")
        delay = 1000 * (long["started_at_s"] - raw["trigger"][0]["seconds"])
        before, after = row["cpu_idle_before_percent"], row["cpu_idle_after_percent"]
        arms[label] = {
            "timing_eligible": min(before, after) >= TIMING_FLOOR and max(before, after) <= 100,
            "idle_before_percent": before, "idle_after_percent": after,
            "exact_output_ids": sum(len(x["output_token_ids"]) for x in (short, long)),
            "short_tail_gap_max_ms": max(gaps),
            "short_tail_gap_p95_ms": bench.percentile(gaps, 0.95),
            "long_offered_ttft_ms": delay + long["ttft_ms"],
            "observed_normal_pressure_samples": len(samples),
        } | verified

    pairs = []
    for index in (1, 2):
        physical, layers = arms[f"physical-{index}"], arms[f"layers-{index}"]
        pairs.append({
            "order": index,
            "timing_eligible": physical["timing_eligible"] and layers["timing_eligible"],
            "short_tail_max_gap_change_percent":
                100 * (layers["short_tail_gap_max_ms"] / physical["short_tail_gap_max_ms"] - 1),
            "long_offered_ttft_change_percent":
                100 * (layers["long_offered_ttft_ms"] / physical["long_offered_ttft_ms"] - 1),
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
