"""Check both process orders of the native Flash Next logit-selection trial."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / "flash-next-logit-native-m5-20261123"
ROLES = ("baseline", "candidate")
FIELDS = ("prompt_sha256", "output_tokens", "finish_reason", "verifier_schedule", "verified_steps")
COUNTERS = ("accepted_steps", "verified_steps", "emitted_tokens")


def observations(path):
    with tarfile.open(path) as archive:
        result = []
        for label in ROLES:
            with archive.extractfile(f"raw/observed-{label}/rows-required.jsonl") as data:
                rows = [json.loads(line) for line in data]
            result.append({row["id"]: ([row[key] for key in FIELDS],
                                       [row["counters"][key] for key in COUNTERS]) for row in rows})
    return result


def replay(reverse):
    script = FIRST / "replay.py"
    first = FIRST / "evidence.tar.xz"
    forward_bytes = subprocess.check_output([sys.executable, "-B", script])
    if forward_bytes != (FIRST / "comparison.json").read_bytes():
        raise ValueError("Baseline-first replay changed")
    forward = json.loads(forward_bytes)
    backward = json.loads(subprocess.check_output([sys.executable, "-B", script, "--archive", reverse]))
    expected, *other = observations(first) + observations(reverse)
    if (forward["order"] != list(ROLES) or backward["order"] != list(reversed(ROLES))
            or forward["source_builds"] != backward["source_builds"]
            or len(expected) != 10 or any(observed != expected for observed in other)
            or forward["matched_output_tokens"] != backward["matched_output_tokens"]):
        raise ValueError("Process order changed an executable, prompt, output, finish, or verifier count")
    return {"timing_eligible": forward["timing_eligible"] and backward["timing_eligible"],
            "matched_output_tokens_per_arm": backward["matched_output_tokens"],
            "baseline_first": forward, "candidate_first": backward}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True, allow_nan=False))
