"""Verify the terminal ordinary-reference capacity check."""

from datetime import datetime
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
from tarfile import open as open_tar

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parents[1] / "flash-next-native-resident-refresh-m5-20261122"
SIGNED = ROOT.parents[1] / "qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archived(path, name):
    with open_tar(path) as archive:
        member = archive.getmember(name)
        require(member.isfile(), f"Invalid archive member: {name}")
        with archive.extractfile(member) as stream:
            return stream.read()


def replay():
    bootstrap = json.loads((ROOT.parent / "boot-preflight.json").read_bytes())
    preflight = json.loads((ROOT / "reference-preflight.json").read_bytes())
    require(preflight == {k: v for k, v in bootstrap.items() if k != "capacity"},
            "The launched owner used different frozen inputs")
    signed_raw = SIGNED.read_bytes()
    require(sha256(signed_raw).hexdigest() == preflight["signed_reference_sha256"],
            "The signed memory anchor changed")
    signed = json.loads(signed_raw)
    capsule = ROOT.parent / "launcher-inputs.tar.xz"
    source = archived(capsule, "capacity.py")
    inventory = dict(line.split()[::-1] for line in archived(capsule, "inputs.sha256").decode().splitlines())
    require(sha256(source).hexdigest() == inventory["capacity.py"], "Capacity classifier changed")
    module = {}
    exec(compile(source, "capacity.py", "exec"), module)
    classify = module["recovery_capacity"]

    binding = json.loads((ROOT / "native-matrix.json").read_bytes())
    native = json.loads((NATIVE / "comparison.json").read_bytes())
    matrix = archived(NATIVE / "evidence.tar.xz", "observed/generate-matrix/result.json")
    require(binding["source_commit"] == native["build"]["commit"]
            and binding["sha256"] == native["observed_sha256"]["generate-matrix"] == sha256(matrix).hexdigest()
            and json.loads(matrix)["hardware"]["os"] == preflight["macos"],
            "The ordinary-reference owner did not bind the archived native workload")
    require((ROOT / "status.txt").read_text() == "2\n"
            and json.loads((ROOT / "outcome.json").read_bytes()) ==
                {"phase": "capacity-wait-expired", "launched": False},
            "The recovery did not end before launch")
    rows = [json.loads(line) for line in decompress((ROOT / "headroom.jsonl.gz").read_bytes()).splitlines()]
    moments = [datetime.fromisoformat(row["utc"]) for row in rows]
    require(moments and all(a < b for a, b in zip(moments, moments[1:])), "Unordered capacity trace")
    wall_seconds = (moments[-1] - moments[0]).total_seconds()
    require(wall_seconds >= 1800, "The capacity wait ended before the planned window")
    for row in rows:
        decision = classify(signed["hardware"]["memory_bytes"], signed["peak_mlx_memory_bytes"],
                            row["pressure"], row["largest_other_process_rss_bytes"])
        require(all(row[key] == value for key, value in decision.items())
                and decision["reason"] == "background-rss-exceeds-signed-margin",
                "A recorded wait sample was eligible or had another reason")
    limits = [row["largest_background_rss_bytes"] for row in rows]
    return {
        "scope": "capacity check before ordinary reference launch",
        "ordinary_reference_launched": False,
        "qualification": False,
        "os": preflight["macos"],
        "samples": len(rows),
        "wall_seconds": wall_seconds,
        "sampled_normal_pressure": len(rows),
        "eligible_samples": 0,
        "observed_signed_margin_bytes": rows[0]["observed_signed_margin_bytes"],
        "background_rss_bytes_min": min(limits),
        "background_rss_bytes_max": max(limits),
        "native_matrix_sha256": binding["sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
