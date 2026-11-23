"""Replay the qualifying Flash Next timing repeat and its startup routes."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FRESH = ROOT.parent / "flash-next-probe-followup-fresh-m5-20261123" / "evidence.tar.xz"
FROZEN = {
    "seal_native.py": "c37e52aafce58ce9894835db3d6f44a014c6c6fce6d9ab7e210dbd38784d9c92",
    "inventory.json": "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6",
    "signed-native.json": "0f97478d5d9ca0f523d4b7f90b205d7177d06036dc023642c36d8a7e005815dd",
}
FILES = {
    "driver/verify_matrix.py", "driver/verify_routes.py", "raw/build.json",
    "raw/cpu-intervals.jsonl", "raw/observed/comparison.json", "raw/route-proof.json", "raw/status.txt",
    "raw/observed/generate-matrix/raw-runs.json", "raw/observed/generate-matrix/result.json",
    *(f"raw/logs/server-{label}.log" for label in ("warm", "disabled-1", "disabled-2", "default-1", "default-2")),
}


def read(archive, path):
    member = archive.getmember(path)
    if not member.isfile():
        raise ValueError(f"Expected a regular file: {path}")
    with archive.extractfile(member) as stream:
        return stream.read()


def host_observations(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    previous = None
    for row in rows:
        deltas = row["deltas"]
        if (previous and row["previous_phase"] != previous or not 0 < row["interval_seconds"] < 15
                or set(deltas) != {"user", "system", "idle", "nice"}
                or any(type(value) is not int or value < 0 for value in deltas.values())
                or not sum(deltas.values())
                or not isclose(row["cpu_idle_percent"], 100 * deltas["idle"] / sum(deltas.values()))):
            raise ValueError("Invalid passive CPU observation")
        previous = row["phase"]
    results = {}
    for label in ("matrix", "after-last-matrix-cell"):
        group = [row for row in rows if all(row[key].startswith(label) for key in ("phase", "previous_phase"))]
        ticks = sum(sum(row["deltas"].values()) for row in group)
        if not ticks:
            raise ValueError(f"No passive observations in phase {label}")
        results[label] = {
            "observations": len(group), "seconds": sum(row["interval_seconds"] for row in group),
            "weighted_cpu_idle_percent": 100 * sum(row["deltas"]["idle"] for row in group) / ticks,
            "observations_below_90_percent": sum(row["cpu_idle_percent"] < 90 for row in group),
        }
    return results


def replay(path):
    with (tarfile.open(path) as archive, tarfile.open(FRESH) as fresh,
          TemporaryDirectory(prefix="ax-flash-next-matrix-replay-") as directory):
        names = archive.getnames()
        if set(names) != FILES or len(names) != len(FILES) or read(archive, "raw/status.txt") != b"0\n":
            raise ValueError("A repeat file is missing, duplicated or incomplete")
        if read(archive, "raw/build.json") != read(fresh, "raw/build.json"):
            raise ValueError("The timing repeat used a different signed binary")
        stage = Path(directory)
        for name in names:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read(archive, name))
        for name, expected in FROZEN.items():
            contents = read(fresh, f"driver/{name}")
            if sha256(contents).hexdigest() != expected:
                raise ValueError(f"Frozen matrix input changed: {name}")
            (stage / "driver" / name).write_bytes(contents)

        def check(script, args, evidence):
            actual = subprocess.check_output([sys.executable, "-B", stage / "driver" / script, *args],
                                             env={**os.environ, "PYTHONPATH": str(REPO)})
            result = json.loads(actual)
            if result != json.loads((stage / evidence).read_bytes()):
                raise ValueError(f"The signed repeat and replay differ: {script}")
            return result

        matrix = check("verify_matrix.py", [stage / "raw/observed/generate-matrix", stage / "raw/build.json"],
                       "raw/observed/comparison.json")
        routes = check("verify_routes.py", [stage / "raw/logs"], "raw/route-proof.json")
        passive = host_observations(read(archive, "raw/cpu-intervals.jsonl"))
    return {"matrix": matrix, "routes": routes, "passive_host_observations": passive}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    result = replay(cli.parse_args().archive)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
