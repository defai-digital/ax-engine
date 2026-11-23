"""Rebuild the Flash Next release board from the archived M5 receipts."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
REPO = ROOT.parents[2]
FRESH = RESULTS / "flash-next-probe-followup-fresh-m5-20261123"
RETAINED = RESULTS / "flash-next-probe-followup-retained-m5-20261122"
QA = RESULTS / "flash-next-probe-followup-qa-m5-20261122"
LIFECYCLE = RESULTS / "flash-next-probe-followup-lifecycle-m5-20261122"
REFERENCE = RESULTS / "flash-next-ordinary-reference-m5-20261121" / "partial-result.json"
REFERENCE_SHA256 = "868bdf9cd3937b3b49323e75fe38979a206ba16e8d529efa54c016820322673d"
PINS = {
    "check_flash_next_release_gates.py": "3041a6fbd8099193b32c817ca830661b07a5da0e400a6523f9d05212b5631ead",
    "flash_next_release_thresholds.json": "17a39e0229cf734dca04ffa5abb4718a6c7b1c9c773891385a37396a1625345b",
    "qualify_qwen38_flash_next.py": "ff020ed48cdc87e56d56a3213f0485e56a2dbf7c1f1c60587d0174c5fc9f0b4c",
}


def copy_member(archive, member, destination):
    with archive.extractfile(member) as stream:
        contents = stream.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)
    return contents


def verify_artifacts(fresh, retained, qa, lifecycle):
    for folder in (fresh, retained, qa, lifecycle):
        actual = subprocess.check_output([sys.executable, "-B", folder / "replay.py"])
        if actual != (folder / "comparison.json").read_bytes():
            raise ValueError(f"Archived evidence did not replay: {folder.name}")


def assemble(target, fresh, retained, qa, lifecycle, reference):
    with (tarfile.open(fresh / "evidence.tar.xz") as cold,
          tarfile.open(retained / "evidence.tar.xz") as warm,
          tarfile.open(qa / "evidence.tar.xz") as cohort):
        build_bytes = copy_member(cold, "raw/build.json", target / "build.json")
        with warm.extractfile("build.json") as stream:
            if stream.read() != build_bytes:
                raise ValueError("Fresh and retained evidence used different signed binaries")
        for name in ("result.json", "native-support/result.json"):
            copy_member(cold, f"raw/delivery/{name}", target / "delivery" / name)
        for name in ("result.json", "raw-runs.json"):
            copy_member(warm, f"observed/generate-matrix/{name}", target / "generate-matrix" / name)
        copy_member(cohort, "result.json", target / "qa" / "result.json")
        for member in cohort.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 2 and parts[0] in ("default", "disabled") and parts[1].endswith(".json"):
                copy_member(cohort, member, target / "qa" / member.name)
    for source, subdir in ((lifecycle / "lifecycle.json", "lifecycle"), (reference, "reference")):
        path = target / subdir / "result.json"
        path.parent.mkdir()
        path.write_bytes(source.read_bytes())
    return json.loads(build_bytes)


def replay(fresh=FRESH, retained=RETAINED, qa=QA, lifecycle=LIFECYCLE, reference=REFERENCE):
    scripts = REPO / "scripts"
    for name, digest in PINS.items():
        if sha256((scripts / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The frozen release input changed: {name}")
    if sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The archived partial reference changed")
    verify_artifacts(fresh, retained, qa, lifecycle)
    with TemporaryDirectory(prefix="ax-flash-next-board-") as directory:
        target = Path(directory)
        build = assemble(target, fresh, retained, qa, lifecycle, reference)
        verdict_path = target / "verdict.json"
        run = subprocess.run([sys.executable, "-B", scripts / "check_flash_next_release_gates.py",
                              "--evidence-dir", target, "--output", verdict_path],
                             capture_output=True)
        if run.returncode not in (0, 1) or not verdict_path.is_file():
            raise RuntimeError(run.stderr.decode())
        verdict = json.loads(verdict_path.read_bytes())
        if verdict["release_ready"] != (run.returncode == 0):
            raise ValueError("The release gate exit code disagrees with its verdict")
    del verdict["source"]
    return {**verdict, "native_build_commit": build["commit"],
            "native_server_sha256": build["server_sha256"], "frozen_inputs_sha256": PINS}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
