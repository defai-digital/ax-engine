"""Replay the consolidated Flash Next native profile through the frozen release evaluator."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from runpy import run_path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SERVER = "24eeca1e3ce4a1015c3a9c4581ac6931ad6df023195abd2d72c29975e4bb3efd"
CANDIDATE = "3fc0a02cd75ec05226f87a1a9e47a00ac1389d31"
PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MODES = ("disabled", "default")
EVIDENCE = ("delivery/result.json", "delivery/native-support/result.json", "lifecycle/result.json",
            "qa/result.json", "generate-matrix/result.json", "generate-matrix/raw-runs.json")
STAGED = ("status.txt", "build.json", "inputs.sha256", "inventory.json", "dry-run.json",
          "run_resumed_native.sh", "resume-inputs.sha256", "signed-native.json", "seal_native.py")
QA_ROW = re.compile(r"qa/(?:default|disabled)/[0-9]{3}-[a-z0-9_-]+\.json")


def digest(data):
    return sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def materialize(archive_path, directory):
    expected = set(STAGED) | set(EVIDENCE) | {"delivery/download.ndjson", "portable-sha256.json"}
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = {item.name for item in members}
        need(len(names) == len(members) and expected <= names, "An archive member is missing or repeated")
        qa = names - expected
        need(len(qa) == 210 and all(QA_ROW.fullmatch(name) for name in qa), "Unexpected QA row inventory")
        need(all(len([name for name in qa if name.startswith(f"qa/{mode}/")]) == 105 for mode in MODES),
             "Incomplete QA arm")
        for item in members:
            relative = PurePosixPath(item.name)
            need(item.isfile() and not relative.is_absolute() and ".." not in relative.parts,
                 "Unsafe report member")
            with archive.extractfile(item) as handle:
                body = handle.read()
            need(not any(token in body for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1")),
                 f"Private runtime identifiers in {item.name}")
            path = directory / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)


def read(directory, name):
    return json.loads((directory / name).read_bytes())


def check_native(directory):
    build = read(directory, "build.json")
    inventory = digest((directory / "inventory.json").read_bytes())
    need((directory / "status.txt").read_bytes() == b"0\n", "The resumed workflow did not complete")
    need(build["commit"] == CANDIDATE and build["server_sha256"] == SERVER
         and build["profile"] == "release-server" and build["codesign_verified"] is True,
         "Unexpected signed native build")
    need(build["rustc"].startswith("rustc 1.99.1 ") and build["mlx"] == build["mlx_metal"] == "0.32.4",
         "Frozen native toolchain changed")
    inputs = {name: checksum for checksum, name in
              (row.split() for row in (directory / "inputs.sha256").read_text().splitlines())}
    for name in ("build.json", "dry-run.json", "inventory.json", "signed-native.json", "seal_native.py"):
        need(digest((directory / name).read_bytes()) == inputs[f"./{name}"], f"Bundle input changed: {name}")
    need(inputs["./ax-engine-server"] == SERVER and inputs["./ax-engine"] == build["cli_sha256"],
         "Binary manifest and run provenance differ")
    resume_checksum = digest((directory / "run_resumed_native.sh").read_bytes())
    need((directory / "resume-inputs.sha256").read_text() == f"{resume_checksum}  run_resumed_native.sh\n",
         "Resumed launcher changed")
    portable = read(directory, "portable-sha256.json")
    need(set(portable) <= {"delivery/result.json", "delivery/download.ndjson"}, "Unexpected path normalization")
    for key, changes in portable.items():
        need(changes["portable"] == digest((directory / key).read_bytes())
             and re.fullmatch(r"[a-f0-9]{64}", changes["original"]) is not None, "Invalid portability receipt")
    for name in EVIDENCE[:-1]:
        leg = read(directory, name)
        need(leg["server_sha256"] == SERVER and leg["inventory_sha256"] == inventory and leg["revision"] == PIN,
             f"Source, inventory or revision changed: {name}")
        need(leg["hardware"]["soc"] == "Apple M5 Max" and leg["hardware"]["memory_bytes"] == 128 * 2**30
             and leg["hardware"]["os"] == "27.2.1" and leg["hardware"]["storage"]["internal"] is True,
             f"Hardware scope changed: {name}")
    delivery = read(directory, "delivery/result.json")
    controls = read(directory, "delivery/native-support/result.json")
    need(delivery["completed"] and delivery["fresh_cache"] and not delivery["resumed"]
         and delivery["download"]["ax_engine_sha256"] == build["cli_sha256"], "Fresh delivery evidence changed")
    need(controls["functional_controls_passed"] and set(controls["results"]) == {"disabled", "required", "default"}
         and all(len(rows) == 7 for rows in controls["results"].values()), "Incomplete native support controls")
    transfer = ROOT.parent / "flash-next-fresh-delivery-38c90b27a-m5-20261122/replay.py"
    need(digest(transfer.read_bytes()) == "69e89be57f5a70f19d294d2f069993bf94f3c01a919667dffb6e342d16a37a93",
         "Archived delivery replay changed")
    original_transfer = run_path(str(transfer))
    counters = original_transfer["check_transfer"]((directory / "delivery/download.ndjson").read_bytes(),
                                                   delivery, 132261877478)
    oracle = read(directory, "signed-native.json")
    native = run_path(str(directory / "seal_native.py"))
    qa_matches = {}
    matrix_matches = {}
    matrix = read(directory, "generate-matrix/raw-runs.json")
    for mode in MODES:
        rows = native["prompt_rows"](directory / "qa" / mode)
        need(rows.keys() == oracle["qa"][mode].keys(), "The signed native QA roster changed")
        qa_matches[mode] = sum(rows[key] == oracle["qa"][mode][key] for key in rows)
        need(len(matrix[mode]) == 2, "Incomplete raw matrix repeats")
        matrix_matches[mode] = sum(native["matrix_rows"](run) == oracle["matrix"][mode]
                                   for run in matrix[mode])
    run_summary = {"delivery_progress_counters": counters, "signed_native_qa_matches": qa_matches,
                   "signed_native_matrix_process_matches": matrix_matches,
                   "native_http_cases": sum(map(len, controls["results"].values()))}
    return build, inventory, run_summary


def replay(archive, capture=None):
    board = run_path(str(ROOT.parent / "flash-next-guarded-reader-board-m5-20261123/replay.py"))
    board["verify_evaluator_inputs"]()
    reference, scope = board["reference_input"](capture)
    with TemporaryDirectory(prefix="ax-flash-consolidated-native-") as work:
        directory = Path(work)
        materialize(archive, directory)
        build, inventory, native_checks = check_native(directory)
        (directory / "reference").mkdir()
        (directory / "reference/result.json").write_bytes(reference)
        output = directory / "verdict.json"
        proc = subprocess.run([sys.executable, "-E", "-s", "-B", REPO / "scripts/check_flash_next_release_gates.py",
                               "--evidence-dir", directory, "--output", output], capture_output=True)
        need(proc.returncode in (0, 1) and output.is_file(), proc.stderr.decode())
        verdict = read(directory, "verdict.json")
        need(verdict["release_ready"] == (proc.returncode == 0)
             and verdict["evidence_sha256"]["reference"] == digest(reference), "Evaluator seal differs")
        verdict.pop("source")
    return {**verdict, "native_build": build, "inventory_sha256": inventory,
            "reference_input": scope, "native_checks": native_checks}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(ROOT / "evidence.tar.xz", args.ordinary_capture), sort_keys=True, indent=2))
