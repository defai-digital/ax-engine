"""Evaluate the signed guarded-reader Flash Next release evidence on M5."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RESULTS = ROOT.parent
SCRIPTS = REPO / "scripts"
PARTIAL = RESULTS / "flash-next-ordinary-reference-m5-20261121" / "partial-result.json"
PARTIAL_SHA256 = "868bdf9cd3937b3b49323e75fe38979a206ba16e8d529efa54c016820322673d"
SERVER_SHA256 = "c1a7c3f9dfb81613fc370758a0eece5f6105683d53baab7ea4bc3a371cb81de2"
ARCHIVES = {
    "fresh": "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b",
    "lifecycle": "6e19ebf041c98180f55bb762508182ea18575aa44bfab87c4c2c337a6c06935e",
    "matrix": "2e102e2027fb0e6faf4017da435b73964671af878accd16086316612d9ac2cac",
    "qa": "5c0e94079426f40ab4791550403e1eabf37293295804b63285b7ce57f60d8658",
}
VERIFIER_MANIFESTS = {
    "fresh": "f7becb620f69d32d1bd34fd255ca9c0a3bb14e6a8e3f30b0c78cbbcaec3bfb55",
    "lifecycle": "9cd10cd93cb660e616d44d9ba4882990af448f75ceccada53887de7b008385a2",
    "matrix": "4708e8748dd25b00c04184622b7dc14cab63a3980e7fe254fc3a48cc244e8bb6",
    "qa": "a1da1025c58616274e2a09d0ad0bd428f247bef60e175b58b96dac1e75fb3132",
}
PYTHON = (sys.executable, "-E", "-s", "-B")
PYTHON_ENV = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
PYTHON_ENV.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
EVALUATOR_INPUTS = {
    "check_flash_next_release_gates.py": "3041a6fbd8099193b32c817ca830661b07a5da0e400a6523f9d05212b5631ead",
    "flash_next_release_thresholds.json": "17a39e0229cf734dca04ffa5abb4718a6c7b1c9c773891385a37396a1625345b",
    "qualify_qwen38_flash_next.py": "ff020ed48cdc87e56d56a3213f0485e56a2dbf7c1f1c60587d0174c5fc9f0b4c",
}
REFERENCE_RAW = {
    "result": "reference/result.json",
    "monitor": "monitor/status.json",
    "outcome": "outcome.json",
    "preflight": "reference-preflight.json",
    "heartbeat": "heartbeat-admission.json",
    "native_input": "native-matrix.json",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def reference_input(capture):
    if capture is None:
        raw = PARTIAL.read_bytes()
        require(digest(raw) == PARTIAL_SHA256, "The retained ordinary reference partial changed")
        return raw, "archived-partial"
    verification = json.loads((capture / "verified-reference.json").read_bytes())
    capture = capture / "raw"
    raw = {name: (capture / path).read_bytes() for name, path in REFERENCE_RAW.items()}
    require(verification["sha256"] == {name: digest(value) for name, value in raw.items()},
            "The collected ordinary reference differs from the verified capture")
    require((capture / "status.txt").read_bytes() == b"0\n",
            "The ordinary reference process did not complete")
    result = json.loads(raw["result"])
    require(result["completed"] is True and result["phase"] == "complete"
            and not result.get("mode") and result["qualification"] is False,
            "The completed ordinary MLX-VLM profile is required")
    require((verification["qa_rows"], verification["matrix_rows"]) == (105, 15)
            and len(result["qa"]["rows"]) == 105 and len(result["matrix"]["cells"]) == 15
            and verification["runtime_files_checked"] == result["runtime_files_bound"] > 0,
            "The collected ordinary reference roster or runtime binding changed")
    return raw["result"], "verified-ordinary-capture"


def verified_archive(kind, staging):
    source = RESULTS / f"flash-next-guarded-reader-{kind}-m5-20261123"
    manifest = (source / "SHA256SUMS").read_bytes()
    require(digest(manifest) == VERIFIER_MANIFESTS[kind], f"The sealed {kind} verifier manifest changed")
    hashes = {name: value for value, name in map(str.split, manifest.decode().splitlines())}
    archive = (source / "evidence.tar.xz").read_bytes()
    require(digest(archive) == hashes["evidence.tar.xz"] == ARCHIVES[kind], f"The sealed {kind} archive changed")
    verifier = (source / "replay.py").read_bytes()
    expected = (source / "comparison.json").read_bytes()
    require(digest(verifier) == hashes["replay.py"] and digest(expected) == hashes["comparison.json"],
            f"The sealed {kind} replay or expected result changed")
    write(staging, "replay.py", verifier)
    write(staging, "evidence.tar.xz", archive)
    actual = subprocess.check_output([*PYTHON, staging / "replay.py", "--repo", REPO,
                                      "--archive", staging / "evidence.tar.xz"], env=PYTHON_ENV)
    require(actual == expected and json.loads(actual)["server_sha256"] == SERVER_SHA256,
            f"The archived {kind} evidence did not replay for the signed build")
    return tarfile.open(fileobj=BytesIO(archive))


def member(archive, name):
    entry = archive.getmember(name)
    require(entry.isfile(), f"Expected a regular member {name}")
    with archive.extractfile(entry) as stream:
        return stream.read()


def write(target, path, data):
    path = target / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def assemble(target, reference):
    members = {
        "fresh": ("", ("delivery/result.json", "delivery/native-support/result.json")),
        "lifecycle": ("lifecycle/", ("result.json",)),
        "matrix": ("generate-matrix/", ("result.json", "raw-runs.json")),
        "qa": ("qa/", ("result.json",)),
    }
    for kind, (prefix, paths) in members.items():
        with verified_archive(kind, target / "verify" / kind) as archive:
            for name in paths:
                write(target, prefix + name, member(archive, name))
            if kind == "qa":
                for name in archive.getnames():
                    parts = Path(name).parts
                    if len(parts) == 2 and parts[0] in {"default", "disabled"} and name.endswith(".json"):
                        write(target, prefix + name, member(archive, name))
    write(target, "reference/result.json", reference)


def verify_evaluator_inputs():
    for name, expected in EVALUATOR_INPUTS.items():
        require(digest((SCRIPTS / name).read_bytes()) == expected, f"The frozen input changed: {name}")


def replay(capture=None):
    verify_evaluator_inputs()
    reference, reference_scope = reference_input(capture)
    with TemporaryDirectory(prefix="ax-flash-next-guarded-board-") as work:
        target = Path(work)
        assemble(target, reference)
        verdict_path = target / "verdict.json"
        run = subprocess.run([*PYTHON, SCRIPTS / "check_flash_next_release_gates.py",
                              "--evidence-dir", target, "--output", verdict_path],
                             capture_output=True, env=PYTHON_ENV)
        if run.returncode not in (0, 1) or not verdict_path.is_file():
            raise RuntimeError(run.stderr.decode())
        verdict = json.loads(verdict_path.read_bytes())
        require(verdict["release_ready"] == (run.returncode == 0)
                and verdict["evidence_sha256"]["reference"] == digest(reference),
                "The release evaluator did not bind its verdict to the capture")
    verify_evaluator_inputs()
    del verdict["source"]
    return {**verdict, "native_server_sha256": SERVER_SHA256,
            "reference_input": reference_scope, "native_archives_sha256": ARCHIVES,
            "frozen_inputs_sha256": EVALUATOR_INPUTS,
            "native_verifier_manifests_sha256": VERIFIER_MANIFESTS}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--ordinary-capture", type=Path, help="directory of the verified ordinary-reference collection")
    args = cli.parse_args()
    print(json.dumps(replay(args.ordinary_capture), indent=2, sort_keys=True, allow_nan=False))
