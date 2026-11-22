"""Replay de-identified Flash Next mixed-serving observations over the frozen workload."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PRIOR_WORKLOAD = ROOT.parent / "flash-next-mxfp4-mixed-serving-m5-20261121/evidence.tar.xz"
FROZEN = {
    "mixed-corpus.jsonl": "mixed-corpus.jsonl",
    "solo-reference.json": "mixed-result/solo-1.json",
    "workload.json": "workload.json",
}
BINARY = "ax-engine-server-layer-http"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def need(valid, message):
    if not valid:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def replay(root, previous, allow_partial=False):
    if not allow_partial:
        need((root / "status.txt").read_text() == "0\n", "The measured controller failed")
    hashes = {name: checksum for checksum, name in
              (line.split() for line in (root / "inputs.sha256").read_text().splitlines())}
    extras = {"summarize_sustained.py", "seal_sustained.py"}
    with TemporaryDirectory() as temp:
        files = Path(temp)
        with tarfile.open(root / "driver.tar.xz") as driver:
            expected = (hashes.keys() - FROZEN.keys() - {BINARY}) | extras
            need(set(driver.getnames()) == expected, "Driver members differ")
            for member in driver:
                need(member.isfile() and Path(member.name).name == member.name, "Unexpected driver path")
                raw = driver.extractfile(member).read()
                if member.name not in extras:
                    need(digest(raw) == hashes[member.name], "Measured driver bytes differ")
                (files / member.name).write_bytes(raw)
        with tarfile.open(previous) as baseline:
            for name, member in FROZEN.items():
                raw = baseline.extractfile(member).read()
                need(digest(raw) == hashes[name], "Frozen workload or solo token evidence differs")
                (files / name).write_bytes(raw)
        observed = files / "observed"
        observed.mkdir()
        with tarfile.open(root / "evidence.tar.xz") as evidence:
            for member in evidence:
                need(member.isfile() and Path(member.name).name == member.name, "Unexpected evidence path")
                (observed / member.name).write_bytes(evidence.extractfile(member).read())
        receipt = json.loads((observed / "receipt.json").read_bytes())
        build = json.loads((files / "build.json").read_bytes())
        need(receipt["build"] == build and build["server_sha256"] == hashes[BINARY]
             and receipt["inventory_sha256"] == hashes["mxfp4-inventory.json"]
             and receipt["manifest"]["model-manifest.json"] == MANIFEST,
             "The result does not bind the measured pack and executable")
        command = [sys.executable, "-B", str(files / "summarize_sustained.py"), str(observed)]
        if allow_partial:
            command.append("--allow-partial")
        return subprocess.check_output(command)


if __name__ == "__main__":
    sys.stdout.buffer.write(replay(ROOT, PRIOR_WORKLOAD))
