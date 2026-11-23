"""Reproduce guarded mixed-serving evidence from the original request receipts."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
FROZEN = {
    "mixed-corpus.jsonl": "mixed-corpus.jsonl",
    "solo-reference.json": "mixed-result/solo-1.json",
    "workload.json": "workload.json",
}
BINARY = "ax-engine-server-resident-mixed"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def unpack(archive, dest, expected=None):
    members = set()
    for item in archive:
        require(item.isfile() and Path(item.name).name == item.name and item.name not in members,
                "Unexpected portable evidence path")
        members.add(item.name)
        (dest / item.name).write_bytes(archive.extractfile(item).read())
    require(expected is None or expected == members, "Portable archive roster differs")


def replay(root, repo):
    candidates = repo / "benchmarks" / "results"
    baseline = candidates / "flash-next-mxfp4-mixed-serving-m5-20261121" / "evidence.tar.xz"
    proof = candidates / "flash-next-native-mixed-admission-m5-20261122" / "comparison.json"
    require((root / "status.txt").read_text() == "0\n", "Measured controller failed")
    receipts = {name: checksum for checksum, name in
                map(str.split, (root / "natural-inputs.sha256").read_text().splitlines())}
    with TemporaryDirectory(prefix="flash-next-natural-evidence-") as temporary:
        files = Path(temporary)
        offline = {"summarize_natural.py", "seal_natural.py"}
        expected = receipts.keys() - FROZEN.keys() - {BINARY}
        with tarfile.open(root / "driver.tar.xz") as driver:
            unpack(driver, files, expected | offline)
        require(all(digest((files / name).read_bytes()) == receipts[name] for name in expected)
                and (files / "admission-proof.json").read_bytes() == proof.read_bytes(),
                "Measured driver or exact guarded-admission proof differs")
        with tarfile.open(baseline) as oracle:
            for name, member in FROZEN.items():
                raw = oracle.extractfile(member).read()
                require(digest(raw) == receipts[name], "Frozen corpus or solo output oracle differs")
                (files / name).write_bytes(raw)
        observed = files / "observed"
        observed.mkdir()
        with tarfile.open(root / "evidence.tar.xz") as evidence:
            unpack(evidence, observed)
        receipt = json.loads((observed / "receipt.json").read_bytes())
        build = json.loads((files / "candidate-build.json").read_bytes())
        require(receipt["build"] == build and build["codesign_verified"] is True
                and build["server_sha256"] == receipts[BINARY]
                and receipt["inventory_sha256"] == receipts["mxfp4-inventory.json"]
                and receipt["manifest"]["model-manifest.json"] == MANIFEST,
                "Live receipt does not bind the tested runtime and pack")
        return subprocess.check_output([sys.executable, "-B", files / "summarize_natural.py", observed])


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    options = cli.parse_args()
    sys.stdout.buffer.write(replay(ROOT, options.repo))
