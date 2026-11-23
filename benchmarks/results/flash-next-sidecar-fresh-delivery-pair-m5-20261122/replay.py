"""Replay both native Flash Next launches after empty-cache delivery."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
STOCK = Path("benchmarks/results/flash-next-fresh-delivery-38c90b27a-m5-20261122")
ARMS = ("mapped", "prefetch")
SOURCE_FILES = ("result.json", "download.ndjson", "native-support/result.json", "native-support/host-samples.json",
                "native-support/server-required-kill-switch.log", "native-support/server-disabled.log",
                "native-support/server-required.log", "native-support/server-default.log")


def digest(raw):
    return sha256(raw).hexdigest()


def need(valid, reason):
    if not valid:
        raise ValueError(reason)


def complete_archive(archive_path, repo, pins):
    with tarfile.open(archive_path) as archive, TemporaryDirectory(prefix="flash-next-fresh-replay-") as temp:
        names = [item.name for item in archive]
        need(len(names) == len(set(names)) and all(item.isfile() for item in archive),
             "Unexpected archive entry")

        def content(name):
            return archive.extractfile(name).read()

        manifest = content("harness-inputs.sha256")
        build = content("build.json")
        need(digest(manifest) == pins["manifest"] and digest(build) == pins["build"]
             and content("status.txt") == b"0\n", "Frozen launcher, build or native success record differs")
        need(digest((repo / STOCK / "replay.py").read_bytes()) == pins["stock_replay"],
             "Transfer validation source differs")
        expected_hashes = {name: checksum for checksum, name in map(str.split, manifest.decode().splitlines())}
        required = {"status.txt", "harness-inputs.sha256", "build.json", "driver/audit_pair.py",
                    "observed/pair.json", "observed/source-sha256.json"}
        required.update([f"driver/{name}" for name in expected_hashes])
        required.update(f"observed/{arm}/{name}" for arm in ARMS for name in SOURCE_FILES)
        need(set(names) == required, "Fresh-delivery evidence roster differs")
        need(all(digest(content(f"driver/{name}")) == checksum for name, checksum in expected_hashes.items()),
             "Frozen staged driver or inputs differ")
        with tarfile.open(repo / STOCK / "evidence.tar.xz") as prior:
            need(content("driver/stock-native.json") == prior.extractfile("native-support.json").read(),
                 "The retained native response oracle differs")
        provenance = json.loads(content("observed/source-sha256.json"))
        need(list(provenance) == list(ARMS), "Missing source provenance for a delivery arm")
        for arm in ARMS:
            need(set(provenance[arm]) == set(SOURCE_FILES)
                 and all(digest(content(f"observed/{arm}/{name}")) == provenance[arm][name]["portable"]
                         for name in SOURCE_FILES), "Portable delivery observation changed")
        dest = Path(temp)
        for name in names:
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content(name))
        return subprocess.check_output([sys.executable, "-B", dest / "driver/audit_pair.py", dest / "observed",
                                        dest / "build.json", "--repo", repo], cwd=dest,
                                       env={"PYTHONPATH": str(dest / "driver")})


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit("Replay requires Python 3.12 or newer.")
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    pins = json.loads((ROOT / "archive-pins.json").read_bytes())
    sys.stdout.buffer.write(complete_archive(args.archive, args.repo.resolve(), pins))
