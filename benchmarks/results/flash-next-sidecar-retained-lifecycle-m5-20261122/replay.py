"""Replay signed-build receipts for the retained Flash Next startup comparison."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
HARNESS_SHA = "b42cbdca4ed6112205eadab0d37284fc20f32bbf6753abcb05bd8fccf7e9cad9"
BUILD_SHA = "4f8a34afc5b4f3aa49610cbc3bcdb8e094231c9b1297f22284f020908f727772"
ORACLE = "benchmarks/results/flash-next-native-lifecycle-38c90b27a-m5-20261122/lifecycle.json"


def digest(data):
    return sha256(data).hexdigest()


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def replay(package, repo):
    with tarfile.open(package) as archive, TemporaryDirectory(prefix="flash-next-lifecycle-evidence-") as temp:
        names = [item.name for item in archive]
        need(len(names) == len(set(names)) and all(item.isfile() for item in archive),
             "Duplicate or non-file members in retained evidence")

        def contents(name):
            return archive.extractfile(name).read()

        manifest = contents("harness-inputs.sha256")
        build = contents("build.json")
        need(digest(manifest) == HARNESS_SHA and digest(build) == BUILD_SHA and contents("status.txt") == b"0\n",
             "Frozen harness, exact signed build, or success receipt differs")
        hashes = {name: checksum for checksum, name in map(str.split, manifest.decode().splitlines())}
        evidence = ["observed/pair.json"]
        for tag in ("mapped", "prefetch"):
            evidence += [f"observed/{tag}/{name}" for name in
                         ("result.json", "host-pressure.json", "server-disabled.log", "server-default.log")]
        expected = {"harness-inputs.sha256", "build.json", "status.txt", "driver/assess.py", *evidence}
        expected.update(f"driver/{name}" for name in hashes)
        need(set(names) == expected, "The retained evidence roster changed")
        for name, checksum in hashes.items():
            need(digest(contents(f"driver/{name}")) == checksum, f"Frozen staged input differs for {name}")
        need(contents("driver/oracle.json") == (repo / ORACLE).read_bytes(),
             "The original lifecycle receipt is different")
        root = Path(temp)
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents(name))
        return subprocess.check_output([sys.executable, "-B", root / "driver" / "assess.py",
                                        root / "observed", root / "build.json"],
                                       env={"PYTHONPATH": str(root / "driver")})


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    options = cli.parse_args()
    sys.stdout.buffer.write(replay(options.archive, options.repo))
