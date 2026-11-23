"""Recheck the signed automatic-reader launch after an independent delivery."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
TRANSFER_REPLAY = Path("benchmarks/results/flash-next-fresh-delivery-38c90b27a-m5-20261122/replay.py")
TRANSFER_REPLAY_SHA256 = "69e89be57f5a70f19d294d2f069993bf94f3c01a919667dffb6e342d16a37a93"
PINNED = {
    "raw/harness-inputs.sha256": "8a00b964b18fc170c7a6adea8c9398133495d2d3f03e9af5d6169f0119b7dec0",
    "raw/build.json": "3091f8f2f78087bb8ba34e0c156fcb49b9058eee3aa0884cbb106a6344eda2d2",
    "driver/verify_auto.py": "df1e0f2d300341b854174e94d08a9cec12cadc98a38ad30539636193c7963891"
}
RAW = {
    "auto-delivery.json", "build.json", "harness-inputs.sha256",
    "delivery/result.json", "delivery/download.ndjson", "delivery/native-support/result.json",
    "delivery/native-support/host-samples.json", "status.txt",
    *(f"delivery/native-support/server-{mode}.log"
      for mode in ("disabled", "required", "default", "required-kill-switch")),
}


def replay(archive_path, repo):
    with tarfile.open(archive_path) as archive, TemporaryDirectory(prefix="ax-flash-auto-replay-") as directory:
        files = archive.getmembers()
        names = {entry.name for entry in files}
        if len(files) != len(names) or any(
            not entry.isfile() or PurePosixPath(entry.name).is_absolute()
            or ".." in PurePosixPath(entry.name).parts
            or str(PurePosixPath(entry.name)) != entry.name for entry in files
        ):
            raise ValueError("Unsafe or duplicate archive entry")

        def read(name):
            with archive.extractfile(name) as stream:
                return stream.read()

        if read("raw/status.txt") != b"0\n":
            raise ValueError("The independent cold-delivery process did not finish")
        if any(sha256(read(name)).hexdigest() != digest for name, digest in PINNED.items()):
            raise ValueError("Pinned launcher, build or verifier changed")
        if sha256((repo / TRANSFER_REPLAY).read_bytes()).hexdigest() != TRANSFER_REPLAY_SHA256:
            raise ValueError("Frozen transfer replay changed")
        manifest = {PurePosixPath(name).as_posix(): digest for digest, name in
                    map(str.split, read("raw/harness-inputs.sha256").decode().splitlines())}
        expected = ({f"raw/{name}" for name in RAW} | {f"driver/{name}" for name in manifest}
                    | {"driver/verify_auto.py"})
        if names != expected or any(sha256(read(f"driver/{name}")).hexdigest() != digest
                                    for name, digest in manifest.items()):
            raise ValueError("Frozen driver or evidence roster changed")
        dest = Path(directory)
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read(name))
        return subprocess.check_output(
            [sys.executable, "-B", dest / "driver/verify_auto.py", dest / "raw",
             dest / "raw/build.json", "--repo", repo],
            cwd=dest,
        )


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit("Replay requires Python 3.12 or newer.")
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    args = cli.parse_args()
    sys.stdout.buffer.write(replay(args.archive, args.repo.resolve()))
