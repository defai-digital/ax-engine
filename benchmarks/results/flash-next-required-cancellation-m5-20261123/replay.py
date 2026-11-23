"""Replay checked native Flash Next cancellation observations."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def replay():
    with (TemporaryDirectory(prefix="flash-next-primary-cancellation-") as temp,
          tarfile.open(HERE / "evidence.tar.xz") as source):
        root = Path(temp)
        names = source.getnames()
        assert len(names) == len(set(names))
        for item in source:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert path.name != "ax-engine-server"
        source.extractall(root, filter="data")
        capture, bundle = root / "capture", root / "bundle"
        raw = (bundle / "inputs.sha256").read_bytes()
        assert (capture / "inputs.sha256").read_bytes() == raw
        entries = {name: digest for digest, name in map(str.split, raw.decode().splitlines())}
        assert {path.name for path in bundle.iterdir()} == entries.keys() - {"ax-engine-server"} | {"inputs.sha256"}
        for name, digest in entries.items():
            if name != "ax-engine-server":
                assert sha256((bundle / name).read_bytes()).hexdigest() == digest, name
        build = json.loads((bundle / "build.json").read_bytes())
        assert build["codesign_verified"] is True and build["server_sha256"] == entries["ax-engine-server"]
        assert build["patch_sha256"] == entries["source.patch"]
        return subprocess.check_output([sys.executable, "-B", root / "compare.py", capture])


if __name__ == "__main__":
    sys.stdout.buffer.write(replay())
