"""Verify the frozen M2 storage comparison and its guarded AX workload binding."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
MATRIX = HERE.parent / "flash-next-guarded-reader-matrix-m5-20261123" / "evidence.tar.xz"


def replay():
    with (TemporaryDirectory(prefix="flash-next-storage-parity-replay-") as scratch,
          tarfile.open(HERE / "evidence.tar.xz") as archive,
          tarfile.open(MATRIX) as native):
        root = Path(scratch)
        names = archive.getnames()
        assert len(set(names)) == len(names)
        for entry in archive:
            name = PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and ".." not in name.parts
        archive.extractall(root, filter="data")
        source = json.loads((root / "source.json").read_bytes())
        assert sha256((root / "run_flash_next_affine8_reference.py").read_bytes()).hexdigest() == source["runner_sha256"]
        assert sha256((root / "watch.py").read_bytes()).hexdigest() == source["watch_sha256"]
        assert sha256(native.extractfile("result.json").read()).hexdigest() == source["matrix_sha256"]
        assert (root / "status.txt").read_text().strip() == "0"
        return subprocess.check_output([sys.executable, "-B", root / "compare_storage.py", root])


if __name__ == "__main__":
    sys.stdout.buffer.write(replay())
