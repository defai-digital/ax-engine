"""Verify and replay the signed M5 QSA serving crossover/archive."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def verify(root):
    assert (root / "status.txt").read_text().strip() == "0"
    entries = {name: digest for digest, name in map(
        str.split, (root / "inputs.sha256").read_text().splitlines()
    )}
    for name, digest in entries.items():
        if name != "ax-engine-server":
            assert sha256((root / name).read_bytes()).hexdigest() == digest, name
    build = json.loads((root / "build.json").read_bytes())
    assert entries["ax-engine-server"] == build["binary_sha256"]
    assert entries["source.patch"] == build["patch_sha256"]
    assert build["profile"] == "release-server" and build["mlx"] == "0.32.4"
    assert build["rustc"].startswith("rustc 1.99.1 ")
    assert {p.name for p in (root / "observed").iterdir()} == {
        "stock-first", "paired-first", "paired-second", "stock-second",
    }
    portable_prompts = json.loads((root / "prompts.json").read_bytes())
    lengths = {name: len(prompt) for name, prompt in portable_prompts.items()}
    assert sorted(lengths.values()) == [512, 2048, 8192, 8192]
    assert portable_prompts["long-replay"] != portable_prompts["long-holdout"]
    rows = [
        json.loads(line)
        for arm in sorted((root / "observed").iterdir())
        for line in (arm / "requests.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 20 and sum(row["warmup"] for row in rows) == 4
    for row in rows:
        assert len(row["output_tokens"]) == 64
    return subprocess.check_output([sys.executable, "-E", "-s", "-B", root / "compare_http.py", root, "--m5"])


def replay():
    with (TemporaryDirectory(prefix="flash-next-qsa-http-m5-") as scratch,
          tarfile.open(HERE / "evidence.tar.xz") as archive):
        root = Path(scratch)
        names = archive.getnames()
        assert len(names) == len(set(names))
        for member in archive:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            assert path.name != "ax-engine-server"
        archive.extractall(root, filter="data")
        return verify(root)


if __name__ == "__main__":
    sys.stdout.buffer.write(replay())
