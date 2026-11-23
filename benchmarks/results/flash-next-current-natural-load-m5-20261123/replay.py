"""Recalculate the archived M5 natural-load comparison from raw SSE traces."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def main():
    with (TemporaryDirectory(prefix="flash-next-natural-replay-") as scratch,
          tarfile.open(ROOT / "evidence.tar.xz") as archive):
        work = Path(scratch)
        names = archive.getnames()
        assert len(set(names)) == len(names)
        for entry in archive:
            target = PurePosixPath(entry.name)
            assert entry.isfile() and not target.is_absolute() and ".." not in target.parts
            assert target.name != "ax-engine-server"
        archive.extractall(work, filter="data")
        entries = {name: digest for digest, name in map(str.split, (work / "inputs.sha256").read_text().splitlines())}
        assert entries.keys() - {"ax-engine-server"} == {p.name for p in (work / "bundle").iterdir()}
        for name, digest in entries.items():
            if name != "ax-engine-server":
                assert sha256((work / "bundle" / name).read_bytes()).hexdigest() == digest, name
        build = json.loads((work / "build.json").read_bytes())
        assert build["codesign_verified"] and entries["ax-engine-server"] == build["server_sha256"]
        assert entries["source.patch"] == build["patch_sha256"]
        note = json.loads((work / "redactions.json").read_bytes())
        for arm, receipt in note.items():
            actual = work / "observed" / arm / f"{arm}.json"
            assert sha256(actual.read_bytes()).hexdigest() == receipt["archived_sha256"]
            observed = json.loads(actual.read_bytes())
            assert observed["target"]["base_url"] == "[loopback]"
            assert observed["corpus"]["path"] == "mixed-corpus.jsonl"
        result = subprocess.check_output([sys.executable, "-B", work / "compare.py", work])
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    main()
