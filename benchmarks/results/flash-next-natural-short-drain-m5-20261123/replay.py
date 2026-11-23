"""Recompute the four M5 natural-load decode-drain receipts from original traces."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def replay():
    with (TemporaryDirectory(prefix="flash-next-natural-drain-replay-") as scratch,
          tarfile.open(HERE / "evidence.tar.xz") as archive):
        root = Path(scratch)
        names = archive.getnames()
        assert len(set(names)) == len(names)
        for entry in archive:
            name = PurePosixPath(entry.name)
            assert entry.isfile() and not name.is_absolute() and ".." not in name.parts
            assert name.name != "ax-engine-server"
        archive.extractall(root, filter="data")
        manifest = (root / "inputs.sha256").read_bytes()
        assert (root / "bundle/inputs.sha256").read_bytes() == manifest
        entries = {name: digest for digest, name in map(str.split, manifest.decode().splitlines())}
        bundle = root / "bundle"
        assert {item.name for item in bundle.iterdir()} == entries.keys() - {"ax-engine-server"} | {"inputs.sha256"}
        for name, digest in entries.items():
            if name != "ax-engine-server":
                assert sha256((bundle / name).read_bytes()).hexdigest() == digest, name
        build = json.loads((root / "build.json").read_bytes())
        assert build["codesign_verified"] and build["server_sha256"] == entries["ax-engine-server"]
        assert build["patch_sha256"] == entries["source.patch"]
        redactions = json.loads((root / "redactions.json").read_bytes())
        assert set(redactions) == {"mixed-first", "drain-first", "drain-second", "mixed-second"}
        for arm, note in redactions.items():
            observed = root / "observed" / arm
            data = (observed / f"{arm}.json").read_bytes()
            assert sha256(data).hexdigest() == note["archived_sha256"]
            artifact = json.loads(data)
            assert artifact["target"]["base_url"] == "[loopback]"
            assert artifact["corpus"]["path"] == "mixed-corpus.jsonl"
        decision = json.loads((root / "dispatch-decision.json").read_bytes())
        targeted = HERE.parent / "flash-next-priority-short-drain-m5-20261123" / "comparison.json"
        assert decision["start"] is True
        raw_targeted = targeted.read_bytes()
        assert sha256(raw_targeted).hexdigest() == decision["crossover_sha256"]
        targeting = json.loads(raw_targeted)
        assert targeting["build"] == build and targeting["qualification"] is False
        assert decision["eligible_targeted_orders"] == [
            name for name, row in targeting["comparisons"].items() if row["timing_eligible"]
        ]
        assert decision["eligible_targeted_orders"]
        return subprocess.check_output([sys.executable, "-B", root / "compare.py", root])


if __name__ == "__main__":
    sys.stdout.buffer.write(replay())
