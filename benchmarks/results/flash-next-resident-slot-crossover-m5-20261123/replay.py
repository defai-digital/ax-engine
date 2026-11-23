"""Verify the archived M5 slot crossover and recompute the comparison."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


def replay():
    with (TemporaryDirectory(prefix="flash-next-slot-replay-") as scratch,
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
        assert build["codesign_verified"]
        assert build["server_sha256"] == entries["ax-engine-server"]
        assert build["patch_sha256"] == entries["source.patch"]
        redactions = json.loads((root / "redactions.json").read_bytes())
        observed = root / "observed"
        assert set(redactions) == {arm.name for arm in observed.iterdir()}
        for arm, note in redactions.items():
            assert note["fields"] == ["target.base_url", "corpus.path"]
            assert len(bytes.fromhex(note["original_sha256"])) == 32
            data = (observed / arm / f"{arm}.json").read_bytes()
            assert sha256(data).hexdigest() == note["archived_sha256"]
            artifact = json.loads(data)
            assert artifact["target"]["base_url"] == "[loopback]"
            assert artifact["corpus"]["path"] == "mixed-corpus.jsonl"
        comparison = json.loads(subprocess.check_output([sys.executable, "-B", root / "compare.py", root]))
        attribution = json.loads(subprocess.check_output(
            [sys.executable, "-B", root / "attribute_prefill.py", root]
        ))
        assert comparison["arms"].keys() == attribution.keys()
        return comparison | {"prefill_gap_attribution": attribution}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
