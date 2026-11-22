"""Exercise selected-page boundaries and copied MLX ownership on an owned fixture."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from base_probe import PAGE, page_state


def run(binary, scratch):
    payload = bytes((offset * 41 + offset // 127) % 256 for offset in range(5 * PAGE + 17))
    extents = ((PAGE + 64, 2 * PAGE + 12), (2 * PAGE + 12, 3 * PAGE + 64))
    plan = tuple((begin, end - begin, "U8", [end - begin]) for begin, end in extents)
    expect_hash = sha256(b"".join(payload[a:b] for a, b in extents)).hexdigest()
    one_page = [(2 * PAGE, PAGE, "U8", [PAGE])]
    neighbors = [(index * PAGE, PAGE, "U8", [PAGE]) for index in (0, 1, 3, 4)]
    arms = {}
    with TemporaryDirectory(dir=scratch) as temp:
        root = Path(temp)
        geometry = root / "geometry.txt"
        geometry.write_text("".join(f"{start} {length} {dtype} 1 {shape[0]}\n"
                                    for start, length, dtype, shape in plan))
        for mode in ("mmap", "mmap-reclaim-private", "mmap-reclaim-shared"):
            path = root / mode
            with path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            with path.open("rb") as stream:
                assert stream.read() == payload
            before = [page_state(path, regions)["resident_pages"] for regions in (one_page, neighbors)]
            child = subprocess.run([binary, path, geometry, mode], capture_output=True, check=True, text=True)
            arrays = json.loads(child.stdout)
            after = [page_state(path, regions)["resident_pages"] for regions in (one_page, neighbors)]
            assert path.read_bytes() == payload and arrays["weights_sha256"] == expect_hash
            assert before == [1, 4] and arrays["reclaim_errno"] == 0
            assert arrays["requested_invalidate_pages"] == int(mode != "mmap")
            arms[mode] = {"pages_before": before, "pages_after": after, "arrays": arrays}
    for key in ("weights_sha256", "metal_sha256", "allocated_bytes", "selected_bytes"):
        assert len({arm["arrays"][key] for arm in arms.values()}) == 1
    assert arms["mmap"]["pages_after"] == [1, 4]
    return {"scope": "synthetic mapped MLX copies on Darwin", "status": "complete",
            "regions": ["selected full page", "excluded and partially selected neighbor pages"],
            "targeted_release_observed": {mode: arm["pages_after"] == [0, 4]
                                          for mode, arm in arms.items()}, "arms": arms}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.binary, args.scratch), indent=2, allow_nan=False))
