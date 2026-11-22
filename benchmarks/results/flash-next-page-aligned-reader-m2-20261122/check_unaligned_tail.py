"""Verify page-bounded staging at an unaligned start and the final EOF page."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    size = 4 * 1024**2 + 257
    offset = 39
    payload = (bytes(range(251)) * (size // 251 + 1))[:size]
    digest = sha256(payload).hexdigest()
    with TemporaryDirectory() as folder:
        source, plan = (Path(folder) / name for name in ("weights.bin", "layout.txt"))
        source.write_bytes(b"\xa7" * offset + payload)
        plan.write_text(f"{offset} {size} U8 1 {size}\n")
        cases = {}
        for mode in ("mmap", "aligned-buffered", "aligned-nocache"):
            child = subprocess.run([binary, source, plan, mode], check=True, capture_output=True)
            row = json.loads(child.stdout)
            del row["elapsed_seconds"]
            if (row["mode"] != mode or row["tensors"] != 1 or row["selected_bytes"] != size
                    or row["weights_sha256"] != digest or row["nocache"] is not (mode == "aligned-nocache")):
                raise ValueError("An aligned read changed the unaligned tensor payload")
            cases[mode] = row
        for result in cases.values():
            for key in ("allocated_bytes", "weights_sha256", "metal_sha256"):
                if cases["mmap"][key] != result[key]:
                    raise ValueError("An aligned read changed owned storage or the post-read Metal check")
        plan.write_text(f"{offset} {size + 1} U8 1 {size + 1}\n")
        broken = subprocess.run([binary, source, plan, "aligned-nocache"], check=False, capture_output=True)
        if broken.returncode != 1 or b"range exceeds validated file bounds" not in broken.stderr:
            raise ValueError("Alignment handling accepted a payload that would cross EOF")
        print(json.dumps({"binary_sha256": sha256(binary.read_bytes()).hexdigest(), "source_offset": offset,
                          "out_of_range_rejected": True, "cases": cases}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
