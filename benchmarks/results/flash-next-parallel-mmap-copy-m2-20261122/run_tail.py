"""Check the paired mapped copy across a non-power-of-two tail."""
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
    size = 64 * 1024**2 + 257
    offset = 37
    expected = sha256()
    block = bytes(range(251)) * 4096
    with TemporaryDirectory() as folder:
        source, plan = (Path(folder) / name for name in ("tail.bin", "layout.txt"))
        with source.open("wb") as stream:
            stream.write(b"\xc3" * offset)
            for start in range(0, size, len(block)):
                content = block[:min(len(block), size - start)]
                expected.update(content)
                stream.write(content)
            stream.write(b"\xe7" * 101)
        plan.write_text(f"{offset} {size} U8 1 {size}\n")
        results = {}
        for mode in ("mmap", "mmap-pair"):
            proc = subprocess.run([binary, source, plan, mode], check=True, capture_output=True)
            row = json.loads(proc.stdout)
            del row["elapsed_seconds"]
            if (row["mode"] != mode or row["tensors"] != 1 or row["selected_bytes"] != size
                    or row["paired_tensors"] != (mode == "mmap-pair") or row["weights_sha256"] != expected.hexdigest()):
                raise ValueError("A mapped byte copy changed the expected tail")
            results[mode] = row
        shared = ("selected_bytes", "allocated_bytes", "weights_sha256", "metal_sha256")
        if any(results["mmap"][key] != results["mmap-pair"][key] for key in shared):
            raise ValueError("Tail allocation or post-unmap Metal output changed")
        invalid = source.stat().st_size - offset + 1
        plan.write_text(f"{offset} {invalid} U8 1 {invalid}\n")
        result = subprocess.run([binary, source, plan, "mmap-pair"], check=False, capture_output=True)
        if result.returncode != 1 or b"range exceeds validated file bounds" not in result.stderr:
            raise ValueError("A mapped copy accepted a source that ended early")
        print(json.dumps({"binary_sha256": sha256(binary.read_bytes()).hexdigest(), "source_offset": offset,
                          "truncated_source_rejected": True, "cases": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
