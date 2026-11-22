"""Verify the real-pack concurrent cold-load control."""
from argparse import ArgumentParser
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def check(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional worktree at the tested source")
    args = parser.parse_args()
    receipt = json.loads((ROOT / "receipt.json").read_bytes())
    build = receipt["build"]
    raw = decompress((ROOT / "test-output.txt.gz").read_bytes())
    check(sha256(raw).hexdigest() == receipt["test_output_sha256"], "Test output changed")
    check(receipt["result"] == 0 and receipt["test"].encode() in raw
          and b"1 passed; 0 failed" in raw
          and b"Two independent Flash Next runners share one weight allocation and match all output IDs." in raw,
          "The pinned native test did not pass")
    check(receipt["pack_revision"] == REVISION and receipt["manifest"]["model-manifest.json"] == MANIFEST
          and build["commit"].startswith("89eac33d2") and build["mlx"] == build["mlx_metal"] == "0.32.4"
          and "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"],
          "Model, binary or build identity changed")
    check(receipt["native_runners"] == 2 and receipt["weight_allocations"] == 1
          and receipt["identical_ids_per_runner"] == 8 and set(receipt["pressure_samples"]) == {1},
          "Unexpected shared allocation, output, or pressure")
    if args.source is not None:
        for name, digest in build["source_sha256"].items():
            check(sha256((args.source / name).read_bytes()).hexdigest() == digest,
                  f"Tested source differs: {name}")
    print("Two cold native sessions, one weight allocation, eight matching token IDs per session.")


if __name__ == "__main__":
    main()
