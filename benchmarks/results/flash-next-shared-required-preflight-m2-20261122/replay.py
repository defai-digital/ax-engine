"""Verify the fixed-pack shared Flash Next preflight controls."""

from argparse import ArgumentParser
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
BASELINE = "fd85d821fa6bfd7fef55475eb6104f498381d022"
FIXED = "e91bf4afb32f5ba08a6e72b1cfb51d631fced3d3"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_control(name, expected_commit):
    receipt = json.loads((ROOT / f"{name}.json").read_bytes())
    raw = decompress((ROOT / f"{name}.txt.gz").read_bytes())
    build = receipt["build"]
    check(sha256(raw).hexdigest() == receipt["test_output_sha256"]
          and b"1 passed; 0 failed" in raw, f"{name}: test output mismatch")
    check(build["commit"] == expected_commit and build["mlx"] == build["mlx_metal"] == "0.32.4"
          and build["rustc"].startswith("rustc 1.99.1 ")
          and build["cargo"].startswith("cargo 1.99.1 "), f"{name}: build mismatch")
    check(receipt["pack_revision"] == REVISION and receipt["inventory_sha256"] == INVENTORY
          and receipt["qualification"] is False, f"{name}: pack or scope mismatch")
    return receipt, raw


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional worktree at the fixed source")
    args = parser.parse_args()
    for name, commit, loaded in (("baseline", BASELINE, True), ("bypassed", FIXED, False)):
        receipt, raw = read_control(name, commit)
        after = str(loaded).lower()
        observed = f"shared_after=MlxSharedWeightsCell {{ loaded: {after} }}"
        elapsed = receipt["elapsed_seconds"]
        check(receipt["manifest_sha256"] == MANIFEST and receipt["publisher_members"] == 49
              and receipt["hardware"] == "Mac Studio M2 Ultra 192 GB"
              and set(receipt["pressure_levels"]) == {1}, f"{name}: environment mismatch")
        check(receipt["shared_loaded_before"] is False and receipt["shared_loaded_after"] is loaded
              and receipt["activation_error"] == "MlxMtpRequiredButUnavailable"
              and b"shared_before=MlxSharedWeightsCell { loaded: false }" in raw
              and f"{observed} elapsed_seconds={elapsed:.6f}".encode() in raw,
              f"{name}: shared state or timing mismatch")
        print(f"{name}: typed rejection in {elapsed:.6f} s, shared weights loaded: {after}")

    receipt, raw = read_control("attached", FIXED)
    build = receipt["build"]
    check(receipt["manifest"]["model-manifest.json"] == MANIFEST
          and receipt["native_runners"] == 2 and receipt["weight_allocations"] == 1
          and receipt["identical_ids_per_runner"] == 8 and set(receipt["pressure_samples"]) == {1}
          and receipt["result"] == 0 and receipt["test"].encode() in raw
          and b"match all output IDs" in raw, "attached: reuse control mismatch")
    if args.source is not None:
        for filename, digest in build["changed_source_sha256"].items():
            check(sha256((args.source / filename).read_bytes()).hexdigest() == digest,
                  f"Tested source differs: {filename}")
    print("attached: two runners, one shared allocation, eight identical output IDs per runner")


if __name__ == "__main__":
    main()
