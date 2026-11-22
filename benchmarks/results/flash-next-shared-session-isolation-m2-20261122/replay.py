"""Replay the real Flash Next cold shared-session isolation control."""

from argparse import ArgumentParser
from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "flash-next-shared-cold-weights-m5-20261122/receipt.json"
TEST = "runner::flash_next_tests::flash_next_real_concurrent_cold_runners_share_weights"
RESULT = "Two cold Flash Next runners share weights and match distinct solo trajectories at the same request ID."


def need(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return sha256(data).hexdigest()


def replay(source=None):
    m5, receipt = read(BASE), read(ROOT / "receipt.json")
    build_path = ROOT / "test-build.json"
    build = read(build_path)
    sums = dict(line.split()[::-1] for line in (ROOT / "inputs.sha256").read_text().splitlines())
    need((ROOT / "status.txt").read_text() == "0\n" and receipt["result"] == 0
         and receipt["scope"] == "native functional session isolation with shared cold weights"
         and receipt["qualification"] is False and receipt["hardware"] ==
             {"soc": "Apple M2 Ultra", "memory_bytes": 192 * 2**30, "os_version": "27.2.1"},
         "The functional control did not complete")
    need(receipt["build"] == build and build["profile"] == "test"
         and build["libmlx_sha256"] == m5["build"]["libmlx_sha256"]
         and build["binary_sha256"] == sums["ax-flash-shared-load-test"]
         and sums["test-build.json"] == digest(build_path.read_bytes())
         and sums["check.py"] == digest((ROOT / "check.py").read_bytes())
         and build["test_patch_sha256"] == sums["test.patch"] == digest(decompress((ROOT / "test.patch.gz").read_bytes())),
         "The staged source or executable binding changed")
    need(receipt["inventory_sha256"] == m5["inventory_sha256"] == sums["inventory.json"]
         and receipt["pack_revision"] == m5["pack_revision"] and receipt["manifest"] == m5["manifest"],
         "The pinned publisher pack changed")
    raw = decompress((ROOT / "test-output.txt.gz").read_bytes())
    output = raw.decode()
    need(digest(raw) == receipt["test_output_sha256"] and receipt["test"] == TEST
         and f"test {TEST} ... ok" in output and "1 passed; 0 failed" in output and RESULT in output,
         "The native test output changed or lacks the isolation result")
    scope = {
        "weight_allocations": 1, "concurrent_runners": 2, "solo_runners": 2,
        "shared_request_id": 1, "matched_solo_ids_per_runner": 8,
    }
    need(all(receipt[key] == expected for key, expected in scope.items())
         and set(receipt["pressure_samples"]) == {1},
         "The observed request isolation or memory pressure changed")
    if source is not None:
        for name, expected in build["source_sha256"].items():
            need(digest((source / name).read_bytes()) == expected, f"Tested source changed: {name}")
    return {
        "scope": receipt["scope"], "qualification": False,
        "source_tree": build["source_tree"], "pack_revision": receipt["pack_revision"],
        "normal_pressure_samples": len(receipt["pressure_samples"]), **scope,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.source), indent=2, allow_nan=False))
