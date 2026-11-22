"""Replay primary-tier concurrent Flash Next shared-weight startup."""

from gzip import decompress
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "flash-next-shared-cold-weights-m2-20261122/receipt.json"
TEST = "runner::flash_next_tests::flash_next_real_concurrent_cold_runners_share_weights"


def check(valid, message):
    if not valid:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def replay():
    m2, m5 = read(PREVIOUS), read(ROOT / "receipt.json")
    build_file = ROOT / "test-build.json"
    build = read(build_file)
    inputs = dict(line.split()[::-1] for line in (ROOT / "inputs-m5.sha256").read_text().splitlines())
    check((ROOT / "status.txt").read_text() == "0\n" and m5["result"] == 0
          and m5["qualification"] is False and m5["scope"] == m2["scope"]
          and m5["hardware"] ==
              {"soc": "Apple M5 Max", "memory_bytes": 128 * 2**30, "os_version": "27.2.1"},
          "The primary-tier functional check did not finish")
    check(m5["build"] == m2["build"] == build
          and inputs["test-build.json"] == sha256(build_file.read_bytes()).hexdigest()
          and inputs["ax-flash-shared-load-test"] == build["binary_sha256"]
          and m5["pack_revision"] == m2["pack_revision"]
          and m5["inventory_sha256"] == m2["inventory_sha256"] == inputs["inventory.json"]
          and m5["manifest"] == m2["manifest"],
          "The pinned native executable or pack changed")
    record = decompress((ROOT / "test-output.txt.gz").read_bytes())
    text = record.decode()
    duration = re.search(r"test result: ok\. 1 passed; 0 failed; [^\n]+; finished in (\d+\.\d+)s", text)
    check(m5["test"] == m2["test"] == TEST
          and sha256(record).hexdigest() == m5["test_output_sha256"]
          and f"test {TEST} ... ok" in text and duration is not None
          and "Two independent Flash Next runners share one weight allocation and match all output IDs." in text,
          "The real-pack test output did not prove a passing shared-weight run")
    check(m5["native_runners"] == 2 and m5["weight_allocations"] == 1
          and m5["identical_ids_per_runner"] == 8 and set(m5["pressure_samples"]) == {1},
          "The observation changed expected allocation, output, or pressure")
    return {
        "scope": m5["scope"], "qualification": False, "same_binary_as_m2": True,
        "pack_revision": m5["pack_revision"], "native_runners": m5["native_runners"],
        "shared_weight_allocations": m5["weight_allocations"],
        "matching_output_ids_per_runner": m5["identical_ids_per_runner"],
        "normal_pressure_samples": len(m5["pressure_samples"]),
        "observed_test_duration_s": float(duration.group(1)),
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
