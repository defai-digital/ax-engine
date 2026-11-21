"""Replay the grouped Flash Next timing and correctness checks from the raw receipt."""

import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT.parent / "flash-next-mxfp4-prefill-partition-m5-20261121/holdout-8192-prompts.jsonl.gz"
INVENTORY = ROOT.parent / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assess():
    build = json.loads((ROOT / "build.json").read_text())
    raw = gzip.decompress((ROOT / "result.json.gz").read_bytes())
    record = json.loads(raw)
    manifest = {path: expected for expected, path in (
        line.split(maxsplit=1) for line in (ROOT / "live-inputs.sha256").read_text().splitlines()
    )}
    inputs = {
        "build.json": (ROOT / "build.json").read_bytes(),
        "grouped-prefill.patch": gzip.decompress((ROOT / "grouped-prefill.patch.gz").read_bytes()),
        "driver/prompts.jsonl": gzip.decompress(PROMPTS.read_bytes()),
        "driver/inventory.json": INVENTORY.read_bytes(),
    }
    binary = "driver/ax-engine-server-grouped-lab"
    assert record["status"] == "complete"
    assert record["binary_sha256"] == build["sha256"][binary] == manifest.pop(binary)
    assert record["prompts_sha256"] == digest(inputs["driver/prompts.jsonl"])
    assert record["inventory_sha256"] == digest(inputs["driver/inventory.json"])
    with tempfile.TemporaryDirectory() as temp:
        driver = Path(temp)
        with tarfile.open(ROOT / "driver-sources.tar.gz") as archive:
            for member in archive:
                data = archive.extractfile(member).read()
                inputs[member.name] = data
                if member.name.startswith("driver/") and member.name.endswith(".py"):
                    (driver / Path(member.name).name).write_bytes(data)
        assert {path: digest(data) for path, data in inputs.items()} == manifest
        sys.path.insert(0, str(driver))
        from run_grouped_prefill import ARMS, PROMPT_LENGTH, gaps
        from verify_flash_next_prefill_split import (
            COUNTERS, PACK_REVISION, SPLIT_PREFIX, metrics, verify_partition, verify_route_overlap,
        )

        assert record["pack_revision"] == PACK_REVISION
        arms = record["arms"]
        assert set(arms) == set(ARMS)

        def drained(counters):
            assert counters.keys() == COUNTERS and not any(counters.values())

        serial = arms["off"]["serial"]
        assert len(serial) == 2
        expected = {row["prompt_id"]: row["output_tokens"] for row in serial}
        assert len(expected) == 2 and all(len(ids) == 96 for ids in expected.values())
        assert all(verify_partition(row, PROMPT_LENGTH, False) == 0 for row in serial)
        drained(arms["off"]["serial_drained"])

        summary = {}
        for name, widths in ARMS.items():
            arm = arms[name]
            assert arm["turn_widths"] == widths
            assert [trial["leader"] for trial in arm["trials"]] == [0, 1]
            leaders = []
            for trial in arm["trials"]:
                leader = trial["leader"]
                requests = trial["requests"]
                assert len(requests) == 2
                for index, row in enumerate(requests):
                    assert row["output_tokens"] == expected[row["prompt_id"]]
                    splits = verify_partition(row, PROMPT_LENGTH, bool(widths), widths)
                    assert splits == (4 if widths and index != leader else 0)
                if widths:
                    verify_route_overlap(trial, widths[0])
                by_request = [gaps(requests[i], requests[1 - i]) for i in range(2)]
                assert by_request == trial["per_request"]
                assert all(
                    row["intervals_at_or_over_100ms"]
                    == row["intervals_at_or_over_100ms_adjacent_to_peer_prefill"]
                    for row in by_request
                )
                assert metrics(trial) == trial["aggregate"]
                drained(trial["drained"])
                leaders.append(by_request[leader])
            samples = arm["memory_samples"]
            assert {row["pressure"] for row in samples} == {1}
            summary[name] = {
                "leader_median_p95_gap_seconds": statistics.median(
                    row["gap_p95_seconds"] for row in leaders
                ),
                "leader_max_gap_seconds": max(row["gap_max_seconds"] for row in leaders),
                "leader_stalls_at_or_over_100ms": [row["intervals_at_or_over_100ms"] for row in leaders],
                "normal_pressure_samples": len(samples),
            }

        canceled = arms["two"]["disconnect"]
        removed = canceled["disconnected"]
        cut = removed["prefill_progress"][0]
        assert not removed["first_output_tokens"] and cut["cursor"] == ARMS["two"][0]
        assert cut["prefill_route"] == {f'{SPLIT_PREFIX}{removed["request_id"]}': 2048}
        assert cut["step_id"] in canceled["shared_steps"]
        for label in ("survivor", "replacement"):
            row = canceled[label]
            assert row["output_tokens"] == expected[row["prompt_id"]]
        drained(canceled["drained_after_disconnect"])
        drained(canceled["drained_after_replacement"])

    return {
        "binary_sha256": record["binary_sha256"],
        "result_sha256": digest(raw),
        "paired_requests_matching_serial": 12,
        "cancellation_survivor_and_replacement_matching_serial": 2,
        "arms": summary,
    }


if __name__ == "__main__":
    print(json.dumps(assess(), indent=2, sort_keys=True))
