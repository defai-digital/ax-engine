#!/usr/bin/env python3
"""Locate remaining layer-zero activation drift after decay projection tiling."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
OPERATOR = HERE.parent / "flash-next-prefill-operator-drift-m5-20261123"
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
ARM = "wide-tiling-residual"


def replay(trace, tiling, original):
    helper = runpy.run_path(str(OPERATOR / "replay.py"))
    member, read, samples = (helper[name] for name in ("member", "read", "samples"))
    assert member(trace, "status.txt") == b"0\n"
    assert member(trace, "host.txt") == member(tiling, "host.txt")
    assert member(trace, "build.json") == member(tiling, "build.json")
    assert member(trace, "source.patch") == member(tiling, "source.patch")
    assert member(trace, "prompt.json") == member(tiling, "prompt.json")
    assert sha256((OPERATOR / "evidence.tar.xz").read_bytes()).hexdigest() == (
        member(trace, "gdn-tiling-operator.sha256").decode().strip()
    )
    for line in member(trace, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert read(trace, "build.json")["binary_sha256"] == expected
        else:
            assert sha256(member(trace, name)).hexdigest() == expected

    root = f"observed/{ARM}"
    receipt = read(trace, f"{root}/receipt.json")
    proof = read(trace, f"{root}/log-proof.json")
    assert proof["test_result"] == "1 passed; 0 failed"
    assert proof["original_sha256"] == receipt["original_test_log_sha256"]
    assert receipt["flags"]["AX_FLASH_NEXT_DUMP_LAYER"] == "0"
    assert receipt["flags"]["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == "1"
    assert receipt["result_sha256"] == sha256(member(trace, f"{root}/result/result.json")).hexdigest()
    for name, expected in receipt["operator_sha256"].items():
        assert sha256(member(trace, f"{root}/operator/{name}")).hexdigest() == expected
    actual = read(trace, f"{root}/result/result.json")
    steady = "observed/wide-tiling/result"
    assert actual == read(tiling, f"{steady}/result.json")
    for pos in (8192, 8193):
        assert member(trace, f"{root}/result/position-{pos}.logits.f32") == member(
            tiling, f"{steady}/position-{pos}.logits.f32"
        )
    run_samples = member(trace, f"{root}/native-samples.jsonl")
    assert sha256(run_samples).hexdigest() == receipt["samples_sha256"]
    parsed = [json.loads(line) for line in run_samples.splitlines()]
    assert len(parsed) == receipt["sample_count"] and parsed
    assert all(row["pressure"] == 1 and row["footprint_bytes"] <= 96 * 2**30 for row in parsed)
    peak = max(row["footprint_bytes"] for row in parsed)
    assert peak == receipt["sampled_footprint_peak_bytes"]

    narrow = read(original, "observed/512/result/result.json")
    expected = samples(original, 512, narrow["cuts"])
    observed = samples(trace, ARM, actual["cuts"])
    assert expected.keys() == observed.keys()
    stages = {}
    for stage in helper["STAGES"]:
        pairs = {}
        for position in helper["POSITIONS"]:
            key = stage, position
            if key not in expected:
                continue
            a, shape = expected[key]
            b, observed_shape = observed[key]
            assert shape == observed_shape
            if a != b:
                av, bv = (struct.unpack(f"<{len(data)//4}f", data) for data in (a, b))
                pairs[str(position)] = max(abs(x - y) for x, y in zip(av, bv, strict=True))
        if any(name == stage for name, _ in expected):
            stages[stage] = pairs
    differences = [stage for stage in helper["STAGES"] if stages.get(stage)]
    return {
        "qualification": False,
        "candidate_ids_and_states_unchanged_by_profiling": True,
        "normal_pressure_samples": len(parsed),
        "sampled_footprint_peak_bytes": peak,
        "positions": helper["POSITIONS"],
        "earliest_sampled_difference": differences[0] if differences else None,
        "maximum_absolute_deltas_by_position": stages,
    }


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz") as trace, tarfile.open(
        TILING / "evidence.tar.xz"
    ) as tiling, tarfile.open(OPERATOR / "evidence.tar.xz") as original:
        print(json.dumps(replay(trace, tiling, original), indent=2, sort_keys=True))
