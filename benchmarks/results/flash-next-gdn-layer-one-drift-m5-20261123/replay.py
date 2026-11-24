#!/usr/bin/env python3
"""Replay the first differing Flash Next layer after decay tiling."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
POSITIONS = [0, 1, 31, 32, 255, 256, 493, 494, 495, 510, 511, 512, 513, 767,
             1023, 1024, 1039, 1040, 1041, 1535, 1536, 1544, 2046, 2047]
STAGES = (
    "embedding", "ple_operator", "attention_hc_read",
    "gdn_query_input", "gdn_key_input", "gdn_value_input", "gdn_log_decay",
    "gdn_beta", "gdn_core_output", "gdn_gated_output", "gdn",
    "qsa_normalized_query", "qsa_normalized_key", "qsa_attention_before_gate", "qsa",
    "attention_hc_write", "mlp_hc_read", "moe_compute", "mlp_hc_write",
)


def replay(trace, tiling, helper):
    member, read, digest = (helper[name] for name in ("member", "read", "digest"))
    assert member(trace, "status.txt") == b"0\n"
    for name in ("host.txt", "build.json", "source.patch", "prompt.json", "prior-ids.json"):
        assert member(trace, name) == member(tiling, name)
    assert read(trace, "observed/receipt.json") == read(tiling, "observed/receipt.json")
    for line in member(trace, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert read(trace, "build.json")["binary_sha256"] == expected
        else:
            assert digest(trace, name) == expected

    both, pressure = [], {}
    for name, previous, switch in (("narrow-layer-one", "narrow-stock", 0),
                                    ("wide-tiling-layer-one", "wide-tiling", 1)):
        root = f"observed/{name}"
        receipt = read(trace, f"{root}/receipt.json")
        flags = receipt["flags"]
        assert flags["AX_FLASH_NEXT_DUMP_LAYER"] == "1"
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(switch)
        assert list(map(int, flags["AX_FLASH_NEXT_DUMP_POSITIONS"].split(","))) == POSITIONS
        proof = read(trace, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == receipt["original_test_log_sha256"]
        assert receipt["result_sha256"] == digest(trace, f"{root}/result/result.json")
        actual = read(trace, f"{root}/result/result.json")
        assert actual == read(tiling, f"observed/{previous}/result/result.json")
        for cut in (8192, 8193):
            file = f"result/position-{cut}.logits.f32"
            assert member(trace, f"{root}/{file}") == member(tiling, f"observed/{previous}/{file}")
        measurement = read(trace, f"{root}/result/tiling.json")
        assert measurement["tiled_decay_projections"] == switch * 144
        assert digest(trace, f"{root}/result/tiling.json") == receipt["tiling_sha256"]
        assert digest(trace, f"{root}/native-samples.jsonl") == receipt["samples_sha256"]
        monitored = [json.loads(row) for row in member(trace, f"{root}/native-samples.jsonl").splitlines()]
        assert monitored and len(monitored) == receipt["sample_count"]
        assert all(r["pressure"] == 1 and r["footprint_bytes"] <= 96 * 2**30 for r in monitored)
        peak = max(r["footprint_bytes"] for r in monitored)
        assert peak == receipt["sampled_footprint_peak_bytes"]
        pressure[name] = {"normal_samples": len(monitored), "peak_footprint_bytes": peak}

        ops = {}
        for file, expected in receipt["operator_sha256"].items():
            full = f"{root}/operator/{file}"
            assert digest(trace, full) == expected
            if not file.endswith(".json"):
                continue
            entry = read(trace, full)
            stage, position = entry["stage"], entry["position"]
            assert stage in STAGES and position in POSITIONS
            assert entry["layer"] == (None if stage == "embedding" else 1)
            start = entry["chunk_start"]
            end = start + entry["chunk_len"]
            assert [start, end] in actual["cuts"]
            assert start + entry["local_index"] == position < end
            key = stage, position
            assert key not in ops
            ops[key] = (entry["shape"], member(trace, full.removesuffix(".json") + ".f32le"))
        assert ops
        both.append(ops)

    narrow, wide = both
    assert narrow.keys() == wide.keys()
    stages = {}
    for stage in STAGES:
        rows = {}
        tested = 0
        for pos in POSITIONS:
            if (stage, pos) not in narrow:
                continue
            tested += 1
            shape_a, a = narrow[(stage, pos)]
            shape_b, b = wide[(stage, pos)]
            assert shape_a == shape_b
            if a != b:
                left, right = (struct.unpack(f"<{len(raw)//4}f", raw) for raw in (a, b))
                rows[str(pos)] = max(abs(x - y) for x, y in zip(left, right, strict=True))
        if tested:
            stages[stage] = {"tested_rows": tested, "different_positions": rows}
    first = next((stage for stage, result in stages.items() if result["different_positions"]), None)
    return {
        "qualification": False,
        "exact_replay_of_both_full_model_arms": True,
        "layer": 1,
        "positions": POSITIONS,
        "pressure": pressure,
        "earliest_sampled_difference": first,
        "stages": stages,
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(TILING / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as trace, tarfile.open(
        TILING / "evidence.tar.xz"
    ) as tiling:
        print(json.dumps(replay(trace, tiling, helper), indent=2, sort_keys=True))
