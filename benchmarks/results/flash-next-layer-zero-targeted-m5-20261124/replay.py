#!/usr/bin/env python3
"""Trace the layer-zero stages preceding the three input differences in layer one."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
CONTROL = HERE.parent / "flash-next-gdn-layer-one-recurrence-m5-20261123"
POSITIONS = [0, 457, 458, 459, 1333, 1334, 1335, 1900, 1901, 1902, 2046]
STAGES = (
    "embedding", "ple_operator", "attention_hc_read",
    "gdn_query_input", "gdn_key_input", "gdn_value_input", "gdn_log_decay",
    "gdn_beta", "gdn_core_output", "gdn_gated_output", "gdn",
    "qsa_normalized_query", "qsa_normalized_key", "qsa_attention_before_gate", "qsa",
    "attention_hc_write", "mlp_hc_read",
    "moe_router_logits", "moe_indices", "moe_down", "moe_routed",
    "moe_shared_gate", "moe_shared_up", "moe_shared_activation", "moe_shared_input",
    "moe_shared_down", "moe_shared_router", "moe_shared_score", "moe_shared_output",
    "moe_delta", "moe_compute", "mlp_hc_write",
)


def verify_projection(component, inventory, narrow, wide):
    assert component["pack_revision"] == inventory["revision"]
    assert component["mlx"] == "0.32.4" and component["mode"] == "affine8/group32"
    assert component["partition"] == [512, 512, 512, 511]
    assert component["verified_positions"] == POSITIONS
    published = {member["name"]: member["publisher_sha256"] for member in inventory["members"]}
    assert component["shards"] and all(published[name] == digest for name, digest in component["shards"].items())
    recorded_inputs, outputs = [], []
    for arm in (narrow, wide):
        recorded_inputs.append({str(pos): sha256(arm["mlp_hc_read", pos][1]).hexdigest()
                                for pos in POSITIONS})
        outputs.append({str(pos): sha256(arm["moe_router_logits", pos][1]).hexdigest()
                        for pos in POSITIONS})
    assert component["input_rows"] == recorded_inputs[0] == recorded_inputs[1]
    assert component["archived_narrow_rows"] == outputs[0]
    assert component["archived_wide_rows"] == outputs[1]
    variants = component["variants"]
    assert set(variants) == {"canonical_tiles", "whole_2047", "padded_2048"}
    different = [pos for pos in POSITIONS if outputs[0][str(pos)] != outputs[1][str(pos)]]
    assert different == [458, 1334, 1335, 1901]
    for name, variant in variants.items():
        narrow_layout = name == "canonical_tiles"
        assert variant["output_rows"] == outputs[0 if narrow_layout else 1]
        assert variant["different_from_narrow"] == ([] if narrow_layout else different)
        assert variant["different_from_wide"] == (different if narrow_layout else [])


def replay(trace, control, helper):
    member, read, digest = (helper[name] for name in ("member", "read", "digest"))
    assert member(trace, "status.txt") == b"0\n"
    assert sha256((CONTROL / "evidence.tar.xz").read_bytes()).hexdigest() == (
        member(trace, "prior-layer-one-recurrence.sha256").decode().strip()
    )
    summary = json.loads((CONTROL / "comparison.json").read_text())
    assert summary["prework"]["attention_input"]["first_positions"] == [458, 1334, 1901]
    for name in ("host.txt", "build.json", "source.patch", "prompt.json", "prior-ids.json"):
        assert member(trace, name) == member(control, name)
    assert read(trace, "observed/receipt.json") == read(control, "observed/receipt.json")
    for line in member(trace, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert read(trace, "build.json")["binary_sha256"] == expected
        else:
            assert digest(trace, name) == expected

    both, pressure = [], {}
    for name, previous, switch in (("narrow-incoming-bridge", "narrow-layer-one-recurrence", 0),
                                    ("wide-tiling-incoming-bridge", "wide-tiling-layer-one-recurrence", 1)):
        root = f"observed/{name}"
        receipt = read(trace, f"{root}/receipt.json")
        flags = receipt["flags"]
        assert flags["AX_FLASH_NEXT_DUMP_LAYER"] == "0"
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(switch)
        assert list(map(int, flags["AX_FLASH_NEXT_DUMP_POSITIONS"].split(","))) == POSITIONS
        proof = read(trace, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == receipt["original_test_log_sha256"]
        assert receipt["result_sha256"] == digest(trace, f"{root}/result/result.json")
        actual = read(trace, f"{root}/result/result.json")
        assert actual == read(control, f"observed/{previous}/result/result.json")
        for cut in (8192, 8193):
            file = f"result/position-{cut}.logits.f32"
            assert member(trace, f"{root}/{file}") == member(control, f"observed/{previous}/{file}")
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
            assert entry["layer"] == (None if stage == "embedding" else 0)
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
    component = json.loads((HERE / "router_projection_component.json").read_text())
    verify_projection(component, read(trace, "mxfp4-inventory.json"), narrow, wide)
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
        "layer": 0,
        "positions": POSITIONS,
        "pressure": pressure,
        "earliest_sampled_difference": first,
        "stages": stages,
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(TILING / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as trace, tarfile.open(
        CONTROL / "evidence.tar.xz"
    ) as control:
        print(json.dumps(replay(trace, control, helper), indent=2, sort_keys=True))
