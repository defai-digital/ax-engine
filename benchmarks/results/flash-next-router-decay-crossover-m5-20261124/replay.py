#!/usr/bin/env python3
"""Verify the Flash Next router and decay projection crossover."""

from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
LAYER_ONE = HERE.parent / "flash-next-gdn-layer-one-recurrence-m5-20261123"
LAYER_ZERO = HERE.parent / "flash-next-layer-zero-targeted-m5-20261124"
ARMS = {
    "narrow-stock": (512, False, False, False, "narrow-stock"),
    "narrow-both": (512, True, True, False, "narrow-stock"),
    "wide-stock-first": (2048, False, False, False, "wide-stock-first"),
    "wide-router": (2048, False, True, False, None),
    "wide-both": (2048, True, True, False, None),
    "wide-decay": (2048, True, False, False, "wide-tiling"),
    "wide-stock-last": (2048, False, False, False, "wide-stock-last"),
    "wide-both-trace": (2048, True, True, True, None),
}
POSITIONS = (0, 457, 458, 459, 1333, 1334, 1335, 1900, 1901, 1902, 2046)
STAGES = ("mlp_hc_read", "moe_router_logits", "moe_indices", "moe_routed", "mlp_hc_write")


def operator_rows(archive, arm, helper):
    root = f"observed/{arm}"
    read, digest = helper["read"], helper["digest"]
    receipt = read(archive, f"{root}/receipt.json")
    result = read(archive, f"{root}/result/result.json")
    rows = {}
    for file, expected in receipt["operator_sha256"].items():
        if not file.endswith(".json"):
            continue
        path = f"{root}/operator/{file}"
        assert digest(archive, path) == expected
        row = read(archive, path)
        stage, pos = row["stage"], row["position"]
        if stage not in STAGES:
            continue
        assert row["layer"] == 0 and pos in POSITIONS
        begin = row["chunk_start"]
        assert [begin, begin + row["chunk_len"]] in result["cuts"]
        assert begin + row["local_index"] == pos < begin + row["chunk_len"]
        raw = file.removesuffix(".json") + ".f32le"
        assert digest(archive, f"{root}/operator/{raw}") == receipt["operator_sha256"][raw]
        assert (stage, pos) not in rows
        rows[(stage, pos)] = (row["shape"], receipt["operator_sha256"][raw])
    assert set(rows) == {(stage, pos) for stage in STAGES for pos in POSITIONS}
    return rows


def replay(trace, tiling, layer_one, layer_zero, helper, layer_helper):
    member, read, digest = (helper[name] for name in ("member", "read", "digest"))
    assert member(trace, "status.txt") == b"0\n"
    for source, name in ((TILING, "gdn-tiling-trial"), (LAYER_ONE, "gdn-layer-one-recurrence"),
                         (LAYER_ZERO, "layer-zero-targeted")):
        assert sha256((source / "evidence.tar.xz").read_bytes()).hexdigest() == (
            member(trace, f"prior-{name}.sha256").decode().strip()
        )
    for file in ("host.txt", "prompt.json", "prior-ids.json", "mxfp4-inventory.json"):
        assert member(trace, file) == member(tiling, file)
    build = read(trace, "build.json")
    suite = read(trace, "observed/receipt.json")
    assert suite["qualification"] is False and suite["build"] == build
    assert suite["manifest"] == read(tiling, "observed/receipt.json")["manifest"]
    assert suite["manifest"]["model-manifest.json"] == build["manifest_sha256"]
    assert suite["prompt_sha256"] == digest(trace, "prompt.json")
    assert suite["prior_ids_sha256"] == digest(trace, "prior-ids.json")
    assert build["codesign_verified"] and build["profile"] == "release-server"
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert build["patch_sha256"] == digest(trace, "source.patch")
    for line in member(trace, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert build["binary_sha256"] == expected
        else:
            assert digest(trace, name) == expected

    results, timings = {}, {}
    for name, (width, decay, router, captured, old) in ARMS.items():
        root = f"observed/{name}"
        receipt = read(trace, f"{root}/receipt.json")
        flags = receipt["flags"]
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(int(decay))
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_MOE_ROUTER"] == str(int(router))
        assert flags["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(width)
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY"] == str(int(captured))
        assert flags["AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX"] == "1"
        for flag in ("GROUPED_EVAL", "PAIRED_GATHER", "INDEXED_PREFILL"):
            assert flags[f"AX_MLX_FLASH_NEXT_QSA_{flag}"] == "0"
        proof = read(trace, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == receipt["original_test_log_sha256"]
        for key, file in (
            ("result_sha256", "result/result.json"),
            ("tiling_sha256", "result/tiling.json"),
            ("samples_sha256", "native-samples.jsonl"),
        ):
            assert receipt[key] == digest(trace, f"{root}/{file}")
        result = read(trace, f"{root}/result/result.json")
        assert result["width"] == width and result["qualification"] is False
        assert len(result["output_ids"]) == 64
        assert result["cuts"] == read(tiling, f"observed/{'narrow-stock' if width == 512 else 'wide-stock-first'}/result/result.json")["cuts"]
        for witness in result["witnesses"][-2:]:
            decision = witness["logits"]
            token = result["output_ids"][witness["position"] - 8192]
            assert decision["token"] == token
            blob = decision["logits_f32_le"]
            raw = member(trace, f"{root}/result/{blob['file']}")
            assert len(raw) == blob["bytes"] and sha256(raw).hexdigest() == blob["sha256"]
            values = struct.unpack(f"<{decision['shape'][1]}f", raw)
            assert all(map(math.isfinite, values))
            assert max(range(len(values)), key=values.__getitem__) == token
        if old:
            for file in ("result.json", "position-8192.logits.f32", "position-8193.logits.f32"):
                assert member(trace, f"{root}/result/{file}") == member(
                    tiling, f"observed/{old}/result/{file}"
                )
        measurements = read(trace, f"{root}/result/tiling.json")
        assert measurements["tiled_decay_projections"] == (144 if decay and width == 2048 else 0)
        assert measurements["tiled_router_projections"] == (192 if router and width == 2048 else 0)
        assert 0 < measurements["first_token_ns"] <= measurements["total_ns"]
        assert 0 < measurements["mlx_peak_bytes"] <= 96 * 2**30
        monitored = [json.loads(row) for row in member(trace, f"{root}/native-samples.jsonl").splitlines()]
        assert monitored and len(monitored) == receipt["sample_count"]
        assert all(row["pressure"] == 1 and row["footprint_bytes"] <= 96 * 2**30 for row in monitored)
        peak = max(row["footprint_bytes"] for row in monitored)
        assert peak == receipt["sampled_footprint_peak_bytes"]
        timings[name] = measurements | {"normal_samples": len(monitored), "peak_footprint_bytes": peak}
        results[name] = result

    trace_arm = "wide-both-trace"
    assert results[trace_arm] == results["wide-both"]
    root = f"observed/{trace_arm}"
    receipt = read(trace, f"{root}/receipt.json")
    expected_operator = operator_rows(layer_zero, "narrow-incoming-bridge", helper)
    actual_operator = operator_rows(trace, trace_arm, helper)
    for key in expected_operator:
        assert expected_operator[key][0] == actual_operator[key][0]
    changed_operator = {
        stage: [pos for pos in POSITIONS if expected_operator[(stage, pos)] != actual_operator[(stage, pos)]]
        for stage in STAGES
    }
    assert receipt["recurrence_sha256"] == digest(trace, f"{root}/result/recurrence.json")
    actual_layer_one = read(trace, f"{root}/result/recurrence.json")
    expected_layer_one = read(layer_one, "observed/narrow-layer-one-recurrence/result/recurrence.json")
    assert actual_layer_one["layer"] == expected_layer_one["layer"] == 1
    assert actual_layer_one["partition"] == [2047]
    cmp_rows = layer_helper["compare"]
    gdn = {
        name: {
            key: cmp_rows(expected_layer_one[name][key], actual_layer_one[name][key])
            for key in expected_layer_one[name]
        } for name in ("prework_rows", "input_rows")
    }
    gdn["output"] = cmp_rows(expected_layer_one["original_output"], actual_layer_one["original_output"])
    gdn["same_final_state"] = (
        expected_layer_one["original_boundary_states"][-1] == actual_layer_one["original_boundary_states"][-1]
    )
    compare = helper["compare_results"]
    return {
        "qualification": False,
        "source_base": build["source_base"],
        "exact_archived_stock_and_decay_controls": True,
        "measurements": timings,
        "wider_vs_narrow": {
            name: compare(results["narrow-stock"], results[name])
            for name in ("wide-stock-first", "wide-router", "wide-decay", "wide-both")
        },
        "combined_vs_wide_stock": compare(results["wide-stock-first"], results["wide-both"]),
        "combined_layer_zero_different_positions": changed_operator,
        "combined_layer_one_vs_narrow": gdn,
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(TILING / "replay.py"))
    layer_helper = runpy.run_path(str(LAYER_ONE / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as current, tarfile.open(
        TILING / "evidence.tar.xz"
    ) as old, tarfile.open(LAYER_ONE / "evidence.tar.xz") as one, tarfile.open(
        LAYER_ZERO / "evidence.tar.xz"
    ) as zero:
        print(json.dumps(replay(current, old, one, zero, helper, layer_helper), indent=2, sort_keys=True))
