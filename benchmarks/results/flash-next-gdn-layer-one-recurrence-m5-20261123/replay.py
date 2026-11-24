#!/usr/bin/env python3
"""Compare the full layer-one recurrence inputs with the signed Flash Next controls."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile

HERE = Path(__file__).resolve().parent
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
LAYER = HERE.parent / "flash-next-gdn-layer-one-drift-m5-20261123"
PREFIX = 2047
PROJECTION = (
    "attention_input", "qkv_projection", "convolved_query", "convolved_key",
    "convolved_value",
)
INPUTS = ("query", "key", "value", "decay", "beta")
DECAY = ("projection", "biased_projection", "negative_abs_exp", "log1p", "softplus", "log_decay", "decay")
REPLAYS = ("whole", "split_lazy", "split_fenced")
ARMS = (
    ("narrow-layer-one-recurrence", "narrow-layer-one", 512, [512, 512, 512, 511]),
    ("wide-tiling-layer-one-recurrence", "wide-tiling-layer-one", 2048, [2047]),
)


def compare(a, b):
    assert a["shape"] == b["shape"]
    assert a["shape"][:2] == [1, PREFIX]
    assert len(a["rows"]) == len(b["rows"]) == PREFIX
    changed = [i for i, (x, y) in enumerate(zip(a["rows"], b["rows"], strict=True)) if x != y]
    return {
        "different_rows": len(changed),
        "first_different_row": changed[0] if changed else None,
        "last_different_row": changed[-1] if changed else None,
        "first_positions": changed[:16],
    }


def with_state(a, b):
    return compare(a["output"], b["output"]) | {"same_final_state": a["final_state"] == b["final_state"]}


def replay(trace, prior, helper):
    member, read, digest = (helper[name] for name in ("member", "read", "digest"))
    assert member(trace, "status.txt") == b"0\n"
    for evidence, name in ((TILING, "gdn-tiling-trial"), (LAYER, "gdn-layer-one-drift")):
        assert sha256((evidence / "evidence.tar.xz").read_bytes()).hexdigest() == (
            member(trace, f"prior-{name}.sha256").decode().strip()
        )
    for name in ("host.txt", "prompt.json", "prior-ids.json", "mxfp4-inventory.json"):
        assert member(trace, name) == member(prior, name)
    suite = read(trace, "observed/receipt.json")
    build = read(trace, "build.json")
    assert suite["qualification"] is False and suite["build"] == build
    assert suite["manifest"] == read(prior, "observed/receipt.json")["manifest"]
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

    captured, pressure, witnesses = {}, {}, {}
    for name, old, width, partition in ARMS:
        root, old = f"observed/{name}", f"observed/{old}"
        receipt = read(trace, f"{root}/receipt.json")
        proof = read(trace, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == receipt["original_test_log_sha256"]
        flags = receipt["flags"]
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_LAYER"] == "1"
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY"] == "1"
        assert flags["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(width)
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(int(width == 2048))
        for flag in ("GROUPED_EVAL", "PAIRED_GATHER", "INDEXED_PREFILL"):
            assert flags[f"AX_MLX_FLASH_NEXT_QSA_{flag}"] == "0"
        assert flags["AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX"] == "1"
        for key, file in (
            ("result_sha256", "result/result.json"),
            ("tiling_sha256", "result/tiling.json"),
            ("recurrence_sha256", "result/recurrence.json"),
            ("samples_sha256", "native-samples.jsonl"),
        ):
            assert receipt[key] == digest(trace, f"{root}/{file}")
        for file in ("result.json", "position-8192.logits.f32", "position-8193.logits.f32"):
            assert member(trace, f"{root}/result/{file}") == member(prior, f"{old}/result/{file}")
        measurement = read(trace, f"{root}/result/tiling.json")
        assert measurement["tiled_decay_projections"] == (144 if width == 2048 else 0)
        monitored = [json.loads(row) for row in member(trace, f"{root}/native-samples.jsonl").splitlines()]
        assert monitored and len(monitored) == receipt["sample_count"]
        assert all(row["pressure"] == 1 and row["footprint_bytes"] <= 96 * 2**30 for row in monitored)
        peak = max(row["footprint_bytes"] for row in monitored)
        assert peak == receipt["sampled_footprint_peak_bytes"]
        pressure[name] = {"normal_samples": len(monitored), "peak_footprint_bytes": peak}
        value = read(trace, f"{root}/result/recurrence.json")
        assert value["layer"] == 1 and value["partition"] == partition
        assert value["recurrence"] == "mlx_gated_delta_update"
        assert set(value["prework_rows"]) == set(PROJECTION)
        assert set(value["input_rows"]) == set(INPUTS)
        assert set(value["decay_stages"]) == set(DECAY)
        assert set(value["replays"]) == set(REPLAYS)
        assert len(value["original_boundary_states"]) == len(partition) + 1
        query = value["input_rows"]["query"]["shape"]
        val = value["input_rows"]["value"]["shape"]
        assert value["original_boundary_states"][0] == sha256(
            bytes(4 * query[0] * query[-1] * val[-2] * val[-1])
        ).hexdigest()
        captured[name] = value

        rows = value["prework_rows"] | value["input_rows"]
        sampled = {
            "attention_hc_read": rows["attention_input"],
            "gdn_query_input": rows["convolved_query"],
            "gdn_key_input": rows["convolved_key"],
            "gdn_value_input": rows["convolved_value"],
            "gdn_log_decay": value["decay_stages"]["log_decay"]["rows"],
            "gdn_beta": rows["beta"],
            "gdn_core_output": value["original_output"],
        }
        original_receipt = read(prior, f"{old}/receipt.json")
        total = 0
        for file, expected in original_receipt["operator_sha256"].items():
            if not file.endswith(".json"):
                continue
            entry = read(prior, f"{old}/operator/{file}")
            pos, stage = entry["position"], entry["stage"]
            if pos >= PREFIX or stage not in sampled:
                continue
            raw = file.removesuffix(".json") + ".f32le"
            assert sampled[stage]["rows"][pos] == original_receipt["operator_sha256"][raw]
            assert digest(prior, f"{old}/operator/{raw}") == original_receipt["operator_sha256"][raw]
            total += 1
        assert total == 7 * 23
        witnesses[name] = total

    a, b = captured.values()
    return {
        "qualification": False,
        "layer": 1,
        "source_base": build["source_base"],
        "exact_full_model_arm_repeats": True,
        "matched_prior_operator_rows": witnesses,
        "pressure": pressure,
        "prework": {name: compare(a["prework_rows"][name], b["prework_rows"][name]) for name in PROJECTION},
        "recurrence_inputs": {name: compare(a["input_rows"][name], b["input_rows"][name]) for name in INPUTS},
        "decay_stages": {name: compare(a["decay_stages"][name]["rows"], b["decay_stages"][name]["rows"])
                          for name in DECAY},
        "live_recurrence": with_state(
            {"output": a["original_output"], "final_state": a["original_boundary_states"][-1]},
            {"output": b["original_output"], "final_state": b["original_boundary_states"][-1]},
        ),
        "same_input_replays_vs_live": {
            name: {
                key: with_state({
                    "output": arm["original_output"], "final_state": arm["original_boundary_states"][-1],
                }, arm["replays"][key]) for key in REPLAYS
            } for name, arm in captured.items()
        },
        "same_input_splits_vs_whole": {
            name: {key: with_state(arm["replays"]["whole"], arm["replays"][key])
                   for key in ("split_lazy", "split_fenced")} for name, arm in captured.items()
        },
        "fixed_partition_cross_arm_replays": {
            key: with_state(a["replays"][key], b["replays"][key]) for key in REPLAYS
        },
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(TILING / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as current, tarfile.open(
        LAYER / "evidence.tar.xz"
    ) as previous:
        print(json.dumps(replay(current, previous, helper), indent=2, sort_keys=True))
