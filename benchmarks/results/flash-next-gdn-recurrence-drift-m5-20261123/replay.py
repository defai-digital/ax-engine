#!/usr/bin/env python3
"""Verify signed Flash Next layer-zero GDN inputs and same-state replay."""

from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "flash-next-prefill-geometry-divergence-m5-20261123"
OPERATOR = HERE.parent / "flash-next-prefill-operator-drift-m5-20261123"
PREFIX = 2047
VARIANTS = ("whole", "split_lazy", "split_fenced")
INPUTS = ("query", "key", "value", "decay", "beta")


def member(archive, name):
    item = archive.extractfile(name)
    assert item is not None
    return item.read()


def read(archive, name):
    return json.loads(member(archive, name))


def validate(record):
    shape, rows = record["shape"], record["rows"]
    assert shape[:2] == [1, PREFIX] and all(dim > 0 for dim in shape)
    assert len(rows) == PREFIX and all(re.fullmatch(r"[0-9a-f]{64}", row) for row in rows)


def compare(left, right):
    assert left["shape"] == right["shape"]
    differences = [
        i for i, (x, y) in enumerate(zip(left["rows"], right["rows"], strict=True))
        if x != y
    ]
    return {
        "different_rows": len(differences),
        "first_different_row": differences[0] if differences else None,
        "last_different_row": differences[-1] if differences else None,
    }


def run(archive, earlier, traced):
    prior = runpy.run_path(str(GEOMETRY / "replay.py"))
    live = prior["replay"](archive, prior["PRIOR"])
    samples = runpy.run_path(str(OPERATOR / "replay.py"))["samples"]
    arms, comparisons, peak, prior_core_rows = {}, {}, {}, 0
    for width in (512, 2048):
        root = f"observed/{width}"
        filename = f"{root}/result/result.json"
        live_result = read(archive, filename)
        assert live_result == read(earlier, filename) == read(traced, filename)
        for position in (8192, 8193):
            logits = f"{root}/result/position-{position}.logits.f32"
            assert member(archive, logits) == member(earlier, logits)
        receipt = read(archive, f"{root}/receipt.json")
        assert receipt["flags"]["AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY"] == "1"
        assert receipt["flags"]["AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX"] == "1"
        body = member(archive, f"{root}/result/recurrence.json")
        assert sha256(body).hexdigest() == receipt["recurrence_sha256"]
        arm = json.loads(body)
        partition = [512, 512, 512, 511] if width == 512 else [2047]
        assert arm["recurrence"] == "mlx_gated_delta_update"
        assert arm["layer"] == 0 and arm["partition"] == partition
        states = arm["original_boundary_states"]
        assert len(states) == len(partition) + 1
        assert all(re.fullmatch(r"[0-9a-f]{64}", state) for state in states)
        assert sorted(arm["input_rows"]) == sorted(INPUTS)
        for values in arm["input_rows"].values():
            validate(values)
        assert sorted(arm["replays"]) == sorted(VARIANTS)
        actual = arm["original_output"]
        validate(actual)
        for (stage, pos), (data, _) in samples(traced, width, live_result["cuts"]).items():
            if stage == "gdn_core_output" and pos < PREFIX:
                assert actual["rows"][pos] == sha256(data).hexdigest()
                prior_core_rows += 1
        q_shape, v_shape = (arm["input_rows"][key]["shape"] for key in ("query", "value"))
        zero_bytes = 4 * q_shape[0] * q_shape[3] * v_shape[2] * v_shape[3]
        assert states[0] == sha256(bytes(zero_bytes)).hexdigest()
        for replay in arm["replays"].values():
            validate(replay["output"])
            assert re.fullmatch(r"[0-9a-f]{64}", replay["final_state"])
        whole = arm["replays"]["whole"]
        comparisons[str(width)] = {
            "replays_against_original": {
                name: compare(actual, variant["output"]) | {
                    "same_final_state": states[-1] == variant["final_state"],
                }
                for name, variant in arm["replays"].items()
            },
            "splits_against_same_input_whole": {
                name: compare(whole["output"], variant["output"]) | {
                    "same_final_state": whole["final_state"] == variant["final_state"],
                }
                for name, variant in arm["replays"].items() if name != "whole"
            },
        }
        arms[width] = arm
        peak[str(width)] = receipt["sampled_footprint_peak_bytes"]

    assert prior_core_rows == 22
    narrow, wide = arms[512], arms[2048]
    assert narrow["original_boundary_states"][0] == wide["original_boundary_states"][0]
    return {
        "qualification": False,
        "shared_prefix_tokens": PREFIX,
        "source_base": live["source_base"],
        "macos": live["macos"],
        "matched_http_token_ids": live["matched_http_token_ids"],
        "normal_pressure_samples": live["normal_pressure_samples"],
        "matched_prior_core_rows": prior_core_rows,
        "peak_sampled_footprint_bytes": peak,
        "input_comparisons": {
            key: compare(narrow["input_rows"][key], wide["input_rows"][key]) for key in INPUTS
        },
        "original_output_comparison": compare(narrow["original_output"], wide["original_output"]),
        "same_original_final_state": (
            narrow["original_boundary_states"][-1] == wide["original_boundary_states"][-1]
        ),
        "same_input_replay": comparisons,
        "replay_cross_arm_comparisons": {
            name: compare(narrow["replays"][name]["output"], wide["replays"][name]["output"]) | {
                "same_final_state": (
                    narrow["replays"][name]["final_state"] == wide["replays"][name]["final_state"]
                ),
            }
            for name in VARIANTS
        },
    }


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz") as current, tarfile.open(
        GEOMETRY / "evidence.tar.xz"
    ) as prior_capture, tarfile.open(OPERATOR / "evidence.tar.xz") as traced_capture:
        print(json.dumps(run(current, prior_capture, traced_capture), indent=2, sort_keys=True))
