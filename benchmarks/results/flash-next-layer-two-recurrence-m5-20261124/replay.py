#!/usr/bin/env python3
"""Verify the first differing GDN layer after the combined projection trial."""

from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile

HERE = Path(__file__).resolve().parent
CROSSOVER = HERE.parent / "flash-next-router-decay-crossover-m5-20261124"
TILING = HERE.parent / "flash-next-gdn-tiling-trial-m5-20261123"
LAYER_ONE = HERE.parent / "flash-next-gdn-layer-one-recurrence-m5-20261123"
ARMS = (("narrow-layer-two", "narrow-stock", 512, [512, 512, 512, 511]),
        ("wide-both-layer-two", "wide-both", 2048, [2047]))


def replay(archive, crossover, helper, layer_helper):
    member, read, digest = (helper[name] for name in ("member", "read", "digest"))
    assert member(archive, "status.txt") == b"0\n"
    assert sha256((CROSSOVER / "evidence.tar.xz").read_bytes()).hexdigest() == (
        member(archive, "prior-router-decay-crossover.sha256").decode().strip()
    )
    for file in ("host.txt", "prompt.json", "prior-ids.json", "build.json", "source.patch",
                 "mxfp4-inventory.json", "observed/receipt.json"):
        assert member(archive, file) == member(crossover, file)
    build = read(archive, "build.json")
    for line in member(archive, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert build["binary_sha256"] == expected
        else:
            assert digest(archive, name) == expected

    captured, pressure = {}, {}
    for name, prior, width, partition in ARMS:
        root = f"observed/{name}"
        receipt = read(archive, f"{root}/receipt.json")
        flags = receipt["flags"]
        active = width == 2048
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(int(active))
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_MOE_ROUTER"] == str(int(active))
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_LAYER"] == "2"
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY"] == "1"
        assert flags["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(width)
        assert flags["AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX"] == "1"
        for flag in ("GROUPED_EVAL", "PAIRED_GATHER", "INDEXED_PREFILL"):
            assert flags[f"AX_MLX_FLASH_NEXT_QSA_{flag}"] == "0"
        proof = read(archive, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == receipt["original_test_log_sha256"]
        for key, file in (
            ("result_sha256", "result/result.json"),
            ("recurrence_sha256", "result/recurrence.json"),
            ("tiling_sha256", "result/tiling.json"),
            ("samples_sha256", "native-samples.jsonl"),
        ):
            assert receipt[key] == digest(archive, f"{root}/{file}")
        for file in ("result.json", "position-8192.logits.f32", "position-8193.logits.f32"):
            assert member(archive, f"{root}/result/{file}") == member(
                crossover, f"observed/{prior}/result/{file}"
            )
        measurement = read(archive, f"{root}/result/tiling.json")
        assert measurement["tiled_decay_projections"] == active * 144
        assert measurement["tiled_router_projections"] == active * 192
        samples = [json.loads(row) for row in member(archive,f"{root}/native-samples.jsonl").splitlines()]
        assert samples and len(samples) == receipt["sample_count"]
        assert all(x["pressure"] == 1 and x["footprint_bytes"] <= 96 * 2**30 for x in samples)
        assert max(x["footprint_bytes"] for x in samples) == receipt["sampled_footprint_peak_bytes"]
        pressure[name] = {"normal_samples": len(samples), "peak_footprint_bytes": receipt["sampled_footprint_peak_bytes"]}
        data = read(archive, f"{root}/result/recurrence.json")
        assert data["layer"] == 2 and data["partition"] == partition
        assert data["recurrence"] == "mlx_gated_delta_update"
        assert len(data["original_boundary_states"]) == len(partition) + 1
        captured[name] = data

    a, b = captured.values()
    compare, with_state = (layer_helper[key] for key in ("compare", "with_state"))
    assert a["original_boundary_states"][0] == b["original_boundary_states"][0]
    output = lambda data: {"output": data["original_output"], "final_state": data["original_boundary_states"][-1]}
    return {
        "qualification": False,
        "layer": 2,
        "exact_archived_full_model_repeats": True,
        "pressure": pressure,
        "prework": {key: compare(a["prework_rows"][key], b["prework_rows"][key]) for key in a["prework_rows"]},
        "inputs": {key: compare(a["input_rows"][key], b["input_rows"][key]) for key in a["input_rows"]},
        "decay_stages": {key: compare(a["decay_stages"][key]["rows"], b["decay_stages"][key]["rows"])
                         for key in a["decay_stages"]},
        "live": with_state(output(a), output(b)),
        "same_input_replay_vs_live": {
            name: {kind: with_state(output(data), replayed) for kind, replayed in data["replays"].items()}
            for name, data in captured.items()
        },
        "same_partition_cross_arm": {kind: with_state(a["replays"][kind], b["replays"][kind]) for kind in a["replays"]},
    }


if __name__ == "__main__":
    helper = runpy.run_path(str(TILING / "replay.py"))
    layer_helper = runpy.run_path(str(LAYER_ONE / "replay.py"))
    with tarfile.open(HERE / "evidence.tar.xz") as trace, tarfile.open(
        CROSSOVER / "evidence.tar.xz"
    ) as crossover:
        print(json.dumps(replay(trace, crossover, helper, layer_helper), indent=2, sort_keys=True))
