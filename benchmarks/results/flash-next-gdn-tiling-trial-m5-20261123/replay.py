#!/usr/bin/env python3
"""Verify the signed Flash Next decay projection tiling trial."""

from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import struct
import tarfile

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "flash-next-prefill-geometry-divergence-m5-20261123"
DECAY = HERE.parent / "flash-next-gdn-decay-projection-m5-20261123"
RECURRENCE = HERE.parent / "flash-next-gdn-recurrence-drift-m5-20261123"
ARMS = {
    "narrow-stock": (512, False, False),
    "narrow-switch": (512, True, False),
    "wide-stock-first": (2048, False, False),
    "wide-tiling": (2048, True, False),
    "wide-stock-last": (2048, False, False),
    "wide-tiling-trace": (2048, True, True),
}


def member(archive, name):
    file = archive.extractfile(name)
    assert file is not None
    return file.read()


def read(archive, name):
    return json.loads(member(archive, name))


def digest(archive, name):
    return sha256(member(archive, name)).hexdigest()


def compare_results(left, right):
    changed = [
        i for i, (a, b) in enumerate(zip(left["output_ids"], right["output_ids"], strict=True))
        if a != b
    ]
    pairs = [{item["position"]: item for item in arm["witnesses"]} for arm in (left, right)]
    states = {}
    for position in sorted(pairs[0].keys() & pairs[1].keys()):
        a, b = (p[position] for p in pairs)
        layers = [
            i for i, (x, y) in enumerate(zip(a["layer_hashes"], b["layer_hashes"], strict=True))
            if x != y
        ]
        states[str(position)] = {
            "changed_layer_count": len(layers),
            "first_changed_layer": layers[0] if layers else None,
            "first_layer_matches": a["layer_hashes"][0] == b["layer_hashes"][0],
            "same_decision_logits": a["logits"] == b["logits"],
        }
    return {
        "different_token_count": len(changed),
        "first_different_token": changed[0] if changed else None,
        "witnesses": states,
    }


def replay(archive, earlier, decay):
    require_prior = runpy.run_path(str(GEOMETRY / "replay.py"))
    require_prior["replay"](earlier, require_prior["PRIOR"])
    for name, sealed in ((GEOMETRY, "gdn-tiling-geometry"), (DECAY, "gdn-tiling-decay")):
        assert sha256((name / "evidence.tar.xz").read_bytes()).hexdigest() == (
            member(archive, f"{sealed}.sha256").decode().strip()
        )

    assert member(archive, "status.txt") == b"0\n"
    assert member(archive, "prompt.json") == member(earlier, "prompt.json")
    assert member(archive, "prior-ids.json") == member(earlier, "prior-ids.json")
    assert member(archive, "mxfp4-inventory.json") == member(earlier, "mxfp4-inventory.json")
    soc, memory, system, internal, ssd = member(archive, "host.txt").decode().splitlines()
    assert (soc, int(memory), internal, ssd) == ("Apple M5 Max", 128 * 2**30, "true", "true")
    build = read(archive, "build.json")
    assert build["mlx"] == build["mlx_metal"] == "0.32.4"
    assert "1.99.1" in build["rustc"] and "1.99.1" in build["cargo"]
    assert build["profile"] == "release-server" and build["codesign_verified"]
    assert build["patch_sha256"] == digest(archive, "source.patch")
    receipt = read(archive, "observed/receipt.json")
    assert receipt["build"] == build and receipt["qualification"] is False
    assert receipt["manifest"] == read(earlier, "observed/receipt.json")["manifest"]
    assert receipt["prompt_sha256"] == digest(archive, "prompt.json")
    assert receipt["prior_ids_sha256"] == digest(archive, "prior-ids.json")
    for line in member(archive, "inputs.sha256").decode().splitlines():
        expected, name = line.split()
        if name == "ax-flash-prefill-geometry-test":
            assert build["binary_sha256"] == expected
        else:
            assert digest(archive, name) == expected

    results, measurements = {}, {}
    for arm, (width, tiled, trace) in ARMS.items():
        root = f"observed/{arm}"
        record = read(archive, f"{root}/receipt.json")
        flags = record["flags"]
        assert flags["AX_FLASH_NEXT_TRIAL_TILE_GDN_DECAY"] == str(int(tiled))
        assert flags["AX_FLASH_NEXT_GDN_GEOMETRY_REPLAY"] == str(int(trace))
        assert flags["AX_FLASH_NEXT_RECORD_GDN_TILING"] == "1"
        assert flags["AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX"] == "1"
        assert flags["AX_MLX_LONG_PROMPT_PREFILL_CHUNK"] == str(width)
        for disabled in ("GROUPED_EVAL", "PAIRED_GATHER", "INDEXED_PREFILL"):
            assert flags[f"AX_MLX_FLASH_NEXT_QSA_{disabled}"] == "0"
        proof = read(archive, f"{root}/log-proof.json")
        assert proof["test_result"] == "1 passed; 0 failed"
        assert proof["original_sha256"] == record["original_test_log_sha256"]
        for key, file in (
            ("tiling_sha256", "result/tiling.json"), ("result_sha256", "result/result.json"),
            ("samples_sha256", "native-samples.jsonl"),
        ):
            assert record[key] == digest(archive, f"{root}/{file}")
        samples = [json.loads(row) for row in member(archive, f"{root}/native-samples.jsonl").splitlines()]
        assert len(samples) == record["sample_count"] and samples
        peak = max(row["footprint_bytes"] for row in samples)
        assert peak == record["sampled_footprint_peak_bytes"] <= 96 * 2**30
        assert all(row["pressure"] == 1 for row in samples)
        measure = read(archive, f"{root}/result/tiling.json")
        assert 0 < measure["first_token_ns"] <= measure["total_ns"]
        assert 0 < measure["mlx_peak_bytes"] <= 96 * 2**30
        assert (measure["tiled_decay_projections"] > 0) == (tiled and width == 2048)
        measurements[arm] = measure | {"sampled_footprint_peak_bytes": peak, "samples": len(samples)}

        result = read(archive, f"{root}/result/result.json")
        assert result["width"] == width and result["qualification"] is False
        assert len(result["output_ids"]) == 64
        for witness in result["witnesses"][-2:]:
            decision = witness["logits"]
            token = result["output_ids"][witness["position"] - 8192]
            assert decision["token"] == token
            blob = decision["logits_f32_le"]
            raw = member(archive, f"{root}/result/{blob['file']}")
            assert len(raw) == blob["bytes"] and sha256(raw).hexdigest() == blob["sha256"]
            values = struct.unpack(f"<{decision['shape'][1]}f", raw)
            assert all(map(math.isfinite, values))
            assert max(range(len(values)), key=values.__getitem__) == token
        original = read(earlier, f"observed/{width}/result/result.json")
        assert result["cuts"] == original["cuts"]
        if width == 512 or not tiled:
            assert result == original
        results[arm] = result

    assert results["wide-tiling"] == results["wide-tiling-trace"]
    root = "observed/wide-tiling-trace"
    record = read(archive, f"{root}/receipt.json")
    rerun = f"{root}/result/recurrence.json"
    assert record["recurrence_sha256"] == digest(archive, rerun)
    candidate = read(archive, rerun)
    canon = read(decay, "observed/512/result/recurrence.json")
    stock = read(decay, "observed/2048/result/recurrence.json")
    assert candidate["layer"] == 0 and candidate["partition"] == [2047]
    compare = runpy.run_path(str(RECURRENCE / "replay.py"))["compare"]

    def gdn_compare(arm):
        return {
            "projection": compare(arm["decay_stages"]["projection"]["rows"],
                                  candidate["decay_stages"]["projection"]["rows"]),
            "decay": compare(arm["input_rows"]["decay"], candidate["input_rows"]["decay"]),
            "recurrence": compare(arm["original_output"], candidate["original_output"]),
            "same_final_state": arm["original_boundary_states"][-1] == candidate["original_boundary_states"][-1],
        }

    return {
        "qualification": False,
        "macos": system,
        "source_base": build["source_base"],
        "stock_repeats_match": True,
        "narrow_switch_matches": True,
        "measurements": measurements,
        "wide_candidate_vs_wide_stock": compare_results(results["wide-stock-first"], results["wide-tiling"]),
        "wide_stock_vs_narrow": compare_results(results["narrow-stock"], results["wide-stock-first"]),
        "wide_candidate_vs_narrow": compare_results(results["narrow-stock"], results["wide-tiling"]),
        "layer_zero_gdn": {
            "candidate_vs_narrow": gdn_compare(canon),
            "candidate_vs_wide_stock": gdn_compare(stock),
        },
    }


if __name__ == "__main__":
    with tarfile.open(HERE / "evidence.tar.xz") as fresh, tarfile.open(
        GEOMETRY / "evidence.tar.xz"
    ) as earlier, tarfile.open(DECAY / "evidence.tar.xz") as decay:
        print(json.dumps(replay(fresh, earlier, decay), indent=2, sort_keys=True))
