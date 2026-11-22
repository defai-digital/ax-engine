"""Recompute the bounded two-layer gate/up experiment."""
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
INVENTORY = (ROOT.parent / "qualification" / "2026-10-04-flash-next-mxfp4-native-support"
             / "pack-inventory.json")
CASES = ((1, 512, 10), (511, 512, 512), (512, 512, 512), (512, 64, 64))
MODES = {"split", "packed", "scratch"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay(report):
    require(report["qualification"] is False and report["versions"]["mlx"] == "0.32.4"
            and report["versions"]["mlx-metal"] == "0.32.4"
            and report["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and sha256((ROOT / "probe.py").read_bytes()).hexdigest() == report["probe_sha256"],
            "Probe, version or checkpoint identity changed")
    require(report["publisher_shards_verified"] == ["model-00001-of-00025.safetensors",
                                                      "model-00021-of-00025.safetensors"]
            and sha256(INVENTORY.read_bytes()).hexdigest() == report["inventory_sha256"],
            "Missing publisher binding")
    require([layer["layer"] for layer in report["layer_results"]] == [0, 32], "Missing layer")
    output = []
    prefill = []
    for layer in report["layer_results"]:
        require(layer["packed_bytes"] == 891_289_600 and layer["pack_seconds"] > 0,
                "Invalid packed pair allocation")
        for row, geometry in zip(layer["cases"], CASES, strict=True):
            require((row["rows"], row["route_pool"], row["distinct_experts"]) == geometry
                    and row["sorted_indices"] is (row["rows"] > 1), "Changed routing shape")
            require(row["bit_identity"] == {mode: dict.fromkeys(
                ("gate", "up", "activation", "down"), True) for mode in ("packed", "scratch")},
                "Packing changed projection or activation bits")
            times = row["elapsed_seconds"]
            require(times.keys() == MODES and all(len(t) == 6 and all(
                math.isfinite(v) and v > 0 for v in t) for t in times.values()), "Invalid timings")
            measured = {mode: median(values) for mode, values in times.items()}
            require(measured == row["median_seconds"], "Raw timings disagree with medians")
            peaks = row["peak_additional_bytes"]
            require(peaks.keys() == MODES and all(type(value) is int and value >= 0
                                                 for value in peaks.values()), "Invalid memory sample")
            result = {"layer": layer["layer"], "tokens": row["rows"], "route_pool": row["route_pool"],
                      "median_seconds": measured,
                      "relative_packed_wall": measured["packed"] / measured["split"],
                      "relative_scratch_wall": measured["scratch"] / measured["split"],
                      "scratch_minus_split_bytes": peaks["scratch"] - peaks["split"]}
            output.append(result)
            if row["rows"] > 1:
                prefill.append(result)
    return {"scope": "Routed component only", "all_exact": True, "cases": output,
            "prefill_packed_wall_ratio_range": [min(row["relative_packed_wall"] for row in prefill),
                                                max(row["relative_packed_wall"] for row in prefill)],
            "prefill_scratch_wall_ratio_range": [min(row["relative_scratch_wall"] for row in prefill),
                                                 max(row["relative_scratch_wall"] for row in prefill)]}


if __name__ == "__main__":
    require((ROOT / "status.txt").read_bytes() == b"0\n", "Probe did not exit successfully")
    print(json.dumps(replay(json.loads((ROOT / "result.json").read_bytes())),
                     indent=2, sort_keys=True, allow_nan=False))
