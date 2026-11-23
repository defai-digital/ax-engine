"""Verify Flash Next sorted expert tail evidence and timing from raw samples."""
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent


def clock_ratio(scope):
    assert scope["fused_bit_differences"] == 0
    times = scope["wall_us"]
    assert set(times) == {"eager", "fused"}
    assert all(len(arm) == 10 and all(math.isfinite(t) and t > 0 for t in arm) for arm in times.values())
    ratio = median(times["fused"]) / median(times["eager"])
    assert ratio == scope["fused_over_eager"]
    return ratio


def replay(path):
    record = json.loads((path / "result.json").read_bytes())
    assert record["harness_sha256"] == hashlib.sha256((path / "real_pack.py").read_bytes()).hexdigest()
    assert record["kernel_sha256"] == hashlib.sha256((path / "probe.py").read_bytes()).hexdigest()
    assert record["mlx"] == "0.32.4" and record["qualification"] is False
    assert record["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert len(set(record["shards"])) == len(record["shards"]) == 2
    assert record["round_order"] == ["eager", "fused", "fused", "eager"] * 5
    assert record["ps"] == [1, 1, 1] and (path / "status.txt").read_text().strip() == "0"
    idle = []
    for phase in ("before", "after"):
        rows = (path / f"idle-{phase}.txt").read_text().splitlines()
        assert len(rows) == 2
        match = re.search(r"([\d.]+)% idle", rows[-1])
        assert match is not None
        idle.append(float(match[1]))
    assert min(idle) >= 90.0

    rows = []
    assert [layer["layer"] for layer in record["layers"]] == [0, 32]
    for layer in record["layers"]:
        keys = [(case["width"], case["route_pool"], case["seed"]) for case in layer["cases"]]
        assert keys == [(511, 512, 37), (512, 512, 43), (512, 64, 53)]
        for case in layer["cases"]:
            tail, full = case["tail"], case["routed_projections_and_tail"]
            elements = case["width"] * 2560
            assert tail["elements"] == full["elements"] == elements
            assert 0 < tail["float_acc_bit_differences"] < elements
            assert case["distinct_experts"] == case["route_pool"]
            rows.append({"layer": layer["layer"], "width": case["width"], "route_pool": case["route_pool"],
                         "exact_output_elements": elements, "float_acc_differences": tail["float_acc_bit_differences"],
                         "tail_ratio": clock_ratio(tail), "projections_and_tail_ratio": clock_ratio(full)})
    return {"qualification": False, "macos": record["macos"], "soc": record["soc"],
            "cpu_idle_percent": idle, "cases": rows,
            "exact_output_elements": sum(row["exact_output_elements"] for row in rows),
            "float_acc_differences": sum(row["float_acc_differences"] for row in rows)}


if __name__ == "__main__":
    print(json.dumps(replay(HERE), indent=2, sort_keys=True, allow_nan=False))
