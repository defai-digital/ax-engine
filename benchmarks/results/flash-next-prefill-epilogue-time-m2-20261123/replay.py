"""Verify the measured Flash Next SwiGLU timing ratios from raw samples."""
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

HERE = Path(__file__).resolve().parent
SCOPES = ("epilogue_only", "gate_up_epilogue_down")


def interval(values):
    return [min(values), max(values)]


def replay(path):
    recorded = json.loads((path / "result.json").read_bytes())
    assert recorded["probe_sha256"] == hashlib.sha256((path / "time_epilogue.py").read_bytes()).hexdigest()
    assert (path / "status.txt").read_text().strip() == "0"
    assert recorded["qualification"] is False
    assert recorded["mlx"] == "0.32.4"
    assert recorded["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert len(recorded["shards"]) == len(set(recorded["shards"])) == 2
    assert recorded["memory_pressure"] == [1, 1, 1]
    assert recorded["round_order"] == ["eager", "compiled", "compiled", "eager"] * 5
    idle = []
    for phase in ("before", "after"):
        rows = (path / f"idle-{phase}.txt").read_text().splitlines()
        assert len(rows) == 2
        match = re.search(r"([\d.]+)% idle", rows[-1])
        assert match is not None
        idle.append(float(match[1]))
    assert min(idle) >= 90.0
    observed = []
    assert [row["layer"] for row in recorded["layers"]] == [0, 32]
    for layer in recorded["layers"]:
        shapes = [(row["rows"], row["route_pool"]) for row in layer["cases"]]
        assert shapes == [(1, 512), (511, 512), (512, 512), (512, 64)]
        for case in layer["cases"]:
            assert 1 <= case["distinct_experts"] <= case["route_pool"]
            row = {"layer": layer["layer"], "rows": case["rows"], "route_pool": case["route_pool"]}
            assert set(case["scopes"]) == set(SCOPES)
            for key in SCOPES:
                scope = case["scopes"][key]
                assert scope["bitwise_exact"] is True
                samples = scope["wall_us"]
                assert set(samples) == {"eager", "compiled"}
                assert all(len(times) == 10 and all(math.isfinite(t) and t > 0 for t in times)
                           for times in samples.values())
                ratio = median(samples["compiled"]) / median(samples["eager"])
                assert ratio == scope["compiled_over_eager"]
                row[key] = ratio
            observed.append(row)
    prefill = [row for row in observed if row["rows"] > 1]
    return {
        "qualification": False, "soc": recorded["soc"], "macos": recorded["macos"],
        "endpoint_cpu_idle_percent": idle, "cases": observed,
        "prefill_case_count": len(prefill),
        "prefill_compiled_over_eager_range": {key: interval([row[key] for row in prefill])
                                              for key in SCOPES},
    }


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE),
                     indent=2, sort_keys=True, allow_nan=False))
