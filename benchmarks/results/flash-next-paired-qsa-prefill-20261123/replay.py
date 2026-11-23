"""Validate paired QSA prefill output, timings, and pack bindings."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_arm(result):
    assert result["qualification"] is False and result["final_state_and_logits_exact"] is True
    arms = result["arms"]
    assert [arm["paired"] for arm in arms] == [True, False, False, True]
    expected_widths = [512, 512, 512, 511, 1] * 4
    for arm in arms:
        assert (arm["paired_dispatches"] > 0) == arm["paired"]
        assert isinstance(arm["peak_bytes"], int) and 0 < arm["peak_bytes"] < 96 * 2**30
        assert math.isfinite(arm["seconds"]) and arm["seconds"] > 0
        offset = 0
        for (observed, width, seconds), expected in zip(arm["chunks"], expected_widths, strict=True):
            assert observed == offset and width == expected
            assert math.isfinite(seconds) and seconds > 0
            offset += width
        assert offset == 8192 and sum(chunk[2] for chunk in arm["chunks"]) <= arm["seconds"]
    return arms


def verify_device(work, mode, build):
    path = work / mode
    assert (path / "status.txt").read_text().strip() == "0"
    receipt = read(path / "receipt.json")
    assert receipt["build"] == build and receipt["qualification"] is False
    assert receipt["manifest"]["model-manifest.json"] == build["manifest_sha256"]
    assert receipt["storage"]["internal"] and receipt["storage"]["solid_state"]
    soc = "Apple M5 Max" if mode == "m5" else "Apple M2 Ultra"
    ram = 128 if mode == "m5" else 192
    assert receipt["soc"] == soc and receipt["memory_bytes"] == ram * 2**30
    assert receipt["result_sha256"] == sha(path / "result.json")
    observed = read(path / "result.json")
    arms = verify_arm(observed)
    samples = [json.loads(line) for line in (path / "pressure.jsonl").read_text().splitlines()]
    assert len(samples) == receipt["pressure_samples"] > 1
    assert all(s["pressure"] == 1 and 0 <= s["footprint_bytes"] <= 96 * 2**30 for s in samples)
    # A boundary sample can race process startup or exit.
    assert all(s["footprint_bytes"] > 0 for s in samples[1:-1])
    idle = receipt["cpu_idle_percent"]
    assert len(idle) == 2 and all(0 <= value <= 100 for value in idle)
    stock = [arm["seconds"] for arm in arms if not arm["paired"]]
    paired = [arm["seconds"] for arm in arms if arm["paired"]]
    return {
        "soc": soc, "memory_bytes": receipt["memory_bytes"], "macos": receipt["macos"],
        "cpu_idle_percent": idle, "normal_pressure_samples": len(samples),
        "endpoint_timing_eligible": min(idle) >= 90,
        "paired_seconds": paired, "stock_seconds": stock,
        "paired_over_stock": median(paired) / median(stock),
        "ordered_pair_ratios": [paired[0] / stock[0], paired[1] / stock[1]],
        "paired_dispatches": [arm["paired_dispatches"] for arm in arms if arm["paired"]],
        "paired_peak_bytes": [arm["peak_bytes"] for arm in arms if arm["paired"]],
        "stock_peak_bytes": [arm["peak_bytes"] for arm in arms if not arm["paired"]],
        "final_state_and_logits_exact": True,
    }


def replay(work):
    build = read(work / "build.json")
    assert build["profile"] == "release-server" and build["mlx"] == "0.32.4"
    assert build["rustc"].startswith("rustc 1.99.1 ") and build["cargo"].startswith("cargo 1.99.1 ")
    patch = gzip.decompress((work / "source.patch.gz").read_bytes())
    assert build["patch_sha256"] == hashlib.sha256(patch).hexdigest()
    assert build["prompt_sha256"] == sha(work / "prompt.json")
    source = work.parent / "flash-next-shared-step-clock-m5-20261123" / "fixtures.json"
    case = read(source)["cases"][1]
    assert case["observed_mixed_prefill_width"] == 511
    assert read(work / "prompt.json") == case["long_prompt_ids"]
    component = (work / "m4-component.txt").read_text()
    assert "soc=Apple M4 Pro" in component and "memory_bytes=68719476736" in component
    assert f'{build["binary_sha256"]}  ax-flash-paired-qsa-test' in component
    assert f'{build["libmlx_sha256"]}  mlx/libmlx.dylib' in component
    assert "1 passed; 0 failed" in component
    return {
        "qualification": False,
        "build": build,
        "m4_component_pass": True,
        "results": {mode: verify_device(work, mode, build) for mode in ("m2", "m5")},
    }


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]) if len(sys.argv) == 2 else HERE),
                     indent=2, sort_keys=True, allow_nan=False))
