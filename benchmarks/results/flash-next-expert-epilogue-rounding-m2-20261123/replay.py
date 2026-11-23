"""Recheck the two-layer Flash Next epilogue probe without loading weights."""

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = ((1, 512), (511, 512), (512, 512), (512, 64))
VARIANTS = ("compiled_with_bf16_activation", "late_bf16_round")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay(root=ROOT):
    assert (root / "status.txt").read_bytes() == b"0\n"
    raw = json.loads((root / "observed.json").read_bytes())
    inventory = json.loads((root / "mxfp4-inventory.json").read_bytes())
    assert raw["qualification"] is False and raw["macos"] == "27.2.1"
    assert raw["versions"]["mlx"] == raw["versions"]["mlx-metal"] == "0.32.4"
    assert raw["pack_revision"] == inventory["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert raw["probe_sha256"] == digest(root / "probe.py")
    assert raw["reused_probe_sha256"] == digest(
        root.parent / "flash-next-packed-expert-probe-m2-20261122" / "probe.py")
    assert raw["inventory_sha256"] == digest(root / "mxfp4-inventory.json")
    members = {row["name"]: row for row in inventory["members"]}
    shards = raw["publisher_shards_verified_before_and_after"]
    assert len(members) == 49 and len(shards) == len(set(shards)) == 2
    assert all(members[name]["publisher_sha256"] == members[name]["sha256"] for name in shards)
    assert raw["sampled_pressure_levels"] == [1] * 6
    assert [row["layer"] for row in raw["layer_results"]] == [0, 32]

    summary = {variant: {stage: {"elements": 0, "different_bits": 0, "max_abs_difference": 0}
                         for stage in ("activation", "down")} for variant in VARIANTS}
    for layer in raw["layer_results"]:
        assert [tuple(case[key] for key in ("rows", "route_pool")) for case in layer["cases"]] == list(CASES)
        for case in layer["cases"]:
            assert 10 <= case["distinct_experts"] <= case["route_pool"]
            assert set(case["variants"]) == set(VARIANTS)
            for variant, stages in case["variants"].items():
                assert set(stages) == {"activation", "down"}
                for stage, result in stages.items():
                    width = 640 if stage == "activation" else 2560
                    assert result["elements"] == case["rows"] * 10 * width
                    assert 0 <= result["different_bits"] <= result["elements"]
                    assert result["max_abs_difference"] >= 0
                    overall = summary[variant][stage]
                    for field in ("elements", "different_bits"):
                        overall[field] += result[field]
                    overall["max_abs_difference"] = max(overall["max_abs_difference"], result["max_abs_difference"])
    assert all(item["different_bits"] == item["max_abs_difference"] == 0
               for item in summary[VARIANTS[0]].values())
    assert all(item["different_bits"] > 0 and item["max_abs_difference"] > 0
               for item in summary[VARIANTS[1]].values())
    return {"scope": raw["scope"], "qualification": False, "hardware": "Mac Studio M2 Ultra 192 GB",
            "pack_revision": raw["pack_revision"], "normal_pressure_samples": 6,
            "cases": 8, "stages": summary}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
