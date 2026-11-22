"""Compare the unchanged M5 mmap control across the two read-ahead campaigns."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "flash-next-mmap-read-ahead-m5-20261122"
INPUTS = {
    "original": (
        "evidence.tar.xz",
        "e85cbede1315b72b7be5648a025c840b169cf6e1118448fe7ba306b0674c9462",
        ("mmap-short", "advised-short", "advised-long", "mmap-long"),
        "failed",
    ),
    "reversed": (
        "evidence-reverse.tar.xz",
        "6354f36fd607925891b2399e60c53565be4cfa4014008d8c0ad96cf5738b0c64",
        ("advised-short", "mmap-short", "mmap-long", "advised-long"),
        "complete",
    ),
}
LENGTHS = (512, 2048, 8192)
STABLE_FIELDS = (
    "prompt_hash",
    "prompt_tokens",
    "phase",
    "trial",
    "output_tokens",
    "finish_reason",
    "status",
    "verified_steps",
    "route_errors",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_receipt(campaign):
    name, digest, order, status = INPUTS[campaign]
    path = SOURCE / name
    require(sha256(path.read_bytes()).hexdigest() == digest, f"Input changed for {campaign}")
    with tarfile.open(path) as archive:
        receipt = json.load(archive.extractfile("receipt.json"))
        inputs = {name: value for value, name in map(str.split,
                  archive.extractfile("inputs.sha256").read().decode().splitlines())}
    require(receipt["status"] == status and tuple(receipt["arms"]) == order,
            f"Observed process order or terminal state changed for {campaign}")
    require(inputs["run_read_ahead.py"] == receipt["runner_sha256"],
            f"Measurement runner provenance changed for {campaign}")
    return receipt


def warm_cells(arm):
    cells = {}
    for size in LENGTHS:
        cases = [row for row in arm["cases"] if row["phase"] == "warm" and row["prompt_tokens"] == size]
        require([row["trial"] for row in cases] == [1, 2, 3], f"Invalid trials at {size} tokens")
        timings = [row["timing"] for row in cases]
        require(all(
            all(value > 0 and isfinite(value) for value in (row["ttft_seconds"], row["decode_seconds"]))
            and row["decode_tokens"] == 31 for row in timings
        ), f"Invalid timing observation at {size} tokens")
        cells[size] = {
            "median_ttft_seconds": median(row["ttft_seconds"] for row in timings),
            "median_decode_tokens_per_second": median(row["decode_tokens"] / row["decode_seconds"] for row in timings),
        }
    return cells


def replay(original, reversed_order):
    keys = ("build", "hardware", "macos_version", "pack_revision", "inventory_sha256",
            "model_manifest_sha256")
    require(all(original[key] == reversed_order[key] for key in keys),
            "The two campaigns used different serving inputs")
    require(original["build"]["commit"] == "c2358dd814f5bb4a11fce0f22d9d79c2809f63ea",
            "The tested source changed")
    original_arm, reversed_arm = (receipt["arms"]["mmap-short"] for receipt in (original, reversed_order))
    fields = ("initial_selected_page_samples", "profile", "model", "context_overflow")
    require(all(original_arm[key] == reversed_arm[key] for key in fields),
            "The mapped controls used different initial page samples, models or context profiles")
    stable = []
    measurements = {}
    for name, arm in (("original", original_arm), ("reversed", reversed_arm)):
        idle = (arm["quiet_baseline"]["cpu_idle_percent"], arm["post_run_cpu_idle_percent"])
        require(arm["loader_mode"] == "mmap" and arm["pressure_levels"] == [1]
                and all(90 <= value <= 100 for value in idle), "The mapped control was timing-ineligible")
        require(arm["health_ready"] == arm["health_after_requests"] == "ok", "Native health state changed")
        require(len(arm["cases"]) == 14 and arm["cases"][0]["prompt_tokens"] == 512,
                "Incomplete short-first serving workload")
        require(all(len(row["output_tokens"]) == 32 and row["verified_steps"] == row["route_errors"] == 0
                    for row in arm["cases"]), "Output geometry or direct route changed")
        stable.append([tuple(row[key] for key in STABLE_FIELDS) for row in arm["cases"]])
        measurements[name] = {"cpu_idle_percent": idle, "warmed": warm_cells(arm)}
    require(stable[0] == stable[1], "Output IDs or route fields differ between mapped controls")
    first, second = (campaign["warmed"] for campaign in measurements.values())
    observed_change = {
        size: {field: round(100 * (second[size][field] / value - 1), 3)
               for field, value in readings.items()}
        for size, readings in first.items()
    }
    return {
        "scope": "unchanged mmap short-first controls, between two campaigns",
        "qualification": False,
        "source_commit": original["build"]["commit"],
        "matched_output_ids": 448,
        "process_position": {"original": 1, "reversed": 2},
        "measurements": measurements,
        "reversed_relative_to_original_percent": observed_change,
    }


if __name__ == "__main__":
    result = replay(load_receipt("original"), load_receipt("reversed"))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
