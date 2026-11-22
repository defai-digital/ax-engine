"""Recompute the paired default-path Flash Next pre-read measurements."""
import argparse
import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
ARMS = {"mmap-short": 512, "prefetch-short": 512, "prefetch-long": 8192, "mmap-long": 8192}
WIDTHS = (512, 2048, 8192)
BYTES = 5_214_301_696
VOCAB = 248320
CURRENT_DELIVERY = "flash-next-sidecar-prefetch-delivery-m5-20261122"
TOKEN_ORACLE = "flash-next-sidecar-reader-warm-m5-20261122"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def read(archive, name):
    item = archive.getmember(name)
    require(item.isfile(), f"Invalid raw evidence {name}")
    return archive.extractfile(item).read()


def expected_cases(first):
    return [(first, 0, "first"), (512, 100, "warmup"),
            *((length, trial, "warm") for trial in (1, 2, 3) for length in WIDTHS),
            (8192 if first == 512 else 512, 0, "parity"), (2048, 0, "parity"), (32736, 0, "boundary")]


def verify_routes(log, mode, arm, selection):
    phases = {}
    for key in ("resident_tensors", "mtp_sidecar"):
        values = re.findall(fr'phase="{key}" elapsed_us=(\d+)', log)
        require(len(values) == 1 and int(values[0]) > 0, f"Ambiguous {key} phase")
        phases[key] = int(values[0]) / 1e6
    shards = [[name, int(count)] for name, count in re.findall(
        r'Flash Next resident shard shard=Some\("([^"/]+)"\) tensor_count=(\d+)', log)]
    reads = re.findall(r"Flash Next MTP sidecar prefetch completed bytes=(\d+) elapsed_ms=([\d.]+)", log)
    enabled = mode == "prefetch"
    require(shards == selection and arm["shard_count"] == len(shards)
            and arm["tensor_count"] == sum(count for _, count in shards)
            and "decision=resident" in log and "effective_sequence_cap=Some(1)" in log
            and all(pattern not in log for pattern in ("prefetch failed", "positional read lab"))
            and len(reads) == int(enabled) and arm["prefetched_sidecar"] is enabled
            and all(int(size) == BYTES and 0 < float(timing) < float("inf") for size, timing in reads)
            and arm["sidecar_prefetch_seconds"] == (float(reads[0][1]) / 1000 if reads else 0)
            and phases["resident_tensors"] == arm["load_phase_seconds"]
            and phases["mtp_sidecar"] == arm["sidecar_load_seconds"]
            and 0 < sum(phases.values()) < arm["ready_seconds"] < float("inf"),
            "Reader, startup phase, residency or selected publisher tensors changed")


def verify_samples(raw, arm):
    rows = [json.loads(line) for line in raw.splitlines()]
    require(rows and sorted({row["pressure"] for row in rows}) == arm["pressure_levels"] == [1]
            and all(0 <= row["elapsed_seconds"] < float("inf") for row in rows)
            and max(row["reference_rss_bytes"] for row in rows if "reference_rss_bytes" in row)
            == arm["peak_sampled_rss_bytes"], "Pressure or sampled RSS changed")
    for key in ("physical_footprint_bytes", "kernel_lifetime_peak_footprint_bytes"):
        values = [row[key] for row in (*rows, arm["process_before_stop"]) if key in row]
        require(max(values, default=None) == arm[f"max_observed_{key}"], "Footprint changed")


def timing(rows, matrix):
    result = {"ttft_seconds": median(row["timing"]["ttft_seconds"] for row in rows),
              "decode_tokens_per_second": median(matrix.decode_rate(row["timing"]) for row in rows)}
    require(all(isfinite(value) and value > 0 for value in result.values()), "Invalid serving timing")
    return result


def replay(capture, repository):
    benchmarks = repository / "benchmarks" / "results"
    script = repository / "scripts" / "run_flash_next_generate_matrix.py"
    sys.path.insert(0, str(script.parent))
    import run_flash_next_generate_matrix as matrix

    baseline = json.loads(gzip.decompress((benchmarks / TOKEN_ORACLE / "receipt.json.gz").read_bytes()))
    oracle = {}
    for arm in baseline["arms"].values():
        for row in arm["cases"]:
            require(oracle.setdefault(row["prompt_hash"], row["output_tokens"]) == row["output_tokens"],
                    "Historical fixed-workload copies disagree")

    with tarfile.open(capture) as archive, tarfile.open(benchmarks / CURRENT_DELIVERY / "evidence.tar.xz") as delivery:
        receipt = json.loads(read(archive, "receipt.json"))
        exact_build = read(archive, "build.json")
        delivered = read(delivery, "native-comparison.json")
        bound = {name: sha for sha, name in map(str.split, read(archive, "warm-inputs.sha256").decode().splitlines())}
        require(exact_build == read(delivery, "build.json")
                and receipt["build"] == json.loads(exact_build)
                and receipt["build"]["server_sha256"] == bound["ax-engine-server-prefetch"]
                and receipt["libmlx_sha256"] == receipt["build"]["libmlx_sha256"]
                and digest(exact_build) == bound["build.json"] and digest(delivered) == bound["delivered.json"]
                and digest(script.read_bytes()) == bound["scripts/run_flash_next_generate_matrix.py"]
                and receipt["inventory_sha256"] == bound["mxfp4-inventory.json"]
                and receipt["runner_sha256"] == bound["run_sidecar_prefetch_warm.py"]
                == digest(read(archive, "run_sidecar_prefetch_warm.py"))
                and receipt["pack_revision"] == json.loads(delivered)["revision"]
                and receipt["model_manifest_sha256"] == bound["model-manifest.json"] == MANIFEST,
                "Build, exact workload or pack changed")
        selection_raw = read(archive, "expected-resident-selection.json")
        require(digest(selection_raw) == bound["expected-resident-selection.json"], "Expected shard selection changed")
        selection = json.loads(selection_raw)
        require(receipt["launch"] == "ordinary-32k-resident-single-request"
                and list(receipt["arms"]) == list(ARMS), "Missing arms or altered product launch")
        stats, idle = {}, {}
        for label, first in ARMS.items():
            arm = receipt["arms"][label]
            mode = label.split("-")[0]
            require(arm["loader_mode"] == mode
                    and arm["model"] == {"id": "qwen3.8-flash-next", "context_length": 32768}
                    and arm["profile"] == {"effective_sequence_cap": 1, "expert_residency": "resident"}
                    and arm["health_ready"] == arm["health_after_requests"] == "ok"
                    and arm["context_overflow"] == {"http_status": 400, "error_code": "context_length_exceeded"},
                    f"Single-stream product controls changed in {label}")
            verify_routes(read(archive, f"{label}.log").decode(), mode, arm, selection)
            verify_samples(read(archive, f"{label}-samples.jsonl"), arm)
            idle[label] = [arm["quiet_baseline"]["cpu_idle_percent"], arm["post_run_cpu_idle_percent"]]
            require(all(isfinite(x) and 0 <= x <= 100 for x in idle[label]), "Invalid CPU idle sample")
            rows = arm["cases"]
            for row, geometry in zip(rows, expected_cases(first), strict=True):
                length, trial, phase = geometry
                key = matrix.ids_sha(matrix.workload_ids(VOCAB, length, trial))
                require((row["prompt_tokens"], row["trial"], row["phase"], row["prompt_hash"]) == (*geometry, key)
                        and row["route_errors"] == row["verified_steps"] == 0
                        and len(row["output_tokens"]) == 32 and row["output_tokens"] == oracle[key],
                        f"Request identity, route or independent output oracle changed in {label}")
            first_request = timing(rows[:1], matrix)
            stats[label] = {
                "ready_seconds": arm["ready_seconds"], "load_phase_seconds": arm["load_phase_seconds"],
                "sidecar_load_seconds": arm["sidecar_load_seconds"], "first_prompt_tokens": first,
                "first_request": first_request,
                "ready_seconds_plus_first_ttft": arm["ready_seconds"] + first_request["ttft_seconds"],
                "warmed_medians": {str(length): timing([
                    row for row in rows if row["prompt_tokens"] == length and row["phase"] == "warm"
                ], matrix) for length in WIDTHS},
            }
        code = int(read(archive, "status.txt"))
        eligible = [name for name, samples in idle.items() if min(samples) >= 90]
        ratios = {}
        for order in ("short", "long"):
            candidate = stats[f"prefetch-{order}"]["warmed_medians"]
            control = stats[f"mmap-{order}"]["warmed_medians"]
            ratios[order] = {length: {metric: candidate[length][metric] / value for metric, value in values.items()}
                             for length, values in control.items()}
        measured = {
            "matched_output_ids_per_arm": 448, "arms": stats, "warm_candidate_to_control_ratios": ratios,
            "slowest_pread_to_fastest_mmap_ready_ratio": max(stats[f"prefetch-{x}"]["ready_seconds"] for x in ("short", "long"))
            / min(stats[f"mmap-{x}"]["ready_seconds"] for x in ("short", "long")),
            "warmed_timing_band_met": all(row["ttft_seconds"] <= 1.03 and row["decode_tokens_per_second"] >= 0.97
                                          for order in ratios.values() for row in order.values()),
        }
        if code == 0:
            require(receipt["status"] == "complete" and len(eligible) == 4
                    and measured == receipt["diagnostic_comparison"], "Complete comparison changed")
        else:
            require(receipt["status"] == "failed" and len(eligible) < 4
                    and "profile, pressure or lifecycle changed" in read(archive, "owner-tail.txt").decode()
                    and "diagnostic_comparison" not in receipt,
                    "Nonzero receipt needs separate diagnosis")
        return {"qualification": False, "build": receipt["build"], "timing_eligible_arms": eligible,
                "cpu_idle_percent": idle, "controller_status": code, **measured}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--bundle", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.bundle, args.repo), indent=2, sort_keys=True, allow_nan=False))
