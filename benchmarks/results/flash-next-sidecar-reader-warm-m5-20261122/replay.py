"""Replay the Flash Next sidecar-only startup and warmed-serving checks."""
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "scripts"))
import run_flash_next_generate_matrix as matrix  # noqa: E402

VOCAB = 248320
WIDTHS = (512, 2048, 8192)
ARMS = {"mmap-short": 512, "pread-short": 512, "pread-long": 8192, "mmap-long": 8192}
RAW_SHA256 = "9596e4fbb63496221edbd5040342cd83f2555f46555c539cc12986ccacf66843"


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def workload(first):
    other = 8192 if first == 512 else 512
    return [(first, 0, "first"), (512, 100, "warmup"),
            *((length, trial, "warm") for trial in (1, 2, 3) for length in WIDTHS),
            (other, 0, "parity"), (2048, 0, "parity"), (32736, 0, "boundary")]


def summarize_times(rows):
    values = {"ttft_seconds": median(row["timing"]["ttft_seconds"] for row in rows),
              "decode_tokens_per_second": median(matrix.decode_rate(row["timing"]) for row in rows)}
    require(all(math.isfinite(value) and value > 0 for value in values.values()),
            "Nonpositive or invalid serving duration")
    return values


def replay(receipt):
    require(receipt["status"] == "complete" and list(receipt["arms"]) == list(ARMS),
            "Four-arm receipt did not complete in the recorded order")
    require(receipt["launch"] == "ordinary-32k-resident-single-request",
            "Unexpected Flash Next launch contract")
    require(receipt["scope"] == "guarded mmap trunk with isolated MTP sidecar reader comparison",
            "Unexpected reader comparison scope")
    require(sha256((ROOT / "run_sidecar_warm.py").read_bytes()).hexdigest() == receipt["runner_sha256"],
            "Measured runner bytes changed")
    selection = json.loads((ROOT / "expected-resident-selection.json").read_bytes())
    require(receipt["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and receipt["inventory_sha256"] == "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
            and receipt["build"]["server_sha256"] == "891b756e7472e929c3c3b179a4cec945fd4c4eb7c30512156145ee8629109480",
            "Pack inventory or measured server does not match the control")
    arms, outputs, result = receipt["arms"], {}, {}
    first = arms["mmap-short"]
    for label, prompt_length in ARMS.items():
        arm = arms[label]
        mode = label.split("-")[0]
        require(arm["loader_mode"] == mode and arm["positional_sidecar_reads"] is (mode == "pread")
                and "positional_payload_reads" not in arm,
                f"Wrong loader for {label}")
        require(arm["profile"] == {"expert_residency": "resident", "effective_sequence_cap": 1}
                and arm["model"] == {"id": "qwen3.8-flash-next", "context_length": 32768},
                f"Unexpected residency or request scope for {label}")
        require(arm["pressure_levels"] == [1] and all(90 <= idle <= 100 for idle in (
                arm["quiet_baseline"]["cpu_idle_percent"], arm["post_run_cpu_idle_percent"])),
                f"Host conditions outside bounds for {label}")
        require(arm["health_ready"] == arm["health_after_requests"] == "ok"
                and arm["context_overflow"] == {"http_status": 400, "error_code": "context_length_exceeded"},
                f"Lifecycle or context boundary failed for {label}")
        require(arm["shard_count"] == len(selection)
                and arm["tensor_count"] == sum(count for _, count in selection),
                f"Loaded tensor inventory differs for {label}")
        rows = arm["cases"]
        for row, expected in zip(rows, workload(prompt_length), strict=True):
            width, trial, phase = expected
            prompt_hash = matrix.ids_sha(matrix.workload_ids(VOCAB, width, trial))
            require((row["prompt_tokens"], row["trial"], row["phase"], row["prompt_hash"])
                    == (*expected, prompt_hash), f"Unexpected workload row for {label}")
            tokens = row["output_tokens"]
            require(row["route_errors"] == row["verified_steps"] == 0 and len(tokens) == 32
                    and all(type(token) is int and 0 <= token < VOCAB for token in tokens)
                    and outputs.setdefault(prompt_hash, tokens) == tokens,
                    f"Route or token identity changed for {label}")
        first_time = summarize_times(rows[:1])
        ready = arm["ready_seconds"]
        phase_time = arm["load_phase_seconds"]
        sidecar = arm["sidecar_load_seconds"]
        require(0 < phase_time and 0 < sidecar and phase_time + sidecar < ready and math.isfinite(ready),
                f"Invalid readiness for {label}")
        result[label] = {"ready_seconds": ready, "load_phase_seconds": phase_time, "sidecar_load_seconds": sidecar,
                         "first_prompt_tokens": prompt_length, "first_request": first_time,
                         "ready_seconds_plus_first_ttft": ready + first_time["ttft_seconds"],
                         "warmed_medians": {str(length): summarize_times([
                             row for row in rows if row["phase"] == "warm" and row["prompt_tokens"] == length
                         ]) for length in WIDTHS}}
    readiness = max(result[label]["ready_seconds"] for label in ("pread-short", "pread-long"))
    control = min(result[label]["ready_seconds"] for label in ("mmap-short", "mmap-long"))
    ratios = {}
    for order in ("short", "long"):
        candidate = result[f"pread-{order}"]["warmed_medians"]
        control_medians = result[f"mmap-{order}"]["warmed_medians"]
        ratios[order] = {length: {metric: candidate[length][metric] / value
                                 for metric, value in timing.items()}
                         for length, timing in control_medians.items()}
    comparison = {"arms": result, "matched_output_ids_per_arm": len(first["cases"]) * 32,
                  "slowest_pread_to_fastest_mmap_ready_ratio": readiness / control,
                  "warm_candidate_to_control_ratios": ratios,
                  "warmed_timing_band_met": all(
                      value["ttft_seconds"] <= 1.03 and value["decode_tokens_per_second"] >= 0.97
                      for order in ratios.values() for value in order.values())}
    require(comparison == receipt["diagnostic_comparison"], "Raw rows and recorded comparison disagree")
    return comparison


if __name__ == "__main__":
    raw = gzip.decompress((ROOT / "receipt.json.gz").read_bytes())
    require(sha256(raw).hexdigest() == RAW_SHA256, "Native receipt SHA-256 changed")
    print(json.dumps(replay(json.loads(raw)), indent=2, sort_keys=True, allow_nan=False))
