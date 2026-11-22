"""Evaluate the counterbalanced Flash Next warmup diagnostic from its raw cells."""

from argparse import ArgumentParser
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import re

from run_warmup import (INVENTORY, MANIFEST, MODES, ORDER, REVISION, digest, need,
                        report_rows, warmup_phases)


def finite_time(seconds):
    return type(seconds) in (float, int) and isfinite(seconds) and seconds > 0


def before_worker_seconds(contents, worker_seconds):
    log = re.sub(r"\x1b\[[0-9;]*m", "", contents)
    rows = log.splitlines()
    plan = [line.split()[0] for line in rows if "memory plan:" in line]
    warmed = [line.split()[0] for line in rows if "completed worker-path startup warmup" in line]
    need(len(plan) == len(warmed) == 1, "The startup trace cannot bind one memory plan and worker phase")
    elapsed = (datetime.fromisoformat(warmed[0]) - datetime.fromisoformat(plan[0])).total_seconds()
    interval = elapsed - worker_seconds
    need(finite_time(interval), "The memory-plan-to-worker interval is invalid")
    return interval

def compare(output, build, thresholds, *, modes=MODES, receipt="result.json"):
    result = json.loads((output / receipt).read_bytes())
    minimum = thresholds["timing_host"]["min_cpu_idle_percent"]
    readiness_bound = thresholds["readiness"]["warm_seconds_max"]
    need(result["scope"] == "M5 startup warmup comparison" and result["qualification"] is False
         and result["build"] == build and result["publisher_revision"] == REVISION
         and result["inventory_sha256"] == INVENTORY and result["manifest"] == MANIFEST,
         "Unexpected startup diagnostic scope, executable or pack")
    expected = [(mode, f"{mode}-{index}", single)
                for mode in modes for index, single in enumerate(ORDER, 1)]
    arms = result["arms"]
    need([(row["mode"], row["name"], row["single"]) for row in arms] == expected,
         "Incomplete or reordered startup comparison")
    mode_results = {}
    for mode in modes:
        observed = []
        identity = None
        for arm in (row for row in arms if row["mode"] == mode):
            label = arm["name"]
            stream, log = output / f"{label}.json", output / f"server-{label}.log"
            need(arm["raw_sha256"] == digest(stream) and arm["log_sha256"] == digest(log),
                 "A measured workload or warmup trace changed")
            raw = json.loads(stream.read_bytes())
            stable, scores = report_rows(raw, mode)
            routes = raw["trajectories"] + raw["cells"]
            if mode == "required":
                need(all(row["verified_steps"] > 0 and row["mtp"]["available"]
                         and row["mtp"]["requested"] and row["mtp"]["active"] for row in routes),
                     "A Required request completed without the verifier")
            else:
                need(all(row["verified_steps"] == 0 for row in routes),
                     "A default request activated the verifier")
            contents = log.read_text()
            phases = warmup_phases(contents, arm["single"])
            need(all(arm[key] == value for key, value in (phases | scores).items())
                 and arm["ready_seconds"] == raw["ready_seconds"] and finite_time(raw["ready_seconds"]),
                 "Startup or HTTP timing does not replay")
            if identity is None:
                identity = stable
            need(stable == identity, "The warmup treatment changed prompt, output or stop")
            idle = arm["host_idle_percent"]
            need(len(idle) == 2 and all(type(x) in (float, int) and 0 <= x <= 100 for x in idle)
                 and arm["normal_pressure_samples"] >= 2,
                 "Missing host measurements")
            eligible = all(minimum <= x for x in idle)
            need(eligible == arm["timing_eligible"], "Timing eligibility does not replay")
            observed.append(arm | {"plan_to_worker_start_seconds":
                                  before_worker_seconds(contents, phases["worker_elapsed_us"] / 1e6)})

        ratios = []
        for baseline, candidate in ((observed[0], observed[1]), (observed[3], observed[2])):

            eligible = baseline["timing_eligible"] and candidate["timing_eligible"]
            result_pair = {
                "order": sorted((baseline["name"], candidate["name"])),
                "timing_eligible": eligible,
            }
            if eligible:
                timing_cells = {}
                for length, values in baseline["scored"].items():
                    candidate_scores = candidate["scored"][length]
                    timing_cells[length] = {
                        "warm_ttft_ratio": candidate_scores["ttft_median_seconds"] / values["ttft_median_seconds"],
                        "warm_decode_ratio": candidate_scores["decode_median_tok_s"] / values["decode_median_tok_s"],
                    }
                readiness_delta = candidate["ready_seconds"] - baseline["ready_seconds"]
                result_pair.update(
                    candidate_observed_within_warm_bound=candidate["ready_seconds"] <= readiness_bound,
                    ready_delta_s=readiness_delta,
                    launch_to_first_token_delta_s=readiness_delta + candidate["first_ttft_seconds"]
                                                  - baseline["first_ttft_seconds"],
                    worker_delta_s=(candidate["worker_elapsed_us"] - baseline["worker_elapsed_us"]) / 1e6,
                    http_delta_s=(candidate["http_elapsed_us"] - baseline["http_elapsed_us"]) / 1e6,
                    plan_to_worker_start_delta_s=candidate["plan_to_worker_start_seconds"]
                                                 - baseline["plan_to_worker_start_seconds"],
                    lengths=timing_cells,
                )
            ratios.append(result_pair)
        mode_results[mode] = {
            "matching_http_responses_per_arm": len(identity["outputs"]),
            "matching_output_ids_per_arm": sum(len(row[1]) for row in identity["outputs"]),
            "pairs": ratios,
        }
    return {"scope": "M5 same-snapshot warmup diagnostic", "qualification": False, "modes": mode_results}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("output", "build", "thresholds"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.output, json.loads(args.build.read_bytes()),
                             json.loads(args.thresholds.read_bytes())), indent=2, allow_nan=False))
