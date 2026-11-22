"""Compare one and three Flash Next worker warmup passes before HTTP readiness."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median
import subprocess
from threading import Event
from unittest.mock import patch

import check_flash_next_native_support as native
import run_flash_next_generate_matrix as matrix

REVISION = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = {"model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"}
INVENTORY = "97b890234af731ac6e07d2ed7dbe32b4b0fcb1f0aff01addaaa340a838b28cc6"
CONTEXT = 32768
ORDER = (False, True, True, False)
MODES = ("default", "required")
PROMPTS = matrix.PROMPTS[:2]
LENGTHS = matrix.LENGTHS
LOG_FIELDS = re.compile(r'\b(phase|passes|streams|elapsed_us|shape|generated)=(?:"([^"]+)"|(\w+))')


def need(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pressure():
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n",
                                       "kern.memorystatus_vm_pressure_level"], text=True))


def monitor(done, samples):
    while not done.wait(1):
        samples.append(pressure())


def warmup_phases(log, single):
    rows = []
    for line in log.splitlines():
        if "startup warmup" in line or "warmup request finished" in line:
            fields = {key: quoted or value for key, quoted, value in LOG_FIELDS.findall(line)}
            if "phase" in fields:
                rows.append(fields)
    worker = [row for row in rows if row["phase"] == "worker"]
    http = [row for row in rows if row["phase"] == "http"]
    attempts = [row for row in rows if row["phase"] == "worker_request"]
    passes = 1 if single else 3
    need(len(worker) == len(http) == 1 and int(worker[0]["passes"]) == passes
         and int(http[0]["streams"]) == 3, "Startup profiling did not exercise the chosen warmup policy")
    need([int(x["shape"]) for x in attempts] == [0, 1, 2] * passes
         and all(x["generated"] in ("true", "false") for x in attempts),
         "The full worker workload did not reach the generation service")
    timing = {"worker_elapsed_us": int(worker[0]["elapsed_us"]),
              "http_elapsed_us": int(http[0]["elapsed_us"])}
    need(all(v > 0 for v in timing.values()), "Missing startup phase timing")
    completed = [sum(x["shape"] == str(i) and x["generated"] == "true" for x in attempts)
                 for i in range(3)]
    need(all(completed), "At least one worker warmup shape never generated an output")
    return {"worker_passes": passes, "worker_completed_by_shape": completed,
            "http_streams": 3} | timing


def report_rows(run, mode):
    need(run["mode"] == mode and run["verifier"] == "canonical", "Unexpected serving route")
    cells, trajectories = run["cells"], run["trajectories"]
    schedule = [(length, index, index < matrix.WARMUPS) for length in LENGTHS
                for index in range(matrix.WARMUPS + matrix.MEASURED)]
    need([(c["length"], c["run"], c["warmup"]) for c in cells] == schedule
         and [t["prompt"] for t in trajectories] == list(PROMPTS),
         "Request corpus changed")
    expected_keys = {f"traj-{i:02d}" for i in range(len(PROMPTS))}
    expected_keys.update(f"cell-{length}-{index}" for length, index, _ in schedule)
    need(run["prompt_id_hashes"].keys() == expected_keys
         and all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in run["prompt_id_hashes"].values()),
         "Input token digests do not cover the fixed request roster")
    rows = trajectories + cells
    need(all(r["status"] == "finished" and r["route_errors"] == 0 and r["output_tokens"] for r in rows)
         and all(len(r["output_tokens"]) == matrix.OUTPUT_TOKENS for r in cells),
         "Request completion, route errors or token budgets changed")
    verified = sum(r["verified_steps"] for r in rows)
    need((mode == "required" and verified > 0) or (mode == "default" and verified == 0),
         "The configured MTP policy was not exercised")
    first = trajectories[0]["timing"]["ttft_seconds"]
    need(first is not None and isfinite(first) and first > 0, "Invalid first request timing")
    scores = {}
    for length in LENGTHS:
        scored = [r["timing"] for r in cells if r["length"] == length and not r["warmup"]]
        ttft = [t["ttft_seconds"] for t in scored]
        rates = [matrix.decode_rate(t) for t in scored]
        need(all(v is not None and isfinite(v) and v > 0 for v in (*ttft, *rates)),
             "Invalid scored TTFT or decode timing")
        scores[str(length)] = {"ttft_median_seconds": median(ttft), "decode_median_tok_s": median(rates)}
    stable = {"prompts": run["prompt_id_hashes"],
              "outputs": [[r[k] for k in ("prompt_tokens", "output_tokens", "finish_reason")] for r in rows]}
    return stable, {"first_ttft_seconds": first, "verified_steps": verified, "scored": scores}


def measure(args, mode, name, single):
    idle_before = None if args.functional else native.wait_for_quiet()["cpu_idle_percent"]
    samples = [pressure()]
    need(samples[0] == 1, "The benchmark requires normal memory pressure")
    requests, advertised = native.request, []

    def verify_model(base, path, *positional, **options):
        raw = requests(base, path, *positional, **options)
        if path == "/v1/models":
            advertised.extend(v["context_length"] for v in json.loads(raw)["data"])
        return raw

    done = Event()
    with ThreadPoolExecutor(max_workers=1) as worker:
        sampling = worker.submit(monitor, done, samples)
        try:
            with patch.object(matrix, "TOTAL_BLOCKS", CONTEXT // 16), patch.object(native, "request", verify_model):
                raw = matrix.run_mode(
                    args.server, args.pack, args.output, mode, PROMPTS, LENGTHS, label=name,
                    extra_env={
                        "AX_SERVER_FLASH_NEXT_STARTUP_WARMUP_ONCE": str(int(single)),
                        "RUST_LOG": "info,ax_engine_server::startup_warmup=debug",
                    },
                )
        finally:
            done.set()
            sampling.result()
    samples.append(pressure())
    idle_after = None if args.functional else native.cpu_idle_percent()
    need(advertised == [CONTEXT] and all(p == 1 for p in samples),
         "Context envelope or sampled memory pressure changed")
    log = args.output / f"server-{name}.log"
    phases = warmup_phases(log.read_text(), single)
    stable, scores = report_rows(raw, mode)
    case = args.output / f"{name}.json"
    case.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    return stable, {
        "name": name, "mode": mode, "single": single, "ready_seconds": raw["ready_seconds"],
        "raw_sha256": digest(case), "log_sha256": digest(log),
        "host_idle_percent": [idle_before, idle_after], "normal_pressure_samples": len(samples),
        "timing_eligible": not args.functional and all(90 <= v <= 100 for v in (idle_before, idle_after)),
    } | phases | scores


def run(args):
    build = json.loads(args.build.read_bytes())
    inventory = json.loads(args.inventory.read_bytes())
    need(digest(args.server) == build["server_sha256"] and digest(args.inventory) == INVENTORY
         and inventory["revision"] == REVISION, "Build or publisher identity changed")
    if not args.functional:
        native.storage_info(args.pack)
    need(native.validate_inventory(args.pack, inventory) == MANIFEST, "The frozen delivered manifest changed")
    args.output.mkdir()
    evidence = {"scope": "M2 functional warmup" if args.functional else "M5 startup warmup comparison", "qualification": False, "build": build,
                "publisher_revision": REVISION, "inventory_sha256": INVENTORY, "manifest": MANIFEST, "arms": []}
    for mode in ("default",) if args.functional else MODES:
        baseline = None
        for index, single in enumerate(ORDER[:2] if args.functional else ORDER, 1):
            label = f"{mode}-{index}"
            stable, stats = measure(args, mode, label, single)
            if baseline is None:
                baseline = stable
            need(baseline == stable, f"{mode}: prompt, generation or finish reason changed")
            evidence["arms"].append(stats)
            text = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
            (args.output / "progress.json").write_text(text)
            print(label, stats["worker_elapsed_us"], stats["http_elapsed_us"], flush=True)
    need(native.validate_inventory(args.pack, inventory) == MANIFEST, "Publisher bytes changed during the run")
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("server", "build", "pack", "inventory", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--functional", action="store_true")
    run(parser.parse_args())
