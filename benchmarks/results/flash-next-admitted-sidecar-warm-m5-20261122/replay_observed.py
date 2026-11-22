"""Reconcile a warmed repeat with its passive host CPU tick samples."""

from collections import defaultdict
import gzip
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent


def main():
    warm = runpy.run_path(str(ROOT / "replay.py"))
    common = runpy.run_path(str(warm["PROTOTYPE"]))
    require = common["require"]
    observer = runpy.run_path(str(ROOT / "observe_warm_cpu.py"))
    inputs = {
        "observed/receipt.json": gzip.decompress((ROOT / "observed-receipt.json.gz").read_bytes()),
        "host-cpu-ticks.jsonl": gzip.decompress((ROOT / "observed-cpu-ticks.jsonl.gz").read_bytes()),
        "observe_warm_cpu.py": (ROOT / "observe_warm_cpu.py").read_bytes(),
        "status.txt": (ROOT / "observed-status.txt").read_bytes(),
    }
    hashes = {name: sha256(data).hexdigest() for name, data in inputs.items()}
    sealed = {name: digest for digest, name in (line.split() for line in
              (ROOT / "observed.sha256").read_text().splitlines())}
    require(hashes == sealed, "Observed serving evidence changed")
    build = json.loads(warm["STARTUP"].read_bytes())["build"]
    result = warm["replay"](json.loads(inputs["observed/receipt.json"]), common, build)
    require(inputs["status.txt"] == b"1\n" and result["ineligible_arms"] == list(common["ARMS"]),
            "The warmed repeat did not preserve its four timing misses")
    rows = [json.loads(line) for line in inputs["host-cpu-ticks.jsonl"].splitlines()]
    require(len(rows) > 1 and rows[-1]["phase"] == "failed", "Incomplete passive terminal samples")
    per_phase = defaultdict(lambda: [0, 0, 0])
    previous = None
    for row in rows:
        delta = row["deltas"]
        if previous is not None:
            require(delta == observer["tick_delta"](previous["ticks"], row["ticks"])
                    and row["previous_phase"] == previous["phase"]
                    and isclose(row["interval_seconds"], row["monotonic_seconds"]
                                - previous["monotonic_seconds"]), "Discontinuous CPU tick observations")
        total = sum(delta.values())
        require(total > 0 and row["interval_seconds"] > 0
                and isclose(row["host_idle_percent"], 100 * delta["idle"] / total), "An idle sample changed")
        if row["phase"] == row["previous_phase"]:
            phase = per_phase[row["phase"]]
            phase[0] += 1
            phase[1] += delta["idle"]
            phase[2] += total
        previous = row
    result["same_phase_cpu_intervals"] = {
        name: {"samples": count, "idle_percent": 100 * idle / total}
        for name, (count, idle, total) in per_phase.items()
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
