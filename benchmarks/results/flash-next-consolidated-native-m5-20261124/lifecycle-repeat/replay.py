"""Recheck the current signed native lifecycle under its repeated host conditions."""

from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
BENCH = NATIVE.parent
REPO = NATIVE.parents[2]
MODES = ("disabled", "default")


def replay():
    refresh = run_path(str(NATIVE / "replay.py"))
    board = run_path(str(BENCH / "flash-next-guarded-reader-board-m5-20261123/replay.py"))
    controls = run_path(str(BENCH / "flash-next-guarded-reader-lifecycle-m5-20261123/replay.py"))
    board["verify_evaluator_inputs"]()
    reference, scope = board["reference_input"](None)
    need = refresh["need"]
    repeated_raw = (HERE / "result.json").read_bytes()
    repeated = json.loads(repeated_raw)
    with TemporaryDirectory(prefix="ax-flash-consolidated-lifecycle-") as directory:
        work = Path(directory)
        refresh["materialize"](NATIVE / "evidence.tar.xz", work)
        refresh["check_native"](work)
        original = refresh["read"](work, "lifecycle/result.json")
        need(original["host_idle_after"] < 90, "Original CPU endpoint changed")
        stable_fields = ("repo_id", "revision", "modes", "actions", "counters", "total_blocks", "ordinary_launch",
                         "expert_stream", "server_sha256", "inventory_sha256", "hardware", "lifecycle_passed")
        need(all(original[key] == repeated[key] for key in stable_fields), "Lifecycle workload or build changed")
        matching = {}
        for left, right in zip(original["runs"], repeated["runs"], strict=True):
            mode = left["mode"]
            need(mode == right["mode"] and mode in MODES and mode not in matching, "Unexpected lifecycle arm")
            stable = controls["stable_action"]
            need(list(map(stable, left["actions"])) == list(map(stable, right["actions"])),
                 f"Repeated {mode} lifecycle actions differ")
            need(left["model"] == right["model"] and right["final_verified_steps"] == 0,
                 f"Repeated {mode} model identity or default verifier policy changed")
            matching[mode] = len(right["actions"])
        need(matching == {mode: 7 for mode in MODES}, "Incomplete lifecycle controls")
        first_idle = [original["host_quiet"]["cpu_idle_percent"], original["host_idle_after"]]
        observed = [repeated["host_quiet"]["cpu_idle_percent"], repeated["host_idle_after"]]
        (work / "lifecycle/result.json").write_bytes(repeated_raw)
        (work / "reference").mkdir()
        (work / "reference/result.json").write_bytes(reference)
        output = work / "verdict.json"
        proc = subprocess.run([sys.executable, "-E", "-s", "-B", REPO / "scripts/check_flash_next_release_gates.py",
                               "--evidence-dir", work, "--output", output], capture_output=True)
        need(proc.returncode in (0, 1) and output.is_file(), proc.stderr.decode())
        verdict = refresh["read"](work, "verdict.json")
        need(verdict["release_ready"] == (proc.returncode == 0)
             and verdict["evidence_sha256"]["lifecycle"] == sha256(repeated_raw).hexdigest(),
             "Evaluator did not bind the repeat")
    return {"candidate": refresh["CANDIDATE"], "reference_input": scope, "first_cpu_idle_percent": first_idle,
            "repeat_cpu_idle_percent": observed, "repeat_timing_eligible": all(90 <= item <= 100 for item in observed),
            "repeat_sha256": sha256(repeated_raw).hexdigest(), "matching_actions": matching,
            "repeat_readiness_seconds": {run["mode"]: run["ready_seconds"] for run in repeated["runs"]},
            "release_ready": verdict["release_ready"], "gates": verdict["gates"]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
