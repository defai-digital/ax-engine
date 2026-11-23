"""Replay the current-main Flash Next lifecycle against the stable signed receipt."""
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
PREVIOUS = RESULTS / "flash-next-probe-followup-lifecycle-m5-20261122"
PREVIOUS_BUILD = RESULTS / "flash-next-natural-prefix-switch-controls-m5-20261123"
MODES = {"default", "disabled"}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def replay():
    prior_tools = run_path(str(PREVIOUS / "replay.py"))
    stable_action = prior_tools["stable_action"]
    gates = prior_tools["gates"]
    previous_raw = (PREVIOUS / "lifecycle.json").read_bytes()
    previous = json.loads(previous_raw)
    build = read(ROOT / "build.json")
    receipt = read(ROOT / "lifecycle.json")
    capture = read(ROOT / "capture.json")
    with tarfile.open(PREVIOUS_BUILD / "evidence.tar.xz") as archive:
        with archive.extractfile("build.json") as stream:
            signed_switch = json.load(stream)
    need(build["commit"] == build["source_base"]
         and build["patch_sha256"] == sha256(b"").hexdigest()
         and build["codesign_verified"] is True and capture["codesign_verified"] is True
         and build["profile"] == "release-server"
         and build["cargo"].startswith("cargo 1.99.1 ")
         and build["rustc"].startswith("rustc 1.99.1 ")
         and build["mlx"] == build["mlx_metal"] == "0.32.4",
         "Expected an unpatched, signed current-main release-server build")
    need(receipt["server_sha256"] == build["server_sha256"] == capture["server_sha256"]
         == signed_switch["server_sha256"], "Runtime binary digest disagreement")
    need(all(receipt[key] == previous[key] for key in ("hardware", "revision", "inventory_sha256"))
         and receipt["inventory_sha256"] == capture["inventory_sha256"]
         and capture["inventory_members"] == 49
         and capture["final_pressure"] == 1
         and receipt["qualification"] is False and receipt["release_ready"] is False,
         "Expected the same retained pack, M5 SKU, and functional validation profile")
    need(capture["manifest_sha256"] == "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef",
         "Retained manifest digest disagreement")
    thresholds = read(gates.THRESHOLDS)
    gate, = gates.gate_lifecycle(receipt, thresholds)
    need(gate["passed"], gate["detail"])
    runs = {run["mode"]: run for run in receipt["runs"]}
    old_runs = {run["mode"]: run for run in previous["runs"]}
    need(len(receipt["runs"]) == len(receipt["modes"]) == 2
         and set(runs) == set(receipt["modes"]) == MODES, "Expected both direct modes")
    def stable_rows(run):
        return [{key: value for key, value in stable_action(row).items() if key != "mode"}
                for row in run["actions"]]

    need(stable_rows(runs["default"]) == stable_rows(runs["disabled"]),
         "Current direct modes disagree on stable lifecycle observations")
    summary = {}
    for mode in sorted(runs):
        run = runs[mode]
        rows = run["actions"]
        need(run["final_verified_steps"] == 0 and
             list(map(stable_action, rows)) == list(map(stable_action, old_runs[mode]["actions"])),
             f"Changed stable HTTP lifecycle behavior in {mode}")
        log = (ROOT / f"server-{mode}.log").read_bytes()
        need(b"decision=resident" in log, f"Expected resident expert loading for {mode}")
        marks = prior_tools["extract"](log)
        prefetch = "prefetch_complete" in marks["timestamps"]
        intervals = prior_tools["measure"](marks["timestamps"], run["ready_seconds"], prefetch)
        summary[mode] = {"matching_actions": len(rows), "ready_seconds": run["ready_seconds"],
                         "resident": True, "final_verified_steps": run["final_verified_steps"],
                         "sidecar_prefetch": prefetch,
                         "within_staged_readiness_bound":
                             run["ready_seconds"] <= thresholds["readiness"]["warm_seconds_max"],
                         "startup_intervals": intervals}
    return {"source": build["commit"], "server_sha256": build["server_sha256"],
            "prior_lifecycle_sha256": sha256(previous_raw).hexdigest(),
            "lifecycle_gate": gate, "runs": summary,
            "cpu_idle_percent": [receipt["host_quiet"]["cpu_idle_percent"], receipt["host_idle_after"]],
            "final_pressure": capture["final_pressure"], "release_ready": False}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
