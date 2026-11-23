"""Recheck the guarded-reader M5 lifecycle against prior signed control actions."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
FRESH_SHA = "9e5eaff70d2d365c3f49ba7c1809b121aec5d447ccda2bfed14f826e4ed8230b"
PRIOR_SHA = "7924bbe552fe85e16d80f61b5f4db55ef860cd11526dc61625a7c209ed6a3ba1"
MODES = ("disabled", "default")
MEMBERS = {"result.json", *(f"server-{mode}.log" for mode in MODES)}


def member(archive, name):
    entry = archive.getmember(name)
    assert entry.isfile()
    with archive.extractfile(entry) as stream:
        return stream.read()


def stable_action(row):
    value = dict(row)
    usage = value.get("usage", {})
    if "ax_engine" in usage:
        value["usage"] = {**usage, "ax_engine": {
            key: data for key, data in usage["ax_engine"].items()
            if key not in {"prefill_time_us", "decode_time_us"}
        }}
    return value


def replay(repo, capture):
    bench = repo / "benchmarks/results"
    previous = bench / "flash-next-probe-followup-lifecycle-m5-20261122"
    fresh_path = bench / "flash-next-guarded-reader-fresh-m5-20261123/evidence.tar.xz"
    previous_raw = (previous / "lifecycle.json").read_bytes()
    assert sha256(previous_raw).hexdigest() == PRIOR_SHA
    assert sha256(fresh_path.read_bytes()).hexdigest() == FRESH_SHA
    sys.path.insert(0, str(repo / "scripts"))
    import check_flash_next_release_gates as gates
    import run_flash_next_qa_cohort as qa

    prior_runs = {row["mode"]: row for row in json.loads(previous_raw)["runs"]}
    with tarfile.open(capture) as archive, tarfile.open(fresh_path) as fresh:
        names = archive.getnames()
        assert len(names) == len(MEMBERS) and set(names) == MEMBERS
        data = {name: member(archive, name) for name in names}
        delivery = json.loads(member(fresh, "delivery/result.json"))
        manifest = {name: digest for digest, name in map(str.split, member(fresh, "inputs.sha256").decode().splitlines())}
    assert all(token not in value for value in data.values() for token in (b"/Users/", b"/private/tmp/", b"127.0.0.1"))
    for name in ("run_flash_next_lifecycle.py", "check_flash_next_release_gates.py", "flash_next_release_thresholds.json"):
        path = f"scripts/{name}"
        assert sha256((repo / path).read_bytes()).hexdigest() == manifest[path]
    result = json.loads(data["result.json"])
    assert all(result[name] == delivery[name] for name in ("hardware", "revision", "inventory_sha256", "server_sha256"))
    assert result["server_sha256"] == manifest["ax-engine-server"]
    assert result["ordinary_launch"] is True and result["total_blocks"] == 2048
    assert result["qualification"] is False and result["release_ready"] is False
    assert result["modes"] == list(MODES) == [row["mode"] for row in result["runs"]]
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    gate, = gates.gate_lifecycle(result, thresholds)
    assert gate["passed"], gate["detail"]
    report = {}
    for run in result["runs"]:
        mode = run["mode"]
        expected = qa.ORDINARY_MODEL_ID if mode == "default" else qa.MODEL_ID
        assert run["model"] == {"id": expected, "context_length": 32768}
        assert run["final_verified_steps"] == 0
        assert list(map(stable_action, run["actions"])) == list(map(stable_action, prior_runs[mode]["actions"]))
        log = re.sub(r"\x1b\[[0-9;]*m", "", data[f"server-{mode}.log"].decode())
        assert "decision=resident" in log and "context_tokens=32768 effective_sequence_cap=Some(1)" in log
        report[mode] = {"actions_matching": len(run["actions"]), "ready_seconds": run["ready_seconds"],
                        "final_verified_steps": run["final_verified_steps"]}
    readiness, = gates.gate_readiness(delivery, [(mode, run["ready_seconds"]) for mode, run in report.items()], thresholds)
    idle = [result["host_quiet"]["cpu_idle_percent"], result["host_idle_after"]]
    floor = thresholds["timing_host"]["min_cpu_idle_percent"]
    return {"scope": "Guarded-reader ordinary native lifecycle on M5", "release_ready": False,
            "server_sha256": result["server_sha256"], "lifecycle_sha256": sha256(data["result.json"]).hexdigest(),
            "lifecycle": gate, "readiness": readiness, "runs": report, "timing_cpu_idle_percent": idle,
            "timing_eligible": all(floor <= value <= 100 for value in idle)}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.repo.resolve(), args.archive), sort_keys=True, indent=2, allow_nan=False))
