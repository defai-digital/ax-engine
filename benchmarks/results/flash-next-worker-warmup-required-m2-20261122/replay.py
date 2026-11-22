"""Replay the Flash Next Required MTP worker warmup comparison."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT.parent / "flash-next-worker-warmup-once-m2-20261122"
REQUIRED_ARMS = (("required-1", False), ("required-2", True))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Missing replay module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checksums(root):
    return dict(line.split()[::-1] for line in (root / "inputs-m2.sha256").read_text().splitlines())


def replay():
    baseline = load("default_warmup_verifier", DEFAULT / "replay.py")
    need = baseline.need
    hashes = checksums(ROOT)
    previous = checksums(DEFAULT)
    previous.pop("run_m2.sh")
    for name in ("run_required.py", "run_m2.sh"):
        need(hashes.pop(name) == baseline.digest(ROOT / name), f"Updated Required source differs: {name}")
    need(hashes == previous, "The Required run changed the sealed binary, inventory or shared code")
    build = baseline.read(DEFAULT / "build.json")
    need(build["server_sha256"] == hashes["ax-engine-server-startup"]
         and build["driver_sha256"] == hashes["run_warmup.py"] == baseline.digest(DEFAULT / "run_warmup.py")
         and baseline.digest(DEFAULT / "build.json") == hashes["build.json"],
         "The Required run did not bind the original executable and driver")
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        baseline.unpack(baseline.DEPENDENCIES, folder, baseline.HARNESS)
        need(all(baseline.digest(folder / name) == hashes[name] for name in baseline.HARNESS),
             "A shared Required helper differs")
        sys.path[:0] = [str(DEFAULT), str(folder)]
        run = load("required_warmup_driver", ROOT / "run_required.py")
        import run_warmup as runner

        need(run.ARMS == REQUIRED_ARMS and run.MODE == "required", "The staged arm order changed")
        labels = {label for label, _ in REQUIRED_ARMS}
        names = {"result.json", "status.txt",
                 *(f"{label}.json" for label in labels),
                 *(f"server-{label}.log" for label in labels)}
        baseline.unpack(ROOT / "evidence.tar.xz", folder, names)
        result = baseline.read(folder / "result.json")
        need((folder / "status.txt").read_text() == "0\n" and result["qualification"] is False
             and result["scope"] == "M2 Ultra 192 GB functional Required MTP worker warmup"
             and result["build"] == build and result["publisher_revision"] == runner.REVISION
             and result["inventory_sha256"] == runner.INVENTORY == hashes["inventory.json"]
             and result["manifest"] == runner.MANIFEST, "The Required run did not finish against the sealed pack")
        need([(row["name"], row["single"]) for row in result["arms"]] == list(REQUIRED_ARMS)
             and all(type(row["single"]) is bool for row in result["arms"]),
             "The arm order changed")
        stable, comparison = [], {}
        for row in result["arms"]:
            label = row["name"]
            raw = folder / f"{label}.json"
            log = folder / f"server-{label}.log"
            need(row["mode"] == run.MODE and row["timing_eligible"] is False
                 and row["host_idle_percent"] == [None, None] and row["normal_pressure_samples"] >= 2
                 and baseline.digest(raw) == row["raw_sha256"] and baseline.digest(log) == row["log_sha256"],
                 "The Required observation or host scope changed")
            raw_data = baseline.read(raw)
            stable.append(run.identity(raw_data))
            phases = runner.warmup_phases(log.read_text(), row["single"])
            _, scores = runner.report_rows(raw_data, run.MODE)
            need(row["ready_seconds"] == raw_data["ready_seconds"]
                 and all(row[key] == value for key, value in (phases | scores).items()),
                 "The Required summary does not replay")
            comparison[label] = {
                "responses": len(stable[-1]["outputs"]),
                "generated_ids": sum(len(output[1]) for output in stable[-1]["outputs"]),
                "verified_steps": row["verified_steps"],
                "accepted_draft_tokens": sum(route["mtp"]["accepted_tokens"] for route in stable[-1]["routing"]),
                "ready_seconds": row["ready_seconds"],
                "normal_pressure_samples": row["normal_pressure_samples"],
            } | phases
        need(stable[0] == stable[1], "Warmup changed Required output, stop state or verifier routing")
        return {"scope": result["scope"], "qualification": False, "arms": comparison}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
