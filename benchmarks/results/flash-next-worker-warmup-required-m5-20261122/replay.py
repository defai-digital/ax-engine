"""Replay the complete M5 Flash Next warmup comparison using its original phases."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DIRECT = ROOT.parent / "flash-next-worker-warmup-default-m5-20261122"
M2_DIRECT = ROOT.parent / "flash-next-worker-warmup-once-m2-20261122"
M2_REQUIRED = ROOT.parent / "flash-next-worker-warmup-required-m2-20261122"
THRESHOLDS = ROOT.parents[2] / "scripts/flash_next_release_thresholds.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Missing frozen verifier: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay():
    prior = load("default_phase", M2_DIRECT / "replay.py")
    need, read, digest = prior.need, prior.read, prior.digest
    hashes = dict(line.split()[::-1] for line in (ROOT / "inputs.sha256").read_text().splitlines())
    build = read(DIRECT / "build.json")
    need(hashes == dict(line.split()[::-1] for line in (DIRECT / "inputs.sha256").read_text().splitlines())
         and hashes["build.json"] == digest(DIRECT / "build.json")
         and build["driver_sha256"] == hashes["run_warmup.py"] == digest(M2_DIRECT / "run_warmup.py")
         and build["server_sha256"] == hashes["ax-engine-server-startup"],
         "The complete run changed the default-phase executable or sealed inputs")
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        prior.unpack(prior.DEPENDENCIES, scratch, prior.HARNESS)
        need(all(digest(scratch / name) == hashes[name] for name in prior.HARNESS),
             "A frozen workload helper changed")
        for folder, mode, extra in ((DIRECT, "default", {"progress.json"}),
                                    (ROOT, "required", {"result.json", "status.txt"})):
            names = extra | {f"{mode}-{i}.json" for i in range(1, 5)} | {
                f"server-{mode}-{i}.log" for i in range(1, 5)
            }
            prior.unpack(folder / "evidence.tar.xz", scratch, names)
        observed = read(scratch / "result.json")
        interim = read(scratch / "progress.json")
        need((scratch / "status.txt").read_text() == "0\n" and observed["arms"][:4] == interim["arms"]
             and {k: v for k, v in observed.items() if k != "arms"} ==
                 {k: v for k, v in interim.items() if k != "arms"},
             "The complete run did not preserve the original default phase")
        thresholds_sha = "17a39e0229cf734dca04ffa5abb4718a6c7b1c9c773891385a37396a1625345b"
        need(digest(THRESHOLDS) == thresholds_sha, "Readiness guard changed")
        sys.path[:0] = [str(scratch), str(M2_DIRECT), str(DIRECT)]
        from compare_m5 import compare
        report = compare(scratch, build, read(THRESHOLDS))
        required = load("required_control", M2_REQUIRED / "run_required.py")
        source_checks = dict(line.split()[::-1] for line in
                             (M2_REQUIRED / "inputs-m2.sha256").read_text().splitlines())
        need(digest(M2_REQUIRED / "run_required.py") == source_checks["run_required.py"],
             "The per-request Required verifier changed")
        signatures = [required.identity(read(scratch / f"required-{i}.json")) for i in range(1, 5)]
        report["modes"]["required"]["identical_per_request_verifier_records"] = all(
            sig["routing"] == signatures[0]["routing"] for sig in signatures
        )
        return report | {
            "normal_pressure_samples": sum(row["normal_pressure_samples"] for row in observed["arms"]),
            "required_ready_s_in_process_order": [row["ready_seconds"] for row in observed["arms"][4:]],
            "required_worker_s_in_process_order": [row["worker_elapsed_us"] / 1e6 for row in observed["arms"][4:]],
            "required_cpu_idle_percent": [row["host_idle_percent"] for row in observed["arms"][4:]],
            "required_verified_steps": [row["verified_steps"] for row in observed["arms"][4:]],
        }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
