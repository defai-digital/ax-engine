"""Replay the Flash Next current-source native campaign on the standard M5 profile."""

from hashlib import file_digest
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
ARMS = ("disabled", "default")
PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = {"model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"}


def need(valid, message):
    if not valid:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def unpack(path, directory):
    with tarfile.open(path) as archive:
        for part in archive:
            relative = Path(part.name)
            dest = directory / relative
            need(part.isfile() and not relative.is_absolute() and ".." not in relative.parts
                 and not dest.exists(), "Unexpected campaign member")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(part).read())


def identities(driver):
    build = read(driver / "build.json")
    inventory = digest(driver / "inventory.json")
    hashes = {}
    for line in (driver / "inputs.sha256").read_text().splitlines():
        checksum, name = line.split()
        name = Path(name)
        need(name not in hashes and not name.is_absolute() and ".." not in name.parts,
             "Unexpected or repeated checksum member")
        hashes[name] = checksum
    binary = Path("ax-engine-server")
    need(hashes.pop(binary) == build["server_sha256"], "The live arm hashes disagree with the binary")
    need(all(digest(driver / name) == sha for name, sha in hashes.items()),
         "The measured driver, inventory or workload changed")
    need(build["profile"] == "release-server" and build["mlx"] == build["mlx_metal"] == "0.32.4"
         and build["rustc"].startswith("rustc 1.99.1 ") and build["cargo"].startswith("cargo 1.99.1 "),
         "Unexpected compiler or MLX identity")
    return build, inventory


def load_arm(observed, name, build, inventory):
    result = read(observed / name / "result.json")
    hardware = result["hardware"]
    need(result["revision"] == PIN and result["server_sha256"] == build["server_sha256"]
         and result["inventory_sha256"] == inventory and hardware["soc"] == "Apple M5 Max"
         and hardware["memory_bytes"] == 128 * 2**30 and result["qualification"] is False
         and result["release_ready"] is False, f"{name}: model, host or build differs")
    return result


def functional_phases(observed, driver, seal, build, inventory):
    support = load_arm(observed, "native-support", build, inventory)
    need(support["functional_controls_passed"]
         and support["native_manifest_before"] == support["native_manifest_after"] == MANIFEST
         and support["negative_control_result"]["activation_error_observed"]
         and set(support["results"]) == {"disabled", "required", "default"}
         and all(len(v) == 7 for v in support["results"].values()),
         "Native HTTP or activation safety controls failed")
    skip = read(observed / "explicit-skip.json")
    need(skip["completed"] and skip["code"] == 1
         and skip["payload_markers"] and set(skip["payload_markers"].values()) == {0},
         "The native bypass conflict accessed a forbidden startup phase")
    qa = load_arm(observed, "qa", build, inventory)
    oracle = read(driver / "signed-native.json")
    need(qa["ordinary_launch"] is True
         and {arm: seal.prompt_rows(observed / "qa" / arm) for arm in ARMS} == oracle["qa"],
         "Raw current QA rows differ from the frozen direct baseline")
    lifecycle = load_arm(observed, "lifecycle", build, inventory)
    need(lifecycle["lifecycle_passed"]
         and {r["mode"] for r in lifecycle["runs"]} == set(ARMS)
         and all(r["exit_code_after_stop"] == r["final_verified_steps"] == 0
                 and len(r["actions"]) == 7
                 and all(a["drained"] and set(a["drained"].values()) == {0} for a in r["actions"])
                 for r in lifecycle["runs"]),
         "Native lifecycle controls are incomplete or leaked work")
    return {"native-support": support, "qa": qa, "lifecycle": lifecycle}


def replay_archive(root):
    with TemporaryDirectory() as directory:
        work = Path(directory)
        unpack(root / "evidence.tar.xz", work)
        driver, observed = work / "driver", work / "observed"
        sys.path.insert(0, str(driver))
        import seal_native as seal
        build, inventory = identities(driver)
        need((work / "status.txt").read_text() == "0\n", "The live controller did not complete")
        rows = functional_phases(observed, driver, seal, build, inventory)
        matrix = load_arm(observed, "generate-matrix", build, inventory)
        repeats = read(observed / "generate-matrix/raw-runs.json")
        oracle = read(driver / "signed-native.json")["matrix"]
        need(set(matrix["runs"]) == repeats.keys() == oracle.keys()
             and all(len(repeats[arm]) == 2 and all(seal.matrix_rows(r) == oracle[arm] for r in repeats[arm])
                     and seal.matrix_rows(matrix["runs"][arm]) == oracle[arm] for arm in ARMS),
             "At least one independent matrix output differs")
        need(all(r["verified_steps"] == r["route_errors"] == 0 for runs in repeats.values() for process in runs
                 for r in (*process["trajectories"], *process["cells"])),
             "A direct stream had verifier activity or route errors")
        rows["generate-matrix"] = matrix
        quiet = {name: [rows[name]["host_quiet"]["cpu_idle_percent"], rows[name]["host_idle_after"]]
                 for name in ("qa", "lifecycle", "generate-matrix")}
        summary = {
            "scope": "Current-source native controls on the resident Flash Next standard",
            "qualification": False, "build": build, "pack_revision": PIN,
            "manifest_sha256": MANIFEST["model-manifest.json"], "inventory_sha256": inventory,
            "reference_control_sha256": digest(driver / "signed-native.json"),
            "observed_sha256": {name: digest(observed / name / "result.json") for name in rows},
            "native_http_cases": 21, "matched_qa_rows": 210, "matched_independent_matrix_arrays": 124,
            "readiness_seconds": rows["native-support"]["readiness_seconds"], "quiet_samples": quiet,
            "timing_eligible_legs": [name for name, samples in quiet.items() if all(90 <= n <= 100 for n in samples)],
        }
        need(summary == read(observed / "comparison.json"),
             "Replay and the online native campaign summary disagree")
        return summary


if __name__ == "__main__":
    print(json.dumps(replay_archive(ROOT), indent=2, allow_nan=False))
