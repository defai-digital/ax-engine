"""Verify a complete native Flash Next logit-selection comparison."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PRIOR = ROOT.parent / "flash-next-canonical-cost-m5-20261123"
FRESH = ROOT.parent / "flash-next-probe-followup-fresh-m5-20261123" / "evidence.tar.xz"
DRIVER = {
    "check_flash_next_native_support.py", "compare_cost.py", "qualify_qwen38_flash_next.py",
    "run_cost.py", "run_flash_next_generate_matrix.py", "run_native_m5.sh",
    "run_native_reductions.py", "verify_routes.py",
}
BUILDS = {"baseline": "build.json", "candidate": "candidate/build.json"}
BINARIES = {"baseline": "ax-engine-server", "candidate": "candidate/ax-engine-server"}
OBSERVATIONS = {"rows-required.jsonl", "server-required.log"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(archive, name):
    item = archive.getmember(name)
    require(item.isfile(), f"Expected a regular archive member: {name}")
    with archive.extractfile(item) as stream:
        return stream.read()


def install(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def derive(root, result, expected):
    sys.path.insert(0, str(root))
    from run_native_reductions import compare, load_workloads, raw_rows
    from run_cost import matrix
    from verify_routes import route

    workloads, panels = load_workloads()
    labels = result["order"]
    require(labels in (["baseline", "candidate"], ["candidate", "baseline"]), "Invalid process order")
    if labels[0] == "candidate":
        workloads.reverse()
    require(result["qualification"] is False and result["request_policy"] == "required"
            and result["workload_panels"] == panels and set(result["arms"]) == BUILDS.keys(),
            "The native trial changed its run contract")
    records, idle = {}, {}
    for label in labels:
        arm = result["arms"][label]
        path = root / f"observed-{label}"
        log = path / "server-required.log"
        require(arm["mode"] == "required" and route(log.read_bytes(), log.name) == arm["startup_route"],
                f"Startup route evidence changed: {label}")
        idle[label] = arm["host_cpu_idle_percent"]
        samples = idle[label]
        require(len(samples) == 2 and all(0 <= value <= 100 for value in samples)
                and arm["timing_eligible"] is all(value >= 90 for value in samples),
                f"Invalid CPU idle evidence: {label}")
        warmup = arm["warmup"]
        require(warmup["id"] == "fixed-512" and warmup["budget"] == 64
                and warmup["pressure"] == [1, 1] and warmup["route_errors"] == 0
                and len(warmup["output_tokens"]) == 64, f"Warmup contract changed: {label}")
        matrix.check_verifier_schedule(warmup, "required", "canonical")
        records[label] = raw_rows(path)
        for row, (panel, identity, _, budget) in zip(records[label], workloads, strict=True):
            require((row["panel"], row["id"], row["budget"]) == (panel, identity, budget)
                    and row["status"] == "finished" and row["pressure"] == [1, 1]
                    and row["route_errors"] == 0 and row["mtp"]["requested"] is True
                    and row["verified_steps"] == row["counters"]["verified_steps"]
                    and 0 <= row["counters"]["accepted_steps"] <= row["verified_steps"],
                    f"Execution route changed: {label}/{identity}")
            matrix.check_verifier_schedule(row, "required", "canonical")
            for key in ("prompt_sha256", "output_tokens", "finish_reason"):
                require(row[key] == expected[identity][key], f"Original direct output changed: {label}/{identity}/{key}")
    eligible = all(arm["timing_eligible"] for arm in result["arms"].values())
    panels = compare(records["baseline"], records["candidate"], panels)
    require(result["timing_eligible"] is eligible and result["panels"] == panels,
            "The native comparison summary does not match the observations")
    return {"order": labels, "cpu_idle_percent": idle, "timing_eligible": eligible,
            "source_builds": result["source_builds"],
            "matched_output_tokens": sum(len(row["output_tokens"]) for row in records["baseline"]),
            "panels": panels}


def replay(path):
    previous = run_path(str(PRIOR / "replay.py"))
    prior_archive = PRIOR / "evidence.tar.xz"
    require(previous["replay"](prior_archive) == (PRIOR / "comparison.json").read_bytes(),
            "The paired direct serving reference changed")
    with (tarfile.open(path) as archive, tarfile.open(FRESH) as fresh, tarfile.open(prior_archive) as prior,
          TemporaryDirectory(prefix="flash-next-native-logits-") as directory):
        driver = {name: read(archive, f"driver/{name}") for name in DRIVER}
        staged = {name: read(archive, f"raw/{name}") for name in BUILDS.values()}
        staged.update({"inventory.json": read(fresh, "driver/inventory.json"), **driver})
        staged.update({f"panels/{name}.jsonl": (REPO / file).read_bytes()
                       for name, file in previous["CORPORA"].items()})
        raw = {f"observed-{label}/{name}": read(archive, f"raw/observed-{label}/{name}")
               for label in BUILDS for name in OBSERVATIONS}
        roster = ({f"driver/{name}" for name in driver}
                  | {f"raw/{name}" for name in raw.keys() | set(BUILDS.values())}
                  | {"raw/status.txt", "raw/inputs.sha256", "raw/result-native.json", "candidate.patch"})
        require(set(archive.getnames()) == roster and len(archive.getnames()) == len(roster)
                and read(archive, "raw/status.txt") == b"0\n", "Unexpected or incomplete evidence archive")
        checksums = read(archive, "raw/inputs.sha256").decode().splitlines()
        manifest = {name: digest for digest, name in map(str.split, checksums)}
        builds = {label: json.loads(staged[name]) for label, name in BUILDS.items()}
        base, candidate = builds["baseline"], builds["candidate"]
        require(set(manifest) == staged.keys() | set(BINARIES.values()) and len(checksums) == len(manifest)
                and all(sha256(data).hexdigest() == manifest[name] for name, data in staged.items())
                and all(builds[label]["server_sha256"] == manifest[name] for label, name in BINARIES.items()),
                "Staged input or executable digest changed")
        require(staged[BUILDS["baseline"]] == read(fresh, "raw/build.json")
                and candidate["source_base"] == base["commit"] and candidate["tree"] != base["tree"]
                and candidate["server_sha256"] != base["server_sha256"]
                and all(candidate[key] == base[key] for key in
                        ("mlx", "mlx_metal", "libmlx_sha256", "profile", "rustc", "cargo", "codesign_verified"))
                and candidate["codesign_verified"] is True
                and sha256(read(archive, "candidate.patch")).hexdigest() == candidate["patch_sha256"],
                "The comparison changed the signed baseline, candidate patch or toolchain")
        expected = {row["id"]: row for row in map(json.loads, read(prior, "raw/rows-disabled.jsonl").splitlines())}
        result = json.loads(read(archive, "raw/result-native.json"))
        require(result["source_builds"] == builds and result["inventory_sha256"] == manifest["inventory.json"]
                and result["native_manifest"] == json.loads(read(prior, "raw/result.json"))["manifest"],
                "The recorded trial changed a build or the pinned pack")
        root = Path(directory)
        install(root, {**staged, **raw})
        return derive(root, result, expected)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    print(json.dumps(replay(parser.parse_args().archive), indent=2, sort_keys=True, allow_nan=False))
