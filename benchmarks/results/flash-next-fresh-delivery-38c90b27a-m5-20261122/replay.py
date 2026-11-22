"""Replay fresh MXFP4 delivery and the delivered native HTTP controls."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_flash_next_native_support as native
import check_flash_next_release_gates as gates

BUILD = "38c90b27a3abd3c94b592aabbbc884d441451511"
PREVIOUS = ROOT.parent / "flash-next-mxfp4-current-delivery-m5-20261122" / "evidence.tar.xz"


def require(passed, message):
    if not passed:
        raise ValueError(message)


def member(archive, name):
    row = archive.getmember(name)
    require(row.isfile(), f"Invalid evidence member: {name}")
    with archive.extractfile(row) as data:
        return data.read()


def check_transfer(raw, delivery, asset_bytes):
    events = [json.loads(line) for line in raw.splitlines()]
    terminal = delivery["download"]["terminal"]
    require(len(events) > 1 and events[-1] == terminal and terminal["schema_version"] == "ax.download_model.v1"
            and terminal["status"] == "ready" and terminal["repo_id"] == native.PRIMARY_REPO
            and terminal["revision"] == terminal["resolved_revision"] == native.PACK_REVISION
            and terminal["manifest_present"] is True and terminal["config_present"] is True
            and terminal["errors"] == [], "Downloader did not record the pinned terminal result")
    progress = events[:-1]
    phases = [row["phase"] for row in progress if row["bytes_total"] is None]
    counters = [(row["bytes_done"], row["bytes_total"]) for row in progress if row["bytes_total"] is not None]
    require(all(row["event"] == "progress" for row in progress) and phases == ["inventory", "validate", "ready"]
            and counters and all(type(done) is int and 0 <= done <= total == asset_bytes for done, total in counters)
            and all(a[0] <= b[0] for a, b in zip(counters, counters[1:])) and counters[-1][0] == asset_bytes,
            "Downloader progress does not cover all published bytes in order")
    return len(counters)


def stable_usage(response):
    usage = dict(response["usage"])
    if "ax_engine" in usage:
        usage["ax_engine"] = {key: value for key, value in usage["ax_engine"].items()
                              if key not in ("prefill_time_us", "decode_time_us")}
    return usage


def check_cases(observed, prior, modes):
    activations = {}
    require(list(observed["results"]) == modes == observed["modes"]
            and observed["functional_controls_passed"] is True
            and observed["negative_control_result"]["activation_error_observed"] is True,
            "Native mode order or required activation rejection changed")
    for mode in modes:
        rows = observed["results"][mode]
        require([row["name"] for row in rows] == [name for name, _, _ in native.CASES],
                f"Incomplete native control set for {mode}")
        active = 0
        for row, oracle in zip(rows, prior["results"][mode], strict=True):
            response = row["response"]
            native.check_answer(row["name"], response)
            expected_delta = native.check_activation(row["metrics_before"], row["metrics_after"], mode == "required",
                                                     response["usage"]["completion_tokens"], row["request"]["max_tokens"])
            require(row["request"] == oracle["request"] and response["choices"] == oracle["response"]["choices"]
                    and stable_usage(response) == stable_usage(oracle["response"])
                    and expected_delta == row["activation_delta"] == oracle["activation_delta"],
                    f"Delivered response or native counters changed for {mode}/{row['name']}")
            active += expected_delta[native.VERIFIED]
        activations[mode] = int(active)
    return activations


def replay(archive_path=ROOT / "evidence.tar.xz"):
    threshold_file = SCRIPTS / "flash_next_release_thresholds.json"
    thresholds = json.loads(threshold_file.read_bytes())
    modes = thresholds["delivery"]["functional_controls_on_delivered_copy"]
    with tarfile.open(PREVIOUS) as earlier, tarfile.open(archive_path) as archive:
        prior_raw = member(earlier, "native-support.json")
        prior = json.loads(prior_raw)
        hashes = {name: digest for digest, name in
                  (line.split() for line in member(archive, "inputs.sha256").decode().splitlines())}
        build_raw = member(archive, "build.json")
        build = json.loads(build_raw)
        require(sha256(build_raw).hexdigest() == hashes["build.json"] and build["commit"] == BUILD
                and build["profile"] == "release-server" and build["rustc"].startswith("rustc 1.99.1 ")
                and build["mlx"] == build["mlx_metal"] == "0.32.4", "Wrong tested runtime build")
        for name in ("check_flash_next_native_support.py", "qualify_qwen38_flash_next.py", "run_flash_next_delivery.py"):
            require(sha256((SCRIPTS / name).read_bytes()).hexdigest() == hashes[f"scripts/{name}"],
                    f"The sealed {name} changed")
        provenance = json.loads(member(archive, "source-sha256.json"))
        require(set(provenance) == {"delivery.json", "download.ndjson", "native-support.json"}
                and all(sha256(member(archive, name)).hexdigest() == row["portable"]
                        for name, row in provenance.items())
                and provenance["native-support.json"]["portable"] == provenance["native-support.json"]["original"],
                "A portable receipt changed")
        delivery_raw = member(archive, "delivery.json")
        observed_raw = member(archive, "native-support.json")
        delivery, observed = json.loads(delivery_raw), json.loads(observed_raw)
        require(delivery["completed"] is True and delivery["fresh_cache"] is True and delivery["resumed"] is False
                and delivery["previous_attempts"] == [] and delivery["metadata_preflight_passed"] is True,
                "Evidence is not a complete fresh delivery")
        require(delivery["server_sha256"] == build["server_sha256"] == hashes["ax-engine-server"]
                and delivery["download"]["ax_engine_sha256"] == build["cli_sha256"] == hashes["ax-engine"]
                and delivery["inventory_sha256"] == prior["inventory_sha256"] == build["inventory_sha256"]
                == hashes["inventory.json"], "Delivery, executable or pack inventory identity changed")
        require(delivery["hardware"] == observed["hardware"] == prior["hardware"]
                and delivery["revision"] == observed["revision"] == native.PACK_REVISION
                and not gates.delivery_control_problems(delivery, observed_raw),
                "Raw native support does not match the delivery and primary host")
        manifest = {"model-manifest.json": delivery["verification"]["native_manifest_sha256"]}
        require(all(observed[f"native_manifest_{stage}"] == prior[f"native_manifest_{stage}"] == manifest
                    for stage in ("before", "ready", "after")), "Delivered manifest changed")
        download = member(archive, "download.ndjson")
        counters = check_transfer(download, delivery, thresholds["target"]["published_bytes"])
        activation = check_cases(observed, prior, modes)
        delivery_gate, = gates.gate_delivery(delivery, thresholds)
        readiness, = gates.gate_readiness(delivery, [], thresholds)
        require(delivery_gate["passed"], "Publisher verification did not pass")
        timing = {}
        intervals = {}
        for mode, row in json.loads(member(archive, "startup-events.json")).items():
            require(gates.is_sha256(row["raw_log_sha256"]), "Startup log checksum missing")
            stamps = row["utc"]
            names = ("memory_plan", "runner_admission", "http_listening")
            require(tuple(stamps) == names, "Startup event order changed")
            plan, target, listening = [datetime.fromisoformat(stamps[name].replace("Z", "+00:00")) for name in names]
            loading = (target - plan).total_seconds()
            server = (listening - target).total_seconds()
            require(0 < loading and 0 < server
                    and loading + server <= observed["readiness_seconds"][mode] + 0.001,
                    "Startup log intervals disagree with the readiness observation")
            timing[mode] = {"plan_to_runner_admission_seconds": round(loading, 3),
                            "runner_admission_to_listening_seconds": round(server, 3)}
            intervals[mode] = (plan, listening)
        require(list(timing) == modes, "Startup timing modes changed")
        samples = [json.loads(line) for line in member(archive, "host-samples.jsonl").splitlines()]
        require(samples and all(row.get("event") == "sample" for row in samples)
                and samples[-1]["after"]["delivery_completed"] is True
                and any(row["before"]["modes"] == {} for row in samples)
                and all(set(row["after"]["modes"]).issubset(modes) for row in samples)
                and any(row["after"]["modes"].keys() == set(modes) for row in samples),
                "Passive observer did not cover the startup sequence through delivery completion")
        idle = [row["cpu_idle_percent"] for row in samples if row["cpu_idle_percent"] is not None]
        levels = sorted({row["memory_pressure_level"] for row in samples if row["memory_pressure_level"] is not None})
        startup_idle = {}
        for mode, (start, end) in intervals.items():
            window = [row for row in samples if start <= datetime.fromisoformat(row["utc"]) < end]
            require(window and all(row["cpu_idle_percent"] is not None for row in window),
                    f"No valid host samples inside {mode} startup")
            startup_idle[mode] = {"count": len(window),
                                  "min_percent": min(row["cpu_idle_percent"] for row in window),
                                  "pressure_levels": sorted({row["memory_pressure_level"] for row in window})}
        return {"scope": "Current-build M5 fresh delivery and delivered-copy controls",
                "release_ready": False, "build": build, "hardware": delivery["hardware"], "delivery": delivery_gate,
                "delivered_copy_readiness": readiness, "readiness_seconds": observed["readiness_seconds"],
                "download_progress_counters": counters, "manifest": manifest, "server_log_intervals": timing,
                "functional_controls": sum(len(observed["results"][mode]) for mode in modes), "verified_steps": activation,
                "passive_host_samples": {"count": len(samples), "idle_min_percent": min(idle),
                                         "idle_max_percent": max(idle), "pressure_levels": levels,
                                         "startup_idle": startup_idle},
                "evidence_sha256": {"delivered_controls": sha256(observed_raw).hexdigest(),
                                    "prior_delivered_controls": sha256(prior_raw).hexdigest(),
                                    "thresholds": sha256(threshold_file.read_bytes()).hexdigest()}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive), indent=2, sort_keys=True, allow_nan=False))
