"""Replay fixed-context buffered and mapped Flash Next starts against the signed oracle."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from runpy import run_path
from statistics import median
import sys
import tarfile
from tempfile import TemporaryDirectory

ARMS = ("mapped-first", "buffered-first", "buffered-second", "mapped-second")
OBSERVED = ARMS[:3]
ARM_MODE = {arm: arm.split("-")[0] for arm in ARMS}
READER = {"mapped": "Mapped", "buffered": "AlignedBuffered"}
SAME_HOST = "flash-next-main-lifecycle-m5-20261123"
ORACLE = "flash-next-aligned-native-default-m5-20261122"
ORACLE_SHA = "28767a98ea7a13989905c7d4c94719937e3eeb8af8f6a29419548884f4316d03"
STABLE_FIELDS = ("prompt_hash", "prompt_tokens", "trial", "phase", "output_tokens",
                 "finish_reason", "status", "verified_steps", "route_errors")
INPUT_BINARIES = {"ax-engine-server", "loader_footprint.dylib"}
INPUT_FILES = {
    "build.json", "mxfp4-inventory.json", "source.patch", "check_flash_next_native_support.py",
    "qualify_qwen38_flash_next.py", "run_cache_policy.py", "run_load_order.py",
    "run_guarded.sh", "run_owner.sh", "loader_footprint.c", "model-manifest.json",
    "expected-resident-selection.json", "resident-page-selection.json",
    "scripts/check_flash_next_native_support.py", "scripts/qualify_qwen38_flash_next.py",
    "scripts/run_flash_next_generate_matrix.py", "scripts/watch_flash_next_reference.py",
}
PUBLISHER_PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
DERIVED_MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stable_rows(arm):
    return [tuple(row[name] for name in STABLE_FIELDS) for row in arm["cases"]]


def older_outputs(results_root):
    archive_path = results_root / ORACLE / "evidence.tar.xz"
    need(digest(archive_path) == ORACLE_SHA, "Earlier signed response oracle changed")
    with tarfile.open(archive_path) as archive:
        with archive.extractfile("default/receipt.json") as stream:
            receipt = json.load(stream)
    return stable_rows(receipt["arms"]["mmap-short"])


def replay_one(capture, record, arm, tools, build, inputs, target, expected_rows):
    mode = ARM_MODE[arm]
    need(record["status"] == "complete" and set(record["arms"]) == {arm}
         and record["build"] == build and record["macos_version"] == target["os"]
         and record["launch"] == "default-32k-resident-single-request"
         and record["policy"] == "default"
         and record["pack_revision"] == PUBLISHER_PIN
         and record["inventory_sha256"] == inputs["mxfp4-inventory.json"]
         and record["model_manifest_sha256"] == inputs["model-manifest.json"]
         and record["runner_sha256"] == inputs["run_cache_policy.py"]
         and record["footprint_probe_sha256"] == inputs["loader_footprint.dylib"]
         and record["footprint_probe_source_sha256"] == inputs["loader_footprint.c"]
         and record["resident_page_selection_sha256"] == inputs["resident-page-selection.json"]
         and record["libmlx_sha256"] == build["libmlx_sha256"],
         f"The selected arm is missing or has changed staged inputs: {arm}")
    row = record["arms"][arm]
    need(row["loader_mode"] == mode and row["resident_reader"] == READER[mode]
         and stable_rows(row) == expected_rows
         and row["shard_count"] == 15 and row["tensor_count"] == 1942
         and row["attached_sidecar_depth"] == 1,
         f"Canonical selection or M5 oracle disagrees in {arm}")
    log = (capture / f"{arm}.log").read_text()
    traced = {}
    tools["attach_loader_trace"](log, mode, "default", traced)
    need(all(row[key] == value for key, value in traced.items())
         and "Flash Next MTP sidecar prefetch completed" not in log
         and float(row["ready_seconds"]) >= row["load_phase_seconds"] > 0
         and isfinite(row["ready_seconds"]), f"Inconsistent startup trace in {arm}")
    samples = [json.loads(line) for line in (capture / f"{arm}-samples.jsonl").read_text().splitlines()]
    peak = max((point.get("reference_rss_bytes", 0) for point in samples), default=0)
    idle = [row["quiet_baseline"]["cpu_idle_percent"], row["post_run_cpu_idle_percent"]]
    need(samples and {point["pressure"] for point in samples} == {1}
         and row["pressure_levels"] == [1] and peak == row["peak_sampled_rss_bytes"] > 0
         and all(0 <= sample <= 100 for sample in idle),
         f"Host observations changed in {arm}")
    return row, {"pressure_samples": len(samples),
                 "max_sampled_server_rss_gib": round(peak / 1024**3, 3),
                 "cpu_idle_percent": idle,
                 "same_host_oracle_tokens": sum(len(case["output_tokens"]) for case in row["cases"])}


def warm_summary(row, driver):
    lengths = driver["TIMED_LENGTHS"]
    matrix = driver["matrix"]
    warm = {}
    for length in lengths:
        samples = [case["timing"] for case in row["cases"]
                   if case["phase"] == "warm" and case["prompt_tokens"] == length]
        need(len(samples) == 3, "Incomplete warm timing distribution")
        warm[length] = {"ttft_seconds": median(item["ttft_seconds"] for item in samples),
                        "decode_tokens_per_second": median(matrix.decode_rate(item) for item in samples)}
    return warm


def ratios(before, after):
    return {length: {name: after[length][name] / old for name, old in values.items()}
            for length, values in before.items()}


def compare(capture, results_root):
    build = read(capture / "build.json")
    owner = read(capture / "capture.json")
    target = read(results_root / SAME_HOST / "lifecycle.json")["hardware"]
    inputs_rows = (capture / "inputs.sha256").read_text().splitlines()
    inputs = dict(line.split()[::-1] for line in inputs_rows)
    need(len(inputs_rows) == len(inputs) and set(inputs) == INPUT_FILES | INPUT_BINARIES
         and all(digest(capture / name) == inputs[name] for name in INPUT_FILES),
         "A staged runner or pack input has changed")
    need(owner["initial_owner_status"] == 1 and tuple(owner["order"]) == OBSERVED
         and owner["retained_original_files_verified"] == 9
         and owner["codesign_verified"] is build["codesign_verified"] is True
         and owner["server_sha256"] == build["server_sha256"] == inputs["ax-engine-server"]
         and owner["form_factor"] == target["form_factor"] and owner["soc"] == target["soc"]
         and owner["memory_gib"] * 1024**3 == target["memory_bytes"] and owner["os"] == target["os"]
         and build["profile"] == "release-server"
         and build["cargo"].startswith("cargo 1.99.1 ") and build["rustc"].startswith("rustc 1.99.1 ")
         and build["mlx"] == build["mlx_metal"] == "0.32.4"
         and build["patch_sha256"] == inputs["source.patch"],
         "Candidate executable, host or exact source patch changed")
    inventory = read(capture / "mxfp4-inventory.json")
    need(build["inventory_sha256"] == inputs["mxfp4-inventory.json"]
         and inputs["model-manifest.json"] == DERIVED_MANIFEST
         and len(inventory["members"]) == 49 and inventory["revision"] == PUBLISHER_PIN,
         "Publisher manifest or inventory changed")
    sys.path.insert(0, str(capture))
    try:
        tools = run_path(str(capture / "run_cache_policy.py"))
        need(tuple(label for label, _, _ in tools["ARMS"]) == ARMS
             and all(length == 512 for _, _, length in tools["ARMS"]),
             "Runner no longer uses one fixed prompt geometry")
        expected_rows = older_outputs(results_root)
        summaries = {}
        for arm in OBSERVED:
            folder = capture / "arms" / arm
            need(owner["prior_reference_admission"][arm] ==
                 {"pressure": 1, "eligible_for_attempt": False,
                  "reason": "background-rss-exceeds-signed-margin"},
                 f"The independent reference had priority before {arm}")
            row, host = replay_one(
                folder, read(folder / "receipt.json"), arm, tools, build, inputs, target, expected_rows
            )
            summaries[arm] = {**host, "ready_seconds": row["ready_seconds"],
                              "resident_seconds": row["load_phase_seconds"], "warmed": warm_summary(row, tools)}
    finally:
        sys.path.pop(0)
    mapped, buffered, repeat = (summaries[arm] for arm in OBSERVED)
    relative = ratios(mapped["warmed"], buffered["warmed"])
    idle = all(sample >= 90 for arm in (mapped, buffered) for sample in arm["cpu_idle_percent"])
    pass_steady = idle and all(row["ttft_seconds"] <= 1.03
                               and row["decode_tokens_per_second"] >= 0.97 for row in relative.values())
    return {"observed_arms": summaries, "source_base": build["source_base"],
            "server_sha256": build["server_sha256"], "source_patch_sha256": build["patch_sha256"],
            "first_pair_eligible": idle, "first_pair_warmed_band_passed": pass_steady,
            "first_pair_readiness_reduction_seconds": round(mapped["ready_seconds"] - buffered["ready_seconds"], 3),
            "buffered_vs_mapped_first_warmed_ratios": relative,
            "buffered_second_vs_first_warmed_ratios": ratios(buffered["warmed"], repeat["warmed"]),
            "missing_arm": "mapped-second", "release_ready": False}


def verify_m4(root, m5):
    record = read(root / "m4-controls.json")
    raw = root / "m4-controls.log"
    required = {
        "io::tests::aligned_loader_native_read_rejects_short_and_unrepresentable_ranges",
        "io::tests::aligned_filtered_loader_preserves_unaligned_weights_and_owned_arrays",
        "weights::qwen4_exp::tests::aligned_reader_requires_a_requested_resident_mxfp4_trunk",
        "weights::qwen4_exp::tests::aligned_reader_keeps_the_mapped_resident_tensor_selection",
    }
    names = {"engine-mlx-tests", "mlx-sys-tests", "mlx/libmlx.dylib", "mlx/libjaccl.dylib", "mlx/mlx.metallib"}
    hashes = record["input_sha256"]
    log = raw.read_text()
    need(record["source_base"] == m5["source_base"]
         and record["source_patch_sha256"] == m5["source_patch_sha256"]
         and record["m5_server_sha256"] == m5["server_sha256"]
         and record["rustc"].startswith("rustc 1.99.1 ")
         and record["mlx"] == record["mlx_metal"] == "0.32.4"
         and record["hardware"] == "Mac mini M4 Pro 64 GiB" and record["os"] == "27.0"
         and record["native_exit"] == 0 and record["test_executable_signature_verified"] is True
         and record["result_sha256"] == digest(raw)
         and set(hashes) == names and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())
         and all(f"{name}: OK" in log.splitlines() for name in names)
         and set(re.findall(r"^test (\S+) \.\.\. ok$", log, re.MULTILINE)) == required
         and len(re.findall(r"^test result: ok\. 2 passed; 0 failed; 0 ignored;", log, re.MULTILINE)) == 2,
         "Supported-tier compatibility evidence changed")
    return {"hardware": record["hardware"], "os": record["os"], "controls_passed": len(required),
            "same_candidate_source": True, "scope": "Focused native loader and reader routing"}


def replay():
    root = Path(__file__).resolve().parent
    files = {*INPUT_FILES, "capture.json", "inputs.sha256"}
    for arm in OBSERVED:
        files.update(f"arms/{arm}/{name}" for name in ("receipt.json", f"{arm}.log", f"{arm}-samples.jsonl"))
    with TemporaryDirectory(prefix="flash-next-buffered-three-") as stage:
        path = Path(stage)
        with tarfile.open(root / "evidence.tar.xz") as archive:
            entries = archive.getmembers()
            need(len(entries) == len(files) and {entry.name for entry in entries} == files
                 and all(entry.isfile() for entry in entries), "Unexpected staged archive roster")
            for entry in entries:
                target = path / entry.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as stream:
                    target.write_bytes(stream.read())
        m5 = compare(path, root.parent)
        build = read(path / "build.json")
        need(read(root / "m4-controls.json")["input_sha256"]["mlx/libmlx.dylib"] == build["libmlx_sha256"],
             "M4 and M5 used different MLX shared libraries")
        return {**m5, "supported_m4": verify_m4(root, m5)}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
