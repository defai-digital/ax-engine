"""Replay the follow-up's exact native matrix and explicit mapped reader control."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
FROZEN = RESULTS / "flash-next-auto-reader-fresh-m5-20261122" / "evidence.tar.xz"
HTTP = RESULTS / "flash-next-probe-followup-http-m5-20261122" / "evidence.tar.xz"
QA = RESULTS / "flash-next-probe-followup-qa-m5-20261122" / "evidence.tar.xz"
LIFECYCLE = RESULTS / "flash-next-probe-followup-lifecycle-m5-20261122" / "lifecycle.json"
shared = runpy.run_path(str(RESULTS / "flash-next-auto-reader-board-m5-20261122" / "replay.py"))
member, require, signature, gates = (shared[key] for key in ("member", "require", "signature", "gates"))
from check_flash_next_native_support import CASES, check_activation, check_answer, is_target_hardware


def stable_usage(response):
    usage = dict(response["usage"])
    if "ax_engine" in usage:
        usage["ax_engine"] = {key: value for key, value in usage["ax_engine"].items()
                              if key not in ("prefill_time_us", "decode_time_us")}
    return usage


def replay(evidence):
    with (tarfile.open(FROZEN) as frozen, tarfile.open(HTTP) as http,
          tarfile.open(QA) as qa, tarfile.open(evidence) as archive):
        build_bytes = member(http, "build.json")
        build = json.loads(build_bytes)
        native_http = member(http, "native-support/result.json")
        native = json.loads(member(archive, "observed/comparison.json"))
        retained = json.loads(member(archive, "retained-result.json"))
        matrix_raw = member(archive, "observed/generate-matrix/result.json")
        repeats_raw = member(archive, "observed/generate-matrix/raw-runs.json")
        matrix, repeat = json.loads(matrix_raw), json.loads(repeats_raw)
        require(build["codesign_verified"] is True and member(archive, "build.json") == build_bytes
                and member(archive, "status.txt") == b"0\n"
                and member(qa, "build.json") == build_bytes
                and native["build"] == build and retained["native"] == native
                and native["qualification"] is False and retained["qualification"] is False,
                "Retained runs do not bind the completed signed follow-up")
        require(native["observed_sha256"] == {
            key: sha256(body).hexdigest() for key, body in (
                ("native-support", native_http), ("qa", member(qa, "result.json")),
                ("lifecycle", LIFECYCLE.read_bytes()), ("generate-matrix", matrix_raw))},
            "Raw follow-up receipt changed")
        require(matrix["server_sha256"] == build["server_sha256"]
                and matrix["inventory_sha256"] == native["inventory_sha256"]
                and is_target_hardware(matrix["hardware"]), "Native timing run used a different target")
        oracle_raw = member(frozen, "driver/signed-native.json")
        require(sha256(oracle_raw).hexdigest() == native["reference_control_sha256"],
                "The signed output oracle changed")
        oracle = json.loads(oracle_raw)["matrix"]
        require(repeat.keys() == oracle.keys() and all(len(runs) == 2 and
                all(signature(run) == oracle[mode] for run in runs) for mode, runs in repeat.items())
                and not gates.matrix_repeat_problems(matrix, repeat)
                and native["matched_independent_matrix_arrays"] == 124,
                "An independently repeated output or selected process differs")
        log = member(archive, "opt-out/server-default.log")
        require(retained["automatic_override"] == "absent" and retained["opt_out_override"] == "0"
                and sha256(log).hexdigest() == retained["opt_out_log_sha256"]
                and b"Flash Next MTP sidecar prefetch" not in log
                and b"preview listening bind_address=" in log,
                "The explicit opt-out did not finish through the mapped reader")
        original = json.loads(native_http)["results"]["default"]
        require(len(CASES) == len(original) == retained["opt_out_matched_cases"], "Opt-out roster changed")
        for (case, _, _), before in zip(CASES, original, strict=True):
            row = json.loads(member(archive, f"opt-out/default-{case}.json"))
            result = row["response"]
            check_answer(case, result)
            steps = check_activation(row["metrics_before"], row["metrics_after"], False,
                                     result["usage"]["completion_tokens"], row["request"]["max_tokens"])
            require(row["name"] == before["name"] == case and row["request"] == before["request"]
                    and result["choices"] == before["response"]["choices"]
                    and stable_usage(result) == stable_usage(before["response"])
                    and steps == row["activation_delta"] == before["activation_delta"],
                    "The mapped response changed")

    cpu = [matrix["host_quiet"]["cpu_idle_percent"], matrix["host_idle_after"]]
    thresholds = json.loads(gates.THRESHOLDS.read_bytes())
    require(native["quiet_samples"]["generate-matrix"] == cpu, "Matrix host samples changed")
    return {
        "source_commit": build["commit"], "qualification": False, "matched_output_arrays": 124,
        "matrix_cpu_idle_percent": cpu,
        "matrix_timing_eligible": all(thresholds["timing_host"]["min_cpu_idle_percent"] <= n <= 100 for n in cpu),
        "matrix_readiness_seconds": {mode: matrix["runs"][mode]["ready_seconds"] for mode in gates.AX_ARMS},
        "opt_out_matched_cases": retained["opt_out_matched_cases"],
        "opt_out_readiness_seconds": retained["opt_out_readiness_seconds"],
        "opt_out_cpu_idle_percent": retained["opt_out_cpu_idle_percent"],
        "opt_out_pressure": retained["opt_out_pressure"],
        "raw_repeats_sha256": sha256(repeats_raw).hexdigest(),
    }


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = cli.parse_args()
    print(json.dumps(replay(args.archive), indent=2, sort_keys=True, allow_nan=False))
