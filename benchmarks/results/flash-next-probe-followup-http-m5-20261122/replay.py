"""Replay the signed follow-up's retained HTTP and supported-tier controls."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "flash-next-auto-reader-fresh-m5-20261122" / "evidence.tar.xz"
BASE_MANIFEST_HASH = "aa11ae0fe99d3c0667123373bea2c98721447f610e2206c8c6130620014148c0"
BUILD_HASH = "3091f8f2f78087bb8ba34e0c156fcb49b9058eee3aa0884cbb106a6344eda2d2"
COMMIT = "26c0ccefe6f35072a576c368aaceb37c1aa10ae0"
MODES = ("disabled", "required", "default", "required-kill-switch")
PLATFORMS = (
    ("m4-pro", "Apple M4 Pro", 64),
    ("m4-macos26", "Apple M4", 16),
)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def member(archive, name):
    row = archive.getmember(name)
    ensure(row.isfile(), f"Expected a regular evidence member: {name}")
    with archive.extractfile(row) as stream:
        return stream.read()


def pins(raw):
    return {name.removeprefix("./"): digest for digest, name in map(str.split, raw.decode().splitlines())}


def platform_controls(archive):
    result = {}
    for label, expected_soc, memory in PLATFORMS:
        stem = f"platform/{label}"
        log = member(archive, stem + ".log").decode()
        ensure(member(archive, stem + ".status") == b"0\n" and log.startswith("sidecar-tests: OK\n"),
               "Focused test run or signed executable verification failed")
        header = re.search(r"^(Apple M4(?: Pro)?), ([0-9]+) GiB, macOS ([0-9.]+),", log, re.M)
        ensure(header is not None and header[1] == expected_soc and int(header[2]) == memory
               and (label != "m4-macos26" or header[3].startswith("26.")),
               "Focused test ran on the wrong supported tier")
        tests = re.findall(r"^test result: ok\. (\d+) passed; 0 failed;", log, re.M)
        ensure(tests == ["23", "1"], "The two focused suites did not both pass")
        digest = pins(member(archive, stem + ".sha256"))["sidecar-tests"]
        result[label] = {"soc": expected_soc, "memory_gib": memory, "os": header[3],
                         "focused_tests": sum(map(int, tests)), "test_binary_sha256": digest}
    ensure(len({row["test_binary_sha256"] for row in result.values()}) == 1,
           "Supported-tier test executable differs")
    return result


def replay(evidence):
    expected = {"build.json", "harness-inputs.sha256", "explicit-skip.json", "native-support/result.json"}
    expected.update(f"native-support/server-{mode}.log" for mode in MODES)
    expected.update(f"platform/{label}.{kind}"
                    for label, _, _ in PLATFORMS for kind in ("log", "status", "sha256"))
    with (tarfile.open(BASE) as base, tarfile.open(evidence) as candidate,
          TemporaryDirectory(prefix="ax-flash-followup-http-") as directory):
        ensure(sorted(candidate.getnames()) == sorted(expected), "Unexpected follow-up evidence roster")
        manifest = member(base, "raw/harness-inputs.sha256")
        ensure(sha256(manifest).hexdigest() == BASE_MANIFEST_HASH, "The frozen response verifier changed")
        staging = Path(directory)
        source = pins(manifest)
        inputs = {"compare_delivery.py", "stock-native.json",
                  *(name for name in source if name.startswith("scripts/"))}
        staged_pins = pins(member(candidate, "harness-inputs.sha256"))
        for name in inputs:
            rel = PurePosixPath(name)
            ensure(not rel.is_absolute() and ".." not in rel.parts, "Invalid harness member path")
            raw = member(base, "driver/" + name)
            ensure(sha256(raw).hexdigest() == source[name] == staged_pins[name],
                   "Follow-up used different gate or oracle input")
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        sys.path.insert(0, str(staging))
        from compare_delivery import native, verify_controls, verify_route

        build_bytes = member(candidate, "build.json")
        build = json.loads(build_bytes)
        ensure(sha256(build_bytes).hexdigest() == BUILD_HASH and build["commit"] == COMMIT
               and build["codesign_verified"] is True, "Follow-up release build changed")
        controls = member(candidate, "native-support/result.json")
        result = json.loads(controls)
        ensure(result["server_sha256"] == build["server_sha256"] and native.is_target_hardware(result["hardware"]),
               "Retained HTTP evidence used a different runtime or host")
        steps = verify_controls(result, json.loads((staging / "stock-native.json").read_bytes()))
        ensure(steps == {"disabled": 0, "required": 107, "default": 0}, "Unexpected MTP activation")
        routes = staging / "logs"
        routes.mkdir()
        for mode in MODES:
            name = f"server-{mode}.log"
            (routes / name).write_bytes(member(candidate, "native-support/" + name))
        prefetch = verify_route(routes, True)
        skip = json.loads(member(candidate, "explicit-skip.json"))
        ensure(skip["completed"] is True and skip["code"] == 1 and skip["payload_markers"]
               and set(skip["payload_markers"].values()) == {0},
               "Explicit Required-sidecar bypass entered payload startup")
        return {"commit": COMMIT, "qualification": False, "verified_http_cases": 21,
                "readiness_seconds": result["readiness_seconds"], "prefetch_seconds": prefetch,
                "verifier_steps": steps, "explicit_skip_before_payload": True,
                "native_receipt_sha256": sha256(controls).hexdigest(),
                "supported_tier_checks": platform_controls(candidate)}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence.tar.xz")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive), indent=2, sort_keys=True, allow_nan=False))
