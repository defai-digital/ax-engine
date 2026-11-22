"""Verify the observed native M2 resident reader routing and HTTP outputs."""

from gzip import decompress
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
SCRIPTS = ROOT.parents[2] / "scripts"
PREVIOUS = RESULTS / "flash-next-aligned-native-http-m2-20261122/native-result.json"
INVENTORY = RESULTS / "qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json"
SELECTION = RESULTS / "flash-next-sidecar-reader-warm-m5-20261122/expected-resident-selection.json"
PIN = "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
MANIFEST = "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
ARMS = {
    "default-mapped": ("default", False, False, "Mapped"),
    "default-buffered": ("default", True, False, "AlignedBuffered"),
    "default-both": ("default", True, True, "AlignedNoCache"),
    "required-buffered": ("required", True, False, "AlignedBuffered"),
}

sys.path.insert(0, str(SCRIPTS))
import check_flash_next_native_support as native


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stable(row):
    response = row["response"]
    usage = response["usage"]
    return (
        row["name"], row["request"], response["choices"],
        tuple(usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")),
        row["activation_delta"],
    )


def replay(root=ROOT):
    build, trace, software = (read(root / name) for name in ("build.json", "resident-traces.json", "software.json"))
    result = json.loads(decompress((root / "native-result.json.gz").read_bytes()))
    inputs = {name: value for value, name in map(str.split, (root / "inputs.sha256").read_text().splitlines())}
    for staged, path in (
        ("../build.json", root / "build.json"), ("inventory.json", INVENTORY),
        ("prior-functional.json", PREVIOUS), ("run_buffered_smoke.py", root / "run_buffered_smoke.py"),
        ("check_flash_next_native_support.py", Path(native.__file__)),
        ("qualify_qwen38_flash_next.py", SCRIPTS / "qualify_qwen38_flash_next.py"),
    ):
        require(inputs[staged] == digest(path), f"Staged input changed for {staged}")
    require(build["profile"] == "release-server" and build["rustc"].startswith("rustc 1.99.1 ")
            and build["commit"].startswith("23c5c8cb8")
            and build["mlx"] == build["mlx_metal"] == "0.32.4"
            and build["inventory_sha256"] == inputs["inventory.json"]
            and build["server_sha256"] == inputs["../ax-engine-server"],
            "The measured release-server image or pack changed")
    require(build["post_link_strip"] == {
        "tool": "llvm-objcopy from Rust 1.99.1", "flag": "--strip-debug",
        "codesign_verified": True, "mach_o_text_unchanged": True,
    }, "Missing pinned post-link strip verification")
    require((root / "status.txt").read_text() == "0\n" and result["completed"] is True
            and result["scope"] == "M2 Ultra buffered and no-cache reader routing"
            and result["qualification"] is False and result["build"] == build
            and result["pack_revision"] == PIN and result["manifest"] == {"model-manifest.json": MANIFEST},
            "The native controls or publisher recheck did not complete")
    require(software == {
        "rust_passed": 4790, "rust_failed": 0, "rust_ignored": 107,
        "python_passed": 345, "python_skipped": 7, "workspace_status": 0,
    }, "The pinned software suite did not pass")

    prior, selection = read(PREVIOUS)["runs"], read(SELECTION)
    require(tuple(result["runs"]) == tuple(trace) == tuple(ARMS), "The observed process order changed")
    comparison = {}
    for label, (policy, buffered, nocache, reader) in ARMS.items():
        arm, observed = result["runs"][label], trace[label]
        require(arm["mode"] == policy and arm["requested_buffered"] is buffered
                and arm["requested_nocache"] is nocache and arm["reader"] == reader
                and observed["reader"] == reader and observed["aligned_nocache"] is nocache
                and observed["resident"] is True and observed["shards"] == selection,
                f"{label}: reader selection or page policy changed")
        rows = arm["rows"]
        require(arm["native_shards"] == 15 and arm["native_tensors"] == 1942
                and arm["pressure"] == ["1", "1"] and arm["matches"] == len(rows) == 7,
                f"{label}: resident coverage or HTTP control count changed")
        for row in rows:
            native.check_answer(row["name"], row["response"])
            steps = native.check_activation(row["metrics_before"], row["metrics_after"], policy == "required",
                row["response"]["usage"]["completion_tokens"], row["request"]["max_tokens"])
            require(steps == row["activation_delta"], f"{label}: activation counters changed")
        require(list(map(stable, rows)) == list(map(stable, prior[f"{policy}-mapped"]["rows"])),
                f"{label}: output, usage or activation counters differ from the frozen mapped control")
        verified = sum(row["activation_delta"][native.VERIFIED] for row in rows)
        require(arm["verified_steps"] == verified == (107 if policy == "required" else 0),
                f"{label}: unexpected MTP verifier activity")
        require(0 < arm["ready_seconds"] and isfinite(arm["ready_seconds"]),
                f"{label}: invalid readiness observation")
        comparison[label] = {field: arm[field] for field in ("reader", "matches", "verified_steps", "ready_seconds")}
    return {"scope": result["scope"], "qualification": False, "commit": build["commit"],
            "software": software, "arms": comparison}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
