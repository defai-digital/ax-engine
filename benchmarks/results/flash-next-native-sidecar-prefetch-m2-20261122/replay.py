"""Verify the original M2 functional comparison and phase measurements."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
ARMS = ("default-mapped", "default-prefetch", "required-prefetch", "required-mapped")
CASES = ("count", "repeat", "extend", "json", "stream", "after_stream", "one_token")
BYTE_COUNT = 5_214_301_696
VERIFIED = "ax_engine_flash_next_mtp_verified_steps_total"
COUNTERS = (VERIFIED, "ax_engine_flash_next_mtp_attach_failed_total",
            "ax_engine_flash_next_mtp_prefill_absorb_failures_total", "ax_engine_flash_next_mtp_step_errors_total")
MANIFEST = {"model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"}


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def read_archive(path):
    contents = {}
    with tarfile.open(path) as archive:
        for member in archive:
            require(member.isfile() and member.name == Path(member.name).name
                    and member.name not in contents, "Unsafe or duplicate evidence member")
            with archive.extractfile(member) as source:
                contents[member.name] = source.read()
    return contents


def digest(data):
    return sha256(data).hexdigest()


def valid_seconds(value):
    return type(value) in (int, float) and isfinite(value) and value > 0


def signature(rows):
    output = []
    for case in rows:
        before, after = case["metrics_before"], case["metrics_after"]
        delta = {metric: after[metric] - before[metric] for metric in COUNTERS}
        require(case["activation_delta"] == delta and all(delta[m] == 0 for m in COUNTERS[1:]),
                "Route counters or successful activation changed")
        response = case["response"]
        usage = response["usage"]
        require(0 < usage["completion_tokens"] <= case["request"]["max_tokens"]
                and usage["prompt_tokens"] > 0, "Request usage does not match its budget")
        output.append((case["request"], response["choices"], delta,
                       usage["prompt_tokens"], usage["completion_tokens"]))
    return output


def replay(bundle=ROOT / "evidence.tar.xz"):
    data = read_archive(bundle)
    r, b, inventory, traces = (json.loads(data[name]) for name in (
        "receipt.json", "build.json", "inventory.json", "startup-traces.json"))
    inputs = {name: sha for sha, name in map(str.split, data["prepared.sha256"].decode().splitlines())}
    require(set(inputs) == {"run_prefetch.py", "check_flash_next_native_support.py",
                            "qualify_qwen38_flash_next.py", "inventory.json"}
            and all(digest(data[name]) == sha for name, sha in inputs.items())
            and b["driver_sha256"] == inputs["run_prefetch.py"]
            and b["runtime_diff_sha256"] == digest(data["candidate.patch"])
            and b["profile"] == "release-server" and b["mlx"] == b["mlx_metal"] == "0.32.4"
            and b["rustc"].startswith("rustc 1.99.1") and b["cargo"].startswith("cargo 1.99.1")
            and b["codesign_verified"] is True and r["build"] == b,
            "Build, driver or exact source changed")
    require(data["status.txt"] == b"0\n" and r["status"] == "complete" and r["qualification"] is False
            and len(inventory["members"]) == 49 and r["manifest"] == MANIFEST
            and r["pack_revision"] == inventory["revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and r["inventory_sha256"] == inputs["inventory.json"]
            and tuple(r["arms"]) == ARMS, "Model, arm order or completion receipt changed")
    result = {}
    outputs = {}
    for arm in ARMS:
        values = r["arms"][arm]
        trace = traces[arm]
        require(tuple(row["name"] for row in values["cases"]) == CASES
                and trace["resident"] is True and trace["prefetch_failed"] is False
                and {name: int(amount) for name, amount in trace["phases"].items()} == values["phases_us"]
                and [[int(size), float(elapsed)] for size, elapsed in trace["reads"]] == values["prefetch"],
                "Measured startup or the named controls changed")
        enabled = arm.endswith("-prefetch")
        prefetch = values["prefetch"]
        require((len(prefetch) == 1 and prefetch[0][0] == BYTE_COUNT
                 and valid_seconds(prefetch[0][1])) if enabled else prefetch == [],
                 "Unexpected sidecar pre-read geometry")
        phases = values["phases_us"]
        require(set(phases) == {"resident_tensors", "mtp_sidecar"} and all(valid_seconds(x) for x in phases.values())
                and valid_seconds(values["ready_seconds"]), "Invalid startup timing")
        rows = signature(values["cases"])
        steps = sum(row[2][VERIFIED] for row in rows)
        require(steps == values["verified_steps"] == (107 if arm.startswith("required") else 0),
                "Required verifier coverage or direct-mode policy changed")
        outputs[arm] = rows
        result[arm] = {"ready_seconds": values["ready_seconds"],
                       "sidecar_seconds": phases["mtp_sidecar"] / 1e6,
                       "trunk_seconds": phases["resident_tensors"] / 1e6,
                       "prefetch_seconds": prefetch[0][1] / 1000 if enabled else 0,
                       "verified_steps": steps}
    require(all(outputs[mode+"-mapped"] == outputs[mode+"-prefetch"] for mode in ("default", "required")),
            "Paired stable HTTP responses or activation differ")
    skip = r["required_skip"]
    require(skip["exit_code"] != 0 and valid_seconds(skip["seconds"])
            and skip["activation_error"] is traces["required_skip"]["activation_error"] is True
            and traces["required_skip"]["prefetch_completed"] is False,
            "Explicit skip no longer fails closed")
    return {"qualification": False, "completed_controls": 28, "matching_paired_controls": 14,
            "all_publisher_members_verified": 49, "build": b, "arms": result,
            "required_skip": skip, "hardware": r["hardware"]}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
