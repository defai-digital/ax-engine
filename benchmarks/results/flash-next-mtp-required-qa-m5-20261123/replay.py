"""Verify the M5 Required-policy cohort against its frozen signed inputs."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
M2 = HERE.parent / "flash-next-mtp-required-qa-m2-20261123"


def read(root, name):
    return (root / name).read_bytes()


def cases(output):
    return {row["name"]: row for line in output.splitlines() if (row := json.loads(line))["kind"] == "case"}


def replay():
    check = runpy.run_path(str(M2 / "replay.py"))["check"]
    inputs = dict(
        (name, digest) for digest, name in map(str.split, read(HERE, "inputs.sha256").decode().splitlines())
    )
    local = read(HERE, "build.json")
    build = json.loads(local)
    assert local == read(M2, "build.json") and build["codesign_verified"]
    assert inputs["mtp-audit"] == build["server_test_sha256"]
    for name in ("required-qa-inputs-receipt.json", "mxfp4-inventory.json"):
        assert sha256(read(M2, name)).hexdigest() == inputs[name]
    assert sha256(local).hexdigest() == inputs["build.json"]
    patch = gzip.decompress(read(M2, "source.patch.gz"))
    assert sha256(patch).hexdigest() == inputs["source.patch"] == build["patch_sha256"]
    cohort = gzip.decompress(read(M2, "required-qa-inputs.json.gz"))
    receipt = json.loads(read(M2, "required-qa-inputs-receipt.json"))
    assert sha256(cohort).hexdigest() == inputs["required-qa-inputs.json"] == receipt["audit_input_sha256"]
    output = gzip.decompress(read(HERE, "required-qa-output.jsonl.gz"))
    assert sha256(output).hexdigest() == read(HERE, "output.sha256").decode().split()[0]
    assert read(HERE, "status.txt").strip() == b"0"
    assert read(HERE, "pack-before.json") == read(HERE, "pack-after.json")
    pack = json.loads(read(HERE, "pack-before.json"))
    assert pack["inventory_sha256"] == inputs["mxfp4-inventory.json"]
    assert pack["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert pack["verified_members"] == 49
    samples = [json.loads(line) for line in read(HERE, "pressure.jsonl").splitlines()]
    assert samples and all(row["pressure"] == 1 for row in samples)

    current, earlier = cases(output), cases(gzip.decompress(read(M2, "required-qa-output.jsonl.gz")))
    assert current.keys() == earlier.keys()
    audit = check(cohort, output)
    assert audit["cases"] == audit["same_direct_output"] == audit["both_eos"] == 105
    assert audit["all_post_first_reconciled"] == audit["mtp_decisions_observed"] == 105
    changed = {
        key: [name for name, row in current.items() if row[key] != earlier[name][key]]
        for key in ("decisions", "direct_tokens", "mtp_tokens")
    }
    long = current["long_context_record_mass"]
    return {
        "audit": audit,
        "cross_hardware_changed_cases": changed,
        "long_lookup": {key: long[key] for key in ("input_tokens", "accepted", "verified_steps")},
        "normal_pressure_samples": len(samples),
        "peak_sampled_rss_kib": max(row["rss_kib"] for row in samples),
        "qualification": False,
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, sort_keys=True))
