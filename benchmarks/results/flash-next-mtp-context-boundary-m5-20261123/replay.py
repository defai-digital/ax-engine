"""Replay both hardware runs of the Flash Next 32K same-state audit."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def differences(left, right):
    return [index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]


def replay(raw, repo):
    results = repo / "benchmarks/results"
    m2 = results / "flash-next-mtp-context-boundary-m2-20261123"
    prior_m5 = results / "flash-next-expanded-mtp-audit-m5-20261122"
    verifier = runpy.run_path(str(m2 / "replay.py"))
    require = verifier["require"]
    provenance = read(m2 / "cases-provenance.json")
    verified_m2 = verifier["verify"](m2, repo, provenance)
    require(verified_m2 == read(m2 / "comparison.json"), "The M2 comparison changed")

    m5_receipt = read(raw / "receipt.json")
    original_m5 = read(prior_m5 / "receipt.json")
    m2_receipt = read(m2 / "receipt.json")
    require((raw / "status.txt").read_bytes() == b"0\n"
            and m5_receipt["status"] == "complete" and m5_receipt["test_exit"] == 0
            and m5_receipt["qualification"] is False and m5_receipt["mtp_cost_policy_disabled"] is True,
            "The M5 owned model test did not finish")
    for key in ("build", "test_sha256", "pack_revision", "inventory_sha256", "native_manifest"):
        require(m5_receipt[key] == original_m5[key] == m2_receipt[key],
                f"The M5 test changed its signed input: {key}")
    require(m5_receipt["hardware"] == original_m5["hardware"]
            and m5_receipt["pressure_samples"] and set(m5_receipt["pressure_samples"]) == {1},
            "The M5 hardware or memory-pressure envelope changed")

    report = gzip.decompress((raw / "report.json.gz").read_bytes())
    inputs = verifier["workloads"](repo, provenance)
    require(sha256(report).hexdigest() == m5_receipt["report_sha256"]
            and sha256(inputs).hexdigest() == m5_receipt["cases_sha256"] == m2_receipt["cases_sha256"],
            "M5 prompt or terminal report changed")
    m5_raw = json.loads(report)
    m2_raw = json.loads(gzip.decompress((m2 / "report.json.gz").read_bytes()))
    previous = results / "flash-next-expanded-mtp-audit-m2-20261122" / "replay.py"
    checked = runpy.run_path(str(previous))["replay"](m5_raw, json.loads(inputs), provenance)
    for proof, here, other in zip(checked, m5_raw["cases"], m2_raw["cases"], strict=True):
        require(proof["verified_output_positions"] == proof["budget"] - 1 and proof["accepted_drafts"] > 0,
                "The M5 audit did not exercise every position after the first token")
        require(here["name"] == other["name"] == proof["name"], "The cross-device cases differ")
        proof["direct_differences_between_devices"] = differences(here["direct_tokens"], other["direct_tokens"])
        proof["mtp_differences_between_devices"] = differences(here["mtp_tokens"], other["mtp_tokens"])
    return {"qualification": False, "build": m5_receipt["build"]["commit"],
            "hardware": m5_receipt["hardware"], "pack_revision": m5_receipt["pack_revision"],
            "pressure_samples": len(m5_receipt["pressure_samples"]), "m5_report_sha256": m5_receipt["report_sha256"],
            "m2_report_sha256": verified_m2["report_sha256"], "cases": checked}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--raw", type=Path, default=ROOT)
    cli.add_argument("--repo", type=Path, default=ROOT.parents[2])
    args = cli.parse_args()
    print(json.dumps(replay(args.raw, args.repo), indent=2, sort_keys=True, allow_nan=False))
