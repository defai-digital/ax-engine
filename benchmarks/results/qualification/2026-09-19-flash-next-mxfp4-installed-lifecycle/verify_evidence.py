"""Reconstruct the published installed lifecycle; no model or release qualification."""
import gzip
import hashlib
import json
from pathlib import Path

import lifecycle_checks

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load():
    summary = json.loads((ROOT / "summary.json").read_text())
    for name, row in summary["artifacts"].items():
        require(Path(name).name == name, "Invalid artifact name")
        raw = (ROOT / name).read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                "Changed evidence artifact")
    raw = (ROOT / "lifecycle.json.gz").read_bytes()
    require(raw[4:8] == bytes(4), "Nondeterministic gzip timestamp")
    return summary, json.loads(gzip.decompress(raw))


def verify(summary, evidence):
    require(summary["source_commit"] == evidence["source_commit"] == lifecycle_checks.SOURCE,
            "Wrong source identity")
    require(summary["wheel_sha256"] == evidence["wheel_sha256"] == lifecycle_checks.WHEEL,
            "Wrong wheel identity")
    for record in (summary, evidence, evidence["root_review"]):
        require(record["qualification"] is False and record["release_ready"] is False,
                "Lifecycle evidence cannot promote a release")
    gates = {name: "not_assessed" for name in ("MTP-S", "MTP-P", "MTP-D")}
    require(summary["mtp_certification"] == gates, "Unjustified MTP certification")
    rows = evidence["records"]
    computed = lifecycle_checks.terminal(rows["output/result.json"], rows["contract.json"],
                                        rows["launch.json"], rows["transfer-receipt.json"])
    require(computed == evidence["root_review"], "Published verdict differs from raw reconstruction")
    require(summary["requests_validated"] == computed["requests_validated"] == 14
            and summary["payload_prepost_files"] == computed["payload_prepost_files"] == 47
            and summary["installed_runtime_files"] == 52, "Changed evidence scope")
    return dict(installed_lifecycle_passed=True, requests_validated=14,
                mtp_certification=gates, qualification=False, release_ready=False)


if __name__ == "__main__":
    print(json.dumps(verify(*load()), indent=2))
