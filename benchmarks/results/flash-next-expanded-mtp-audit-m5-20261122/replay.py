"""Reconcile the M5 long-workload verifier decisions with their frozen inputs."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
M2 = ROOT.parent / "flash-next-expanded-mtp-audit-m2-20261122"


def main():
    shared = runpy.run_path(str(M2 / "replay.py"))
    require = shared["require"]
    provenance = json.loads((M2 / "cases-provenance.json").read_bytes())
    prompts = shared["bound_prompts"](provenance)
    run = json.loads((ROOT / "receipt.json").read_bytes())
    raw = gzip.decompress((ROOT / "report.json.gz").read_bytes())
    build = run["build"]
    m2_run = json.loads((M2 / "receipt.json").read_bytes())
    require(run["status"] == "complete" and run["test_exit"] == 0
            and run["qualification"] is False and run["mtp_cost_policy_disabled"] is True
            and (ROOT / "status.txt").read_text() == "0\n", "M5 runner audit did not complete")
    require(build == m2_run["build"]
            and run["test_sha256"] == build["test_sha256"], "Compiled audit changed across hosts")
    require(run["hardware"]["soc"] == "Apple M5 Max"
            and run["hardware"]["memory_bytes"] == 128 << 30
            and run["hardware"]["form_factor"] == "MacBook Pro", "Wrong M5 reference hardware")
    require(run["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
            and run["inventory_sha256"] == sha256(shared["INVENTORY"].read_bytes()).hexdigest()
            and run["native_manifest"] == {
                "model-manifest.json": "62f61c17cf571afc24dfb3002afd7c3c9b038b0eb80cde182f143266954e61ef"
            }, "Publisher pack or delivered manifest changed")
    require(sha256(raw).hexdigest() == run["report_sha256"]
            and sha256(prompts).hexdigest() == run["cases_sha256"] == provenance["cases_sha256"],
            "Raw decisions or prompt inputs changed")
    require(run["pressure_samples"] and all(level == 1 for level in run["pressure_samples"]),
            "Missing or elevated memory pressure readings")
    cases = json.loads(prompts)
    m5 = json.loads(raw)
    m2_raw = gzip.decompress((M2 / "report.json.gz").read_bytes())
    require(sha256(m2_raw).hexdigest() == m2_run["report_sha256"], "M2 verifier evidence changed")
    m2 = json.loads(m2_raw)
    summary = shared["replay"](m5, cases, provenance)
    require(summary == json.loads((ROOT / "comparison.json").read_bytes()), "M5 verifier replay changed")
    cross_chip = []
    for a, b in zip(m5["cases"], m2["cases"], strict=True):
        require(a["name"] == b["name"], "Cross-chip cases differ")
        changed = [idx for idx, (left, right) in enumerate(zip(a["direct_tokens"], b["direct_tokens"],
                                                             strict=True)) if left != right]
        cross_chip.append({"name": a["name"], "changed_direct_positions": changed})
    print(json.dumps({"m5_decisions": summary, "cross_chip": cross_chip}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
