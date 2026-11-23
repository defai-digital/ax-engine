"""Verify the same-state Flash Next MTP audit at long-context boundaries."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = "flash-next-expanded-mtp-audit-m2-20261122"
EXPECTED_INPUTS = [("long_context_record_mass", 29_774, 64), ("matrix-32640-1", 32_640, 128)]


def digest(data):
    return sha256(data).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def workloads(repo, provenance):
    sys.path.insert(0, str(repo / "scripts"))
    import run_flash_next_generate_matrix as matrix

    qa = repo / ("benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa/"
                 "full-qa-inputs.json")
    cases_bytes = qa.read_bytes()
    vocabulary = provenance["matrix_vocab_size"]
    require(digest(cases_bytes) == provenance["qa_source_sha256"]
            and digest(Path(matrix.__file__).read_bytes()) == provenance["matrix_source_sha256"]
            and vocabulary == 248_320 and matrix.WORKLOAD_SEED == provenance["matrix_workload_seed"],
            "Frozen long prompts or workload generator changed")
    rows = json.loads(cases_bytes)["prepared"]
    ids = next(row["input_ids"] for row in rows if row["id"] == EXPECTED_INPUTS[0][0])
    prompts = (ids, matrix.workload_ids(vocabulary, EXPECTED_INPUTS[1][1], 1))
    cases = []
    for plan, expected, tokens in zip(provenance["cases"], EXPECTED_INPUTS, prompts, strict=True):
        name, length, budget = expected
        require(plan == {"name": name, "budget": budget, "input_tokens": length,
                         "prompt_sha256": matrix.ids_sha(tokens)} and len(tokens) == length,
                f"Prompt geometry changed for {name}")
        cases.append({"name": name, "prompt": tokens, "budget": budget})
    require(sum(EXPECTED_INPUTS[-1][1:]) == 32_768, "Missing the full context boundary")
    return (json.dumps(cases, separators=(",", ":")) + "\n").encode()


def verify(raw, repo, provenance):
    prior = repo / "benchmarks/results" / PREVIOUS
    baseline = json.loads((prior / "receipt.json").read_bytes())
    independent = runpy.run_path(str(prior / "replay.py"))["replay"]
    receipt = json.loads((raw / "receipt.json").read_bytes())
    report = gzip.decompress((raw / "report.json.gz").read_bytes())
    cases = workloads(repo, provenance)
    require((raw / "status.txt").read_bytes() == b"0\n"
            and receipt["status"] == "complete" and receipt["qualification"] is False
            and receipt["test_exit"] == 0 and receipt["mtp_cost_policy_disabled"] is True,
            "The owned long-context audit did not complete")
    require(all(receipt[field] == baseline[field] for field in (
        "build", "test_sha256", "pack_revision", "inventory_sha256", "native_manifest")),
        "Audit changed the signed test executable or packed weights")
    require(receipt["hardware"] == baseline["hardware"], "The M2 functional test platform changed")
    require(receipt["pressure_samples"] and set(receipt["pressure_samples"]) == {1},
            "Observed memory pressure was abnormal")
    require(digest(report) == receipt["report_sha256"]
            and digest(cases) == receipt["cases_sha256"] == provenance["cases_sha256"],
            "Native output or exact prompt input changed")
    cases_proof = independent(json.loads(report), json.loads(cases), provenance)
    require(all(case["verified_output_positions"] == case["budget"] - 1
                and case["accepted_drafts"] > 0 for case in cases_proof),
            "At least one post-first-token position or accepted draft was not audited")
    return {"hardware": receipt["hardware"], "build": receipt["build"]["commit"],
            "pack_revision": receipt["pack_revision"], "qualification": False,
            "report_sha256": receipt["report_sha256"], "cases": cases_proof}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT)
    parser.add_argument("--repo", type=Path, default=ROOT.parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.raw, args.repo, json.loads((ROOT / "cases-provenance.json").read_bytes())),
                     indent=2, sort_keys=True, allow_nan=False))
