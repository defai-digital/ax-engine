"""Check trace reconciliation for the frozen Required-policy runner cohort."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def check_case(case, row):
    assert row["kind"] == "case" and row["name"] == case["name"]
    budget = case["budget"]
    prompt_size = len(case["prompt"])
    assert (row["input_tokens"], row["budget"]) == (prompt_size, budget)
    direct, mtp = row["direct_tokens"], row["mtp_tokens"]
    for name, tokens in (("direct", direct), ("mtp", mtp)):
        reason = row[f"{name}_stop_reason"]
        assert 0 < len(tokens) <= budget
        assert reason == "eos_token" or (reason == "max_output_tokens" and len(tokens) == budget)
    changed = [i for i in range(max(len(direct), len(mtp))) if direct[i:i + 1] != mtp[i:i + 1]]
    assert row["changed_from_direct"] == changed
    verified = row["decisions"]
    activation = row["activation"]
    prefix = "ax_mlx_flash_next_mtp_"
    for suffix in ("step_errors", "cost_parks", "cost_parked_steps", "cost_resumes"):
        assert activation.get(prefix + suffix, 0) == 0, (row["name"], suffix)
    assert len(verified) == row["verified_steps"] == activation.get(prefix + "verified_steps", 0)
    accepted = rejected = covered = 0
    last_end = 1
    for step in verified:
        start = step["position"] - prompt_size + 1
        assert last_end <= start < len(mtp)
        assert step["remaining"] == budget - start and step["primary"] == mtp[start - 1]
        if step["accepted"] == 1:
            assert step["remaining"] > 1 and step["draft"] == step["target"]
            assert step["emitted"] == [step["target"], step["next"]]
            accepted += 1
        else:
            assert step["accepted"] == 0 and step["next"] == step["target"]
            assert step["emitted"] == [step["target"]]
            rejected += step["draft"] is not None
        last_end = start + len(step["emitted"])
        assert last_end <= len(mtp) and step["emitted"] == mtp[start:last_end]
        covered += len(step["emitted"])

    assert accepted == row["accepted"] == activation.get(prefix + "accepted_steps", 0)
    assert rejected == row["rejected"] and covered == row["output_positions_audited"]
    return Counter(cases=1, verified_steps=len(verified), accepted_drafts=accepted,
                   rejected_drafts=rejected, output_positions_audited=covered,
                   output_ids=len(mtp), all_post_first_reconciled=covered == len(mtp) - 1,
                   both_eos=row["direct_stop_reason"] == row["mtp_stop_reason"] == "eos_token",
                   mtp_decisions_observed=bool(verified), same_direct_output=not changed,
                   cost_would_park=activation.get(prefix + "cost_would_parks", 0))


def check(input_bytes, output_bytes):
    cases = json.loads(input_bytes)
    assert len(cases) == 105 and cases[-1]["name"] == "long_context_record_mass"
    *rows, footer = (json.loads(row) for row in output_bytes.splitlines())
    stats = Counter()
    for case, row in zip(cases, rows, strict=True):
        stats.update(check_case(case, row))
    assert footer == {
        "kind": "complete", "schema": "ax-engine.flash-next.runner-required-qa-audit.v1",
        "qualification": False, "cases": stats["cases"], "accepted_drafts": stats["accepted_drafts"],
        "verified_steps": stats["verified_steps"], "prefill_quantum": 2048,
    }
    assert stats["accepted_drafts"] > 0
    return dict(stats)


def main():
    archive = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=archive / "required-qa-inputs.json.gz")
    parser.add_argument("--output", type=Path, default=archive / "required-qa-output.jsonl.gz")
    args = parser.parse_args()
    input_bytes = gzip.decompress(args.input.read_bytes())
    output_bytes = gzip.decompress(args.output.read_bytes())
    receipt = json.loads((archive / "required-qa-inputs-receipt.json").read_bytes())
    assert hashlib.sha256(input_bytes).hexdigest() == receipt["audit_input_sha256"]
    assert hashlib.sha256(output_bytes).hexdigest() == (archive / "output.sha256").read_text().split()[0]
    before = (archive / "pack-before.json").read_bytes()
    assert before == (archive / "pack-after.json").read_bytes()
    pack = json.loads(before)
    assert pack["verified_members"] == 49
    assert pack["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
    assert hashlib.sha256((archive / "mxfp4-inventory.json").read_bytes()).hexdigest() == pack["inventory_sha256"]
    build = json.loads((archive / "build.json").read_bytes())
    assert hashlib.sha256(gzip.decompress((archive / "source.patch.gz").read_bytes())).hexdigest() == build["patch_sha256"]
    assert (archive / "status.txt").read_text().strip() == "0"
    print(json.dumps(check(input_bytes, output_bytes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
