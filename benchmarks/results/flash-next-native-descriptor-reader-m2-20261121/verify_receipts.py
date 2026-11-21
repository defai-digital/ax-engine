"""Independently verify the four native Flash Next sidecar comparison arms."""

from hashlib import file_digest
import json
from pathlib import Path
import sys

root, inventory = map(Path, sys.argv[1:])
result = json.loads((root / "positive/receipt.json").read_text())
assert result["status"] == "complete"
assert result["pack_revision"] == "ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3"
assert result["build"] == json.loads((root / "build.json").read_text())
with (root / "server.patch").open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == result["build"]["source_patch_sha256"]
with (root / "run.py").open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == result["build"]["runner_sha256"]
manifest = json.loads(inventory.read_text())
assert manifest["payload_match"] is True and manifest["revision"] == result["pack_revision"]
assert len(manifest["members"]) == len({k["name"] for k in manifest["members"]}) == 49
lfs = [member for member in manifest["members"] if member["publisher_sha256"] is not None]
assert len(lfs) == 27 and all(member["sha256"] == member["publisher_sha256"] for member in lfs)
sidecar_member, = (member for member in lfs if member["name"] == "mtp.safetensors")
assert sidecar_member["sha256"] == "2f7df97f7a969b3d08b5832e4754af5668e92ada6f473d37da87868c79edcfe0"
compared = ("prompt_hash", "output_tokens", "finish_reason", "status", "mtp",
            "verifier_schedule", "verified_steps", "route_errors")
assert tuple(result["arms"]) == ("disabled", "required")
rows = []
for policy, arms in result["arms"].items():
    order = ["name-order", "descriptor-lazy"]
    if policy == "required":
        order.reverse()
    assert list(arms) == order
    paired = []
    for reader, record in arms.items():
        sidecar = record["sidecar"]
        trunk = record["resident"]
        pages = record["sidecar_initial_page_state"]
        assert pages["selected_pages"] == 318256
        assert 0 <= pages["resident_pages"] <= pages["selected_pages"]
        assert sidecar["tensor_count"] == 31
        assert trunk["tensor_count"] == 1942 and trunk["shard_count"] == 15
        assert record["ready_seconds"] >= trunk["elapsed_seconds"] + sidecar["elapsed_seconds"]
        cases = record["cases"]
        assert [case["prompt_tokens"] for case in cases] == [512, 2048, 8192]
        steps = 0
        for case in cases:
            assert len(case["output_tokens"]) == 32
            assert case["route_errors"] == 0
            assert case["mtp"]["available"] is True
            assert case["mtp"]["active"] is case["mtp"]["requested"] is (policy == "required")
            assert (case["verified_steps"] > 0) is (policy == "required")
            steps += int(case["verified_steps"])
        paired.append([[case[field] for field in compared] for case in cases])
        samples = [json.loads(line) for line in
                   (root / f"positive/{policy}-{reader}-samples.jsonl").read_text().splitlines()]
        assert samples and all(sample["pressure"] == 1 for sample in samples)
        sample_rss = max(sample["reference_rss_bytes"] for sample in samples)
        rows.append({"policy": policy, "reader": reader, "ready_seconds": record["ready_seconds"],
                     "resident_seconds": trunk["elapsed_seconds"], "sidecar_seconds": sidecar["elapsed_seconds"],
                     "peak_sampled_startup_rss_gib": sample_rss / (1 << 30),
                     "sidecar_initial_page_state": pages,
                     "startup_samples": len(samples), "greedy_output_tokens": 96,
                     "verified_steps": steps,
                     "accepted_drafts": sum(case["mtp"]["accepted_tokens"] for case in cases)})
    assert paired[0] == paired[1], policy
fault = json.loads((root / "fault/receipt.json").read_text())
assert fault["status"] == "complete" and fault["server_sha256"] == result["build"]["server_sha256"]
assert fault["pack_revision"] == result["pack_revision"]
assert tuple(fault["arms"]) == ("artifact-integrity", "head-attachment")
for arms in fault["arms"].values():
    assert tuple(arms) == ("required", "disabled")
    assert all(arm["rebound_port"] for arm in arms.values())
    for arm in arms.values():
        if not arm["listener_opened"]:
            assert arm["returncode"] != 0
for arm in fault["arms"]["artifact-integrity"].values():
    assert arm["failure_boundary"] == "manifest" and not arm["listener_opened"]
    assert arm["attachment_failures_logged"] == 0
required, direct = (fault["arms"]["head-attachment"][key] for key in ("required", "disabled"))
assert required["listener_opened"] is False
assert direct["listener_opened"] is True and direct["direct_tokens_match"] is True
assert all(arm["attachment_failures_logged"] == 1 and arm["failure_boundary"] == "head attachment"
           for arm in fault["arms"]["head-attachment"].values())
with (root / "check_fault_policy_initial.py").open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == result["build"]["policy_runner_sha256"]
with (root / "check_fault_policy.py").open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == fault["fault_runner_sha256"]
print(json.dumps({"positive": rows, "fault": fault["arms"]}, indent=2))
