"""Replay the Flash Next natural EOS checks against the pinned pack and prompts."""

import gzip
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
QA = Path("benchmarks/results/qualification/2026-09-19-flash-next-mxfp4-installed-qa/full-qa-inputs.json")
REFERENCE = Path("benchmarks/results/qualification/2026-10-24-flash-next-mxfp4-m5max-release-v7.9.0/reference/result.json")
INVENTORY = Path("benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support/pack-inventory.json")
CASES = ("instruction_sort_numbers", "format_bracket_list", "long_context_record_mass")


def digest(raw):
    return sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def workloads(repo, plan):
    sys.path.insert(0, str(repo / "scripts"))
    from run_flash_next_generate_matrix import ids_sha

    source = (repo / QA).read_bytes()
    baseline = (repo / REFERENCE).read_bytes()
    require(digest(source) == plan["qa_inputs_sha256"]
            and digest(baseline) == plan["signed_ordinary_reference_sha256"], "Frozen prompt sources changed")
    ids = {row["id"]: row["input_ids"] for row in json.loads(source)["prepared"]}
    reference = {row["id"]: row for row in json.loads(baseline)["qa"]["rows"]}
    cases = []
    for name, entry in zip(CASES, plan["cases"], strict=True):
        tokens = ids[name]
        answer = reference[name]
        require(answer["normal_stop"] and answer["terminal_eos_id"] == 248046
                and entry == {"name": name, "input_tokens": len(tokens), "prompt_sha256": ids_sha(tokens),
                              "budget": 64, "signed_reference_natural_stop": answer["terminal_eos_id"]},
                f"Frozen natural-stop case changed: {name}")
        cases.append({"name": name, "prompt": tokens, "budget": 64})
    return (json.dumps(cases, separators=(",", ":")) + "\n").encode()


def decisions(case, inputs, terminal):
    name, length, budget = inputs["name"], len(inputs["prompt"]), inputs["budget"]
    direct, tokens = case["direct_tokens"], case["mtp_tokens"]
    changes = [index for index, (left, right) in enumerate(zip_longest(direct, tokens)) if left != right]
    require(case["name"] == name and case["input_tokens"] == length and case["budget"] == budget
            and case["changed_from_direct"] == changes and case["stop_reason"] == "eos_token",
            f"Direct and MTP report disagree at EOS: {name}")
    for output in (direct, tokens):
        require(1 < len(output) < budget and output[-1] in terminal
                and all(token not in terminal for token in output[:-1])
                and all(type(token) is int and 0 <= token < 248_320 for token in output),
                f"Output did not end at the first configured terminal token: {name}")
    cursor = 1
    accepted = rejected = 0
    route = None
    for step in case["decisions"]:
        start = step["position"] - length + 1
        require(start == cursor and step["remaining"] == budget - start and step["primary"] == tokens[start - 1],
                "A verifier step lacks contiguous emitted output")
        allowed = (step["remaining"] > 1 and step["primary"] not in terminal and step["target"] not in terminal
                   and step["draft"] == step["target"])
        require(step["accepted"] == int(allowed), "Reported acceptance differs from the same-state target")
        emitted = [step["target"], step["next"]] if allowed else [step["target"]]
        if not allowed:
            require(step["next"] == step["target"], "Rejected draft changed the correction")
        cursor += len(emitted)
        require(cursor <= len(tokens) and step["emitted"] == emitted == tokens[start:cursor],
                "A verifier step emitted modified tokens or output beyond EOS")
        accepted += int(allowed)
        rejected += int(step["draft"] is not None and not allowed)
        if cursor == len(tokens):
            route = "bonus" if allowed else "correction"
    require(cursor == len(tokens) and route is not None and accepted > 0
            and len(case["decisions"]) == case["verified_steps"]
            and case["accepted"] == accepted and case["rejected"] == rejected
            and case["output_positions_audited"] == len(tokens) - 1,
            "The terminal decision or acceptance counters did not reconcile")
    return {"name": name, "input_tokens": length, "budget": budget, "output_tokens": len(tokens),
            "terminal_token_id": tokens[-1], "terminal_route": route, "steps": case["verified_steps"],
            "direct_output_tokens": len(direct), "changed_from_direct": changes,
            "accepted_drafts": accepted, "rejected_drafts": rejected}


def verify(raw, repo):
    plan = json.loads((ROOT / "cases-provenance.json").read_bytes())
    receipt = json.loads((raw / "receipt.json").read_bytes())
    inventory = (repo / INVENTORY).read_bytes()
    config = (ROOT / "generation_config.json").read_bytes()
    member = next(item for item in json.loads(inventory)["members"] if item["name"] == "generation_config.json")
    terminal = set(json.loads(config)["eos_token_id"])
    require(digest(config) == member["sha256"] and terminal == {248044, 248046}, "Publisher terminal IDs changed")
    require((raw / "status.txt").read_bytes() == b"0\n" and receipt["status"] == "complete"
            and receipt["test_exit"] == 0 and receipt["qualification"] is False
            and receipt["mtp_cost_policy_disabled"] is True,
            "The owned natural-stop model test did not finish")
    require(receipt["inventory_sha256"] == digest(inventory)
            and receipt["pack_revision"] == json.loads(inventory)["revision"]
            and receipt["test_sha256"] == receipt["build"]["test_sha256"]
            and receipt["pressure_samples"] and set(receipt["pressure_samples"]) == {1},
            "The signed executable, pack or memory envelope changed")
    inputs = workloads(repo, plan)
    report = gzip.decompress((raw / "report.json.gz").read_bytes())
    require(digest(inputs) == plan["cases_sha256"] == receipt["cases_sha256"]
            and digest(report) == receipt["report_sha256"], "The workload or raw native result changed")
    observed = json.loads(report)
    require(observed["schema"] == "ax-engine.flash-next.runner-mtp-greedy-natural-stop-audit.v1"
            and observed["qualification"] is False and observed["prefill_quantum"] == 2048
            and observed["ignore_eos"] is False, "The actual EOS handling was not exercised")
    checked = [decisions(case, expected, terminal) for case, expected in
               zip(observed["cases"], json.loads(inputs), strict=True)]
    require({row["terminal_route"] for row in checked} == {"correction", "bonus"},
            "The frozen corpus lost one of its terminal paths")
    return {"qualification": False, "hardware": receipt["hardware"], "build": receipt["build"],
            "pack_revision": receipt["pack_revision"], "cases": checked,
            "pressure_sample_count": len(receipt["pressure_samples"]), "raw_report_sha256": digest(report)}



def replay(repo):
    build = json.loads((ROOT / "build.json").read_bytes())
    source = json.loads((ROOT / "source.json").read_bytes())
    tree = subprocess.check_output(["git", "-C", repo, "rev-parse", f'{source["commit"]}^{{tree}}'],
                                   text=True).strip()
    require(tree == source["tree"] == build["tree"] and build["codesign_verified"] is True
            and build["profile"] == "release-server" and build["mlx"] == build["mlx_metal"] == "0.32.4",
            "The source tree or executable build receipt changed")
    tiers = {"m2": ("Apple M2 Ultra", 192 << 30), "m5": ("Apple M5 Max", 128 << 30)}
    records = {}
    for tier, (soc, memory) in tiers.items():
        record = verify(ROOT / tier, repo)
        require(record["build"] == build and record["hardware"]["soc"] == soc
                and record["hardware"]["memory_bytes"] == memory,
                "A test did not use the exact compiled source and expected hardware")
        records[tier] = record
    m2, m5 = [json.loads(gzip.decompress((ROOT / tier / "report.json.gz").read_bytes()))
              for tier in tiers]
    cross_device = {}
    for left, right in zip(m2["cases"], m5["cases"], strict=True):
        require(left["name"] == right["name"], "The hardware workload rosters differ")
        cross_device[left["name"]] = {
            mode: [index for index, pair in enumerate(zip_longest(left[f"{mode}_tokens"], right[f"{mode}_tokens"]))
                   if pair[0] != pair[1]]
            for mode in ("direct", "mtp")
        }
    return {"source": source, "qualification": False, "hardware_results": records,
            "cross_device_differences": cross_device}


if __name__ == "__main__":
    result = replay(ROOT.parents[2])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
