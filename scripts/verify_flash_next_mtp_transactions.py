#!/usr/bin/env python3
"""Independently read raw private-cursor transaction diagnostics, not certification."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

_spec = importlib.util.spec_from_file_location(
    "_flash_runner_states", Path(__file__).with_name("verify_flash_next_runner_states.py"))
_states = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_states)
require, ids = _states.require, _states.ids


def integer(value):
    require(type(value) is int and value >= 0, "Invalid counter or position")
    return value


class Reader:
    def __init__(self, root, data):
        self.root, self.data = Path(root), data
        self.checked_states = set()
        self.checked_arrays = {}

    def blob(self, record):
        digest = record["sha256"]
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                "Invalid content digest")
        return _states.read_blob(self.root, record, digest + ".bin")

    def array(self, row):
        shape, dtype = row["shape"], row["dtype"]
        require(isinstance(shape, list) and 1 <= len(shape) <= 3
                and all(type(n) is int and n > 0 for n in shape)
                and math.prod(shape[:-1]) == 1, "Invalid array shape")
        require(dtype in ("Float16", "Float32", "Bfloat16"), "Invalid array dtype")
        record = row["f32_le"]
        key = (record["file"], record["sha256"], integer(record["bytes"]), tuple(shape), dtype)
        if key not in self.checked_arrays:
            raw = self.blob(record)
            require(len(raw) == 4 * math.prod(shape), "Array byte count differs from shape")
            values = [v for v, in struct.iter_unpack("<f", raw)]
            require(all(math.isfinite(v) for v in values), "Nonfinite transaction array")
            if dtype == "Bfloat16":
                require(all(bits & 0xFFFF == 0 for bits, in struct.iter_unpack("<I", raw)),
                        "Array values cannot represent declared BF16")
            elif dtype == "Float16":
                try:
                    restored = b"".join(struct.pack("<f", struct.unpack("<e", struct.pack("<e", v))[0])
                                        for v in values)
                except OverflowError as error:
                    raise ValueError("Array values cannot represent declared F16") from error
                require(restored == raw, "Array values cannot represent declared F16")
            self.checked_arrays[key] = max(range(len(values)), key=values.__getitem__)
        return key, self.checked_arrays[key]

    def state(self, row, tokens, graph):
        position = integer(row["position"])
        require(position == len(tokens), "State position differs from consumed prefix")
        owner = self.data[graph + "_owner"]
        record = row["axkb"]
        identity = (record["file"], record["sha256"], integer(record["bytes"]))
        key = (identity, tuple(tokens), graph)
        if key not in self.checked_states:
            raw = self.blob(record)
            _states.state_layout(raw, position, tokens, self.data[graph + "_initial_histories"])
            require(struct.unpack_from("<Q", raw, 48)[0] == owner, "Wrong state owner")
            self.checked_states.add(key)
        return identity, position

    def cursor(self, row, tokens):
        require(type(row["owner"]) is int and row["owner"] == self.data["head_owner"],
                "Wrong cursor owner")
        proposed, accepted = integer(row["proposed"]), integer(row["accepted"])
        require(accepted <= proposed, "Accepted counter exceeds proposals")
        require(len(row["hidden"]["shape"]) == 3 and row["hidden"]["shape"][:2] == [1, 1],
                "Invalid retained hidden row")
        return (self.state(row["state"], tokens[1:], "head"), self.array(row["hidden"])[0],
                proposed, accepted)

    def checkpoint(self, row, tokens):
        return self.state(row["trunk"], tokens, "trunk"), self.cursor(row["cursor"], tokens)

    def output(self, row, tokens):
        return (self.state(row["state"], tokens, "trunk"), self.array(row["logits"])[0],
                self.array(row["hidden"])[0], self.array(row["stream_hidden"])[0])

    def case(self, case, tokens, expected_before=None):
        require(ids(case["tokens"]) == tokens, "Wrong case prefix")
        primary = ids([case["primary"]])[0]
        remaining = integer(case["remaining"])
        terminal = ids(case["terminal_ids"])
        before = self.checkpoint(case["before"], tokens)
        require(expected_before is None or before == expected_before, "Broken trajectory checkpoint")
        require(self.checkpoint(case["after_oracle"], tokens) == before,
                "Oracle mutated its input checkpoint")
        require(self.state(case["input_after"], tokens, "trunk") == before[0],
                "Attempt mutated input trunk")
        require(self.checkpoint(case["initial_after"], tokens) == before,
                "Attempt or retry mutated the saved input checkpoint")
        require(case["retry_error"] is None, "Failed retry")
        error, failure = case["error"], case["failure"]
        require(failure in ("none", "target_1", "target_2", "accepted_catchup"), "Unknown hook")
        if remaining == 0:
            require(failure == "none" and isinstance(error, str) and "budget" in error,
                    "Zero budget did not fail explicitly")
            require(case["reference"] is None and case["actual"] is None
                    and case["retry"] is None and case["attempted_calls"] == [],
                    "Zero budget executed a target or produced output")
            require(self.cursor(case["cursor_after"], tokens) == before[1],
                    "Zero budget mutated cursor")
            return None

        ref = case["reference"]
        correction = self.array(ref["correction_logits"])[1]
        proposal = self.array(ref["proposal_logits"])[1] if remaining > 1 else None
        require(remaining > 1 or ref["proposal_logits"] is None, "One-slot proposal recorded")
        accepted = remaining > 1 and proposal == correction and primary not in terminal and correction not in terminal
        consumed = [primary, correction] if accepted else [primary]
        next_token = self.array(ref["next_logits"])[1]
        require(ref["draft"] is None if proposal is None else ids([ref["draft"]]) == [proposal],
                "Invalid oracle draft token")
        require(ref["accepted"] is accepted
                and ids(ref["consumed"]) == consumed and ids([ref["next"]]) == [next_token],
                "Incorrect oracle decision")
        if not accepted:
            require(self.array(ref["next_logits"])[0] == self.array(ref["correction_logits"])[0],
                    "Rejected or truncated step used a bonus")
        primary_state = self.state(ref["primary_state"], tokens + [primary], "trunk")
        primary_hidden = self.array(ref["primary_hidden"])[0]
        expected = self.checkpoint(ref["checkpoint"], tokens + consumed)
        require(expected[1][2:] == (before[1][2] + int(remaining > 1),
                                    before[1][3] + int(accepted)), "Wrong oracle counters")
        if not accepted:
            require(expected[0] == primary_state and expected[1][1] == primary_hidden,
                    "Rejected step retained a speculative state or row")

        def actual(row):
            require(row["draft"] is None if proposal is None else ids([row["draft"]]) == [proposal],
                    "Invalid cursor draft token")
            require(row["accepted"] is accepted
                    and ids(row["consumed"]) == consumed and ids([row["next"]]) == [next_token],
                    "Invalid cursor decision")
            require(row["target_calls"] == [[t] for t in consumed], "Wrong target call schedule")
            for key in ("correction_logits", "next_logits"):
                require(self.array(row[key])[0] == self.array(ref[key])[0], "Verifier logits differ from ordinary replay")
            require(self.state(row["primary_state"], tokens + [primary], "trunk") == primary_state
                    and self.array(row["primary_hidden"])[0] == primary_hidden,
                    "Primary transition differs from replay")
            require(self.checkpoint(row["checkpoint"], tokens + consumed) == expected,
                    "Private or target state differs from ordinary replay")

        if failure == "none":
            require(error is None and case["retry"] is None, "Unexpected failure or retry")
            actual(case["actual"])
            require(case["attempted_calls"] == [[t] for t in consumed], "Incorrect attempted calls")
            require(self.cursor(case["cursor_after"], tokens + consumed) == expected[1],
                    "Incorrect committed cursor")
        else:
            require(failure == "target_1" or accepted, "Unreachable accepted failure hook")
            require(isinstance(error, str) and "after materialized" in error
                    and case["actual"] is None, "Missing materialized failure")
            require(case["attempted_calls"] == ([[primary]] if failure == "target_1"
                    else [[primary], [correction]]), "Wrong failure boundary")
            require(self.cursor(case["cursor_after"], tokens) == before[1], "Failed transaction changed cursor")
            actual(case["retry"])
        return expected, tokens + consumed, next_token, accepted


def verify_forced(reader, cases, prompt, initial, primary, correction):
    labels = ["matching-draft", "mismatching-draft", "one-slot", "zero-budget",
              "primary-terminal", "correction-terminal"]
    require([c["label"] for c in cases] == labels, "Missing forced verifier controls")
    vocab = reader.data["initial_primary_logits"]["shape"][-1]
    require(vocab > 1, "Insufficient verifier vocabulary")
    drafts = [correction, (correction + 1) % vocab] + [correction] * 4
    terminals = [[], [], [], [], [primary], [correction]]
    for case, draft, budget, terminal in zip(cases, drafts, [3, 3, 1, 0, 3, 3], terminals, strict=True):
        require(ids(case["tokens"]) == prompt and ids([case["primary"], case["draft"]]) == [primary, draft]
                and type(case["remaining"]) is int and case["remaining"] == budget
                and ids(case["terminal_ids"]) == terminal, "Changed forced verifier controls")
        for key in ("before", "after_oracle", "input_after"):
            require(reader.state(case[key], prompt, "trunk") == initial, "Verifier mutated checkpoint")
        if budget == 0:
            require(case["reference"] is None and case["actual"] is None
                    and isinstance(case["error"], str) and "budget" in case["error"]
                    and case["target_calls"] == [], "Zero-budget verifier did work")
            continue
        require(case["error"] is None, "Unexpected verifier failure")
        ref, actual = case["reference"], case["actual"]
        require(reader.array(ref["after_primary"]["logits"])[1] == correction,
                "Changed ordinary correction")
        accepted = budget > 1 and draft == correction and primary not in terminal and correction not in terminal
        consumed = [primary, draft] if accepted else [primary]
        require(actual["accepted"] is accepted and ids(actual["committed"]) == consumed
                and case["target_calls"] == [[t] for t in consumed], "Incorrect verifier commit")
        require(reader.output(actual["after_primary"], prompt + [primary])
                == reader.output(ref["after_primary"], prompt + [primary]), "Wrong primary verifier transition")
        require((ref["after_draft"] is not None) is accepted
                and (actual["after_draft"] is not None) is accepted, "Spurious or missing draft transition")
        if accepted:
            require(reader.output(actual["after_draft"], prompt + consumed)
                    == reader.output(ref["after_draft"], prompt + consumed), "Wrong accepted verifier state")
        last = ref["after_draft"] if accepted else ref["after_primary"]
        require(ids([actual["next"]]) == [reader.array(last["logits"])[1]], "Wrong verifier continuation")


def verify(root):
    root = Path(root)
    data = json.loads((root / "result.json").read_text())
    require(data["schema"] == "ax.flash_next.mtp_transactions.v2", "Unknown schema")
    require(data["qualification"] is False and data["release_ready"] is False
            and data["mtp_certification"] == {g: "not_assessed" for g in ("MTP-S", "MTP-P", "MTP-D")},
            "Unexpected promotion")
    require(data["target_schedule"] == "canonical_singleton" and data["trajectory_steps"] == 8,
            "Changed transaction contract")
    require(type(data["trunk_owner"]) is int and data["trunk_owner"] == 3701
            and type(data["head_owner"]) is int and data["head_owner"] == (3701 ^ 0x51464E4D54500001),
            "Changed or aliased graph owners")
    arm = data["arm"]
    require(arm in ("trained", "permuted", "synthetic_accept", "synthetic_reject"), "Unknown arm")
    require(data["head_permutation_seed"] == (20260916 if arm == "permuted" else None),
            "Changed head control")
    prompt = ids(data["prompt_ids"])
    require(3 <= len(prompt) <= 512, "Invalid prompt length")
    coverage = data["accepted_failure_coverage"]
    require(type(coverage) is bool, "Invalid coverage declaration")
    labels = [f"trajectory-{i}" for i in range(8)] + ["zero-budget", "one-slot",
              "primary-terminal", "correction-terminal", "failure-target-1"]
    if coverage:
        labels += ["failure-target-2", "failure-catchup"]
    cases = data["cases"]
    require([c["label"] for c in cases] == labels, "Incomplete or reordered cases")
    reader = Reader(root, data)
    require(cases[0]["primary"] == reader.array(data["initial_primary_logits"])[1],
            "Initial primary differs from prefill logits")
    tokens, before, previous_next, first_accepted = prompt.copy(), None, None, None
    for case in cases[:8]:
        require(case["remaining"] == 3 and case["terminal_ids"] == [] and case["failure"] == "none",
                "Changed trajectory controls")
        require(previous_next is None or case["primary"] == previous_next, "Broken continuation token")
        result = reader.case(case, tokens, before)
        if result[3] and first_accepted is None:
            first_accepted = (tokens.copy(), reader.checkpoint(case["before"], tokens), case["primary"])
        before, tokens, previous_next, _ = result
    require(coverage is (first_accepted is not None), "False acceptance coverage")
    initial = reader.checkpoint(cases[0]["before"], prompt)
    require(initial[1][2:] == (0, 0), "Nonzero initial counters")
    require(initial[1][1] == reader.array(data["initial_hidden"])[0], "Wrong initial retained row")
    initial_primary = cases[0]["primary"]
    correction = reader.array(cases[0]["reference"]["correction_logits"])[1]
    for case, budget, terminal, hook in zip(cases[8:13], [0, 1, 3, 3, 3],
            [[], [], [initial_primary], [correction], []], ["none"] * 4 + ["target_1"], strict=True):
        require(case["primary"] == initial_primary and case["remaining"] == budget
                and case["terminal_ids"] == terminal and case["failure"] == hook, "Changed boundary controls")
        reader.case(case, prompt, initial)
    if coverage:
        tokens, before, primary = first_accepted
        for case, hook in zip(cases[13:], ["target_2", "accepted_catchup"], strict=True):
            require(case["primary"] == primary and case["remaining"] == 3
                    and case["terminal_ids"] == [] and case["failure"] == hook, "Changed failure controls")
            reader.case(case, tokens, before)
    verify_forced(reader, data["verifier_cases"], prompt, initial[0], initial_primary, correction)
    passed = coverage or arm in ("permuted", "synthetic_reject")
    return {"collection_valid": True, "diagnostic_passed": passed, "arm": arm,
            "cases": len(cases), "accepted_failure_coverage": coverage,
            "terminal_cases_distinct": initial_primary != correction,
            "forced_verifier_controls_assessed": True, "qualification": False,
            "release_ready": False, "mtp_certification": data["mtp_certification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
