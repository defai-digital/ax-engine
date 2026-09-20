"""Offline falsification controls for the private-cursor evidence reader."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "transactions", Path(__file__).with_name("verify_flash_next_mtp_transactions.py"))
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
OWNER = 3701
HEAD = OWNER ^ 0x51464E4D54500001


def tensor(shape):
    count = 1
    for n in shape:
        count *= n
    return (struct.pack("<BB6x4iQ", 10, len(shape), *(shape + [0] * (4 - len(shape))), count * 4)
            + struct.pack("<f", 0.5) * count)


def state(tokens, head=False):
    n = len(tokens)
    raw = b"AXKB" + struct.pack("<IQQQIi", 4, n, 0, 0, 1 if head else 2, 0)
    layer = bytes([4]) + bytes(7) + struct.pack("<Q", HEAD if head else OWNER)
    if not head:
        history = [99, 99]
        for token in tokens:
            history = [99, 99] if token == 99 else [token, history[0]]
        raw += layer + bytes([0, 1]) + tensor([1, 2]) * 2
        raw += bytes([1, 1]) + tensor([1, 2]) + struct.pack("<I2I", 2, *history)
    raw += layer + bytes([1, 1]) + tensor([1, n, 1, 2]) * 2
    raw += bytes([1]) + tensor([1, n, 1, 2]) + bytes([0])
    return raw


class TransactionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data = self.fixture(True)

    def blob(self, raw):
        digest = hashlib.sha256(raw).hexdigest()
        name = digest + ".bin"
        (self.root / name).write_bytes(raw)
        return {"file": name, "sha256": digest, "bytes": len(raw)}

    def array(self, token=None):
        values = [0.5] * 16 if token is None else [0.0] * 32
        if token is not None:
            values[token] = 2.0
        return {"shape": [1, 1, 16] if token is None else [1, 32], "dtype": "Float32",
                "f32_le": self.blob(struct.pack(f"<{len(values)}f", *values))}

    def snapshot(self, tokens, head=False):
        return {"position": len(tokens), "axkb": self.blob(state(tokens, head))}

    def cursor(self, tokens, proposed, accepted):
        return {"state": self.snapshot(tokens[1:], True), "hidden": self.array(),
                "owner": HEAD, "proposed": proposed, "accepted": accepted}

    def checkpoint(self, tokens, proposed, accepted):
        return {"trunk": self.snapshot(tokens), "cursor": self.cursor(tokens, proposed, accepted)}

    def output(self, tokens):
        return {"state": self.snapshot(tokens), "logits": self.array(3),
                "hidden": self.array(), "stream_hidden": self.array()}

    def verifier_cases(self, primary=3):
        cases = []
        for label, draft, budget, terminal in [
            ("matching-draft", 3, 3, []), ("mismatching-draft", 4, 3, []),
            ("one-slot", 3, 1, []), ("zero-budget", 3, 0, []),
            ("primary-terminal", 3, 3, [primary]), ("correction-terminal", 3, 3, [3]),
        ]:
            before = self.snapshot(self.prompt)
            row = {"label": label, "tokens": self.prompt.copy(), "primary": primary, "draft": draft,
                   "remaining": budget, "terminal_ids": terminal, "before": before,
                   "after_oracle": copy.deepcopy(before), "input_after": copy.deepcopy(before),
                   "reference": None, "actual": None, "error": None, "target_calls": []}
            if budget == 0:
                row["error"] = "requires output budget"
            else:
                accepted = draft == 3 and budget > 1 and not terminal
                consumed = [primary, 3] if accepted else [primary]
                reference = {"after_primary": self.output(self.prompt + [primary]),
                             "after_draft": self.output(self.prompt + consumed) if accepted else None}
                row.update(reference=reference, actual={**copy.deepcopy(reference), "accepted": accepted,
                           "committed": consumed, "next": 3}, target_calls=[[t] for t in consumed])
            cases.append(row)
        return cases

    def case(self, label, tokens, proposed, accepted_count, accept=True,
             budget=3, terminal=None, failure="none", primary=3):
        terminal = terminal or []
        before = self.checkpoint(tokens, proposed, accepted_count)
        row = {"label": label, "tokens": tokens.copy(), "primary": primary, "remaining": budget,
               "terminal_ids": terminal, "failure": failure, "before": before,
               "after_oracle": copy.deepcopy(before), "input_after": self.snapshot(tokens),
               "initial_after": copy.deepcopy(before), "retry_error": None,
               "retry": None, "reference": None, "actual": None, "error": None}
        if budget == 0:
            row.update(error="invalid budget", cursor_after=copy.deepcopy(before["cursor"]), attempted_calls=[])
            return row
        accepted = budget > 1 and accept and primary not in terminal and 3 not in terminal
        consumed = [primary, 3] if accepted else [primary]
        after = self.checkpoint(tokens + consumed, proposed + int(budget > 1), accepted_count + int(accepted))
        ref = {"checkpoint": after, "consumed": consumed, "next": 3, "accepted": accepted,
               "draft": (3 if accept else 17) if budget > 1 else None,
               "proposal_logits": self.array(3 if accept else 17) if budget > 1 else None,
               "correction_logits": self.array(3), "next_logits": self.array(3),
               "primary_state": self.snapshot(tokens + [primary]), "primary_hidden": self.array()}
        actual = {k: copy.deepcopy(v) for k, v in ref.items() if k != "proposal_logits"}
        actual["target_calls"] = [[t] for t in consumed]
        row["reference"] = ref
        if failure == "none":
            row.update(actual=actual, attempted_calls=actual["target_calls"],
                       cursor_after=copy.deepcopy(after["cursor"]))
        else:
            row.update(error="injected failure after materialized target", retry=actual,
                       attempted_calls=[[primary]] if failure == "target_1" else [[primary], [3]],
                       cursor_after=copy.deepcopy(before["cursor"]))
        return row

    def fixture(self, accept, initial_primary=3):
        tokens = self.prompt.copy()
        cases = []
        for i in range(8):
            primary = initial_primary if i == 0 else 3
            cases.append(self.case(f"trajectory-{i}", tokens, i, i if accept else 0, accept, primary=primary))
            tokens += [primary, 3] if accept else [primary]
        for label, budget, terminal, hook in [
            ("zero-budget", 0, [], "none"), ("one-slot", 1, [], "none"),
            ("primary-terminal", 3, [initial_primary], "none"), ("correction-terminal", 3, [3], "none"),
            ("failure-target-1", 3, [], "target_1"),
        ]:
            cases.append(self.case(label, self.prompt, 0, 0, accept, budget, terminal, hook, initial_primary))
        if accept:
            for label, hook in [("failure-target-2", "target_2"), ("failure-catchup", "accepted_catchup")]:
                cases.append(self.case(label, self.prompt, 0, 0, True, failure=hook, primary=initial_primary))
        return {"schema": "ax.flash_next.mtp_transactions.v2", "qualification": False, "release_ready": False,
                "mtp_certification": {g: "not_assessed" for g in ("MTP-S", "MTP-P", "MTP-D")},
                "arm": "synthetic_accept" if accept else "synthetic_reject", "head_permutation_seed": None,
                "prompt_ids": self.prompt, "trunk_owner": OWNER, "head_owner": HEAD,
                "trajectory_steps": 8, "target_schedule": "canonical_singleton",
                "initial_primary_logits": self.array(initial_primary), "initial_hidden": self.array(),
                "trunk_initial_histories": [[99, 99], None], "head_initial_histories": [None],
                "accepted_failure_coverage": accept, "cases": cases, "verifier_cases": self.verifier_cases(initial_primary)}

    def verify(self):
        (self.root / "result.json").write_text(json.dumps(self.data))
        return reader.verify(self.root)

    def test_accept_and_reject_controls(self):
        for accepted in (True, False):
            self.data = self.fixture(accepted)
            result = self.verify()
            self.assertTrue(result["diagnostic_passed"])
            self.assertEqual(result["accepted_failure_coverage"], accepted)
            self.assertFalse(result["qualification"])
            self.assertTrue(result["forced_verifier_controls_assessed"])

    def test_missing_natural_acceptance_is_not_trained_pass(self):
        self.data = self.fixture(False)
        self.data["arm"] = "trained"
        self.assertFalse(self.verify()["diagnostic_passed"])

    def test_distinct_primary_and_correction_terminals_are_disclosed(self):
        self.data = self.fixture(True, initial_primary=2)
        self.assertTrue(self.verify()["terminal_cases_distinct"])
        self.data["verifier_cases"][5]["terminal_ids"] = [2]
        with self.assertRaisesRegex(ValueError, "Changed forced verifier controls"):
            self.verify()

    def test_rejects_false_decisions_and_state_ownership(self):
        baseline = copy.deepcopy(self.data)
        for mutate in (
            lambda d: d["cases"][0]["actual"].update(accepted=False),
            lambda d: d["cases"][0]["actual"].update(draft=17),
            lambda d: d["cases"][0]["actual"].update(next=17),
            lambda d: d["cases"][1].update(primary=17),
            lambda d: d["cases"][0]["actual"]["checkpoint"]["cursor"].update(owner=OWNER),
            lambda d: d["cases"][0]["actual"]["checkpoint"]["cursor"].update(proposed=2),
            lambda d: d["cases"][0]["reference"]["checkpoint"]["cursor"].update(proposed=2),
            lambda d: d["cases"][0]["actual"].update(target_calls=[[3, 3]]),
            lambda d: d["cases"][0]["actual"]["checkpoint"]["cursor"]["state"].update(position=100),
            lambda d: d["cases"][12]["cursor_after"].update(proposed=1),
            lambda d: d["cases"][12].update(error="unrelated disk failure"),
            lambda d: d["cases"][12].update(retry_error="retry failure"),
            lambda d: d["cases"][12]["initial_after"]["cursor"].update(proposed=1),
            lambda d: d["cases"][12]["retry"]["checkpoint"]["cursor"].update(accepted=0),
            lambda d: d["cases"][8].update(attempted_calls=[[3]]),
            lambda d: d["cases"][9]["actual"].update(accepted=True),
            lambda d: d["cases"].pop(),
            lambda d: d.update(qualification=True),
        ):
            with self.subTest(mutate=mutate):
                self.data = copy.deepcopy(baseline)
                mutate(self.data)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_valid_digest_does_not_hide_nan_or_owner_corruption(self):
        baseline = copy.deepcopy(self.data)
        for kind in ("nan", "owner"):
            with self.subTest(kind=kind):
                self.data = copy.deepcopy(baseline)
                row = self.data["cases"][0]["before"]["cursor"]["state"]
                raw = bytearray((self.root / row["axkb"]["file"]).read_bytes())
                if kind == "nan":
                    raw[90:94] = struct.pack("<I", 0x7FC00000)
                else:
                    raw[48:56] = struct.pack("<Q", OWNER)
                row["axkb"] = self.blob(raw)
                with self.assertRaisesRegex(ValueError, "Nonfinite|owner"):
                    self.verify()

    def test_wrong_hidden_and_nonfinite_logits_are_rejected(self):
        baseline = copy.deepcopy(self.data)
        for field, bits in (("primary_hidden", 0x3F400000), ("next_logits", 0x7F800000)):
            self.data = copy.deepcopy(baseline)
            row = self.data["cases"][0]["actual"][field]
            raw = bytearray((self.root / row["f32_le"]["file"]).read_bytes())
            raw[:4] = struct.pack("<I", bits)
            row["f32_le"] = self.blob(raw)
            with self.assertRaises(ValueError):
                self.verify()

    def test_initial_decision_and_ple_history_are_bound(self):
        baseline = copy.deepcopy(self.data)
        self.data["initial_primary_logits"] = self.array(17)
        with self.assertRaisesRegex(ValueError, "Initial primary"):
            self.verify()
        self.data = baseline
        row = self.data["cases"][0]["before"]["trunk"]
        raw = bytearray((self.root / row["axkb"]["file"]).read_bytes())
        # Newest PLE history token, after two GDN tensors and the convolution.
        raw[184:188] = struct.pack("<I", 99)
        row["axkb"] = self.blob(raw)
        with self.assertRaisesRegex(ValueError, "PLE history"):
            self.verify()

    def test_forced_verifier_cannot_accept_wrong_or_truncated_drafts(self):
        baseline = copy.deepcopy(self.data)
        for mutate in (
            lambda d: d["verifier_cases"].pop(),
            lambda d: d["verifier_cases"][1].update(draft=3),
            lambda d: d["verifier_cases"][1]["actual"].update(accepted=True),
            lambda d: d["verifier_cases"][2]["actual"].update(committed=[3, 3]),
            lambda d: d["verifier_cases"][3].update(target_calls=[[3]]),
            lambda d: d["verifier_cases"][0]["actual"].update(next=17),
            lambda d: d["verifier_cases"][0]["actual"]["after_draft"]["state"].update(position=10),
            lambda d: d["verifier_cases"][4].update(terminal_ids=[]),
        ):
            with self.subTest(mutate=mutate):
                self.data = copy.deepcopy(baseline)
                mutate(self.data)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_declared_widened_dtype_is_checked(self):
        for dtype in ("Bfloat16", "Float16"):
            with self.subTest(dtype=dtype):
                row = self.array()
                row["dtype"] = dtype
                check = reader.Reader(self.root, self.data)
                check.array(row)
                # A finite F32 value with nonzero low mantissa bits is not a
                # widened half-precision value, even with a correct digest.
                row["f32_le"] = self.blob(struct.pack("<I", 0x3F000001) * 16)
                with self.assertRaisesRegex(ValueError, "declared"):
                    check.array(row)


if __name__ == "__main__":
    unittest.main()
