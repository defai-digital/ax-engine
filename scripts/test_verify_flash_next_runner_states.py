"""Offline mutation controls for the aligned runner evidence reader."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "verify_runner_states", Path(__file__).with_name("verify_flash_next_runner_states.py")
)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def tensor(shape):
    count = 1
    for n in shape:
        count *= n
    return (struct.pack("<BB6x4iQ", 10, len(shape), *(shape + [0] * (4 - len(shape))), count * 4)
            + struct.pack(f"<{count}f", *([0.5] * count)))


def snapshot(tokens):
    p = len(tokens)
    out = b"AXKB" + struct.pack("<IQQQIi", 4, p, 0, 0, 2, 0)
    layer = bytes([4]) + bytes(7) + struct.pack("<Q", 123)
    # One GDN layer carrying PLE, then one QSA layer with index state.
    out += layer + bytes([0, 1]) + tensor([1, 2]) + tensor([1, 2])
    out += bytes([1, 1]) + tensor([1, 2]) + struct.pack("<I2I", 2, *tokens[-2:][::-1])
    out += layer + bytes([1, 1]) + tensor([1, p, 1, 2]) * 2
    out += bytes([1]) + tensor([1, p, 1, 2]) + bytes([0])
    return out


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompt, self.tokens = [1, 2, 3], list(range(4, 12))
        self.data = {
            "schema": "ax.flash_next.runner_aligned_state.v1",
            "qualification": False, "release_ready": False,
            "mtp_certification": {g: "not_assessed" for g in ("MTP-S", "MTP-P", "MTP-D")},
            "max_output_tokens": 8, "block_size_tokens": 4, "target_schedule": "canonical_singleton",
            "prompt_ids": self.prompt, "direct_ids": self.tokens, "mtp_ids": self.tokens.copy(),
            "direct_positions": list(range(4, 11)), "mtp_positions": [3, 5, 7, 9],
            "ple_initial_histories": [[99, 99], None],
            "mtp_routes": [[["ax_mlx_flash_next_mtp_verified_steps", 3]]],
            "diagnostic_passed": True,
            "aligned_states": [self.pair(f"prefix-{p - 3}", p) for p in (5, 7, 9)],
            "continuation": {
                "from_position": 9, "input_token": 10,
                "direct_expected_token": 11, "mtp_expected_token": 11,
                "state": self.pair("continuation", 10),
                "direct": self.decision("direct"), "mtp": self.decision("mtp"),
            },
        }

    def blob(self, name, raw):
        (self.root / name).write_bytes(raw)
        return {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def pair(self, label, p):
        raw = snapshot((self.prompt + self.tokens)[:p])
        return {"position": p, "mtp_position": p, "byte_exact": True,
                **{mode: self.blob(f"{label}.{mode}.axkb", raw) for mode in ("direct", "mtp")}}

    def decision(self, mode):
        values = [0.0] * 16
        values[11] = 2.0
        return {"token": 11, "shape": [1, 16], "logits_f32_le": self.blob(
            f"continuation.{mode}.logits.f32", struct.pack("<16f", *values))}

    def verify(self):
        (self.root / "result.json").write_text(json.dumps(self.data))
        return reader.verify(self.root)

    def test_reconstructs_complete_state_and_continuation(self):
        result = self.verify()
        self.assertTrue(result["diagnostic_passed"])
        self.assertEqual(result["aligned_positions"], [5, 7, 9])
        self.assertFalse(result["qualification"])

    def test_rejects_missing_repeated_or_unaligned_state(self):
        original = copy.deepcopy(self.data)
        for change in (
            lambda d: d["aligned_states"].pop(),
            lambda d: d["aligned_states"].append(d["aligned_states"][-1]),
            lambda d: d["aligned_states"][0].update(mtp_position=6),
            lambda d: d.update(mtp_ids=[99] + self.tokens[1:]),
            lambda d: d.update(continuation=None),
            lambda d: d["continuation"].update(from_position=7),
            lambda d: d["continuation"].update(input_token=12),
            lambda d: d["mtp_positions"].append(9),
        ):
            with self.subTest(change=change):
                self.data = copy.deepcopy(original)
                change(self.data)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_false_pass_and_promotion_fail_closed(self):
        original = copy.deepcopy(self.data)
        for change in (
            lambda d: d.update(qualification=True),
            lambda d: d["mtp_certification"].update({"MTP-S": "passed"}),
            lambda d: d.update(mtp_routes=[]),
            lambda d: d["mtp_routes"].append([["ax_mlx_flash_next_mtp_step_errors", 1]]),
            lambda d: d["aligned_states"][0].update(byte_exact=1),
            lambda d: d.update(diagnostic_passed=1),
            lambda d: d["continuation"]["direct"].update(token=10),
            lambda d: d["continuation"].update(mtp_expected_token=10),
        ):
            with self.subTest(change=change):
                self.data = copy.deepcopy(original)
                change(self.data)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_blob_identity_and_structure_are_checked(self):
        pair = self.data["aligned_states"][0]
        record = pair["direct"]
        path = self.root / record["file"]
        raw = path.read_bytes()
        path.write_bytes(raw + b"x")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.verify()
        for malformed in (raw[:-1], raw + b"x", raw[:8] + struct.pack("<Q", 99) + raw[16:]):
            with self.subTest(size=len(malformed)):
                pair["direct"] = self.blob(record["file"], malformed)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_honest_state_difference_remains_failed_evidence(self):
        pair = self.data["aligned_states"][0]
        path = self.root / pair["mtp"]["file"]
        raw = bytearray(path.read_bytes())
        # First tensor data begins after the AXKB/layer/GDN/tensor headers.
        raw[90:94] = struct.pack("<f", 0.75)
        pair["mtp"] = self.blob(path.name, raw)
        pair["byte_exact"] = False
        self.data["diagnostic_passed"] = False
        result = self.verify()
        self.assertTrue(result["collection_valid"])
        self.assertFalse(result["diagnostic_passed"])

    def test_ple_history_is_newest_first_and_resets_on_eos(self):
        tokens = [1, 99, 3]
        raw = snapshot(tokens)
        self.assertEqual(len(reader.state_layout(raw, 3, tokens, [[99, 99], None])), 2)
        with self.assertRaisesRegex(ValueError, "PLE history"):
            reader.state_layout(raw, 3, [1, 2, 3], [[99, 99], None])

    def test_decision_logits_must_be_finite_and_match_shape(self):
        row = self.data["continuation"]["mtp"]
        for values in ([float("nan")] * 16, [float("inf")] * 16, [0.0] * 15):
            with self.subTest(values=values):
                row["logits_f32_le"] = self.blob("continuation.mtp.logits.f32", struct.pack(f"<{len(values)}f", *values))
                with self.assertRaises(ValueError):
                    self.verify()


if __name__ == "__main__":
    unittest.main()
