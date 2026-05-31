#!/usr/bin/env python3
"""Tests for prepare_qwen36_mtp_sidecar.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("prepare_qwen36_mtp_sidecar.py")
MODULE_SPEC = importlib.util.spec_from_file_location("prepare_qwen36_mtp_sidecar", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
prepare = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["prepare_qwen36_mtp_sidecar"] = prepare
MODULE_SPEC.loader.exec_module(prepare)


class FakeArray:
    def __init__(self, values):
        self.values = values
        self.dtype = "fake"
        self.ndim = 1 if not isinstance(values[0], list) else 2

    def __add__(self, other: float):
        if self.ndim == 1:
            return FakeArray([value + other for value in self.values])
        return FakeArray([[value + other for value in row] for row in self.values])

    def astype(self, _dtype):
        return self

    def tolist(self):
        return self.values


class PrepareQwen36MtpSidecarTests(unittest.TestCase):
    def test_shift_norm_weights_matches_mlx_plus_one_convention(self) -> None:
        tensors = {
            "mtp.norm.weight": FakeArray([0.0, -0.5]),
            "mtp.layers.0.self_attn.q_norm.weight": FakeArray([0.25, -0.25]),
            "mtp.fc.weight": FakeArray([[2.0]]),
        }

        shifted = prepare._shift_norm_weights(tensors)

        self.assertEqual(shifted["mtp.norm.weight"].tolist(), [1.0, 0.5])
        self.assertEqual(shifted["mtp.layers.0.self_attn.q_norm.weight"].tolist(), [1.25, 0.75])
        self.assertEqual(shifted["mtp.fc.weight"].tolist(), [[2.0]])

    def test_runtime_contract_has_mtplx_required_fields(self) -> None:
        contract = prepare._runtime_contract(
            {
                "arch_id": "qwen3-next-mtp",
                "mtp_depth_max": 3,
                "mlx_community_repo": "mlx-community/Qwen3.6-27B-4bit",
            },
            15,
        )

        self.assertEqual(contract["arch_id"], "qwen3-next-mtp")
        self.assertEqual(contract["mtp_tensor_count"], 15)
        self.assertIn("mtplx_version", contract)
        self.assertIn("exactness_baseline", contract)
        self.assertIn("verified_on", contract)
        self.assertEqual(contract["exactness_baseline"]["max_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
