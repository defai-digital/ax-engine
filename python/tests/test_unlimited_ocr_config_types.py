from __future__ import annotations

import unittest

from test_unlimited_ocr import load_module


class SparseConfigTypeTests(unittest.TestCase):
    def test_boolean_and_numeric_scopes_conflict(self) -> None:
        module = load_module()
        for first, second in ((True, 1), (False, 0), (1, 1.0), ([True], [1])):
            for scope in ("language_config", "text_config"):
                with self.subTest(first=first, second=second, scope=scope):
                    with self.assertRaisesRegex(ValueError, "conflicting sparse attention scopes"):
                        module._sparse_attention_config({
                            "sparse_attention_config": {"value": first},
                            scope: {"sparse_attention_config": {"value": second}},
                        })

    def test_equal_scopes_allow_reordered_keys_and_signed_float_zero(self) -> None:
        module = load_module()
        sparse = {"enabled": True, "values": [1, 2.0, {"zero": -0.0}]}
        same = {"values": [1, 2.0, {"zero": 0.0}], "enabled": True}
        self.assertEqual(module._sparse_attention_config({
            "sparse_attention_config": sparse,
            "text_config": {"sparse_attention_config": same},
        }), sparse)
