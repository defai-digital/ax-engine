#!/usr/bin/env python3
"""Unit tests for the shared-prefix serving corpus builder."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("build_serving_shared_prefix_corpus.py")
MODULE_SPEC = importlib.util.spec_from_file_location("build_serving_shared_prefix_corpus", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)


class SharedPrefixServingCorpusTests(unittest.TestCase):
    def test_builder_writes_deterministic_shared_prefix_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "corpus.jsonl"

            with contextlib.redirect_stdout(io.StringIO()):
                code = builder.main_with_args_for_test(
                    [
                        "--output",
                        str(output),
                        "--prompts",
                        "3",
                        "--prefix-tokens",
                        "8",
                        "--suffix-tokens",
                        "4",
                        "--max-output-tokens",
                        "5",
                    ]
                )

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(code, 0)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["input_tokens_count"], 12)
            self.assertEqual(rows[0]["max_output_tokens"], 5)
            self.assertEqual(rows[0]["input_tokens"][:8], rows[1]["input_tokens"][:8])
            self.assertEqual(rows[1]["input_tokens"][:8], rows[2]["input_tokens"][:8])
            self.assertNotEqual(rows[0]["input_tokens"][8:], rows[1]["input_tokens"][8:])
            self.assertEqual(rows[2]["metadata"]["shared_prefix_tokens"], 8)
            self.assertEqual(rows[2]["metadata"]["purpose"], "disk_prefix_cache_serving_soak")

    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaisesRegex(Exception, "positive"):
            builder.positive_int("0")


if __name__ == "__main__":
    unittest.main()
