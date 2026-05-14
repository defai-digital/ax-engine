#!/usr/bin/env python3
"""Unit tests for scripts.update_readme_from_results."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("update_readme_from_results.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "update_readme_from_results", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
updater = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = updater
MODULE_SPEC.loader.exec_module(updater)


def metric(median: float) -> dict[str, float]:
    return {"median": median}


class UpdateReadmeFromResultsTests(unittest.TestCase):
    def test_prefill_table_bolds_faster_ax_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            (results_dir / "gemma-4-e2b-it-4bit.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 128,
                                "prefill_tok_s": metric(100.0),
                                "decode_tok_s": metric(10.0),
                            },
                            {
                                "engine": "ax_engine_mlx",
                                "prompt_tokens": 128,
                                "prefill_tok_s": metric(120.0),
                                "decode_tok_s": metric(12.0),
                                "ttft_ms": metric(1066.7),
                            },
                            {
                                "engine": "mlx_lm",
                                "prompt_tokens": 512,
                                "prefill_tok_s": metric(100.0),
                                "decode_tok_s": metric(10.0),
                            },
                            {
                                "engine": "ax_engine_mlx",
                                "prompt_tokens": 512,
                                "prefill_tok_s": metric(80.0),
                                "decode_tok_s": metric(8.0),
                                "ttft_ms": metric(6400.0),
                            },
                        ]
                    }
                )
                + "\n"
            )

            stdout = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "update_readme_from_results.py",
                        "--results-dir",
                        str(results_dir),
                    ],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                updater.main()

        prefill_table = stdout.getvalue().split("DECODE TABLE", maxsplit=1)[0]

        self.assertIn("**120.0 (+20.0%)**", prefill_table)
        self.assertIn("|        |        | 512 | 100.0 | — | 80.0 (-20.0%) |", prefill_table)
        self.assertNotIn("**80.0 (-20.0%)**", prefill_table)


if __name__ == "__main__":
    unittest.main()
