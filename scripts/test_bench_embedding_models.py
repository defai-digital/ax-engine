#!/usr/bin/env python3
"""Unit tests for bench_embedding_models.py.

No companion test file previously existed; scope limited to the specific
bug fixed here rather than full coverage of the script.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_embedding_models.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_embedding_models", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


class RequiredBackendFailedTests(unittest.TestCase):
    def test_no_errors_is_not_a_required_failure(self) -> None:
        self.assertFalse(mod.required_backend_failed([]))

    def test_required_backend_failure_is_detected(self) -> None:
        self.assertTrue(mod.required_backend_failed(["mlx_lm: connection refused"]))
        self.assertTrue(mod.required_backend_failed(["ax_engine_py: OOM"]))

    def test_optional_backend_with_shared_name_prefix_is_not_required_failure(self) -> None:
        # Regression test: `"mlx_lm" in str(errors)` matched the optional
        # "mlx_lm_batched" backend's own label as a substring of "mlx_lm",
        # incorrectly treating its failure as a required-backend failure.
        self.assertFalse(mod.required_backend_failed(["mlx_lm_batched: OOM"]))
        self.assertFalse(mod.required_backend_failed(["ax_engine_py_batched: OOM"]))

    def test_optional_backend_failure_message_mentioning_required_name(self) -> None:
        # Regression test: a required backend's name appearing inside an
        # unrelated optional backend's error *message* text must not count.
        self.assertFalse(
            mod.required_backend_failed(["mlx_swift: try mlx_lm instead"])
        )

    def test_mixed_required_and_optional_failures(self) -> None:
        self.assertTrue(
            mod.required_backend_failed(["mlx_lm_batched: OOM", "mlx_lm: refused"])
        )


if __name__ == "__main__":
    unittest.main()
