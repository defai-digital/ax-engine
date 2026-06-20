#!/usr/bin/env python3
"""Unit tests for the direct model I/O smoke matrix."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_direct_model_io.py")
MODULE_SPEC = importlib.util.spec_from_file_location("check_direct_model_io", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
check_direct_model_io = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = check_direct_model_io
MODULE_SPEC.loader.exec_module(check_direct_model_io)


class DirectModelIoMatrixTests(unittest.TestCase):
    def test_direct_model_cases_follow_supported_direct_llm_families(self) -> None:
        slugs = {case.slug for case in check_direct_model_io.MODEL_CASES}

        self.assertIn("gemma-4-e2b-it-4bit", slugs)
        self.assertIn("gemma-4-e4b-it-4bit", slugs)
        self.assertIn("gemma-4-26b-a4b-it-4bit", slugs)
        self.assertIn("gemma-4-31b-it-4bit", slugs)
        self.assertIn("qwen3-4b-4bit", slugs)
        self.assertIn("qwen3-5-9b-mlx-4bit", slugs)
        self.assertIn("qwen3-6-27b-4bit", slugs)
        self.assertIn("qwen3-6-35b-a3b-4bit", slugs)
        self.assertIn("qwen3-coder-next-4bit", slugs)
        self.assertIn("glm-4-7-flash-4bit", slugs)

    def test_glm_is_a_direct_model_io_case(self) -> None:
        slugs = {case.slug for case in check_direct_model_io.MODEL_CASES}
        model_ids = {case.model_id for case in check_direct_model_io.MODEL_CASES}

        self.assertIn("glm-4-7-flash-4bit", slugs)
        self.assertIn("mlx-community/GLM-4.7-Flash-4bit", model_ids)


if __name__ == "__main__":
    unittest.main()
