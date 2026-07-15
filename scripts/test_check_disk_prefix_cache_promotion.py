#!/usr/bin/env python3
"""Unit tests for check_disk_prefix_cache_promotion.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check_disk_prefix_cache_promotion.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_disk_prefix_cache_promotion", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(checker)


def base_artifact(**overrides):
    data = {
        "schema": checker.SCHEMA,
        "run_id": "test",
        "ax_commit": "abc",
        "status": "complete",
        "modes": {
            "cold_prefill": {},
            "l1_hit": {},
            "l2_hit_warm_fs": {"filesystem_cache_state": "warm_fs"},
            "l2_hit_cold_fs": {"filesystem_cache_state": "cold_fs"},
            "producer_l2_enabled": {},
        },
        "models": ["gemma"],
        "correctness": {
            "deterministic_match": True,
            "wrong_prefix_hits": 0,
            "corrupt_restores": 0,
        },
        "performance_gates": {
            "decision": "promote",
            "admitted_bucket_improvements": [
                {
                    "prefix_bucket": "8k",
                    "cold_prefill_p95_ttft_ms": 100.0,
                    "l2_hit_p95_ttft_ms": 70.0,
                },
                {
                    "prefix_bucket": "32k",
                    "cold_prefill_p95_ttft_ms": 400.0,
                    "l2_hit_p95_ttft_ms": 280.0,
                },
            ],
        },
    }
    data.update(overrides)
    return data


class CheckDiskPrefixCachePromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name: str, obj: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def test_schema_only_incomplete(self) -> None:
        path = self.write(
            "incomplete.json",
            base_artifact(status="incomplete", performance_gates={"decision": "not_promoted"}),
        )
        result = checker.check_disk_prefix_cache_promotion(
            path, require_performance=False, allow_incomplete=True
        )
        self.assertEqual(result["status"], "incomplete")

    def test_promote_passes(self) -> None:
        path = self.write("ok.json", base_artifact())
        result = checker.check_disk_prefix_cache_promotion(path)
        self.assertEqual(result["status"], "ok")

    def test_loss_to_cold_fails(self) -> None:
        art = base_artifact()
        art["performance_gates"]["admitted_bucket_improvements"][0]["l2_hit_p95_ttft_ms"] = 120.0
        path = self.write("lose.json", art)
        with self.assertRaises(checker.DiskPrefixCachePromotionError) as ctx:
            checker.check_disk_prefix_cache_promotion(path)
        self.assertEqual(ctx.exception.exit_code, 4)

    def test_wrong_schema_fails(self) -> None:
        path = self.write("bad.json", base_artifact(schema="wrong"))
        with self.assertRaises(checker.DiskPrefixCachePromotionError):
            checker.check_disk_prefix_cache_promotion(path, require_performance=False)


if __name__ == "__main__":
    unittest.main()
