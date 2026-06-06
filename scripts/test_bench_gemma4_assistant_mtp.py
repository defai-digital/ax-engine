#!/usr/bin/env python3
"""Tests for bench_gemma4_assistant_mtp.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("bench_gemma4_assistant_mtp")


class Gemma4AssistantMtpBenchTests(unittest.TestCase):
    def test_parse_name_path_map(self) -> None:
        parsed = bench.parse_name_path_map(["26b-a4b-4bit=/tmp/g26", "31b-4bit=/tmp/g31"])
        self.assertEqual(parsed["26b-a4b-4bit"], Path("/tmp/g26"))
        self.assertEqual(parsed["31b-4bit"], Path("/tmp/g31"))

    def test_summarize_artifact_extracts_assistant_and_ngram_telemetry(self) -> None:
        artifact = {
            "build": {"git_tracked_dirty": False},
            "results": [
                {
                    "engine": "ax_engine_gemma4_assistant_mtp_ngram",
                    "decode_tok_s": {"median": 40.0},
                    "prefill_tok_s": {"median": 100.0},
                    "ttft_ms": {"median": 25.0},
                    "ax_decode_claim_status": "ngram_acceleration_effective_throughput",
                    "ax_decode_effective_route": "ngram_verified_bonus_tokens",
                    "ax_mlx_gemma4_assistant_mtp": {
                        "ax_mlx_gemma4_assistant_mtp_draft_tokens": 10,
                        "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 8,
                    },
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_draft_tokens": 12,
                        "ax_mtp_accepted_tokens": 9,
                        "ax_mtp_ngram_submitted_tokens": 4,
                        "ax_mtp_ngram_submitted_accepted_tokens": 3,
                        "ax_mtp_ngram_hit_steps": 2,
                    },
                },
                {
                    "engine": "ax_engine_gemma4_assistant_mtp_ngram",
                    "decode_tok_s": {"median": 50.0},
                    "prefill_tok_s": {"median": 120.0},
                    "ttft_ms": {"median": 20.0},
                    "ax_mlx_gemma4_assistant_mtp": {
                        "ax_mlx_gemma4_assistant_mtp_draft_tokens": 10,
                        "ax_mlx_gemma4_assistant_mtp_accepted_tokens": 7,
                    },
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_draft_tokens": 8,
                        "ax_mtp_accepted_tokens": 6,
                        "ax_mtp_ngram_submitted_tokens": 6,
                        "ax_mtp_ngram_submitted_accepted_tokens": 3,
                        "ax_mtp_ngram_hit_steps": 1,
                    },
                },
                {"engine": "unrelated", "decode_tok_s": {"median": 1.0}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps(artifact))
            summary = bench.summarize_artifact(
                path,
                expected_engine="ax_engine_gemma4_assistant_mtp_ngram",
            )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["decode_tok_s_median"], 45.0)
        self.assertEqual(summary["prefill_tok_s_median"], 110.0)
        self.assertEqual(summary["ttft_ms_median"], 22.5)
        self.assertEqual(summary["assistant_accept_rate"], 15 / 20)
        self.assertEqual(summary["mtp_accept_rate"], 15 / 20)
        self.assertEqual(summary["ngram_accept_rate"], 6 / 10)
        self.assertEqual(summary["ngram_hit_steps"], 3)

    def test_summarize_artifact_fails_closed_on_missing_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({"results": [{"engine": "other"}]}))
            with self.assertRaisesRegex(ValueError, "no rows"):
                bench.summarize_artifact(path, expected_engine="missing")


if __name__ == "__main__":
    unittest.main()
