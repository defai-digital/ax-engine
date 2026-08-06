#!/usr/bin/env python3
"""Tests for the single-client AX vs peer serving benchmark."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench_single_client_mlx_serving as benchmark


class SingleClientServingBenchmarkTests(unittest.TestCase):
    def test_deterministic_prompt_is_stable_and_rep_specific(self) -> None:
        seed_a, prompt_a = benchmark.deterministic_prompt(512, 0)
        seed_b, prompt_b = benchmark.deterministic_prompt(512, 0)
        seed_c, prompt_c = benchmark.deterministic_prompt(512, 1)

        self.assertEqual(seed_a, 151_200)
        self.assertEqual((seed_a, prompt_a), (seed_b, prompt_b))
        self.assertNotEqual(seed_a, seed_c)
        self.assertNotEqual(prompt_a, prompt_c)

    def test_engine_order_is_balanced_across_four_models(self) -> None:
        first_positions = [
            benchmark.engine_order(model_index, rep)[0]
            for model_index in range(4)
            for rep in range(3)
        ]
        self.assertEqual(first_positions.count("ax-engine"), 6)
        self.assertEqual(first_positions.count("mlxcel"), 6)

    def test_sse_decoder_handles_json_and_done(self) -> None:
        with mock.patch.object(benchmark.time, "perf_counter", side_effect=(11.5, 13.0)):
            frames = list(
                benchmark.decode_sse_frames(
                    [
                        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
                        b"\n",
                        b"data: [DONE]\n",
                        b"\n",
                    ],
                    started_at=10.0,
                )
            )
        self.assertEqual(frames[0][0]["choices"][0]["delta"]["content"], "hi")
        self.assertEqual(frames[0][1], 1.5)
        self.assertIsNone(frames[1][0])
        self.assertEqual(frames[1][1], 3.0)

    def test_engine_commands_pin_single_client_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = benchmark.ModelSpec("test", root)
            ax = benchmark.EngineSpec("ax-engine", root / "ax-engine-server")
            peer = benchmark.EngineSpec("mlxcel", root / "mlxcel-server")

            ax_command = benchmark.engine_command(ax, model=model, port=31910)
            peer_command = benchmark.engine_command(peer, model=model, port=31910)

        self.assertIn("--max-concurrent-requests", ax_command)
        self.assertIn("--max-concurrent-requests-per-model", ax_command)
        self.assertIn("--parallel", peer_command)
        self.assertIn("--max-batch-prefill", peer_command)

    def test_hardware_profile_omits_machine_identifiers(self) -> None:
        profile = """Hardware:
      Model Name: MacBook Pro
      Chip: Apple M5 Max
      Serial Number (system): secret
      Hardware UUID: secret
      Provisioning UDID: secret
      Activation Lock Status: Enabled
"""
        sanitized = benchmark.sanitize_hardware_profile(profile)

        self.assertIn("Model Name: MacBook Pro", sanitized)
        self.assertIn("Chip: Apple M5 Max", sanitized)
        self.assertNotIn("secret", sanitized)
        self.assertNotIn("Activation Lock Status", sanitized)

    def test_fixed_generation_rejects_early_eos(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 256.*observed 42"):
            benchmark.require_complete_generation({"completion_tokens": 42}, 256)


if __name__ == "__main__":
    unittest.main()
