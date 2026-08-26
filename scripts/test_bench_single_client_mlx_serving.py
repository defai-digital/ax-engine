#!/usr/bin/env python3
"""Tests for the single-client AX vs peer serving benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("bench_single_client_mlx_serving.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_single_client_mlx_serving", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = benchmark
MODULE_SPEC.loader.exec_module(benchmark)


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
            omlx = benchmark.EngineSpec("omlx", root / "omlx")
            mtplx = benchmark.EngineSpec("mtplx", root / "mtplx")

            ax_command = benchmark.engine_command(ax, model=model, port=31910)
            peer_command = benchmark.engine_command(
                peer,
                model=model,
                port=31910,
                mlxcel_draft_model=root / "mtp-drafter",
            )
            omlx_command = benchmark.engine_command(omlx, model=model, port=31910)
            mtplx_command = benchmark.engine_command(
                mtplx,
                model=model,
                port=31910,
                mtplx_force_unverified=True,
            )

        self.assertIn("--max-concurrent-requests", ax_command)
        self.assertIn("--max-concurrent-requests-per-model", ax_command)
        self.assertIn("--parallel", peer_command)
        self.assertIn("--max-batch-prefill", peer_command)
        self.assertIn("--no-prompt-cache", peer_command)
        self.assertEqual(
            peer_command[peer_command.index("--model-draft") + 1],
            str(root / "mtp-drafter"),
        )
        self.assertEqual(peer_command[peer_command.index("--draft-kind") + 1], "mtp")
        self.assertEqual(peer_command[peer_command.index("--draft-block-size") + 1], "3")
        self.assertIn("--memory-guard", omlx_command)
        self.assertIn("--no-cache", omlx_command)
        self.assertIn("--scheduler-mode", mtplx_command)
        self.assertIn("--ssd-session-cache", mtplx_command)
        self.assertIn("--unsafe-force-unverified", mtplx_command)
        self.assertIn("--yes", mtplx_command)

    def test_three_engine_order_rotates_first_position(self) -> None:
        engines = ("ax-engine", "omlx", "mtplx")
        orders = [benchmark.engine_order(0, rep, engines) for rep in range(3)]

        self.assertEqual([order[0] for order in orders], list(engines))
        self.assertTrue(all(sorted(order) == sorted(engines) for order in orders))

    def test_engine_model_override_preserves_logical_label(self) -> None:
        canonical = benchmark.ModelSpec("qwen", Path("/canonical"))
        override = Path("/runtime-view")

        selected = benchmark.model_for_engine(
            canonical, "omlx", {("omlx", "qwen"): override}
        )

        self.assertEqual(selected, benchmark.ModelSpec("qwen", override))
        self.assertEqual(
            benchmark.model_for_engine(canonical, "mtplx", {}),
            canonical,
        )

    def test_configure_omlx_mtp_persists_isolated_model_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = benchmark.ModelSpec("qwen", root / "models" / "local")

            settings_path = benchmark.configure_omlx_mtp(model)
            settings = json.loads(settings_path.read_text(encoding="utf-8"))

        self.assertEqual(settings_path, root / ".omlx-benchmark-state/model_settings.json")
        self.assertTrue(settings["models"]["local"]["mtp_enabled"])
        self.assertEqual(settings["models"]["local"]["mtp_num_draft_tokens"], 3)

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

    def test_prometheus_metrics_capture_unlabelled_delta_and_latest_gauge(self) -> None:
        parsed = benchmark.parse_unlabelled_prometheus_metrics(
            """# HELP ignored comment
ax_engine_steps_total{model=\"local\"} 19
ax_engine_steps_total 20
ax_engine_mtp_draft_tokens_total 30
ax_engine_mtp_accepted_tokens_total 21
ax_engine_mtp_accept_rate_ewma_x1000 777
not_a_number nope
"""
        )
        self.assertEqual(parsed["ax_engine_steps_total"], 20.0)
        self.assertNotIn('ax_engine_steps_total{model="local"}', parsed)

        delta = benchmark.ax_metric_delta(
            {
                "ax_engine_steps_total": 5.0,
                "ax_engine_mtp_draft_tokens_total": 7.0,
                "ax_engine_mtp_accepted_tokens_total": 4.0,
            },
            parsed,
        )
        self.assertEqual(delta["ax_engine_steps_total"], 15.0)
        self.assertEqual(delta["ax_engine_mtp_draft_tokens_total"], 23.0)
        self.assertEqual(delta["ax_engine_mtp_accepted_tokens_total"], 17.0)
        self.assertEqual(delta["ax_engine_mtp_accept_rate_ewma_x1000"], 777.0)

    def test_quality_smoke_suite_has_general_and_coding_objective_checks(self) -> None:
        dataset = (
            MODULE_PATH.parents[1]
            / "benchmarks/datasets/cross-runtime-quality-smoke-v1.jsonl"
        )

        tasks = benchmark.load_quality_tasks(dataset)

        self.assertEqual(
            {task.profile for task in tasks},
            {"agent-coding", "general"},
        )
        general = next(task for task in tasks if task.profile == "general")
        coding = next(task for task in tasks if task.profile == "agent-coding")
        general_score, general_checks = benchmark.score_quality_task(
            general, "</think>\n5<|eot|>ignored"
        )
        coding_score, coding_checks = benchmark.score_quality_task(
            coding, "```json\n[3, 1, 2]\n```"
        )
        self.assertEqual(general_score, 1.0)
        self.assertEqual(general_checks, {"exact:0": 1.0})
        self.assertEqual(coding_score, 1.0)
        self.assertEqual(coding_checks, {"json-valid:0": 1.0, "json-equals:1": 1.0})

    def test_quality_summary_tracks_passes_and_output_determinism(self) -> None:
        measurements = [
            {
                "engine": "ax-engine",
                "model": "qwen",
                "profile": "general",
                "task_id": "general-01",
                "score": 1.0,
                "passed": True,
                "content_sha256": "same",
            },
            {
                "engine": "ax-engine",
                "model": "qwen",
                "profile": "general",
                "task_id": "general-01",
                "score": 1.0,
                "passed": True,
                "content_sha256": "same",
            },
            {
                "engine": "omlx",
                "model": "qwen",
                "profile": "general",
                "task_id": "general-01",
                "score": 1.0,
                "passed": True,
                "content_sha256": "different-format",
            },
        ]

        summary = benchmark.summarize_quality_measurements(measurements)
        consensus = benchmark.summarize_quality_consensus(measurements)

        self.assertTrue(summary[0]["all_pass"])
        self.assertTrue(summary[0]["deterministic_across_repetitions"])
        self.assertTrue(consensus[0]["all_runtimes_pass"])
        self.assertFalse(consensus[0]["exact_output_match"])


if __name__ == "__main__":
    unittest.main()
