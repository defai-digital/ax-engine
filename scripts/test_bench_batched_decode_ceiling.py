#!/usr/bin/env python3
"""Tests for scripts.bench_batched_decode_ceiling."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_batched_decode_ceiling.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_batched_decode_ceiling",
    SCRIPT_PATH,
)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(bench)


def conditions() -> dict[str, object]:
    return {
        "load_average": {"one_minute": 0.5},
        "power_source": "AC Power",
        "thermal_warning_recorded": False,
        "performance_warning_recorded": False,
        "cpu_power_status_recorded": False,
        "top_processes_cpu": [{"cpu_percent": 0.1, "command": "test"}],
    }


def rows(*, batch8_rate: float = 100.0) -> list[dict[str, object]]:
    return [
        {
            "batch": 1,
            "aggregate_tok_s": 50.0,
            "per_request_tok_s": 50.0,
            "step_us": 20_000.0,
            "scaling_vs_batch1": 1.0,
            "cohort_fnv": "1111111111111111",
        },
        {
            "batch": 2,
            "aggregate_tok_s": 75.0,
            "per_request_tok_s": 37.5,
            "step_us": 26_666.7,
            "scaling_vs_batch1": 1.5,
            "cohort_fnv": "2222222222222222",
        },
        {
            "batch": 4,
            "aggregate_tok_s": 90.0,
            "per_request_tok_s": 22.5,
            "step_us": 44_444.4,
            "scaling_vs_batch1": 1.8,
            "cohort_fnv": "4444444444444444",
        },
        {
            "batch": 8,
            "aggregate_tok_s": batch8_rate,
            "per_request_tok_s": batch8_rate / 8.0,
            "step_us": 8_000_000.0 / batch8_rate,
            "scaling_vs_batch1": batch8_rate / 50.0,
            "cohort_fnv": "8888888888888888",
        },
    ]


def complete_artifact() -> dict[str, object]:
    trials = []
    for repetition in range(1, 6):
        order = (
            bench.POLICIES
            if repetition % 2 == 1
            else tuple(reversed(bench.POLICIES))
        )
        for policy in order:
            policy_value = "1" if policy == "shared" else "0"
            trials.append(
                {
                    "repetition": repetition,
                    "policy": policy,
                    "command": [
                        "/tmp/probe",
                        "/tmp/model",
                        "32",
                    ],
                    "environment": {
                        "AX_MLX_BATCHED_SHARED_PROJ": policy_value,
                        "AX_MLX_BATCHED_PROFILE": "0",
                    },
                    "performance_conditions_start": conditions(),
                    "performance_conditions_end": conditions(),
                    "rows": rows(
                        batch8_rate=100.0 if policy == "shared" else 65.0
                    ),
                }
            )
    return {
        "schema_version": bench.SCHEMA_VERSION,
        "status": "complete",
        "prefill_len": 32,
        "probe_contract": {
            "batches": [1, 2, 4, 8],
            "warmup_steps_per_batch": 8,
            "measured_steps_per_batch": 64,
            "timing_scope": "internal_batched_decode_step_wall",
        },
        "repetitions": 5,
        "cooldown_seconds": 15.0,
        "trial_order": bench.TRIAL_ORDER,
        "max_load_average": bench.DEFAULT_MAX_LOAD_AVERAGE,
        "max_top_process_cpu_percent": (
            bench.DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT
        ),
        "build": {
            "commit": "a" * 40,
            "engine_version": "6.13.2",
            "build_profile": "release",
            "git_tracked_dirty": False,
            "benchmark_binary": "/tmp/probe",
            "benchmark_binary_sha256": "b" * 64,
        },
        "host": {"chip": "Apple M5 Max"},
        "model": {
            "path": "/tmp/model",
            "manifest_sha256": "c" * 64,
        },
        "trials": trials,
    }


class BatchedDecodeCeilingTests(unittest.TestCase):
    def test_parse_probe_output_accepts_complete_matrix(self) -> None:
        output = """\
model: /tmp/model  prefill_len=32
batch  agg_tok_s  per_req_tok_s  step_us  scaling_vs_b1
    1       50.0           50.0    20000   1.00x  cohort_fnv=1111111111111111
    2       75.0           37.5    26667   1.50x  cohort_fnv=2222222222222222
    4       90.0           22.5    44444   1.80x  cohort_fnv=4444444444444444
    8      100.0           12.5    80000   2.00x  cohort_fnv=8888888888888888
"""

        parsed = bench.parse_probe_output(output)

        self.assertEqual([row["batch"] for row in parsed], [1, 2, 4, 8])
        self.assertEqual(parsed[-1]["cohort_fnv"], "8888888888888888")

    def test_parse_probe_output_rejects_partial_matrix(self) -> None:
        with self.assertRaisesRegex(
            bench.BatchedDecodeBenchmarkError,
            "batch matrix mismatch",
        ):
            bench.parse_probe_output(
                "1 50.0 50.0 20000 1.00x cohort_fnv=1111111111111111"
            )

    def test_summary_reports_paired_batch8_ratio(self) -> None:
        artifact = complete_artifact()

        summary = bench.summarize_trials(artifact["trials"])

        paired = summary["paired_batch8_shared_over_row_exact"]
        self.assertAlmostEqual(paired["median_ratio"], 100.0 / 65.0)
        self.assertEqual(paired["wins"], 5)
        self.assertEqual(paired["losses"], 0)

    def test_publication_gate_accepts_complete_evidence(self) -> None:
        artifact = complete_artifact()
        artifact["summary"] = bench.summarize_trials(artifact["trials"])

        self.assertEqual(bench.publication_reasons(artifact), [])

    def test_publication_gate_rejects_cross_policy_token_divergence(self) -> None:
        artifact = complete_artifact()
        artifact["trials"][0]["rows"][-1]["cohort_fnv"] = "9999999999999999"

        reasons = bench.publication_reasons(artifact)

        self.assertIn("batch8_cohort_hash_divergence", reasons)

    def test_publication_gate_rejects_relaxed_load_or_blocked_order(self) -> None:
        artifact = complete_artifact()
        artifact["max_load_average"] = 5.0
        artifact["trials"][0], artifact["trials"][1] = (
            artifact["trials"][1],
            artifact["trials"][0],
        )

        reasons = bench.publication_reasons(artifact)

        self.assertIn("publication_requires_default_load_gate", reasons)
        self.assertIn("trial_sequence_mismatch", reasons)

    def test_policy_environment_removes_inherited_ax_mlx_flags(self) -> None:
        original = bench.os.environ.get("AX_MLX_DENSE_FFN_COMPILE")
        bench.os.environ["AX_MLX_DENSE_FFN_COMPILE"] = "1"
        try:
            env = bench.policy_environment("row_exact")
        finally:
            if original is None:
                del bench.os.environ["AX_MLX_DENSE_FFN_COMPILE"]
            else:
                bench.os.environ["AX_MLX_DENSE_FFN_COMPILE"] = original

        self.assertNotIn("AX_MLX_DENSE_FFN_COMPILE", env)
        self.assertEqual(env["AX_MLX_BATCHED_SHARED_PROJ"], "0")


if __name__ == "__main__":
    unittest.main()
