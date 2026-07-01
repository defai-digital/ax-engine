#!/usr/bin/env python3
"""Tests for the Qwen3.6 MTP benchmark matrix runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_qwen36_mtp_matrix.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_qwen36_mtp_matrix", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
matrix = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_qwen36_mtp_matrix"] = matrix
MODULE_SPEC.loader.exec_module(matrix)


def make_args(root: Path) -> Namespace:
    return Namespace(
        models=["27b", "35b-a3b"],
        bits=[4, 6],
        engines=["ax_engine", "mtplx", "lightning_mlx", "rapid_mlx", "omlx"],
        suites=["flappy"],
        suites_dir=root / "suites",
        output_dir=root / "out",
        hf_cache=root / "hf",
        max_tokens=1000,
        repetitions=5,
        warmup_repetitions=2,
        cooldown=30.0,
        inter_case_cooldown=10.0,
        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        seed=None,
        ax_python=Path("python3"),
        mtplx_python=Path("python3"),
        rapid_python=Path("python3"),
        rapid_source=root / "Rapid-MLX",
        lightning_source=root / "lightning-mlx",
        mtplx_source=root / "MTPLX",
        peer_caches=[root / "hf"],
        mtplx_profile="stable",
        benchmark_contract="apples-to-apples",
        prompt_limit=None,
        prefill_step_size=None,
        lightning_mtp_draft_temperature=0.5,
        lightning_mtp_optimistic=False,
        ax_mtp_optimistic=True,
        lightning_disable_prefix_cache=False,
        lightning_enable_prefix_cache=False,
        base_port=18765,
        no_build_ax_engine=True,
        skip_existing=False,
    )


class Qwen36MtpMatrixTests(unittest.TestCase):
    def test_support_matrix_keeps_only_known_pure_mtp_peer_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            lanes = matrix.build_lanes(args)

        by_key = {(lane.target.key, lane.engine): lane.status for lane in lanes}
        self.assertEqual(by_key[("27b-4bit", "ax_engine")], "supported")
        self.assertEqual(by_key[("27b-4bit", "mtplx")], "supported")
        self.assertEqual(by_key[("27b-4bit", "lightning_mlx")], "supported")
        self.assertEqual(by_key[("35b-a3b-6bit", "mtplx")], "supported")
        self.assertEqual(by_key[("27b-6bit", "mtplx")], "unsupported")
        self.assertEqual(by_key[("35b-a3b-6bit", "lightning_mlx")], "supported")
        self.assertEqual(by_key[("27b-6bit", "lightning_mlx")], "unsupported")
        self.assertEqual(by_key[("27b-4bit", "rapid_mlx")], "unsupported")
        self.assertEqual(by_key[("27b-6bit", "rapid_mlx")], "unsupported")
        self.assertEqual(by_key[("27b-4bit", "omlx")], "unsupported")

    def test_ax_env_sets_optimistic_on_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            env = matrix.ax_env(args)

        self.assertEqual(env["AX_MLX_MTP_OPTIMISTIC"], "1")

    def test_ax_env_disables_optimistic_when_flag_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            args.ax_mtp_optimistic = False
            env = matrix.ax_env(args)

        self.assertEqual(env["AX_MLX_MTP_OPTIMISTIC"], "0")

    def test_ax_command_is_mtp_only_and_disables_ngram_stacking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            target = matrix.TARGETS["27b-6bit"]
            cmd = matrix.ax_command(args, target, "flappy", args.output_dir / "ax.json")

        self.assertIn("--ax-ngram-accel", cmd)
        self.assertIn("--ax-mtp-disable-ngram-stacking", cmd)
        self.assertIn("--ax-mtp-max-depth", cmd)
        self.assertEqual(cmd[cmd.index("--ax-mtp-max-depth") + 1], "3")
        self.assertEqual(cmd[cmd.index("--warmup-repetitions") + 1], "2")
        self.assertNotIn("--ax-direct", cmd)

    def test_mtplx_command_allows_official_artifact_inspection_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            target = matrix.TARGETS["27b-4bit"]
            cmd = matrix.mtplx_command(
                args,
                target,
                "flappy",
                args.output_dir / "mtplx.json",
            )

        assert cmd is not None
        self.assertIn("--allow-unverified-model", cmd)
        self.assertIn("--ignore-eos", cmd)
        self.assertEqual(cmd[cmd.index("--mtp-quant-mode") + 1], "cyankiwi")
        self.assertEqual(cmd[cmd.index("--mtp-quant-policy") + 1], "prequantized-int4")

    def test_mtplx_env_prefers_reference_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root)
            args.mtplx_source.mkdir()

            env = matrix.mtplx_env(args)

        assert env is not None
        self.assertEqual(env["PYTHONPATH"].split(":")[0], str(args.mtplx_source))

    def test_rapid_command_uses_reference_source_and_lightning_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            matrix.RAPID_MODELS["35b-a3b-4bit"] = "local-model"
            target = matrix.TARGETS["35b-a3b-4bit"]
            try:
                cmd = matrix.rapid_command(
                    args,
                    target,
                    "flappy",
                    args.output_dir / "rapid.json",
                )
            finally:
                matrix.RAPID_MODELS.clear()

        assert cmd is not None
        self.assertEqual(cmd[cmd.index("--rapid-source") + 1], str(args.rapid_source))
        self.assertEqual(cmd[cmd.index("--lightning-source") + 1], str(args.lightning_source))
        self.assertEqual(cmd[cmd.index("--rapid-mtp-patch") + 1], "lightning")
        self.assertIn("--ignore-eos", cmd)
        self.assertIn("--require-full-output-tokens", cmd)
        self.assertNotIn("--lightning-mode", cmd)

    def test_peer_optimized_contract_applies_mtplx_and_lightning_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "suites"
            suite_dir.mkdir()
            suite_dir.joinpath("flappy.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"case-{idx}",
                            "category": "c",
                            "prompt": "hello",
                            "max_tokens": 512,
                        }
                    )
                    for idx in range(4)
                )
                + "\n"
            )
            args = make_args(root)
            args.benchmark_contract = "peer-optimized"
            args.max_tokens = 512
            args.prompt_limit = 3
            args.prefill_step_size = 8192
            args.mtplx_profile = "performance-cold"
            args.lightning_mtp_draft_temperature = None
            args.lightning_mtp_optimistic = True
            args.lightning_disable_prefix_cache = True
            target = matrix.TARGETS["27b-4bit"]
            lightning_cmd = matrix.lightning_command(
                args,
                target,
                "flappy",
                args.output_dir / "lightning.json",
            )
            mtplx_cmd = matrix.mtplx_command(
                args,
                target,
                "flappy",
                args.output_dir / "mtplx.json",
            )
            prompt_path = Path(lightning_cmd[lightning_cmd.index("--prompts") + 1])
            prompt_line_count = len(prompt_path.read_text().splitlines())

        assert lightning_cmd is not None
        assert mtplx_cmd is not None
        self.assertEqual(lightning_cmd[lightning_cmd.index("--max-tokens") + 1], "512")
        self.assertEqual(
            lightning_cmd[lightning_cmd.index("--prefill-step-size") + 1], "8192"
        )
        self.assertIn("--mtp-optimistic", lightning_cmd)
        self.assertIn("--disable-prefix-cache", lightning_cmd)
        self.assertNotIn("--mtp-draft-temperature", lightning_cmd)
        self.assertEqual(mtplx_cmd[mtplx_cmd.index("--max-tokens") + 1], "512")
        self.assertEqual(mtplx_cmd[mtplx_cmd.index("--profile") + 1], "performance-cold")
        self.assertEqual(prompt_line_count, 3)
        self.assertEqual(mtplx_cmd[mtplx_cmd.index("--prompts") + 1], str(prompt_path))

    def test_apples_to_apples_contract_disables_lightning_prefix_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))

            matrix.apply_benchmark_contract(args, [])

        self.assertTrue(args.lightning_disable_prefix_cache)

    def test_lightning_prefix_cache_can_be_enabled_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            args.lightning_enable_prefix_cache = True

            matrix.apply_benchmark_contract(args, ["--lightning-enable-prefix-cache"])

        self.assertFalse(args.lightning_disable_prefix_cache)

    def test_lightning_summary_flags_mtp_disabled_logs(self) -> None:
        artifact = {
            "server_log_tail": [
                "WARNING:rapid_mlx.scheduler:[MTP] MTP install skipped; request continues normally without MTP."
            ],
            "results": [{"prompt_id": "case-1", "runs": []}],
        }

        summary = matrix.summarize_lightning_artifact(artifact)

        self.assertEqual(summary["status"], "mtp_disabled")

    def test_summarize_ax_artifact_reports_all_required_metrics(self) -> None:
        artifact = {
            "results": [
                {
                    "engine": "ax_engine_mlx_pure_mtp",
                    "prompt_case_id": "case-1",
                    "decode_tok_s": {"median": 10.0, "values": [9.0, 11.0]},
                    "prefill_tok_s": {"median": 1000.0, "values": [900.0, 1100.0]},
                    "ttft_ms": {"median": 123.0, "values": [120.0, 126.0]},
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_accepted_tokens": 90,
                        "ax_mtp_draft_tokens": 100,
                    },
                    "trials": [
                        {
                            "output_token_ids": list(range(1000, 2000)),
                            "client_wall_ttft_ms": 500.0,
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ax.json"
            path.write_text(json.dumps(artifact))
            summary = matrix.summarize_ax_artifact(artifact, path)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["decode_tok_s"], 10.0)
        self.assertEqual(summary["prefill_tok_s"], 1000.0)
        self.assertEqual(summary["ttft_ms"], 123.0)
        self.assertEqual(summary["accept_rate"], 0.9)

    def test_summarize_ax_artifact_rejects_ngram_telemetry(self) -> None:
        artifact = {
            "results": [
                {
                    "engine": "ax_engine_mlx_pure_mtp",
                    "prompt_case_id": "case-1",
                    "decode_tok_s": {"median": 10.0},
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_accepted_tokens": 90,
                        "ax_mtp_draft_tokens": 100,
                        "ax_mtp_ngram_hit_steps": 1,
                    },
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "not pure MTP"):
            matrix.summarize_ax_artifact(artifact, Path("ax.json"))

    def test_mtplx_summary_derives_prefill_ttft_and_accept_rate(self) -> None:
        artifact = {
            "results": [
                {
                    "prompt_id": "case-1",
                    "prompt_tokens": 200,
                    "summary": {
                        "decode_tok_s": {"median": 20.0},
                        "accept_rate": 0.75,
                    },
                    "runs": [
                        {
                            "measured": True,
                            "prompt_eval_time_s": 0.2,
                        }
                    ],
                }
            ]
        }

        summary = matrix.summarize_mtplx_artifact(artifact)

        self.assertEqual(summary["decode_tok_s"], 20.0)
        self.assertEqual(summary["prefill_tok_s"], 1000.0)
        self.assertEqual(summary["ttft_ms"], 200.0)
        self.assertEqual(summary["accept_rate"], 0.75)

    def test_detect_degenerate_output_flags_whitespace_cycle(self) -> None:
        # 100 tokens of diverse content followed by 900 tokens of repeating 4-cycle.
        # The cycle covers 90% of output — well above the 50% threshold.
        diverse = list(range(1000, 1100))  # 100 unique tokens
        cycle = [248045, 554, 248046, 198] * 225  # 900 tokens
        token_ids = diverse + cycle

        result = matrix.detect_degenerate_output(token_ids)

        self.assertTrue(result["is_degenerate"])
        self.assertEqual(result["cycle_length"], 4)
        self.assertGreater(result["coverage"], 0.5)

    def test_detect_degenerate_output_flags_periodic_whitespace_cycle(self) -> None:
        token_ids = []
        cycle = [248046, 198, 248045, 554]
        for index in range(1000):
            if index % 2 == 0:
                token_ids.append(cycle[index % len(cycle)])
            else:
                token_ids.append(1000 + index)

        result = matrix.detect_degenerate_output(token_ids)

        self.assertTrue(result["is_degenerate"])
        self.assertEqual(result["method"], "periodic_cycle")
        self.assertLessEqual(result["cycle_length"], matrix.DEGENERACY_MAX_CYCLE_LEN)
        self.assertGreaterEqual(
            result["coverage"],
            matrix.DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
        )

    def test_detect_degenerate_output_passes_diverse_output(self) -> None:
        # 1000 diverse tokens — no repeating cycle.
        token_ids = list(range(1000, 2000))

        result = matrix.detect_degenerate_output(token_ids)

        self.assertFalse(result["is_degenerate"])

    def test_detect_degenerate_output_short_sequence_is_not_degenerate(self) -> None:
        token_ids = [1, 2, 3, 4, 5]

        result = matrix.detect_degenerate_output(token_ids)

        self.assertFalse(result["is_degenerate"])

    def test_check_ax_output_degeneracy_reports_per_case(self) -> None:
        artifact = {
            "results": [
                {
                    "engine": "ax_engine_mlx_pure_mtp",
                    "prompt_case_id": "case-1",
                    "trials": [
                        {
                            "output_token_ids": [248045, 554, 248046, 198] * 250,
                        }
                    ],
                }
            ]
        }

        result = matrix.check_ax_output_degeneracy(artifact)

        self.assertTrue(result["degenerate"])
        self.assertEqual(len(result["cases"]), 1)
        self.assertTrue(result["cases"][0]["is_degenerate"])

    def test_mtp_head_provenance_in_plan_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            args.engines = ["ax_engine", "mtplx"]
            lanes = matrix.build_lanes(args)
            matrix.write_plan(args, lanes)
            plan = json.loads((args.output_dir / "plan.json").read_text())

        provenance = plan["contract"]["mtp_head_provenance"]
        self.assertIn("ax_engine", provenance)
        self.assertIn("mtplx", provenance)
        self.assertIn("packaging", provenance["ax_engine"])
        self.assertIn("mtp_precision", provenance["mtplx"])

    def test_degeneracy_gate_config_in_plan_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            lanes = matrix.build_lanes(args)
            matrix.write_plan(args, lanes)
            plan = json.loads((args.output_dir / "plan.json").read_text())

        gate = plan["contract"]["degeneracy_gate"]
        self.assertEqual(gate["max_cycle_len"], matrix.DEGENERACY_MAX_CYCLE_LEN)
        self.assertEqual(gate["coverage_threshold"], matrix.DEGENERACY_COVERAGE_THRESHOLD)
        self.assertEqual(
            gate["periodic_coverage_threshold"],
            matrix.DEGENERACY_PERIODIC_COVERAGE_THRESHOLD,
        )

    def test_seed_in_plan_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            args.seed = 42
            lanes = matrix.build_lanes(args)
            matrix.write_plan(args, lanes)
            plan = json.loads((args.output_dir / "plan.json").read_text())

        self.assertEqual(plan["contract"]["seed"], 42)

    def test_lane_dict_includes_mtp_head_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            lanes = matrix.build_lanes(args)

        ax_lane = next(l for l in lanes if l.engine == "ax_engine")
        lane_dict = matrix.lane_to_dict(ax_lane)
        self.assertIn("mtp_head", lane_dict)
        self.assertIsNotNone(lane_dict["mtp_head"])
        self.assertIn("packaging", lane_dict["mtp_head"])

    def test_summarize_ax_artifact_reports_client_wall_ttft(self) -> None:
        artifact = {
            "results": [
                {
                    "engine": "ax_engine_mlx_pure_mtp",
                    "prompt_case_id": "case-1",
                    "decode_tok_s": {"median": 10.0, "values": [9.0, 11.0]},
                    "prefill_tok_s": {"median": 1000.0, "values": [900.0, 1100.0]},
                    "ttft_ms": {"median": 123.0, "values": [120.0, 126.0]},
                    "ngram_acceleration_telemetry": {
                        "ax_mtp_accepted_tokens": 90,
                        "ax_mtp_draft_tokens": 100,
                    },
                    "trials": [
                        {
                            "output_token_ids": list(range(1000, 2000)),
                            "client_wall_ttft_ms": 1503.0,
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ax.json"
            path.write_text(json.dumps(artifact))
            summary = matrix.summarize_ax_artifact(artifact, path)

        self.assertEqual(summary["client_wall_ttft_ms"], 1503.0)
        self.assertEqual(summary["ttft_scope"], "runner_internal")
        self.assertEqual(summary["client_ttft_scope"], "client_http_wall")
        self.assertFalse(summary["degeneracy_gate"]["degenerate"])

    def test_metric_contract_includes_ttft_scope(self) -> None:
        ax_contract = matrix.metric_contract("ax_engine")
        self.assertIn("ttft_scope", ax_contract)
        self.assertEqual(ax_contract["ttft_scope"], "runner_internal")
        self.assertIn("client_ttft_scope", ax_contract)

        lightning_contract = matrix.metric_contract("lightning_mlx")
        self.assertEqual(lightning_contract["ttft_scope"], "client_http_wall")

    def test_resolve_hf_snapshot_finds_cached_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = (
                root
                / "models--Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")

            resolved = matrix.resolve_hf_snapshot(
                "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
                [root],
            )

        self.assertEqual(resolved, snapshot)


if __name__ == "__main__":
    unittest.main()
