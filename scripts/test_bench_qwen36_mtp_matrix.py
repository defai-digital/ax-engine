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
        warmup_repetitions=1,
        cooldown=30.0,
        inter_case_cooldown=10.0,
        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        ax_python=Path("python3"),
        mtplx_python=Path("python3"),
        rapid_python=Path("python3"),
        lightning_source=root / "lightning-mlx",
        peer_caches=[root / "hf"],
        mtplx_profile="stable",
        lightning_mtp_draft_temperature=0.5,
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
        self.assertEqual(by_key[("27b-4bit", "lightning_mlx")], "unsupported")
        self.assertEqual(by_key[("35b-a3b-6bit", "mtplx")], "supported")
        self.assertEqual(by_key[("27b-6bit", "mtplx")], "unsupported")
        self.assertEqual(by_key[("35b-a3b-6bit", "lightning_mlx")], "unsupported")
        self.assertEqual(by_key[("27b-4bit", "rapid_mlx")], "unsupported")
        self.assertEqual(by_key[("27b-4bit", "omlx")], "unsupported")

    def test_ax_command_is_mtp_only_and_disables_ngram_stacking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            target = matrix.TARGETS["27b-6bit"]
            cmd = matrix.ax_command(args, target, "flappy", args.output_dir / "ax.json")

        self.assertIn("--ax-ngram-accel", cmd)
        self.assertIn("--ax-mtp-disable-ngram-stacking", cmd)
        self.assertIn("--ax-mtp-max-depth", cmd)
        self.assertEqual(cmd[cmd.index("--ax-mtp-max-depth") + 1], "3")
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
        self.assertEqual(cmd[cmd.index("--mtp-quant-mode") + 1], "cyankiwi")

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
