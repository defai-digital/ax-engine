from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts import bench_mtp_6bit_ax_refresh as bench


class BenchMtpRefreshTests(unittest.TestCase):
    def test_bench_command_records_two_warmups_and_greedy_sampler(self) -> None:
        target = bench.Target(
            key="test",
            label="Test",
            mode="MTP",
            model_dir=Path("/models/test"),
            mtp_depth=2,
        )
        args = argparse.Namespace(
            suites_dir=Path("/prompts"),
            generated_tokens=32,
            repetitions=5,
            warmup_repetitions=2,
            cooldown=0.0,
            inter_case_cooldown=0.0,
            approximate_speed_ceiling=False,
        )

        command = bench.bench_cmd(
            target=target,
            suite="sample",
            mode="mtp",
            output_path=Path("/tmp/result.json"),
            args=args,
        )

        warmup_index = command.index("--warmup-repetitions")
        sampling_index = command.index("--ax-sampling")
        self.assertEqual(command[warmup_index + 1], "2")
        self.assertEqual(
            command[sampling_index + 1],
            '{"temperature":0.0,"top_p":1.0,"top_k":0}',
        )

    def test_exact_artifact_validation_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an exact MTP publication candidate"):
            bench.validate_exact_mtp_artifact(Path("artifact.json"), {})

        bench.validate_exact_mtp_artifact(
            Path("artifact.json"),
            {"mtp_correctness_summary": {"publication_candidate": True}},
        )

    def test_approximate_artifact_is_explicit_and_non_publishable(self) -> None:
        artifact = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "publication_candidate": False,
                    "ax_mtp_correctness": {
                        "effective_mode": "approximate_optimistic"
                    },
                }
            ]
        }
        bench.validate_approximate_mtp_artifact(Path("artifact.json"), artifact)

        artifact["results"][0]["publication_candidate"] = True
        with self.assertRaisesRegex(ValueError, "incorrectly marks"):
            bench.validate_approximate_mtp_artifact(Path("artifact.json"), artifact)

    def test_exact_token_equivalence_oracle_rejects_mismatch(self) -> None:
        direct = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "trials": [{"output_token_ids": [1, 2, 3]}],
                }
            ]
        }
        mtp = {
            "results": [
                {
                    "prompt_case_id": "case",
                    "trials": [{"output_token_ids": [1, 2, 4]}],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "token-equivalence oracle failed"):
            bench.validate_exact_token_equivalence(
                Path("direct.json"), direct, Path("mtp.json"), mtp
            )

        mtp["results"][0]["trials"][0]["output_token_ids"] = [1, 2, 3]
        bench.validate_exact_token_equivalence(
            Path("direct.json"), direct, Path("mtp.json"), mtp
        )


if __name__ == "__main__":
    unittest.main()
