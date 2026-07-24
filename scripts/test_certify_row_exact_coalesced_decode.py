import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("certify_row_exact_coalesced_decode.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "certify_row_exact_coalesced_decode", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
certify = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["certify_row_exact_coalesced_decode"] = certify
MODULE_SPEC.loader.exec_module(certify)


class CertifyRowExactCoalescedDecodeTests(unittest.TestCase):
    def scenario(self, *, ragged: bool = False) -> dict[str, object]:
        return {
            "batch": 4,
            "prompt_len": 128,
            "gen_len": 64,
            "prompt_seed": 0,
            "ragged": ragged,
            "sampling": "greedy",
        }

    def context(self) -> dict[str, object]:
        return {
            "model_family": "qwen3_5",
            "artifact_fingerprint_sha256": "artifact",
            "engine_version": "6.11.1",
            "mlx_version": "0.32.0",
            "device_architecture": "applegpu_test",
            "numerics_env_sha256": "environment",
        }

    def test_environment_forces_fail_closed_row_exact_route(self) -> None:
        environment = certify.scenario_environment(
            {
                "AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED": "1",
                "AX_RAGGED": "1",
            },
            self.scenario(),
            coalesced=True,
        )
        self.assertEqual(environment["AX_MLX_BATCHED_DECODE"], "1")
        self.assertNotIn("AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED", environment)
        self.assertNotIn("AX_RAGGED", environment)

    def test_verdict_requires_row_exact_route_and_both_oracles(self) -> None:
        oracle = "HARNESS-FIDELITY: PASS\nBATCHED==SEQUENTIAL: PASS\n"
        self.assertTrue(
            certify.coalesced_passed(
                0, oracle + "ROW-EXACT-COALESCED-PATH: PASS\n"
            )
        )
        self.assertFalse(
            certify.coalesced_passed(
                0,
                oracle
                + "ROW-EXACT-COALESCED-PATH: PASS\n"
                + "BATCHED-PATH: PASS\n",
            )
        )

    def test_decode_metrics_are_parsed(self) -> None:
        metrics = certify.parse_decode_metrics(
            "decode: 2.704s for 256 tokens = 94.7 agg tok/s (batched_flag true)"
        )
        self.assertEqual(metrics["tokens"], 256)
        self.assertEqual(metrics["tokens_per_second"], 94.7)

    def test_evidence_is_fail_closed(self) -> None:
        passing = self.scenario()
        passing["passed"] = True
        failing = self.scenario(ragged=True)
        failing["passed"] = False
        evidence = certify.build_evidence(self.context(), [passing, failing])
        self.assertEqual(evidence["verdict"], "fail")

    def test_write_json_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            certify.write_json(output, {"verdict": "pass"})
            self.assertIn('"verdict": "pass"', output.read_text())
            self.assertFalse(output.with_name("evidence.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
