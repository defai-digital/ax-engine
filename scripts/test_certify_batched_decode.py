import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("certify_batched_decode.py")
MODULE_SPEC = importlib.util.spec_from_file_location("certify_batched_decode", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
certify = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["certify_batched_decode"] = certify
MODULE_SPEC.loader.exec_module(certify)


class CertifyBatchedDecodeTests(unittest.TestCase):
    def scenario(self, *, ragged: bool = False) -> dict[str, object]:
        return {
            "batch": 4,
            "prompt_len": 128,
            "gen_len": 64,
            "prompt_seed": 0,
            "ragged": ragged,
            "sampling": "greedy",
            "passed": True,
        }

    def context(self) -> dict[str, object]:
        return {
            "schema_version": "ax.mlx.batched_decode_certification.v1",
            "model_family": "qwen3",
            "artifact_fingerprint_sha256": "artifact",
            "engine_version": "6.9.0",
            "mlx_version": "0.29.3",
            "device_architecture": "applegpu_test",
            "runtime_contract": "ax.mlx.batched_decode.runtime.v2",
            "numerics_env_sha256": "environment",
            "required_scenarios": [self.scenario()],
        }

    def test_scenario_environment_sets_matrix_shape_and_clears_ragged(self) -> None:
        environment = certify.scenario_environment(
            {"AX_RAGGED": "1"}, self.scenario(ragged=False)
        )
        self.assertEqual(environment["AX_MLX_BATCHED_DECODE"], "1")
        self.assertEqual(environment["AX_BATCH"], "4")
        self.assertEqual(environment["AX_PROMPT_SEED"], "0")
        self.assertEqual(environment["AX_SAMPLING"], "greedy")
        self.assertNotIn("AX_RAGGED", environment)

    def test_scenario_verdict_requires_the_shared_forward(self) -> None:
        oracle_passes = "\n".join(
            [
                "HARNESS-FIDELITY: PASS",
                "BATCHED==SEQUENTIAL: PASS",
            ]
        )
        self.assertFalse(certify.scenario_passed(0, oracle_passes))
        self.assertTrue(
            certify.scenario_passed(0, "BATCHED-PATH: PASS\n" + oracle_passes)
        )

    def test_build_evidence_fails_when_any_scenario_fails(self) -> None:
        passing = self.scenario()
        failed = self.scenario(ragged=True)
        failed["passed"] = False
        evidence = certify.build_evidence(self.context(), [passing, failed])
        self.assertEqual(evidence["verdict"], "fail")

    def test_write_json_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            certify.write_json(output, {"verdict": "pass"})
            self.assertIn('"verdict": "pass"', output.read_text())
            self.assertFalse(output.with_name("evidence.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
