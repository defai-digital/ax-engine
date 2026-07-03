import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("check_decode_hot_path_kernel_admission.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_decode_hot_path_kernel_admission", SCRIPT_PATH
)
assert MODULE_SPEC is not None
checker = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def write(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def valid_promoted_manifest(candidate_id: str) -> dict:
    return {
        "schema_version": checker.SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "title": "Paged decode attention validation",
        "class": "phase1_metal",
        "status": "promoted",
        "production_default": True,
        "feature_flag": "AX_MLX_PAGED_DECODE_ATTENTION",
        "profile": {
            "path": "profile.md",
            "source": "AX_MLX_DECODE_PROFILE",
            "model": "mlx-community/example-4bit",
            "prompt_tokens": 2048,
            "generation_tokens": 128,
            "build_commit": "abc1234",
            "host": "macos-arm64",
            "dominant_stage": "SDPA",
            "stage_wall_share_pct": 31.5,
        },
        "mechanism": {
            "path": "mechanism.md",
            "removed_costs": ["dispatches/token", "KV materialization"],
            "why_mlx_cannot_remove": "AX owns the paged KV layout.",
        },
        "correctness": {
            "path": "correctness.md",
            "oracle": "current_mlx",
            "greedy_parity": "passed",
            "numeric_tolerance": 0.001,
        },
        "microbench": {
            "path": "microbench.json",
            "warmup_runs": 2,
            "measure_runs": 5,
            "host": "macos-arm64",
            "median_speedup_pct": 8.0,
        },
        "real_graph_ab": {
            "path": "e2e-summary.md",
            "baseline_row": "baseline p2048 g128",
            "candidate_row": "candidate p2048 g128",
            "decode_speedup_pct": 5.2,
            "prefill_tok_s": 1200.0,
            "ttft_ms": 120.0,
            "greedy_parity_passed": True,
        },
        "rollback": {
            "default_off": False,
            "fallback": "current MLX attention path",
            "kill_switch": "AX_DISABLE_MLX_PAGED_DECODE_ATTENTION",
            "telemetry_counters": [
                "ax_mlx_paged_decode_attention_attempts",
                "ax_mlx_paged_decode_attention_hits",
                "ax_mlx_paged_decode_attention_fallbacks",
            ],
        },
        "promotion": {
            "path": "promotion-decision.md",
            "decision": "promote",
            "reason": "E2E decode win exceeds the PRD threshold.",
        },
    }


class DecodeHotPathKernelAdmissionTests(unittest.TestCase):
    def test_empty_candidate_root_passes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checked = checker.check_candidates(Path(tmp) / "missing")

        self.assertEqual(checked, [])

    def test_valid_promoted_candidate_passes_full_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-paged-decode-attention"
            candidate_dir = root / candidate_id
            for filename in [
                "profile.md",
                "mechanism.md",
                "correctness.md",
                "microbench.json",
                "e2e-summary.md",
                "promotion-decision.md",
            ]:
                write(candidate_dir / filename)
            manifest = valid_promoted_manifest(candidate_id)
            write_json(candidate_dir / "candidate.json", manifest)

            checked = checker.check_candidates(root)

        self.assertEqual(checked[0]["candidate_id"], candidate_id)
        self.assertTrue(checked[0]["complete"])

    def test_promoted_candidate_requires_real_graph_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-low-speedup"
            candidate_dir = root / candidate_id
            for filename in [
                "profile.md",
                "mechanism.md",
                "correctness.md",
                "microbench.json",
                "e2e-summary.md",
                "promotion-decision.md",
            ]:
                write(candidate_dir / filename)
            manifest = valid_promoted_manifest(candidate_id)
            manifest["real_graph_ab"]["decode_speedup_pct"] = 2.9
            write_json(candidate_dir / "candidate.json", manifest)

            with self.assertRaisesRegex(
                checker.DecodeHotPathAdmissionError,
                "requires >=5% decode speedup",
            ):
                checker.check_candidates(root)

    def test_lower_threshold_requires_ttft_or_variance_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-lower-threshold"
            candidate_dir = root / candidate_id
            for filename in [
                "profile.md",
                "mechanism.md",
                "correctness.md",
                "microbench.json",
                "e2e-summary.md",
                "promotion-decision.md",
            ]:
                write(candidate_dir / filename)
            manifest = valid_promoted_manifest(candidate_id)
            manifest["real_graph_ab"]["decode_speedup_pct"] = 3.2
            manifest["real_graph_ab"]["improved_ttft_or_variance"] = True
            write_json(candidate_dir / "candidate.json", manifest)

            checked = checker.check_candidates(root)

        self.assertEqual(checked[0]["candidate_id"], candidate_id)

    def test_prototype_can_be_partial_but_must_stay_default_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-prototype"
            candidate_dir = root / candidate_id
            write_json(
                candidate_dir / "candidate.json",
                {
                    "schema_version": checker.SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "title": "Quantized projection feasibility",
                    "class": "mlx_sidecar",
                    "status": "prototype",
                    "production_default": False,
                    "feature_flag": "AX_MLX_QPROJ_FEASIBILITY",
                    "profile": {"source": "AX_MLX_DECODE_PROFILE"},
                    "mechanism": {"removed_costs": ["readback elimination"]},
                    "rollback": {
                        "default_off": True,
                        "fallback": "mlx_quantized_matmul",
                        "kill_switch": "AX_DISABLE_MLX_QPROJ_FEASIBILITY",
                        "telemetry_counters": [
                            "ax_mlx_qproj_attempts",
                            "ax_mlx_qproj_hits",
                            "ax_mlx_qproj_fallbacks",
                        ],
                    },
                    "promotion": {
                        "decision": "prototype",
                        "reason": "Admission evidence is still being collected.",
                    },
                },
            )

            checked = checker.check_candidates(root)

        self.assertFalse(checked[0]["complete"])

    def test_production_default_requires_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-bad-default"
            candidate_dir = root / candidate_id
            write_json(
                candidate_dir / "candidate.json",
                {
                    "schema_version": checker.SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "title": "Bad default",
                    "class": "mlx_sidecar",
                    "status": "prototype",
                    "production_default": True,
                    "feature_flag": "AX_MLX_BAD_DEFAULT",
                },
            )

            with self.assertRaisesRegex(
                checker.DecodeHotPathAdmissionError,
                "production_default=true requires status=promoted",
            ):
                checker.check_candidates(root)

    def test_require_complete_forces_partial_manifest_to_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_id = "2026-07-03-partial"
            candidate_dir = root / candidate_id
            write_json(
                candidate_dir / "candidate.json",
                {
                    "schema_version": checker.SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "title": "Partial",
                    "class": "graph_compile",
                    "status": "not_promoted",
                    "production_default": False,
                },
            )

            with self.assertRaisesRegex(
                checker.DecodeHotPathAdmissionError,
                "complete candidate is missing sections",
            ):
                checker.check_candidates(root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
