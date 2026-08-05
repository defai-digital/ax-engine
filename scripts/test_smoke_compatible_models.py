"""Offline tests for scripts/smoke_compatible_models.py."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import smoke_compatible_models as smoke  # noqa: E402


class MatrixValidationTests(unittest.TestCase):
    def test_curated_matrix_is_consistent_with_registry(self) -> None:
        self.assertEqual(smoke.validate_matrix(), [])

    def test_registry_parse_covers_matrix_families(self) -> None:
        tiers = smoke.parse_registry_tiers()
        for model in smoke.SMOKE_MATRIX:
            self.assertIn(model.family, tiers)
            self.assertEqual(tiers[model.family], model.tier, model.slug)

    def test_registry_parse_includes_known_tiers(self) -> None:
        tiers = smoke.parse_registry_tiers()
        self.assertEqual(tiers.get("qwen3"), "certified")
        self.assertEqual(tiers.get("diffusion_gemma"), "experimental")
        self.assertEqual(tiers.get("llama3"), "compatible")

    def test_duplicate_slugs_are_rejected(self) -> None:
        model = smoke.SMOKE_MATRIX[0]
        problems = smoke.validate_matrix([model, model])
        self.assertTrue(any("duplicate" in problem for problem in problems))

    def test_tier_mismatch_is_rejected(self) -> None:
        drifted = smoke.SmokeModel(
            slug="qwen3-4b",
            repo_id="mlx-community/Qwen3-4B-4bit",
            family="qwen3",
            tier="compatible",
        )
        problems = smoke.validate_matrix([drifted])
        self.assertTrue(any("!= registry tier" in problem for problem in problems))

    def test_unknown_family_is_rejected(self) -> None:
        drifted = smoke.SmokeModel(
            slug="mystery",
            repo_id="mlx-community/mystery-7b-4bit",
            family="not_a_family",
            tier="compatible",
        )
        problems = smoke.validate_matrix([drifted])
        self.assertTrue(any("not in ARCHITECTURE_REGISTRY" in p for p in problems))


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "smoke_compatible_models.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_list_exits_zero_and_mentions_tiers(self) -> None:
        completed = self.run_cli("--list")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("certified", completed.stdout)
        self.assertIn("compatible", completed.stdout)
        for model in smoke.SMOKE_MATRIX:
            self.assertIn(model.slug, completed.stdout)

    def test_dry_run_exits_zero(self) -> None:
        completed = self.run_cli("--dry-run")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("matrix validation: ok", completed.stdout)

    def test_models_filter_selects_subset(self) -> None:
        completed = self.run_cli("--models", "llama3.2-1b", "--list")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("llama3.2-1b", completed.stdout)
        self.assertNotIn("ministral-8b", completed.stdout)

    def test_unknown_models_slug_fails_loudly(self) -> None:
        completed = self.run_cli("--models", "nope", "--list")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("unknown --models slugs", completed.stderr)


class CoherenceAssertionTests(unittest.TestCase):
    def model(self) -> smoke.SmokeModel:
        return smoke.SMOKE_MATRIX[0]

    def response(self, content: object) -> dict:
        return {"choices": [{"message": {"content": content}}]}

    def test_non_empty_answer_passes(self) -> None:
        text = smoke.assert_coherent_content(
            self.response("Red, blue, and yellow."), self.model()
        )
        self.assertIn("Red", text)

    def test_empty_content_fails(self) -> None:
        with self.assertRaises(smoke.SmokeFailure):
            smoke.assert_coherent_content(self.response("   "), self.model())

    def test_missing_choices_fails(self) -> None:
        with self.assertRaises(smoke.SmokeFailure):
            smoke.assert_coherent_content({"choices": []}, self.model())

    def test_too_short_content_fails(self) -> None:
        with self.assertRaises(smoke.SmokeFailure):
            smoke.assert_coherent_content(self.response("ok"), self.model())


if __name__ == "__main__":
    unittest.main()
