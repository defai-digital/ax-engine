"""Tests for stability-critical GitHub Actions workflow policy."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class CiWorkflowPolicyTests(unittest.TestCase):
    def test_model_smoke_is_required_on_protected_refs(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn("AX_MODEL_SMOKE_REQUIRED", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("startsWith(github.ref, 'refs/heads/release/')", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertIn("inputs.mlx_model_artifacts_dir != ''", workflow)
        self.assertIn("vars.AX_REQUIRE_MODEL_SMOKE == '1'", workflow)

    def test_missing_required_model_artifacts_fail_closed(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn('if [[ "$AX_MODEL_SMOKE_REQUIRED" == "true" ]]; then', workflow)
        self.assertIn(
            "MLX model artifacts are required for this ref/event but were not mounted.",
            workflow,
        )
        self.assertIn("Real-model smoke checks are required for this ref/event.", workflow)
        self.assertIn("exit 1", workflow)

    def test_missing_optional_model_artifacts_remain_skippable(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn(
            "MLX model artifacts not mounted; skipping optional real-model smoke checks.",
            workflow,
        )
        self.assertIn(
            "optional model-dependent smoke checks skipped",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
