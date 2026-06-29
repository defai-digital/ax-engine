"""Tests for stability-critical GitHub Actions workflow policy."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
NATIVE_DEPS_SCRIPT = ROOT / "scripts" / "install-native-build-deps.sh"


class CiWorkflowPolicyTests(unittest.TestCase):
    def test_model_smoke_is_required_on_release_and_explicit_runs(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn("AX_MODEL_SMOKE_REQUIRED", workflow)
        self.assertIn("startsWith(github.ref, 'refs/heads/release/')", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertIn("inputs.mlx_model_artifacts_dir != ''", workflow)
        self.assertIn("vars.AX_REQUIRE_MODEL_SMOKE == '1'", workflow)

    def test_main_pushes_do_not_require_unmounted_model_artifacts(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertNotIn("github.ref == 'refs/heads/main'", workflow)

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

    def test_aggregate_ci_gate_fails_on_non_successful_needs(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn("contains(needs.*.result, 'failure')", workflow)
        self.assertIn("contains(needs.*.result, 'cancelled')", workflow)
        self.assertIn("contains(needs.*.result, 'skipped')", workflow)
        self.assertIn(
            "One or more CI gates failed, were cancelled, or were skipped.",
            workflow,
        )

    def test_native_dependency_installs_cleanup_untrusted_runner_taps(self) -> None:
        helper = NATIVE_DEPS_SCRIPT.read_text()

        self.assertIn("brew untap --force aws/tap azure/bicep", helper)
        self.assertIn("brew install mlx protobuf", helper)

        workflow_texts = {
            path.name: path.read_text()
            for path in WORKFLOWS_DIR.glob("*.yml")
        }
        direct_install_workflows = [
            name
            for name, text in workflow_texts.items()
            if "brew install mlx protobuf" in text
        ]
        self.assertEqual([], direct_install_workflows)

        for workflow in ("ci.yml", "coverage.yml", "pypi.yml"):
            self.assertIn(
                "bash scripts/install-native-build-deps.sh",
                workflow_texts[workflow],
            )


if __name__ == "__main__":
    unittest.main()
