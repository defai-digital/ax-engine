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

    def test_runtime_validation_uses_macos_26_or_later(self) -> None:
        workflow_texts = {
            path.name: path.read_text() for path in WORKFLOWS_DIR.glob("*.yml")
        }

        for workflow in ("coverage.yml", "brew-release.yml"):
            runner_lines = [
                line.strip()
                for line in workflow_texts[workflow].splitlines()
                if line.strip().startswith("runs-on:")
            ]
            self.assertTrue(runner_lines, f"{workflow} must declare a runner")
            self.assertEqual(
                ["runs-on: macos-26"] * len(runner_lines),
                runner_lines,
                f"{workflow} must not validate AX Engine on Linux or pre-26 macOS",
            )

        ci = workflow_texts["ci.yml"]
        for job_name in (
            "Rust",
            "Scripts and Bench Smoke",
            "Python Package",
            "SDK Clients",
            "Model Smoke",
        ):
            self.assertIn(
                f"name: {job_name}\n    runs-on: macos-26",
                ci,
                f"{job_name} must validate AX Engine on macOS 26",
            )
        self.assertIn(
            "name: Supply Chain\n    runs-on: ubuntu-latest",
            ci,
        )
        self.assertIn("name: CI\n    runs-on: ubuntu-latest", ci)

        pypi = workflow_texts["pypi.yml"]
        build_wheel, publish = pypi.split(
            "  publish:\n    name: Publish to PyPI\n", maxsplit=1
        )
        self.assertIn("runs-on: macos-26", build_wheel)
        self.assertIn("Artifact upload only", publish)

    def test_supply_chain_checks_run_on_linux(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn("name: Supply Chain\n    runs-on: ubuntu-latest", workflow)
        self.assertIn("tool: cargo-deny,cargo-audit", workflow)
        self.assertIn("run: cargo deny check advisories licenses bans sources", workflow)
        self.assertIn("run: cargo audit\n", workflow)
        self.assertNotIn("EmbarkStudios/cargo-deny-action", workflow)

    def test_macos_sdk_gate_uses_supported_go_toolchain_and_all_modules(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn('go-version: "1.25"', workflow)
        self.assertIn("sdk/go/axengine/go.mod", workflow)
        self.assertIn("sdk/go/grpc/go.mod", workflow)
        self.assertIn("working-directory: sdk/go/axengine", workflow)
        self.assertIn("working-directory: sdk/go/grpc", workflow)

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
