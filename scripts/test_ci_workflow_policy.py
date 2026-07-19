"""Tests for stability-critical GitHub Actions workflow policy."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
NATIVE_DEPS_SCRIPT = ROOT / "scripts" / "install-native-build-deps.sh"
PUBLISH_SCRIPT = ROOT / "scripts" / "publish-github-release.sh"
RUST_TOOLCHAIN = ROOT / "rust-toolchain.toml"


class CiWorkflowPolicyTests(unittest.TestCase):
    def test_general_ci_runs_on_branches_but_not_tag_pushes(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn('push:\n    branches:\n      - "**"', workflow)
        self.assertNotIn("push:\n    tags:", workflow)

    def test_model_smoke_is_required_on_release_and_explicit_runs(self) -> None:
        workflow = CI_WORKFLOW.read_text()

        self.assertIn("AX_MODEL_SMOKE_REQUIRED", workflow)
        self.assertIn("startsWith(github.ref, 'refs/heads/release/')", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertIn("inputs.mlx_model_artifacts_dir != ''", workflow)
        self.assertIn("vars.AX_REQUIRE_MODEL_SMOKE == '1'", workflow)

    def test_qa_offline_gate_is_required_in_scripts_job(self) -> None:
        workflow = CI_WORKFLOW.read_text()
        self.assertIn("bash scripts/check-qa.sh", workflow)
        self.assertIn("Run QA harness offline gate", workflow)

    def test_qa_model_gate_runs_when_artifacts_mounted(self) -> None:
        workflow = CI_WORKFLOW.read_text()
        self.assertIn("bash scripts/check-qa-model.sh", workflow)
        self.assertIn("Run QA bank sample + surface probes against real model", workflow)

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

        for workflow in ("coverage.yml",):
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

        brew = workflow_texts["brew-release.yml"]
        self.assertIn("Formula metadata only", brew)
        self.assertIn("runs-on: ubuntu-latest", brew)
        self.assertNotIn("bash scripts/build-pypi-wheel.sh", brew)

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
        self.assertIn("name: Build Python Wheel\n    needs: resolve_candidate", pypi)
        self.assertIn("runs-on: macos-26", pypi)
        self.assertIn("name: Promote validated wheel candidate", pypi)
        _, publish = pypi.split("  publish:\n    name: Publish to PyPI\n", maxsplit=1)
        self.assertIn("Artifact upload only", publish)

        candidate = workflow_texts["release-candidate.yml"]
        self.assertIn("name: Build and validate macOS release candidate", candidate)
        self.assertIn("runs-on: macos-26", candidate)
        self.assertIn("bash scripts/build-pypi-wheel.sh", candidate)

    def test_release_candidate_and_promotion_are_bound_to_exact_sha(self) -> None:
        workflows = {
            path.name: path.read_text() for path in WORKFLOWS_DIR.glob("*.yml")
        }
        candidate = workflows["release-candidate.yml"]
        pypi = workflows["pypi.yml"]

        self.assertIn("ref: ${{ inputs.git_commit }}", candidate)
        self.assertIn("Require successful CI for exact commit", candidate)
        self.assertIn("ax-engine-release-candidate-${{ steps.identity.outputs.commit }}", candidate)
        self.assertIn("ax-engine-pypi-wheel-${{ steps.identity.outputs.commit }}", candidate)
        self.assertIn("shared-key: release-macos-arm64", candidate)
        self.assertIn('ARTIFACT_NAME="ax-engine-pypi-wheel-${RELEASE_SHA}"', pypi)
        self.assertIn("scripts/release_candidate.py verify", pypi)
        self.assertIn("shared-key: release-macos-arm64", pypi)

    def test_homebrew_is_dispatched_only_after_release_assets_are_verified(self) -> None:
        brew = (WORKFLOWS_DIR / "brew-release.yml").read_text()
        publisher = PUBLISH_SCRIPT.read_text()

        self.assertNotIn("push:\n    tags:", brew)
        self.assertIn("workflow_dispatch:", brew)
        self.assertNotIn("for attempt in $(seq 1 30)", brew)
        self.assertLess(
            publisher.rindex("verify_uploaded_release"),
            publisher.index('gh release edit "$TAG"'),
        )
        self.assertLess(
            publisher.index('gh release edit "$TAG"'),
            publisher.index("gh workflow run brew-release.yml"),
        )

    def test_homebrew_verifies_minisign_before_trusting_checksum(self) -> None:
        brew = (WORKFLOWS_DIR / "brew-release.yml").read_text()

        self.assertIn("sudo apt-get install -y minisign", brew)
        self.assertIn("cmp docs/ax-minisign.pub /tmp/ax-minisign.pub", brew)
        self.assertIn("minisign -V", brew)
        self.assertIn("Homebrew release requires Developer ID signing and Apple notarization", brew)
        self.assertLess(brew.index("minisign -V"), brew.index('SHA256="$(awk'))
        self.assertLess(
            brew.index("ACTUAL_SHA256="),
            brew.rindex("      - name: Update Homebrew tap formula"),
        )

    def test_publisher_reuses_exact_sha_ci_by_default(self) -> None:
        publisher = PUBLISH_SCRIPT.read_text()

        self.assertIn("require_green_ci", publisher)
        self.assertIn('-f head_sha="$head_commit"', publisher)
        self.assertIn("--full-local-checks", publisher)
        self.assertIn("prepare_release_candidate", publisher)
        self.assertIn("for _ in {1..15}", publisher)

    def test_rust_toolchain_is_pinned_for_reproducible_release_cache_keys(self) -> None:
        toolchain = RUST_TOOLCHAIN.read_text()

        self.assertRegex(toolchain, r'channel = "\d+\.\d+\.\d+"')
        self.assertNotIn('channel = "stable"', toolchain)
        self.assertIn('components = ["clippy", "rustfmt"]', toolchain)

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
        self.assertIn("brew install protobuf", helper)
        # MLX must come from pip, not Homebrew: the brew formula's deployment
        # target truncates to 26.0 on macOS 26.x, silently compiling out the
        # NAX kernels (see the helper script's comment).
        self.assertNotIn("brew install mlx", helper)
        # And it must be the admitted pinned version (mlx.version at the repo
        # root — the same pin mlx-sys/build.rs enforces at link time), never
        # an unpinned "latest".
        self.assertIn('pip install --upgrade "mlx==${MLX_PIN}"', helper)
        self.assertIn("mlx.version", helper)
        pin = (NATIVE_DEPS_SCRIPT.parent.parent / "mlx.version").read_text().strip()
        self.assertRegex(pin, r"^\d+\.\d+\.\d+$")
        # Subsequent cargo/maturin steps (including fresh-venv maturin develop)
        # must see the resolved paths; CI does this via GITHUB_ENV.
        self.assertIn("MLX_LIB_DIR=", helper)
        self.assertIn("GITHUB_ENV", helper)

        workflow_texts = {
            path.name: path.read_text()
            for path in WORKFLOWS_DIR.glob("*.yml")
        }
        direct_install_workflows = [
            name
            for name, text in workflow_texts.items()
            if "brew install mlx" in text or "brew install protobuf" in text
        ]
        self.assertEqual([], direct_install_workflows)

        for workflow in ("ci.yml", "coverage.yml", "pypi.yml"):
            self.assertIn(
                "bash scripts/install-native-build-deps.sh",
                workflow_texts[workflow],
            )

        # Fresh venvs used for maturin develop must install mlx themselves;
        # install-native-build-deps only targets the host interpreter, and
        # mlx-sys/build.rs prefers the active Python's mlx package.
        coverage = workflow_texts["coverage.yml"]
        self.assertIn("maturin>=1.7,<2", coverage)
        self.assertRegex(coverage, r'pip install .*mlx')

        for path in (
            ROOT / "scripts" / "check-python-preview.sh",
            ROOT / "scripts" / "check-prefix-reuse-equivalence.sh",
        ):
            text = path.read_text()
            self.assertIn("maturin develop", text)
            self.assertRegex(
                text,
                r'pip install .*mlx',
                f"{path.name} must pip-install mlx into its maturin venv",
            )


if __name__ == "__main__":
    unittest.main()
