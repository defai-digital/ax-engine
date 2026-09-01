"""Release-signing and standalone-runtime contract tests."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISH_SCRIPT = os.path.join(REPO_ROOT, "scripts", "publish-github-release.sh")
BREW_RELEASE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "brew-release.sh")


class ReleaseSigningTests(unittest.TestCase):
    def test_publisher_signs_bundled_libraries_without_weakening_validation(self):
        with open(PUBLISH_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn('"$STAGING_DIR/libmlx.dylib"', text)
        self.assertIn('"$STAGING_DIR/libjaccl.dylib"', text)
        self.assertIn("--options runtime", text)
        self.assertNotIn("MACOS_RELEASE_ENTITLEMENTS", text)
        self.assertNotIn('--entitlements "$', text)
        self.assertIn("unexpectedly disables hardened-runtime library validation", text)
        self.assertIn('"disable_library_validation": False', text)
        self.assertIn("verify_pristine_mlx_runtime", text)
        self.assertIn("verify_packaged_mlx_runtime_derivation", text)
        self.assertIn('"dylib_load_commands_preserved": True', text)
        self.assertIn('"metallib_byte_identical": True', text)
        self.assertIn('"kind": "pinned-pypi-wheel"', text)

    def test_publisher_fails_closed_and_verifies_uploaded_release(self):
        with open(PUBLISH_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn("published releases require Minisign", text)
        self.assertIn("published releases require --sign-identity", text)
        self.assertIn("published releases must be notarized", text)
        self.assertIn("TeamIdentifier=$EXPECTED_APPLE_TEAM_ID", text)
        self.assertIn("verify_notarization_log", text)
        self.assertIn("xcrun notarytool log", text)
        self.assertIn('"ticketContents"', text)
        self.assertIn(r"^CDHash=([0-9a-f]+)$", text)
        self.assertIn("notarization_submission_id", text)
        self.assertIn("Apple notarization log contains issues", text)
        self.assertIn("exact arm64 CDHash", text)
        self.assertNotIn("register_notarization_ticket", text)
        self.assertNotIn("spctl --assess", text)
        self.assertNotIn("xattr -w com.apple.quarantine", text)
        self.assertLess(
            text.index("notarize_release_payload"),
            text.rindex("verify_uploaded_release"),
        )
        self.assertIn("release_args+=(--draft)", text)
        self.assertIn(
            "release $TAG is already published; refusing to replace verified assets", text
        )
        self.assertIn("release $TAG is no longer a draft; refusing to publish or mutate it", text)
        self.assertIn('cmp "$REPOSITORY_MINISIGN_PUBLIC_KEY"', text)
        self.assertIn("minisign -V", text)
        self.assertIn("prepare-mlx-release-runtime.sh", text)
        self.assertIn("prepare-standalone-release.sh", text)
        self.assertIn("validate-standalone.sh", text)
        self.assertIn('validate-standalone.sh" --doctor', text)
        self.assertIn("ax.github_release_manifest.v3", text)
        self.assertIn("@loader_path/../libexec", text)
        self.assertLess(
            text.rindex("verify_uploaded_release"), text.index('gh release edit "$TAG"')
        )
        # Server ships with panic=unwind so catch_unwind containment works.
        self.assertIn("--profile release-server", text)
        self.assertIn("target/release-server/ax-engine-server", text)

    def test_publisher_redacts_notary_credentials_and_honors_team_override(self):
        with open(PUBLISH_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn("[App Store Connect API credentials redacted]", text)
        self.assertNotIn("${NOTARY_ARGS[*]}", text)
        self.assertIn("App Store Connect API credentials failed notarization validation", text)
        self.assertIn('grep -F "Authority=Developer ID Application:"', text)
        self.assertNotIn(
            'Authority=Developer ID Application: DEFAI PRIVATE LIMITED', text
        )
        self.assertIn("TeamIdentifier=$EXPECTED_APPLE_TEAM_ID", text)

    def test_publisher_validates_notary_credentials_before_tagging(self):
        with open(PUBLISH_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        resolve_call = text.rindex("\nresolve_notary_args\n")
        validate_call = text.rindex("\nvalidate_notary_credentials\n")
        tag_creation = text.index('run git tag -a "$TAG"')

        self.assertIn("xcrun notarytool history", text)
        self.assertIn("notarization Keychain profile is unavailable", text)
        self.assertLess(resolve_call, validate_call)
        self.assertLess(validate_call, tag_creation)

    def test_publisher_fails_before_tagging_when_notary_profile_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_bin = Path(tmp_dir)
            for command in (
                "cargo",
                "codesign",
                "file",
                "gh",
                "git",
                "install_name_tool",
                "lipo",
                "otool",
                "python3",
                "shasum",
                "tar",
                "zip",
            ):
                stub = fake_bin / command
                stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                stub.chmod(0o755)

            xcrun = fake_bin / "xcrun"
            xcrun.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "notarytool" ] && [ "$2" = "--help" ]; then\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$1" = "notarytool" ] && [ "$2" = "history" ]; then\n'
                "  exit 69\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            xcrun.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            result = subprocess.run(
                [
                    PUBLISH_SCRIPT,
                    "v7.2.1",
                    "--dry-run",
                    "--skip-checks",
                    "--skip-build",
                    "--allow-dirty",
                    "--no-minisign",
                    "--sign-identity",
                    "TEST-CERTIFICATE",
                    "--notary-profile",
                    "missing-profile",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "notarization Keychain profile is unavailable: missing-profile",
            result.stderr,
        )
        self.assertNotIn("git tag", result.stdout)

    def test_legacy_brew_publisher_cannot_mutate_releases(self):
        with open(BREW_RELEASE_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn(
            "scripts/brew-release.sh is a legacy preview and may not publish releases", text
        )
        self.assertIn('if [[ "$DRY_RUN" = false ]]', text)
        self.assertIn("canonical_args=(", text)
        self.assertIn("--dry-run", text)
        self.assertIn('exec "$SCRIPT_DIR/publish-github-release.sh"', text)


if __name__ == "__main__":
    unittest.main()
