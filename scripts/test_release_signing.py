"""Release-signing and standalone-runtime contract tests."""

import os
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
