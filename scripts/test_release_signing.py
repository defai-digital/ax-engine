"""Release-signing contract tests.

The macOS Homebrew package links against Homebrew-provided MLX dylibs. Those
dylibs are ad-hoc signed, so Developer ID hardened-runtime binaries must be
signed with disable-library-validation or dyld rejects them at startup.
"""

import os
import plistlib
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTITLEMENTS = os.path.join(REPO_ROOT, "scripts", "macos-release.entitlements.plist")
PUBLISH_SCRIPT = os.path.join(REPO_ROOT, "scripts", "publish-github-release.sh")
BREW_RELEASE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "brew-release.sh")


class ReleaseSigningTests(unittest.TestCase):
    def test_entitlements_allow_homebrew_mlx_dylibs(self):
        with open(ENTITLEMENTS, "rb") as fh:
            data = plistlib.load(fh)

        self.assertIs(data.get("com.apple.security.cs.disable-library-validation"), True)

    def test_release_scripts_codesign_with_entitlements(self):
        for script in (PUBLISH_SCRIPT, BREW_RELEASE_SCRIPT):
            with self.subTest(script=os.path.basename(script)):
                with open(script, encoding="utf-8") as fh:
                    text = fh.read()
                self.assertIn("MACOS_RELEASE_ENTITLEMENTS=", text)
                self.assertIn('--entitlements "$MACOS_RELEASE_ENTITLEMENTS"', text)

    def test_publisher_fails_closed_and_verifies_uploaded_release(self):
        with open(PUBLISH_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn("published releases require Minisign", text)
        self.assertIn("published releases require --sign-identity", text)
        self.assertIn("published releases must be notarized", text)
        self.assertIn("TeamIdentifier=$EXPECTED_APPLE_TEAM_ID", text)
        # --check-notarization only modifies verification; it must ride on --verify.
        self.assertIn(
            "codesign --verify --strict --check-notarization --verbose=2",
            text,
        )
        self.assertIn("spctl --assess --type execute", text)
        self.assertIn("release_args+=(--draft)", text)
        self.assertIn("release $TAG is already published; refusing to replace verified assets", text)
        self.assertIn("release $TAG is no longer a draft; refusing to publish or mutate it", text)
        self.assertIn("cmp \"$REPOSITORY_MINISIGN_PUBLIC_KEY\"", text)
        self.assertIn("minisign -V", text)
        self.assertLess(text.rindex("verify_uploaded_release"), text.index('gh release edit "$TAG"'))
        # Server ships with panic=unwind so catch_unwind containment works.
        self.assertIn("--profile release-server", text)
        self.assertIn("target/release-server/ax-engine-server", text)

    def test_legacy_brew_publisher_cannot_mutate_releases(self):
        with open(BREW_RELEASE_SCRIPT, encoding="utf-8") as fh:
            text = fh.read()

        self.assertIn("scripts/brew-release.sh is a legacy preview and may not publish releases", text)
        self.assertIn('if [[ "$DRY_RUN" = false ]]', text)
        self.assertIn("--profile release-server", text)


if __name__ == "__main__":
    unittest.main()
