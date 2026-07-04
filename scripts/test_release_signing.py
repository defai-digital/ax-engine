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


if __name__ == "__main__":
    unittest.main()
