"""Tests for Homebrew formula payload normalization."""

from __future__ import annotations

import unittest

from scripts.update_homebrew_bin_install import rewrite_bin_install


class UpdateHomebrewBinInstallTests(unittest.TestCase):
    def test_replaces_complete_multiline_statement(self) -> None:
        formula = '''class AxEngine < Formula
  def install
    bin.install "ax-engine", "ax-engine-server",
                "old-helper.py",
                "old-check.py"

    relink_release_binaries_to_tap_mlx!
  end
end
'''

        updated = rewrite_bin_install(formula, ["ax-engine", "new-helper.py"])

        self.assertIn('    bin.install "ax-engine", "new-helper.py"\n', updated)
        self.assertNotIn("old-helper.py", updated)
        self.assertNotIn("old-check.py", updated)
        self.assertIn("    relink_release_binaries_to_tap_mlx!", updated)

    def test_coalesces_multiple_install_statements(self) -> None:
        formula = '''class AxEngine < Formula
  def install
    bin.install "ax-engine"
    bin.install "ax-engine-server"
  end
end
'''

        updated = rewrite_bin_install(formula, ["ax-engine", "ax-engine-server"])

        self.assertEqual(updated.count("bin.install"), 1)
        self.assertIn('bin.install "ax-engine", "ax-engine-server"', updated)

    def test_rejects_missing_install_statement(self) -> None:
        with self.assertRaisesRegex(ValueError, "no bin.install"):
            rewrite_bin_install("class AxEngine < Formula\nend\n", ["ax-engine"])

    def test_rejects_unsafe_continuation(self) -> None:
        formula = '  bin.install "ax-engine",\n  relink_release_binaries_to_tap_mlx!\n'

        with self.assertRaisesRegex(ValueError, "unsupported bin.install continuation"):
            rewrite_bin_install(formula, ["ax-engine"])


if __name__ == "__main__":
    unittest.main()
