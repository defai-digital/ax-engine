"""Verify child-tool selection even when an unrelated Cargo shadows rustup."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PinnedCargoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts/lib").mkdir(parents=True)
        shutil.copy(ROOT / "scripts/cargo-pinned.sh", self.root / "scripts")
        shutil.copy(ROOT / "scripts/lib/rust-toolchain.sh", self.root / "scripts/lib")
        (self.root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.1"\n')
        self.pinned = self.root / "pinned tools"
        self.shadow = self.root / "shadow"
        self.pinned.mkdir()
        self.shadow.mkdir()
        for tool in ("cargo", "rustc", "rustdoc", "cargo-clippy", "cargo-fmt"):
            self.executable(self.shadow / tool, 'echo "shadow tool invoked" >&2\nexit 91\n')
        self.executable(self.shadow / "rustup", 'printf "%s/rustc\\n" "$TEST_PINNED_DIR"\n')
        for tool in ("rustc", "rustdoc"):
            self.executable(self.pinned / tool, f'echo "{tool} 1.97.1 (test build)"\n')
        for tool in ("cargo-clippy", "cargo-fmt"):
            self.executable(self.pinned / tool, f'echo "{tool} pinned"\n')
        self.executable(self.pinned / "cargo", '''if [ "$1" = "--version" ]; then
  echo "cargo 1.97.1 (test build)"
else
  "$RUSTC" --version
  "$RUSTDOC" --version
  rustc --version
  cargo-clippy --version
  cargo-fmt --version
  printf 'toolchain=%s\\n' "$RUSTUP_TOOLCHAIN"
fi
''')

    @staticmethod
    def executable(path, body):
        path.write_text("#!/bin/sh\nset -eu\n" + body)
        path.chmod(0o755)

    def run_wrapper(self):
        env = dict(os.environ, PATH=f"{self.shadow}:{os.environ['PATH']}",
                   TEST_PINNED_DIR=str(self.pinned), RUSTC=str(self.shadow / "rustc"),
                   RUSTDOC=str(self.shadow / "rustdoc"), RUSTUP_TOOLCHAIN="wrong")
        return subprocess.run(["bash", str(self.root / "scripts/cargo-pinned.sh"), "test"],
                              env=env, text=True, capture_output=True, check=False)

    def test_pins_compiler_docs_and_cargo_subcommands(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cargo-clippy pinned", result.stdout)
        self.assertIn("cargo-fmt pinned", result.stdout)
        self.assertIn("toolchain=1.97.1", result.stdout)
        self.assertNotIn("shadow", result.stdout + result.stderr)

    def test_rejects_wrong_compiler_identity(self):
        self.executable(self.pinned / "rustc", 'echo "rustc 1.98.1 (wrong)"\n')
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected rustc 1.97.1", result.stderr)

    def test_missing_pin_fails_without_running_shadow_tools(self):
        (self.root / "rust-toolchain.toml").unlink()
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("shadow tool invoked", result.stderr)

    def test_missing_pinned_component_does_not_fall_back(self):
        (self.pinned / "cargo-clippy").unlink()
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pinned cargo-clippy is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
