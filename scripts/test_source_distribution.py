"""Check the actual maturin archive, including build-time runtime identity."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("maturin"), "requires maturin")
class SourceDistributionTests(unittest.TestCase):
    def test_build_pins_survive_source_packaging_at_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "maturin", "sdist", "--out", tmp],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archives = list(Path(tmp).glob("*.tar.gz"))
            self.assertEqual(len(archives), 1)
            with tarfile.open(archives[0]) as archive:
                prefix = archive.getnames()[0].split("/")[0]
                self.assertIn(f"{prefix}/crates/mlx-sys/build_pin.rs", archive.getnames())
                for pin in ("mlx.version", "rust-toolchain.toml", "LICENSE",
                            "vendor/block-0.1.6/Cargo.toml", "vendor/block-0.1.6/src/lib.rs"):
                    self.assertEqual(archive.extractfile(f"{prefix}/{pin}").read(),
                                     (ROOT / pin).read_bytes())


if __name__ == "__main__":
    unittest.main()
