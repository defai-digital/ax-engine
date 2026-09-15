#!/usr/bin/env python3
"""Drive shipped ``_cli._download_repo_id`` for the Qwen 3.8 27B primary alias.

Loads ``python/ax_engine/_cli.py`` without the native MLX extension so the pin
can be checked on hosts whose ``_ax_engine`` dylib does not match site MLX.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "python" / "ax_engine"


def load_cli():
    pkg = types.ModuleType("ax_engine")
    pkg.__path__ = [str(PKG_DIR)]
    pkg._bundled_binary = lambda *_args, **_kwargs: None
    sys.modules["ax_engine"] = pkg

    spec_ref = importlib.util.spec_from_file_location(
        "ax_engine._repo_ref", PKG_DIR / "_repo_ref.py"
    )
    assert spec_ref is not None and spec_ref.loader is not None
    repo_ref = importlib.util.module_from_spec(spec_ref)
    sys.modules["ax_engine._repo_ref"] = repo_ref
    spec_ref.loader.exec_module(repo_ref)

    spec = importlib.util.spec_from_file_location("ax_engine._cli", PKG_DIR / "_cli.py")
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules["ax_engine._cli"] = cli
    spec.loader.exec_module(cli)
    return cli


class Qwen38PrimaryAliasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = load_cli()

    def test_download_repo_id_pins_primary_axq_pack(self) -> None:
        repo, profile, revision = self.cli._download_repo_id("qwen3.8-27b:axq")
        self.assertEqual(repo, "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP")
        self.assertEqual(revision, "3e290738e96972307c6aeb9934ab170ca0eae1c1")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.preset, "qwen3.8-27b")
        self.assertIsNone(self.cli._profile_certification(profile))


if __name__ == "__main__":
    unittest.main()
