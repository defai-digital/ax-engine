from __future__ import annotations

import importlib.util
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_download_model():
    path = REPO_ROOT / "scripts" / "download_model.py"
    spec = importlib.util.spec_from_file_location("ax_engine_download_model_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dm = _load_download_model()


class DownloadProgressHelpersTests(unittest.TestCase):
    def test_format_bytes_scales_units(self) -> None:
        self.assertEqual(dm._format_bytes(0), "0 B")
        self.assertEqual(dm._format_bytes(512), "512 B")
        self.assertEqual(dm._format_bytes(1536), "1.5 KiB")
        self.assertEqual(dm._format_bytes(3 * 1024**3), "3.0 GiB")
        self.assertEqual(dm._format_bytes(None), "?")

    def test_render_progress_bar_with_total(self) -> None:
        line = dm._render_progress_bar(
            downloaded=3 * 1024**3,
            total=6 * 1024**3,
            speed=128 * 1024**2,
            eta=24.0,
        )
        self.assertIn("50%", line)
        self.assertIn("3.0 GiB/6.0 GiB", line)
        self.assertIn("128.0 MiB/s", line)
        self.assertIn("ETA", line)
        self.assertTrue(line.startswith("["))

    def test_render_progress_bar_without_total_is_indeterminate(self) -> None:
        line = dm._render_progress_bar(
            downloaded=1024**2, total=None, speed=None, eta=None
        )
        self.assertIn("downloaded", line)
        self.assertIn("-- B/s", line)
        self.assertNotIn("%", line)

    def test_dir_size_bytes_sums_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "a.bin").write_bytes(b"x" * 100)
            nested = root / "sub"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"y" * 50)
            self.assertEqual(dm._dir_size_bytes(root), 150)

    def test_dir_size_bytes_missing_path_is_zero(self) -> None:
        self.assertEqual(dm._dir_size_bytes(pathlib.Path("/no/such/path/here")), 0)


if __name__ == "__main__":
    unittest.main()
