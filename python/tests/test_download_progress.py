from __future__ import annotations

import importlib.util
import io
import pathlib
import tempfile
import unittest
from unittest import mock

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
        line = dm._render_progress_bar(downloaded=1024**2, total=None, speed=None, eta=None)
        self.assertIn("downloaded", line)
        self.assertIn("-- B/s", line)
        self.assertNotIn("%", line)

    def test_dir_size_bytes_sums_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "a.bin").write_bytes(b"x" * 100)
            nested = root / "sub"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"y" * 50)
            self.assertEqual(dm._dir_size_bytes(root), 150)

    def test_dir_size_bytes_missing_path_is_zero(self) -> None:
        self.assertEqual(dm._dir_size_bytes(pathlib.Path("/no/such/path/here")), 0)

    def test_progress_reporter_counts_only_new_physical_cache_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = pathlib.Path(tmp)
            blobs = repo_dir / "blobs"
            blobs.mkdir()
            cached_blob = blobs / "cached"
            cached_blob.write_bytes(b"x" * 100)
            first_snapshot = repo_dir / "snapshots" / "first"
            first_snapshot.mkdir(parents=True)
            try:
                (first_snapshot / "model.safetensors").symlink_to(cached_blob)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            stream = io.StringIO()
            with (
                mock.patch.object(dm, "default_mlx_lm_repo_cache_dir", return_value=repo_dir),
                mock.patch.object(dm.threading, "Thread"),
                dm._ProgressBarReporter("owner/model", 1_000, stream) as reporter,
            ):
                # A new snapshot link to an existing blob consumes no new
                # cache storage and must not advance the reporter.
                second_snapshot = repo_dir / "snapshots" / "second"
                second_snapshot.mkdir()
                (second_snapshot / "model.safetensors").symlink_to(cached_blob)
                downloaded, _speed, _eta = reporter._measure()
                self.assertEqual(downloaded, 0)

                (blobs / "new").write_bytes(b"y" * 25)
                downloaded, _speed, _eta = reporter._measure()
                self.assertEqual(downloaded, 25)

    def test_failed_progress_reporter_does_not_render_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = pathlib.Path(tmp)
            stream = io.StringIO()
            with (
                mock.patch.object(dm, "default_mlx_lm_repo_cache_dir", return_value=repo_dir),
                mock.patch.object(dm.threading, "Thread"),
                self.assertRaisesRegex(RuntimeError, "download failed"),
                dm._ProgressBarReporter("owner/model", 100, stream),
            ):
                raise RuntimeError("download failed")

            output = stream.getvalue()
            self.assertNotIn("100%", output)
            self.assertIn("0%", output)
            self.assertTrue(output.endswith("\n"), "failed display must end its line")

    def test_successful_progress_reporter_renders_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = pathlib.Path(tmp)
            stream = io.StringIO()
            with (
                mock.patch.object(dm, "default_mlx_lm_repo_cache_dir", return_value=repo_dir),
                mock.patch.object(dm.threading, "Thread"),
                dm._ProgressBarReporter("owner/model", 100, stream),
            ):
                pass

            self.assertIn("100%", stream.getvalue())
            self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
