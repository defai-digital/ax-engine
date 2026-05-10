#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("download_model.py")
spec = importlib.util.spec_from_file_location("download_model", SCRIPT_PATH)
download_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_model)


class DownloadModelScriptTest(unittest.TestCase):
    def test_default_destination_uses_huggingface_cache_root(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_model.default_hf_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--mlx-community--Qwen3-4B-4bit",
            )

        with patch.dict(os.environ, {"HF_HOME": "/tmp/hf-home"}, clear=True):
            self.assertEqual(
                download_model.default_hf_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-home/hub/models--mlx-community--Qwen3-4B-4bit"),
            )

        with patch.dict(os.environ, {"HF_HUB_CACHE": "/tmp/hf-hub"}, clear=True):
            self.assertEqual(
                download_model.default_hf_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-hub/models--mlx-community--Qwen3-4B-4bit"),
            )

    def test_json_summary_for_existing_ready_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            (model_dir / "model.safetensors").write_bytes(b"placeholder")
            (model_dir / "model-manifest.json").write_text("{}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "mlx-community/Qwen3-4B-4bit",
                    "--dest",
                    str(model_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            summary = json.loads(result.stdout)
            self.assertEqual(summary["schema_version"], "ax.download_model.v1")
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["dest"], str(model_dir))
            self.assertTrue(summary["manifest_present"])
            self.assertEqual(summary["safetensors_count"], 1)
            self.assertEqual(
                summary["server_command"],
                [
                    "ax-engine-server",
                    "--mlx",
                    "--mlx-model-artifacts-dir",
                    str(model_dir),
                    "--port",
                    "8080",
                ],
            )

    def test_json_summary_for_invalid_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors").write_bytes(b"placeholder")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "mlx-community/Qwen3-4B-4bit",
                    "--dest",
                    str(model_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "invalid")
            self.assertIn("config.json missing", "\n".join(summary["errors"]))


if __name__ == "__main__":
    unittest.main()
