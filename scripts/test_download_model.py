#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("download_model.py")
spec = importlib.util.spec_from_file_location("download_model", SCRIPT_PATH)
download_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_model)


class DownloadModelScriptTest(unittest.TestCase):
    def test_default_destination_uses_mlx_lm_cache_root(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--mlx-community--Qwen3-4B-4bit",
            )

        with patch.dict(os.environ, {"HF_HOME": "/tmp/hf-home"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-home/hub/models--mlx-community--Qwen3-4B-4bit"),
            )

        with patch.dict(os.environ, {"HF_HUB_CACHE": "/tmp/hf-hub"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-hub/models--mlx-community--Qwen3-4B-4bit"),
            )

        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/xdg-cache"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/xdg-cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit"),
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

    def test_progress_json_emits_events_before_summary(self) -> None:
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
                    "--progress-json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0]["event"], "progress")
            self.assertEqual(lines[-1]["status"], "ready")

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

    def test_download_reuses_existing_cache_snapshot(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--mlx-community--Qwen3-4B-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("abc123")
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"placeholder")

            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
            ) -> Path:
                calls.append(model)
                return snapshot

            with patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True), patch.object(
                download_model, "_run_hf_snapshot_download", fake_hf_download
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [])
            self.assertEqual(resolved, snapshot)

    def test_download_uses_huggingface_hub_snapshot_download(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--mlx-community--Qwen3-4B-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            (repo_cache / "refs").mkdir(parents=True)
            (repo_cache / "refs" / "main").write_text("abc123")

            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
            ) -> Path:
                calls.append(model)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
                (snapshot / "model.safetensors").write_bytes(b"placeholder")
                return snapshot

            with patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True), patch.object(
                download_model, "_run_hf_snapshot_download", fake_hf_download
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [repo_id])
            self.assertEqual(resolved, snapshot)

    def test_gemma4_unified_download_does_not_invoke_mlx_lm_generation(self) -> None:
        repo_id = "mlx-community/gemma-4-12B-it-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
            ) -> Path:
                calls.append(model)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text('{"model_type":"gemma4_unified"}')
                (snapshot / "model.safetensors").write_bytes(b"placeholder")
                return snapshot

            with patch.dict(os.environ, {"HF_HOME": tmp}, clear=True), patch.object(
                download_model, "_run_hf_snapshot_download", fake_hf_download
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [repo_id])
            self.assertEqual(resolved, snapshot)

    def test_force_removes_stale_manifest_from_dest(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            # Artifacts left from a prior (possibly different) model.
            (dest / "model-manifest.json").write_text('{"stale":true}')
            (dest / "old.safetensors").write_bytes(b"old")

            snapshot = root / "snapshot"

            def fake_hf_download(
                model,
                *,
                quiet=False,
                progress_json=False,
                progress_bar=False,
            ):
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
                (snapshot / "model.safetensors").write_bytes(b"new")
                return snapshot

            with patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True), patch.object(
                download_model, "_run_hf_snapshot_download", fake_hf_download
            ):
                resolved = download_model.download(repo_id, dest, force=True, quiet=True)

            self.assertEqual(resolved, dest)
            # Stale manifest is dropped so main() regenerates it against the new weights.
            self.assertFalse((dest / "model-manifest.json").exists())
            self.assertTrue((dest / "model.safetensors").exists())

    def test_embedding_repos_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "embedding model downloads are not managed"):
            download_model.download("mlx-community/Qwen3-Embedding-0.6B-8bit", None, quiet=True)

    def test_manifest_generation_uses_local_release_binary_before_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            local_bin = root / "target" / "release" / "generate-manifest"
            local_bin.parent.mkdir(parents=True)
            local_bin.write_text("#!/bin/sh\n")
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch.object(download_model, "REPO_ROOT", root), patch.object(
                download_model.shutil, "which", side_effect=lambda name: "cargo" if name == "cargo" else None
            ), patch.object(download_model.subprocess, "run", fake_run):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            self.assertEqual(calls, [[str(local_bin), str(model_dir)]])

    def test_manifest_generation_falls_back_after_installed_bench_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            local_bin = root / "target" / "debug" / "generate-manifest"
            local_bin.parent.mkdir(parents=True)
            local_bin.write_text("#!/bin/sh\n")
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if command[0] == "ax-engine-bench":
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch.object(download_model, "REPO_ROOT", root), patch.object(
                download_model.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/ax-engine-bench" if name == "ax-engine-bench" else None,
            ), patch.object(download_model.subprocess, "run", fake_run):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            self.assertEqual(
                calls,
                [
                    ["ax-engine-bench", "generate-manifest", str(model_dir)],
                    [str(local_bin), str(model_dir)],
                ],
            )

    def test_manifest_generation_prefers_bundled_binary_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            bundled = "/wheel/ax_engine/_bin/ax-engine-bench"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch.object(
                download_model, "_bundled_bench_bin", return_value=bundled
            ), patch.object(
                download_model.shutil,
                "which",
                side_effect=lambda name: "/usr/bin/ax-engine-bench",
            ), patch.object(download_model.subprocess, "run", fake_run):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            # The bundled binary is used; the stale PATH binary is never invoked.
            self.assertEqual(calls, [[bundled, "generate-manifest", str(model_dir)]])

    def test_validation_rejects_missing_shards_declared_by_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"present")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model-00001-of-00002.safetensors",
                            "model.layers.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )

            errors = download_model._validation_errors(model_dir)

            self.assertEqual(
                errors,
                [
                    f"missing safetensors shard model-00002-of-00002.safetensors in {model_dir}"
                ],
            )

    def test_validation_ignores_stale_index_for_differently_sharded_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"mistral3"}')
            (model_dir / "model-00001-of-00003.safetensors").write_bytes(b"present")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model-00001-of-00010.safetensors"
                        }
                    }
                )
            )

            self.assertEqual(download_model._validation_errors(model_dir), [])

    def test_main_returns_nonzero_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            (model_dir / "model.safetensors").write_bytes(b"placeholder")

            argv = [
                "download_model.py",
                "mlx-community/Qwen3-4B-4bit",
                "--dest",
                str(model_dir),
                "--json",
            ]
            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(
                download_model, "_try_generate_manifest", return_value=False
            ), redirect_stdout(stdout):
                code = download_model.main()

            self.assertEqual(code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "manifest_missing")
            self.assertFalse(summary["manifest_present"])


if __name__ == "__main__":
    unittest.main()
