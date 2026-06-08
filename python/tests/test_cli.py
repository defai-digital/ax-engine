from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import tempfile
import textwrap
import unittest
from unittest import mock

import sys

# Default to importing ax_engine from the source tree (for `maturin develop`
# runs). When validating an installed wheel (AX_ENGINE_RUN_INSTALLED_TESTS=1),
# keep the source off sys.path so the installed package — with its compiled
# _ax_engine extension — is imported instead of the un-built source. Inserting
# it unconditionally shadowed the wheel for the whole discovery process and
# broke the installed-wheel smoke tests.
if os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS") != "1":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "python"))

from ax_engine import _cli


class AxEngineCliTests(unittest.TestCase):
    def capture_main(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = _cli.main(argv)
        return code, out.getvalue()

    def test_download_list_json_shows_targets(self) -> None:
        code, stdout = self.capture_main(["download", "--list", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.download_options.v1")
        self.assertEqual(payload["default_destination"]["kind"], "huggingface_hub_cache")
        self.assertIn("HF_HUB_CACHE", payload["default_destination"]["env"])
        aliases = {target["alias"] for target in payload["targets"]}
        self.assertIn("qwen3.6-35b", aliases)
        self.assertIn("qwen3.6-27b-8bit", aliases)
        self.assertIn("gemma4-e2b-6bit", aliases)

    def test_download_list_text_shows_cache_policy(self) -> None:
        code, stdout = self.capture_main(["download", "--list"])

        self.assertEqual(code, 0)
        self.assertIn("Hugging Face Hub cache", stdout)
        self.assertIn("HF_HUB_CACHE", stdout)
        self.assertIn("--dest only", stdout)

    def test_download_missing_model_shows_targets(self) -> None:
        code, stdout = self.capture_main(["download"])

        self.assertEqual(code, 2)
        self.assertIn("missing model alias or repo id", stdout)
        self.assertIn("qwen3.6-35b", stdout)
        self.assertIn("gemma4-e2b-8bit", stdout)

    def test_download_unknown_alias_shows_targets(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.capture_main(["download", "unknown-model"])

        self.assertIn("unknown model alias", str(raised.exception))
        self.assertIn("qwen3.6-27b-8bit", str(raised.exception))
        self.assertIn("gemma4-e2b-6bit", str(raised.exception))

    def test_serve_dry_run_json_uses_server_preset(self) -> None:
        with mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"):
            code, stdout = self.capture_main(
                [
                    "serve",
                    "qwen36-35b",
                    "--port",
                    "9010",
                    "--dry-run",
                    "--json",
                    "--",
                    "--max-batch-tokens",
                    "1024",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.local_serve_plan.v1")
        self.assertEqual(payload["resolved"]["preset"], "qwen3.6-35b")
        self.assertEqual(payload["server"]["url"], "http://127.0.0.1:9010")
        self.assertEqual(
            payload["server"]["argv"],
            [
                "/opt/bin/ax-engine-server",
                "--host",
                "127.0.0.1",
                "--port",
                "9010",
                "--mlx",
                "--preset",
                "qwen3.6-35b",
                "--resolve-model-artifacts",
                "hf-cache",
                "--max-batch-tokens",
                "1024",
            ],
        )

    def test_serve_dry_run_json_uses_local_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp) / "model"
            model_dir.mkdir()
            with mock.patch.object(_cli, "_server_bin", return_value="ax-engine-server"):
                code, stdout = self.capture_main(
                    ["serve", str(model_dir), "--dry-run", "--json"]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["resolved"]["kind"], "local_dir")
        self.assertIn("--mlx-model-artifacts-dir", payload["server"]["argv"])
        path_index = payload["server"]["argv"].index("--mlx-model-artifacts-dir") + 1
        self.assertEqual(payload["server"]["argv"][path_index], str(model_dir.resolve()))

    def test_download_alias_wraps_download_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            (model_dir / "model.safetensors").write_bytes(b"placeholder")
            (model_dir / "model-manifest.json").write_text("{}")

            (scripts / "download_model.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    p = argparse.ArgumentParser()
                    p.add_argument("repo_id")
                    p.add_argument("--dest")
                    p.add_argument("--force", action="store_true")
                    p.add_argument("--json", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({
                        "schema_version": "ax.download_model.v1",
                        "repo_id": args.repo_id,
                        "dest": __import__("os").environ["FAKE_MODEL_DIR"],
                        "manifest_present": True,
                        "safetensors_count": 1,
                        "config_present": True,
                        "status": "ready",
                        "errors": [],
                        "server_command": ["ax-engine-server"],
                    }))
                    """
                )
            )

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ):
                code, stdout = self.capture_main(["download", "qwen36-35b", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.download_model.v1")
        self.assertEqual(payload["repo_id"], "mlx-community/Qwen3.6-35B-A3B-4bit")
        self.assertEqual(payload["alias"], "qwen3.6-35b")
        self.assertEqual(payload["preset"], "qwen3.6-35b")

    def test_download_qwen36_27b_bit_alias_uses_mlx_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            (scripts / "download_model.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json, os
                    p = argparse.ArgumentParser()
                    p.add_argument("repo_id")
                    p.add_argument("--dest")
                    p.add_argument("--force", action="store_true")
                    p.add_argument("--json", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({
                        "schema_version": "ax.download_model.v1",
                        "repo_id": args.repo_id,
                        "dest": os.environ["FAKE_MODEL_DIR"],
                        "manifest_present": True,
                        "safetensors_count": 1,
                        "config_present": True,
                        "status": "ready",
                        "errors": [],
                        "server_command": ["ax-engine-server"],
                    }))
                    """
                )
            )

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ):
                code, stdout = self.capture_main(["download", "qwen36-27b-8bit", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["repo_id"], "mlx-community/Qwen3.6-27B-8bit")
        self.assertEqual(payload["alias"], "qwen3.6-27b-8bit")
        self.assertNotIn("preset", payload)

    def test_download_gemma4_e2b_bit_alias_uses_mlx_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            (scripts / "download_model.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json, os
                    p = argparse.ArgumentParser()
                    p.add_argument("repo_id")
                    p.add_argument("--dest")
                    p.add_argument("--force", action="store_true")
                    p.add_argument("--json", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({
                        "schema_version": "ax.download_model.v1",
                        "repo_id": args.repo_id,
                        "dest": os.environ["FAKE_MODEL_DIR"],
                        "manifest_present": True,
                        "safetensors_count": 1,
                        "config_present": True,
                        "status": "ready",
                        "errors": [],
                        "server_command": ["ax-engine-server"],
                    }))
                    """
                )
            )

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ):
                code, stdout = self.capture_main(["download", "gemma4-e2b-6bit", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["repo_id"], "mlx-community/gemma-4-e2b-it-6bit")
        self.assertEqual(payload["alias"], "gemma4-e2b-6bit")
        self.assertNotIn("preset", payload)

    def test_serve_download_uses_ready_downloaded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            (model_dir / "model.safetensors").write_bytes(b"placeholder")
            (model_dir / "model-manifest.json").write_text("{}")

            (scripts / "download_model.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json, os
                    p = argparse.ArgumentParser()
                    p.add_argument("repo_id")
                    p.add_argument("--dest")
                    p.add_argument("--force", action="store_true")
                    p.add_argument("--json", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({
                        "schema_version": "ax.download_model.v1",
                        "repo_id": args.repo_id,
                        "dest": os.environ["FAKE_MODEL_DIR"],
                        "manifest_present": True,
                        "safetensors_count": 1,
                        "config_present": True,
                        "status": "ready",
                        "errors": [],
                        "server_command": ["ax-engine-server"],
                    }))
                    """
                )
            )

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ), mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"):
                code, stdout = self.capture_main(
                    ["serve", "qwen36-35b", "--download", "--dry-run", "--json"]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["resolved"]["kind"], "preset")
            self.assertEqual(payload["resolved"]["download"]["dry_run"], True)
            self.assertEqual(payload["server"]["argv"][0], "/opt/bin/ax-engine-server")

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ), mock.patch.object(
                _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
            ), mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp:
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    self.capture_main(["serve", "qwen36-35b", "--download"])

            argv = execvp.call_args.args[1]
            self.assertIn("--preset", argv)
            self.assertIn("qwen3.6-35b", argv)
            path_index = argv.index("--mlx-model-artifacts-dir") + 1
            self.assertEqual(argv[path_index], str(model_dir.resolve()))

    def test_convert_mtplx_json_wraps_prepare_and_provenance_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            output_dir = root / "out"

            (scripts / "prepare_mtp_sidecar.py").write_text(
                textwrap.dedent(
                    """
                    import argparse
                    from pathlib import Path

                    p = argparse.ArgumentParser()
                    p.add_argument("--hf-repo", required=True)
                    p.add_argument("--base", required=True)
                    p.add_argument("--output")
                    p.add_argument("--mtp-depth-max")
                    p.add_argument("--group-size")
                    p.add_argument("--quantize")
                    args = p.parse_args()
                    out = Path(args.output)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "ax_mtp_sidecar_manifest.json").write_text("{}")
                    print("Sidecar ready at:")
                    print(f"  {out}")
                    """
                )
            )
            (scripts / "check_mtp_sidecar_provenance.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    p = argparse.ArgumentParser()
                    p.add_argument("manifest_or_dir")
                    p.add_argument("--json", action="store_true")
                    p.add_argument("--fair-base-only", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({
                        "manifest": str(args.manifest_or_dir) + "/ax_mtp_sidecar_manifest.json",
                        "base_model_id": "mlx-community/Qwen3.6-27B-4bit",
                        "source_model_id": "Qwen/Qwen3.6-27B",
                        "fair_base_only": args.fair_base_only,
                    }))
                    """
                )
            )

            with mock.patch.dict(os.environ, {"AX_ENGINE_REPO_ROOT": str(root)}):
                code, stdout = self.capture_main(
                    [
                        "convert-mtplx",
                        "mlx-community/Qwen3.6-27B-4bit",
                        "--mtp-source",
                        "Qwen/Qwen3.6-27B",
                        "--output",
                        str(output_dir),
                        "--mtp-depth-max",
                        "3",
                        "--quantize",
                        "4",
                        "--fair-base-only",
                        "--json",
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.convert_mtplx.v1")
        self.assertEqual(payload["output_dir"], str(output_dir.resolve()))
        self.assertIn("--quantize", payload["prepare_command"])
        self.assertIn("--fair-base-only", payload["provenance_command"])
        self.assertTrue(payload["provenance"]["fair_base_only"])

    def test_convert_mtplx_uses_model_specific_depth_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            output_dir = root / "out"

            (scripts / "prepare_mtp_sidecar.py").write_text(
                textwrap.dedent(
                    """
                    import argparse
                    from pathlib import Path

                    p = argparse.ArgumentParser()
                    p.add_argument("--hf-repo", required=True)
                    p.add_argument("--base", required=True)
                    p.add_argument("--output")
                    p.add_argument("--mtp-depth-max")
                    p.add_argument("--group-size")
                    args = p.parse_args()
                    out = Path(args.output)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "ax_mtp_sidecar_manifest.json").write_text("{}")
                    print("Sidecar ready at:")
                    print(f"  {out}")
                    """
                )
            )
            (scripts / "check_mtp_sidecar_provenance.py").write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    p = argparse.ArgumentParser()
                    p.add_argument("manifest_or_dir")
                    p.add_argument("--json", action="store_true")
                    args = p.parse_args()
                    print(json.dumps({"manifest": args.manifest_or_dir}))
                    """
                )
            )

            with mock.patch.dict(os.environ, {"AX_ENGINE_REPO_ROOT": str(root)}):
                code, stdout = self.capture_main(
                    [
                        "convert-mtplx",
                        "mlx-community/Qwen3.6-27B-4bit",
                        "--mtp-source",
                        "Qwen/Qwen3.6-27B",
                        "--output",
                        str(output_dir),
                        "--json",
                    ]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["mtp_depth_max"], 3)
        depth_index = payload["prepare_command"].index("--mtp-depth-max") + 1
        self.assertEqual(payload["prepare_command"][depth_index], "3")


if __name__ == "__main__":
    unittest.main()
