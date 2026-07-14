from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import subprocess
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
        self.assertEqual(
            payload["default_destination"]["kind"], "huggingface_hub_cache"
        )
        self.assertIn("HF_HUB_CACHE", payload["default_destination"]["env"])
        aliases = {target["alias"] for target in payload["targets"]}
        self.assertIn("qwen3.5-9b", aliases)
        self.assertIn("qwen3.6-35b", aliases)
        self.assertIn("qwen3.6-27b-8bit", aliases)
        self.assertIn("gemma4-e2b-6bit", aliases)
        self.assertIn("gemma4-12b", aliases)
        self.assertIn("gemma4-12b-6bit", aliases)
        # Secondary direct-mode catalog (Llama / Mistral / GPT-OSS).
        self.assertIn("llama3.1-8b", aliases)
        self.assertIn("llama3.3-70b", aliases)
        self.assertIn("llama4-scout", aliases)
        self.assertIn("mistral-small", aliases)
        self.assertIn("ministral-8b", aliases)
        self.assertIn("devstral-small", aliases)
        self.assertIn("gpt-oss-20b", aliases)
        self.assertIn("gpt-oss-120b", aliases)

    def test_secondary_profile_aliases_resolve_repos(self) -> None:
        cases = {
            "llama3.3-70b": "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "llama3.1-8b-4bit": "mlx-community/Llama-3.1-8B-Instruct-4bit",
            "llama4-scout": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
            "mistral-small": "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
            "ministral-8b": "mlx-community/Ministral-8B-Instruct-2410-4bit",
            "devstral-small": "mlx-community/Devstral-Small-2505-4bit",
            "gpt-oss-20b": "mlx-community/gpt-oss-20b-MXFP4-Q4",
            "gpt-oss-120b-4bit": "mlx-community/gpt-oss-120b-MXFP4-Q4",
        }
        for alias, repo_id in cases.items():
            profile = _cli._profile_for_model(alias)
            self.assertIsNotNone(profile, alias)
            assert profile is not None
            self.assertEqual(profile.repo_id, repo_id, alias)
            self.assertIsNotNone(profile.preset, alias)

    def test_mxfp4_repo_quant_bits(self) -> None:
        profile = _cli._profile_for_model("gpt-oss-20b")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(_cli._profile_quant_bits(profile), 4)

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
        self.assertIn("gemma4-12b", stdout)
        self.assertIn("gemma4-e2b-8bit", stdout)

    def test_download_unknown_alias_shows_targets(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.capture_main(["download", "unknown-model"])

        self.assertIn("unknown model alias", str(raised.exception))
        self.assertIn("qwen3.6-27b-8bit", str(raised.exception))
        self.assertIn("gemma4-12b", str(raised.exception))
        self.assertIn("gemma4-e2b-6bit", str(raised.exception))

    def test_serve_dry_run_json_uses_server_preset(self) -> None:
        with mock.patch.object(
            _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
        ):
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
            with mock.patch.object(
                _cli, "_server_bin", return_value="ax-engine-server"
            ):
                code, stdout = self.capture_main(
                    ["serve", str(model_dir), "--dry-run", "--json"]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["resolved"]["kind"], "local_dir")
        self.assertIn("--mlx-model-artifacts-dir", payload["server"]["argv"])
        path_index = payload["server"]["argv"].index("--mlx-model-artifacts-dir") + 1
        self.assertEqual(
            payload["server"]["argv"][path_index], str(model_dir.resolve())
        )

    def test_serve_dry_run_json_uses_gemma4_12b_server_preset(self) -> None:
        with mock.patch.object(
            _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
        ):
            code, stdout = self.capture_main(
                ["serve", "gemma4-12b", "--port", "9010", "--dry-run", "--json"]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["resolved"]["preset"], "gemma4-12b")
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
                "gemma4-12b",
                "--resolve-model-artifacts",
                "hf-cache",
            ],
        )

    def test_doctor_json_summarizes_bench_doctor(self) -> None:
        bench_report = {
            "schema_version": "ax.engine_bench.doctor.v1",
            "status": "ready",
            "mlx_runtime_ready": True,
            "workflow": {"mode": "installed_tools", "cwd": "/tmp"},
            "host": {
                "supported_mlx_runtime": True,
                "detected_soc": "Apple M3 Max",
                "os": "macos",
                "arch": "aarch64",
            },
            "metal_toolchain": {"fully_available": True},
            "model_artifacts": {
                "selected": True,
                "status": "ready",
                "path": "/models/gemma4-12b",
                "issues": [],
            },
            "issues": [],
        }

        def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
            if command[:2] == ["/opt/bin/ax-engine-bench", "doctor"]:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(bench_report), ""
                )
            if command[1:] == ["--help"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command == ["sw_vers", "-productVersion"]:
                return subprocess.CompletedProcess(command, 0, "15.5\n", "")
            if command == ["sw_vers", "-buildVersion"]:
                return subprocess.CompletedProcess(command, 0, "24F74\n", "")
            if command == ["sysctl", "-n", "hw.memsize"]:
                return subprocess.CompletedProcess(
                    command, 0, str(64 * 1024 * 1024 * 1024), ""
                )
            if command == ["sysctl", "-n", "hw.physicalcpu"]:
                return subprocess.CompletedProcess(command, 0, "16\n", "")
            if command == ["sysctl", "-n", "hw.perflevel0.name"]:
                return subprocess.CompletedProcess(command, 0, "Performance\n", "")
            if command == ["sysctl", "-n", "hw.perflevel0.physicalcpu"]:
                return subprocess.CompletedProcess(command, 0, "12\n", "")
            if command == ["sysctl", "-n", "hw.perflevel1.name"]:
                return subprocess.CompletedProcess(command, 0, "Efficiency\n", "")
            if command == ["sysctl", "-n", "hw.perflevel1.physicalcpu"]:
                return subprocess.CompletedProcess(command, 0, "4\n", "")
            if command[0:2] == ["sysctl", "-n"] and command[2].startswith(
                "hw.perflevel"
            ):
                return subprocess.CompletedProcess(command, 1, "", "unknown oid")
            if command == ["system_profiler", "SPDisplaysDataType"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "Graphics/Displays:\n\n    Apple M3 Max:\n\n      Total Number of Cores: 40\n",
                    "",
                )
            if command == ["system_profiler", "SPHardwareDataType"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "Hardware:\n\n    Hardware Overview:\n\n      Total Number of Cores: 16 (4 Efficiency and 12 Performance)\n      Memory: 64 GB\n",
                    "",
                )
            raise AssertionError(f"unexpected command: {command}")

        with (
            mock.patch.object(
                _cli, "_bench_bin", return_value="/opt/bin/ax-engine-bench"
            ),
            mock.patch.object(
                _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
            ),
            mock.patch.object(_cli, "_package_version", return_value="6.4.5"),
            mock.patch.object(
                _cli, "_run_capture", side_effect=run_capture
            ) as run_capture_mock,
        ):
            code, stdout = self.capture_main(
                [
                    "doctor",
                    "--json",
                    "--mlx-model-artifacts-dir",
                    "/models/gemma4-12b",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.engine.doctor.v1")
        self.assertEqual(payload["result"], "ready")
        self.assertEqual(payload["install"]["version"], "6.4.5")
        self.assertEqual(payload["host"]["os_version"], "15.5")
        self.assertEqual(payload["host"]["os_build"], "24F74")
        self.assertEqual(payload["host"]["ram_gib"], 64)
        self.assertEqual(payload["host"]["cpu_cores"]["performance"], 12)
        self.assertEqual(payload["host"]["cpu_cores"]["efficiency"], 4)
        self.assertEqual(payload["host"]["gpu_cores"], 40)
        self.assertEqual(payload["ready_for"], ["serve", "python_sdk", "model_checks"])
        self.assertEqual(payload["checks"][0]["id"], "server_binary")
        self.assertEqual(payload["checks"][1]["id"], "bench_binary")
        self.assertEqual(payload["checks"][-1]["status"], "ready")
        self.assertEqual(
            payload["source"]["schema_version"], "ax.engine_bench.doctor.v1"
        )
        self.assertEqual(
            payload["next_actions"], ["ax-engine serve /models/gemma4-12b --port 8080"]
        )
        self.assertNotIn("bench_doctor", payload)
        self.assertEqual(
            run_capture_mock.call_args_list[0].args[0],
            [
                "/opt/bin/ax-engine-bench",
                "doctor",
                "--mlx-model-artifacts-dir",
                "/models/gemma4-12b",
                "--json",
            ],
        )

    def test_doctor_verbose_wraps_bench_doctor(self) -> None:
        with (
            mock.patch.object(
                _cli, "_bench_bin", return_value="/opt/bin/ax-engine-bench"
            ),
            mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.capture_main(
                    [
                        "doctor",
                        "--verbose",
                        "--json",
                        "--mlx-model-artifacts-dir",
                        "/models/gemma4-12b",
                    ]
                )

        self.assertEqual(execvp.call_args.args[0], "/opt/bin/ax-engine-bench")
        self.assertEqual(
            execvp.call_args.args[1],
            [
                "/opt/bin/ax-engine-bench",
                "doctor",
                "--json",
                "--mlx-model-artifacts-dir",
                "/models/gemma4-12b",
            ],
        )

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
                code, stdout = self.capture_main(
                    ["download", "qwen36-27b-8bit", "--json"]
                )

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
                code, stdout = self.capture_main(
                    ["download", "gemma4-e2b-6bit", "--json"]
                )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["repo_id"], "mlx-community/gemma-4-e2b-it-6bit")
        self.assertEqual(payload["alias"], "gemma4-e2b-6bit")
        self.assertNotIn("preset", payload)

    def test_download_gemma4_12b_alias_uses_mlx_target(self) -> None:
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
                code, stdout = self.capture_main(["download", "gemma4-12b", "--json"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["repo_id"], "mlx-community/gemma-4-12B-it-4bit")
            self.assertEqual(payload["alias"], "gemma4-12b")
            self.assertEqual(payload["preset"], "gemma4-12b")

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ):
                code, stdout = self.capture_main(
                    ["download", "gemma4-12b-6bit", "--json"]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["repo_id"], "mlx-community/gemma-4-12B-it-6bit")
            self.assertEqual(payload["alias"], "gemma4-12b-6bit")
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

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "AX_ENGINE_REPO_ROOT": str(root),
                        "FAKE_MODEL_DIR": str(model_dir),
                    },
                ),
                mock.patch.object(
                    _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
                ),
            ):
                code, stdout = self.capture_main(
                    ["serve", "qwen36-35b", "--download", "--dry-run", "--json"]
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["resolved"]["kind"], "preset")
            self.assertEqual(payload["resolved"]["download"]["dry_run"], True)
            self.assertEqual(payload["server"]["argv"][0], "/opt/bin/ax-engine-server")

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "AX_ENGINE_REPO_ROOT": str(root),
                        "FAKE_MODEL_DIR": str(model_dir),
                    },
                ),
                mock.patch.object(
                    _cli, "_server_bin", return_value="/opt/bin/ax-engine-server"
                ),
                mock.patch.object(
                    os, "execvp", side_effect=RuntimeError("stop")
                ) as execvp,
            ):
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


class AxEngineInteractiveDownloadTests(unittest.TestCase):
    def capture_main(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = _cli.main(argv)
        return code, out.getvalue()

    def test_download_list_json_includes_mtp_target(self) -> None:
        code, stdout = self.capture_main(["download", "--list", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        targets = {target["alias"]: target for target in payload["targets"]}
        self.assertEqual(targets["gemma4-12b"]["mtp_target"], "gemma-4-12b-4bit")
        self.assertEqual(targets["qwen3.6-35b"]["mtp_target"], "qwen3.6-35b-a3b")
        self.assertIsNone(targets["glm4.7-flash-4bit"]["mtp_target"])
        self.assertIsNone(targets["gemma4-e2b"]["mtp_target"])

    def test_no_model_non_tty_is_not_interactive(self) -> None:
        # stdout is redirected (not a TTY), so the wizard must not engage.
        with mock.patch.object(_cli, "_run_interactive_download") as wizard:
            code, stdout = self.capture_main(["download"])

        wizard.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIn("missing model alias or repo id", stdout)

    def test_no_interactive_flag_blocks_wizard_even_on_tty(self) -> None:
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_run_interactive_download") as wizard,
        ):
            code, _ = self.capture_main(["download", "--no-interactive"])

        wizard.assert_not_called()
        self.assertEqual(code, 2)

    def test_bare_download_on_tty_runs_wizard(self) -> None:
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_run_interactive_download", return_value=0) as wizard,
        ):
            code, _ = self.capture_main(["download"])

        wizard.assert_called_once()
        self.assertEqual(code, 0)

    def test_ui_downloader_requires_tty(self) -> None:
        with mock.patch.object(_cli, "_supports_interactive", return_value=False):
            with self.assertRaises(SystemExit) as raised:
                self.capture_main(["ui-downloader"])

        self.assertIn("interactive terminal", str(raised.exception))

    def test_wizard_flow_invokes_download_with_progress(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "repo_id": "mlx-community/gemma-4-e2b-it-4bit",
            "dest": "/tmp/model",
        }
        inputs = iter(["1", "", "y"])  # select first model, default path, confirm
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(
                _cli, "_download_summary", return_value=(0, summary, "")
            ) as download,
        ):
            code, stdout = self.capture_main(["ui-downloader"])

        self.assertEqual(code, 0)
        download.assert_called_once()
        _, kwargs = download.call_args
        self.assertTrue(kwargs["progress"])
        self.assertIsNone(kwargs["dest"])
        self.assertIn("Status: ready", stdout)

    def _index_of(self, label: str) -> int:
        for index, profile in enumerate(_cli._downloadable_profiles(), start=1):
            if profile.label == label:
                return index
        raise AssertionError(f"profile not found: {label}")

    def test_wizard_mtp_variant_runs_download_mtp(self) -> None:
        idx = self._index_of("gemma4-12b")
        inputs = iter([str(idx), "2", "y"])  # select gemma4-12b, MTP variant, confirm
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(_cli, "_bench_bin", return_value="/fake/ax-engine-bench"),
            mock.patch.object(_cli.subprocess, "run", return_value=completed) as run,
        ):
            code, _ = self.capture_main(["ui-downloader"])

        self.assertEqual(code, 0)
        run.assert_called_once()
        argv = run.call_args[0][0]
        self.assertEqual(
            argv[:3], ["/fake/ax-engine-bench", "download-mtp", "gemma-4-12b-4bit"]
        )

    def test_wizard_direct_variant_on_mtp_model(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "repo_id": "mlx-community/gemma-4-12B-it-4bit",
            "dest": "/tmp/model",
        }
        idx = self._index_of("gemma4-12b")
        inputs = iter([str(idx), "1", "", "y"])  # select, Direct variant, default path, confirm
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(
                _cli, "_download_summary", return_value=(0, summary, "")
            ) as download,
        ):
            code, stdout = self.capture_main(["ui-downloader"])

        self.assertEqual(code, 0)
        download.assert_called_once()
        _, kwargs = download.call_args
        self.assertTrue(kwargs["progress"])
        self.assertIn("Status: ready", stdout)

    def test_wizard_cancel_returns_130(self) -> None:
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", return_value="q"),
            mock.patch.object(_cli, "_download_summary") as download,
        ):
            code, stdout = self.capture_main(["ui-downloader"])

        download.assert_not_called()
        self.assertEqual(code, 130)
        self.assertIn("Cancelled.", stdout)


if __name__ == "__main__":
    unittest.main()
