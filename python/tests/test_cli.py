from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

# Default to importing ax_engine from the source tree (for `maturin develop`
# runs). When validating an installed wheel (AX_ENGINE_RUN_INSTALLED_TESTS=1),
# keep the source off sys.path so the installed package — with its compiled
# _ax_engine extension — is imported instead of the un-built source. Inserting
# it unconditionally shadowed the wheel for the whole discovery process and
# broke the installed-wheel smoke tests.
if os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS") != "1":
    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "python"))

from ax_engine import _cli  # noqa: E402, I001


EXPECTED_AUTOMATOSX_REPOS = {
    "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
    "AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit",
    "AutomatosX/AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP",
    "AutomatosX/AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP",
    "AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit",
    "AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit",
    "AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit",
    "AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ",
    "AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ",
    "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit",
    "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
    "AutomatosX/AX-Qwen3.5-9B-MLX-4bit-MTP",
    "AutomatosX/AX-Qwen3.5-9B-MLX-6bit-MTP",
    "AutomatosX/AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-27B-MLX-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
    "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
    "AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
    "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
    "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
    "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP",
    "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
    "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP",
    "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit",
    "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit",
    "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit",
    "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit",
    "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
    "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
}


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
        targets = payload["targets"]
        self.assertEqual({target["repo_id"] for target in targets}, EXPECTED_AUTOMATOSX_REPOS)
        self.assertEqual(len(targets), 49)
        self.assertTrue(
            all(
                target["alias"].startswith(("ax-", "holo3-", "ornith-"))
                for target in targets
            )
        )
        self.assertTrue(
            all(not target["repo_id"].startswith("mlx-community/") for target in targets)
        )

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

    def test_qwen36_axq_candidates_are_explicit_and_revision_pinned(self) -> None:
        cases = {
            "qwen3.6-27b:axq": (
                "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
                "8c37715c7b5f5ebca00eda6f73be47116a3e4ebc",
                "candidate",
            ),
            "qwen3.6-27b:axq-4bit": (
                "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
                "6182ccbc41c7397ff90670f740c6d9eacfa4b09f",
                "candidate",
            ),
            "qwen3-vl-30b-a3b:axq": (
                "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
                "700ec2c305f5f80e4d7c841c5aec80b050b949c6",
                "candidate",
            ),
            "qwen3-vl-30b-a3b:axq-4bit": (
                "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit",
                "1f4c21a0c9d4347294d3f082928fdfd854284383",
                "candidate",
            ),
            "qwen3.8-27b:axq": (
                "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
                "a5a0b700ea7c5c529c66ca3005b79425ab2f7ea6",
                "candidate",
            ),
            "qwen3.8-27b:axq-4bit": (
                "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
                "7e865596cb32bd41b29c7a25c5b66b9c3ea25e5e",
                "candidate",
            ),
            "ax-qwen3-vl-30b": (
                "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
                "700ec2c305f5f80e4d7c841c5aec80b050b949c6",
                "candidate",
            ),
            "holo3-35b:axq": (
                "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit",
                "e6cc340b04bfcec57544e462ec756e48dd248cf9",
                None,
            ),
            "holo3-35b:axq-4bit": (
                "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit",
                "7b2256130cd55ea6b7489817a9a00c46e9874403",
                None,
            ),
            "ornith-35b:axq": (
                "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit",
                "37361076641d7b7487d1b5ce1b68243ffbdbffe0",
                "candidate",
            ),
            "ornith-35b:axq-4bit": (
                "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit",
                "d7416c665cd8ae6e5fbebc3f17bd547b78cf11fc",
                "candidate",
            ),
            "qwen3.6-35b:axq": (
                "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
                "6a4c220734f81112555ee8783d91e0065c54301c",
                "candidate",
            ),
            "qwen3.6-35b:axq-4bit": (
                "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP",
                "952031cbfbb9cf31414a57eeb681c34dc08ec1e9",
                "candidate",
            ),
            "gemma4-12b:axq": (
                "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP",
                "7ad79df2b0c272431f3e927b133b7dc3d70872f4",
                "candidate",
            ),
            "gemma4-12b:axq-4bit": (
                "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
                "d2a6ac9d59655f0b86a57a64ed85616d0a10e27e",
                "candidate",
            ),
            "gemma4-26b:axq": (
                "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
                "940a60b13e7298140c85d3762492dde6733f8a57",
                "candidate",
            ),
            "gemma4-31b:axq": (
                "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP",
                "7b11bd5179d71a74200fe56075cba5c21212fe6a",
                "candidate",
            ),
        }
        for alias, (repo_id, revision, certification) in cases.items():
            with self.subTest(alias=alias):
                resolved_repo, profile, resolved_revision = _cli._download_repo_id(alias)
                self.assertEqual(resolved_repo, repo_id)
                self.assertEqual(resolved_revision, revision)
                self.assertIsNotNone(profile)
                assert profile is not None
                self.assertEqual(_cli._profile_certification(profile), certification)

        default_profile = _cli._profile_for_model("qwen3.6-27b")
        self.assertIsNotNone(default_profile)
        assert default_profile is not None
        self.assertEqual(default_profile.repo_id, "mlx-community/Qwen3.6-27B-4bit")
        self.assertIsNone(_cli._profile_certification(default_profile))
        ax_qwen35 = _cli._profile_for_model("ax-qwen3.6-35b")
        self.assertIsNotNone(ax_qwen35)
        assert ax_qwen35 is not None
        self.assertEqual(
            ax_qwen35.repo_id,
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
        )
        ax_gemma12 = _cli._profile_for_model("ax-gemma4-12b")
        self.assertIsNotNone(ax_gemma12)
        assert ax_gemma12 is not None
        self.assertEqual(
            ax_gemma12.repo_id,
            "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
        )

        holo3_default = _cli._profile_for_model("holo3-35b")
        self.assertIsNotNone(holo3_default)
        assert holo3_default is not None
        self.assertEqual(
            holo3_default.repo_id,
            "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit",
        )
        self.assertEqual(
            _cli._profile_revision(holo3_default),
            "7b2256130cd55ea6b7489817a9a00c46e9874403",
        )
        self.assertIsNone(_cli._profile_certification(holo3_default))

        ornith_default = _cli._profile_for_model("ornith-35b")
        self.assertIsNotNone(ornith_default)
        assert ornith_default is not None
        self.assertEqual(
            ornith_default.repo_id,
            "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit",
        )
        self.assertEqual(_cli._profile_certification(ornith_default), "candidate")
        self.assertEqual(holo3_default.preset, "holo3-35b")
        self.assertEqual(ornith_default.preset, "ornith-35b")
        for alias, preset in (
            ("holo3-35b:axq", "holo3-35b"),
            ("ornith-35b:axq", "ornith-35b"),
            ("ax-holo3-35b", "holo3-35b"),
            ("ax-ornith-35b", "ornith-35b"),
            ("ax-gemma4-12b", "gemma4-12b"),
            ("ax-gemma4-26b", "gemma4-26b"),
            ("ax-gemma4-31b", "gemma4-31b"),
            ("qwen3.6-27b:axq", "qwen3.6-27b"),
            ("qwen3.6-35b:axq", "qwen3.6-35b"),
            ("ax-qwen3.6-35b", "qwen3.6-35b"),
            ("ax-qwen3.5-9b", "qwen3.5-9b"),
            ("gemma4-12b:axq", "gemma4-12b"),
        ):
            profile = _cli._profile_for_model(alias)
            self.assertIsNotNone(profile, alias)
            assert profile is not None
            self.assertEqual(profile.preset, preset, alias)

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
        self.assertIn("ax-qwen3.6-35b", stdout)
        self.assertIn("ax-gemma4-12b", stdout)
        self.assertIn("ax-diffusiongemma-26b", stdout)

    def test_download_progress_json_requires_model(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code, stdout = self.capture_main(["download", "--progress-json"])

        self.assertEqual(code, 2)
        terminal = json.loads(stdout)
        self.assertEqual(terminal["schema_version"], "ax.download_model.v1")
        self.assertEqual(terminal["status"], "download_failed")
        self.assertIn("--progress-json requires a model", terminal["errors"][0])
        self.assertIn("--progress-json requires a model", stderr.getvalue())

    def test_download_progress_json_normalizes_argument_errors(self) -> None:
        for argv in (
            ["download", "owner/repo", "--progress-json", "--unknown"],
            ["download", "owner/repo", "--progress-json", "--dest"],
        ):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code, stdout = self.capture_main(argv)

                self.assertEqual(code, 2)
                records = [json.loads(line) for line in stdout.splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["schema_version"], "ax.download_model.v1")
                self.assertEqual(records[0]["status"], "download_failed")
                self.assertTrue(records[0]["errors"])
                self.assertIn("error:", stderr.getvalue())

    def test_download_progress_json_rejects_non_streaming_modes(self) -> None:
        for argv, conflict in (
            (["download", "--list", "--progress-json"], "--list"),
            (
                ["download", "owner/repo", "--interactive", "--progress-json"],
                "--interactive",
            ),
        ):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code, stdout = self.capture_main(argv)

                self.assertEqual(code, 2)
                terminal = json.loads(stdout)
                self.assertEqual(terminal["status"], "download_failed")
                self.assertIn(conflict, terminal["errors"][0])
                self.assertIn(conflict, stderr.getvalue())

    def test_download_unknown_alias_shows_targets(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.capture_main(["download", "unknown-model"])

        self.assertIn("unknown model alias", str(raised.exception))
        self.assertIn("ax-qwen3.6-27b-6bit", str(raised.exception))
        self.assertIn("ax-gemma4-12b", str(raised.exception))
        self.assertIn("ax-embeddinggemma-300m", str(raised.exception))

    def test_legacy_alias_is_serve_only_not_managed_download(self) -> None:
        profile = _cli._profile_for_model("qwen36-35b")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.preset, "qwen3.6-35b")

        with self.assertRaisesRegex(SystemExit, "not managed by ax-engine download"):
            _cli._download_repo_id("qwen36-35b")

    def test_download_repo_id_accepts_urls_and_revisions(self) -> None:
        from ax_engine._repo_ref import parse_repo_ref

        cases = [
            ("owner/repo", ("owner/repo", None)),
            ("owner/repo@v1", ("owner/repo", "v1")),
            (
                "https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
                ("AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP", None),
            ),
            ("https://hf.co/owner/repo", ("owner/repo", None)),
            ("huggingface.co/owner/repo/", ("owner/repo", None)),
            ("https://huggingface.co/owner/repo.git", ("owner/repo", None)),
            ("https://huggingface.co/owner/repo/tree/main", ("owner/repo", "main")),
            ("owner/repo@feature/download-ui", ("owner/repo", "feature/download-ui")),
            ("owner/repo@refs%2Fpr%2F123", ("owner/repo", "refs/pr/123")),
            (
                "https://huggingface.co/owner/repo/tree/feature/download-ui",
                ("owner/repo", "feature/download-ui"),
            ),
            (
                "https://huggingface.co/owner/repo/tree/refs%2Fpr%2F123",
                ("owner/repo", "refs/pr/123"),
            ),
            (
                "https://huggingface.co/owner/repo.git/tree/v2?download=true",
                ("owner/repo", "v2"),
            ),
            (
                "https://hf.co:443/owner/repo/tree/v2#files",
                ("owner/repo", "v2"),
            ),
            (
                "https://hf.co/owner/repo?ignored=\tvalue",
                ("owner/repo", None),
            ),
            ("\towner/repo\n", ("owner/repo", None)),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parse_repo_ref(value), expected)

    def test_download_repo_id_rejects_bad_references(self) -> None:
        from ax_engine._repo_ref import parse_repo_ref

        for bad in [
            "",
            "noslash",
            "https://example.com/owner/repo",
            "ftp://huggingface.co/owner/repo",
            "https://huggingface.co:/owner/repo",
            "https://huggingface.co:invalid/owner/repo",
            "https://huggingface.co:65536/owner/repo",
            "https://huggingface.co:443:extra/owner/repo",
            "https://huggingface.\nco/owner/repo",
            "https://huggingface.co/owner/re\tpo",
            "\0https://huggingface.co/owner/repo",
            "https://huggingface.co/owner",
            "https://huggingface.co/owner/repo/blob/main/model.safetensors",
            "C:/owner/repo",
            r"owner\repo/model",
            r"owner/repo@C:\temp",
            "owner/repo/extra/path",
            "owner/repo@",
            "owner//repo",
            ".owner/repo",
            "owner/.repo",
            "owner/repo-",
            "owner/foo..bar",
            "owner/foo--bar",
            f"o/{'x' * 95}",
            "owner/repo.git.git",
            "owner/repo@../escape",
            "owner/repo@feature//escape",
            "owner/repo@feature/.hidden",
            "owner/repo@feature.lock",
            "owner/repo@feature.LOCK",
            "owner/repo@@",
            "owner/repo@feature@{1}",
            "owner/repo@feature\u0080other",
            "\x1cowner/repo",
            "https://huggingface.co/owner/repo/tree/%2e%2e/escape",
            "https://huggingface.co/owner/repo/tree/feature%ZZescape",
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_repo_ref(bad)

        longest = f"o/{'x' * 94}"
        self.assertEqual(parse_repo_ref(longest), (longest, None))

    def test_find_repo_script_does_not_search_cwd_without_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            untrusted_root = root / "untrusted"
            untrusted_script = untrusted_root / "scripts" / "download_model.py"
            untrusted_script.parent.mkdir(parents=True)
            untrusted_script.write_text("raise SystemExit(99)")
            package_dir = root / "venv" / "site-packages" / "ax_engine"
            package_dir.mkdir(parents=True)

            with (
                mock.patch.object(_cli, "__file__", str(package_dir / "_cli.py")),
                mock.patch.object(pathlib.Path, "cwd", return_value=untrusted_root),
                mock.patch.dict(os.environ, {"AX_ENGINE_REPO_ROOT": ""}),
            ):
                self.assertIsNone(_cli._find_repo_script("download_model.py"))

            with (
                mock.patch.object(_cli, "__file__", str(package_dir / "_cli.py")),
                mock.patch.dict(os.environ, {"AX_ENGINE_REPO_ROOT": str(untrusted_root)}),
            ):
                self.assertEqual(_cli._find_repo_script("download_model.py"), untrusted_script)

    def test_parse_download_summary_accepts_progress_then_pretty_json(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "dest": "/tmp/model",
        }
        stdout = "\n".join(
            [
                json.dumps({"event": "progress", "done": 50, "total": 100}),
                "helper note",
                json.dumps(summary, indent=2),
            ]
        )

        self.assertEqual(_cli._parse_download_summary(stdout), summary)
        self.assertIsNone(_cli._parse_download_summary(json.dumps({"status": "ready"})))

    def test_streaming_download_forwards_only_progress_records(self) -> None:
        progress = {"event": "progress", "done": 50, "total": 100, "file": "weights"}
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "dest": "/tmp/model",
        }
        script = (
            "import json;"
            f"print(json.dumps({progress!r}));"
            "print('helper note');"
            f"print(json.dumps({summary!r}))"
        )
        original_popen = subprocess.Popen
        started_processes = []

        def tracking_popen(*args: object, **kwargs: object) -> subprocess.Popen[str]:
            process = original_popen(*args, **kwargs)
            started_processes.append(process)
            return process

        stdout = io.StringIO()
        with (
            mock.patch.object(_cli.subprocess, "Popen", side_effect=tracking_popen),
            contextlib.redirect_stdout(stdout),
        ):
            result = _cli._run_streaming_capture_stdout([sys.executable, "-c", script])

        self.assertEqual(result.returncode, 0)
        self.assertEqual([json.loads(line) for line in stdout.getvalue().splitlines()], [progress])
        self.assertEqual(_cli._parse_download_summary(result.stdout), summary)
        self.assertEqual(len(started_processes), 1)
        process_stdout = started_processes[0].stdout
        self.assertIsNotNone(process_stdout)
        assert process_stdout is not None
        self.assertTrue(process_stdout.closed)

    def test_download_progress_json_emits_compact_terminal_summary(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "repo_id": "owner/repo",
            "revision": "v2",
            "dest": "/tmp/model",
            "status": "ready",
        }
        with mock.patch.object(
            _cli,
            "_download_summary",
            return_value=(0, summary, ""),
        ) as download:
            code, stdout = self.capture_main(["download", "owner/repo@v2", "--progress-json"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), summary)
        self.assertNotIn("\n ", stdout)
        self.assertTrue(download.call_args.kwargs["progress_json"])

    def test_download_progress_json_normalizes_preflight_and_helper_failures(self) -> None:
        cases = (
            (
                "unknown alias",
                ["download", "unknown-model", "--progress-json"],
                None,
                None,
                "unknown model alias",
            ),
            (
                "missing helper",
                ["download", "owner/repo", "--progress-json"],
                "_find_repo_script",
                None,
                "cannot locate scripts/download_model.py",
            ),
            (
                "missing summary",
                ["download", "owner/repo", "--progress-json"],
                "_run_streaming_capture_stdout",
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                "did not emit an ax.download_model.v1 summary",
            ),
            (
                "launch failure",
                ["download", "owner/repo", "--progress-json"],
                "_run_streaming_capture_stdout",
                OSError("permission denied"),
                "permission denied",
            ),
        )
        for name, argv, patched_name, result, expected in cases:
            with self.subTest(name=name):
                stderr = io.StringIO()
                if patched_name is None:
                    patcher = contextlib.nullcontext()
                elif isinstance(result, BaseException):
                    patcher = mock.patch.object(_cli, patched_name, side_effect=result)
                else:
                    patcher = mock.patch.object(_cli, patched_name, return_value=result)
                with patcher, contextlib.redirect_stderr(stderr):
                    code, stdout = self.capture_main(argv)

                self.assertEqual(code, 2)
                records = [json.loads(line) for line in stdout.splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["schema_version"], "ax.download_model.v1")
                self.assertEqual(records[0]["status"], "download_failed")
                self.assertIn(expected, records[0]["errors"][0])
                self.assertIn(expected, stderr.getvalue())

    def test_download_url_forwards_revision_to_helper(self) -> None:
        commands: list[list[str]] = []

        class Result:
            returncode = 0
            stdout = json.dumps({"schema_version": "ax.download_model.v1", "status": "ready"})
            stderr = ""

        def fake_capture(command: list[str]) -> Result:
            commands.append(command)
            return Result()

        with mock.patch.object(_cli, "_run_capture", side_effect=fake_capture):
            code, summary, _ = _cli._download_summary("https://huggingface.co/owner/repo/tree/v2")

        self.assertEqual(code, 0)
        command = commands[0]
        self.assertIn("owner/repo", command)
        self.assertIn("--revision=v2", command)
        assert summary is not None
        self.assertEqual(summary["input"], "https://huggingface.co/owner/repo/tree/v2")
        self.assertEqual(summary["revision"], "v2")

    def test_download_helper_uses_equals_for_option_like_values(self) -> None:
        class Result:
            returncode = 0
            stdout = json.dumps(
                {"schema_version": "ax.download_model.v1", "status": "ready"}
            )
            stderr = ""

        commands: list[list[str]] = []

        def fake_capture(command: list[str]) -> Result:
            commands.append(command)
            return Result()

        with mock.patch.object(_cli, "_run_capture", side_effect=fake_capture):
            _cli._download_summary("owner/repo@-release", dest="-models")

        self.assertIn("--revision=-release", commands[0])
        self.assertIn("--dest=-models", commands[0])
        self.assertNotIn("--revision", commands[0])
        self.assertNotIn("--dest", commands[0])

    def test_serve_dry_run_json_uses_server_preset(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
        ):
            code, stdout = self.capture_main(
                [
                    "serve",
                    "qwen36-35b",
                    "--port",
                    "9010",
                    "--hf-cache-root",
                    cache,
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
        self.assertEqual(payload["resolved"]["kind"], "model_resolution_plan")
        self.assertEqual(payload["resolved"]["preset"], "qwen3.6-35b")
        self.assertEqual(payload["resolved"]["resolution"], "local_cache_then_download")
        self.assertTrue(payload["resolved"]["download"]["required"])
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
                "--mlx-model-artifacts-dir",
                "<resolved-hf-snapshot:mlx-community/Qwen3.6-35B-A3B-4bit>",
                "--max-batch-tokens",
                "1024",
            ],
        )

    def test_serve_dry_run_json_uses_local_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp) / "model"
            model_dir.mkdir()
            with mock.patch.object(_cli, "_server_bin", return_value="ax-engine-server"):
                code, stdout = self.capture_main(["serve", str(model_dir), "--dry-run", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["resolved"]["kind"], "local_dir")
        self.assertEqual(payload["server"]["url"], "http://127.0.0.1:31418")
        self.assertIn("--mlx-model-artifacts-dir", payload["server"]["argv"])
        path_index = payload["server"]["argv"].index("--mlx-model-artifacts-dir") + 1
        self.assertEqual(payload["server"]["argv"][path_index], str(model_dir.resolve()))

    def test_serve_axq_dry_run_uses_pinned_candidate_snapshot(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
        ):
            code, stdout = self.capture_main(
                [
                    "serve",
                    "qwen3.6-27b:axq",
                    "--hf-cache-root",
                    cache,
                    "--dry-run",
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        resolved = payload["resolved"]
        self.assertEqual(resolved["certification"], "candidate")
        self.assertEqual(
            resolved["repo_id"],
            "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        )
        self.assertEqual(
            resolved["revision"],
            "8c37715c7b5f5ebca00eda6f73be47116a3e4ebc",
        )
        self.assertTrue(
            resolved["path"].endswith(
                "snapshots/8c37715c7b5f5ebca00eda6f73be47116a3e4ebc"
            )
        )
        self.assertTrue(resolved["download"]["required"])

    def test_snapshot_cache_requires_every_indexed_weight_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = pathlib.Path(tmp)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.a": "model-00001-of-00002.safetensors",
                            "model.b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "model-00001-of-00002.safetensors").touch()

            self.assertFalse(_cli._snapshot_has_complete_weights(snapshot))

            (snapshot / "model-00002-of-00002.safetensors").touch()
            self.assertTrue(_cli._snapshot_has_complete_weights(snapshot))

    def test_serve_offline_forwards_local_only_to_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = pathlib.Path(tmp) / "snapshot"
            model_dir.mkdir()
            summary = {
                "schema_version": "ax.download_model.v1",
                "repo_id": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
                "revision": "8c37715c7b5f5ebca00eda6f73be47116a3e4ebc",
                "dest": str(model_dir),
                "manifest_present": True,
                "status": "ready",
            }
            with (
                mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
                mock.patch.object(_cli, "_download_summary", return_value=(0, summary, "")) as run,
                mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")),
                self.assertRaisesRegex(RuntimeError, "stop"),
            ):
                self.capture_main(["serve", "qwen3.6-27b:axq", "--offline"])

        self.assertTrue(run.call_args.kwargs["local_only"])
        self.assertTrue(run.call_args.kwargs["allow_unmanaged_alias"])

    def test_serve_unknown_alias_suggests_close_match(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.capture_main(["serve", "qwen3.6-27"])

        message = str(raised.exception)
        self.assertIn("unknown model alias", message)
        self.assertIn("qwen3.6-27b", message)

    def test_serve_dry_run_json_uses_gemma4_12b_server_preset(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
        ):
            code, stdout = self.capture_main(
                [
                    "serve",
                    "gemma4-12b",
                    "--port",
                    "9010",
                    "--hf-cache-root",
                    cache,
                    "--dry-run",
                    "--json",
                ]
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
                "--mlx-model-artifacts-dir",
                "<resolved-hf-snapshot:mlx-community/gemma-4-12B-it-4bit>",
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
                return subprocess.CompletedProcess(command, 0, json.dumps(bench_report), "")
            if command[1:] == ["--help"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command == ["sw_vers", "-productVersion"]:
                return subprocess.CompletedProcess(command, 0, "15.5\n", "")
            if command == ["sw_vers", "-buildVersion"]:
                return subprocess.CompletedProcess(command, 0, "24F74\n", "")
            if command == ["sysctl", "-n", "hw.memsize"]:
                return subprocess.CompletedProcess(command, 0, str(64 * 1024 * 1024 * 1024), "")
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
            if command[0:2] == ["sysctl", "-n"] and command[2].startswith("hw.perflevel"):
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
                    "Hardware:\n\n"
                    "    Hardware Overview:\n\n"
                    "      Total Number of Cores: 16 "
                    "(4 Efficiency and 12 Performance)\n"
                    "      Memory: 64 GB\n",
                    "",
                )
            raise AssertionError(f"unexpected command: {command}")

        with (
            mock.patch.object(_cli, "_bench_bin", return_value="/opt/bin/ax-engine-bench"),
            mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
            mock.patch.object(_cli, "_package_version", return_value="6.4.5"),
            mock.patch.object(_cli, "_run_capture", side_effect=run_capture) as run_capture_mock,
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
        self.assertEqual(payload["source"]["schema_version"], "ax.engine_bench.doctor.v1")
        self.assertEqual(
            payload["next_actions"], ["ax-engine serve /models/gemma4-12b --port 31418"]
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

    def test_doctor_keeps_bundled_runtime_ready_on_bringup_host(self) -> None:
        bench_report = {
            "schema_version": "ax.engine_bench.doctor.v1",
            "status": "bringup_only",
            "mlx_runtime_ready": False,
            "workflow": {"mode": "python_package", "cwd": "/tmp"},
            "host": {
                "supported_mlx_runtime": False,
                "detected_soc": "Apple M4",
                "os": "macos",
                "arch": "aarch64",
            },
            "runtime_assets": {
                "status": "ready",
                "source": "bundled_mlx_runtime",
                "path": "/wheel/ax_engine/.dylibs",
            },
            "metal_toolchain": {"fully_available": False},
            "model_artifacts": {
                "selected": False,
                "status": "not_selected",
                "issues": [],
            },
            "issues": ["Host is outside the supported production MLX profile."],
        }
        passed_probe = {"id": "binary", "status": "pass", "detail": "ok"}

        with (
            mock.patch.object(_cli, "_probe_binary", return_value=passed_probe),
            mock.patch.object(_cli, "_host_system_summary", return_value={}),
        ):
            payload = _cli._user_doctor_report(bench_report)

        checks = {check["id"]: check for check in payload["checks"]}
        self.assertEqual(payload["result"], "degraded")
        self.assertEqual(checks["host"]["status"], "fail")
        self.assertEqual(checks["metal_toolchain"]["status"], "pass")
        self.assertEqual(checks["mlx_runtime"]["status"], "pass")
        self.assertEqual(
            checks["mlx_runtime"]["detail"],
            "Bundled MLX runtime assets available",
        )
        self.assertEqual(
            payload["next_actions"],
            ["Use a supported Apple Silicon host for production MLX workloads."],
        )

    def test_doctor_verbose_wraps_bench_doctor(self) -> None:
        with (
            mock.patch.object(_cli, "_bench_bin", return_value="/opt/bin/ax-engine-bench"),
            mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp,
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
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

    def test_doctor_reports_missing_bundled_bench_without_traceback(self) -> None:
        missing = FileNotFoundError(2, "No such file or directory", "ax-engine-bench")

        def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
            if command[0] == "ax-engine-bench":
                raise missing
            return subprocess.CompletedProcess(command, 0, "", "")

        host = {
            "os": "darwin",
            "arch": "arm64",
            "os_version": "26.2",
            "os_build": "25C56",
            "ram_bytes": 64 * 1024 * 1024 * 1024,
            "ram_gib": 64,
            "cpu_cores": {"physical": 16, "logical": 16},
            "gpu_cores": 40,
        }
        with (
            mock.patch.object(_cli, "_bench_bin", return_value="ax-engine-bench"),
            mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
            mock.patch.object(_cli, "_package_version", return_value="6.9.0"),
            mock.patch.object(_cli, "_host_system_summary", return_value=host),
            mock.patch.object(_cli, "_run_capture", side_effect=run_capture),
        ):
            code, stdout = self.capture_main(["doctor"])

        self.assertEqual(code, 1)
        self.assertIn("Result: not ready", stdout)
        self.assertIn("bench_binary: fail", stdout)
        self.assertIn("ax-engine-bench", stdout)
        self.assertIn("brew reinstall defai-digital/ax-engine/ax-engine", stdout)
        self.assertIn("--force-reinstall", stdout)

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
                code, stdout = self.capture_main(["download", "ax-qwen3.6-35b", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "ax.download_model.v1")
        self.assertEqual(
            payload["repo_id"],
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
        )
        self.assertEqual(payload["alias"], "ax-qwen3.6-35b")
        self.assertEqual(payload["preset"], "qwen3.6-35b")

    def test_download_qwen36_27b_bit_alias_uses_automatosx_target(self) -> None:
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
                code, stdout = self.capture_main(["download", "ax-qwen36-27b-6bit", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["repo_id"], "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP")
        self.assertEqual(payload["alias"], "ax-qwen3.6-27b-6bit")
        self.assertEqual(payload["preset"], "qwen3.6-27b")

    def test_download_diffusiongemma_alias_uses_automatosx_target(self) -> None:
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
                code, stdout = self.capture_main(["download", "ax-diffusiongemma-26b", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertEqual(
            payload["repo_id"],
            "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
        )
        self.assertEqual(payload["alias"], "ax-diffusiongemma-26b")
        self.assertNotIn("preset", payload)

    def test_download_gemma4_12b_aliases_use_automatosx_targets(self) -> None:
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
                code, stdout = self.capture_main(["download", "ax-gemma4-12b", "--json"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(
                payload["repo_id"],
                "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
            )
            self.assertEqual(payload["alias"], "ax-gemma4-12b")
            self.assertEqual(payload["preset"], "gemma4-12b")

            with mock.patch.dict(
                os.environ,
                {"AX_ENGINE_REPO_ROOT": str(root), "FAKE_MODEL_DIR": str(model_dir)},
            ):
                code, stdout = self.capture_main(["download", "ax-gemma4-12b-6bit", "--json"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(
                payload["repo_id"],
                "AutomatosX/AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP",
            )
            self.assertEqual(payload["alias"], "ax-gemma4-12b-6bit")
            self.assertEqual(payload["preset"], "gemma4-12b")

    def test_serve_auto_download_uses_ready_artifacts_without_flag(self) -> None:
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
                    p.add_argument("--progress-bar", action="store_true")
                    p.add_argument("--local-only", action="store_true")
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
                mock.patch.object(_cli, "_server_bin", return_value="/opt/bin/ax-engine-server"),
                mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp,
                self.assertRaisesRegex(RuntimeError, "stop"),
            ):
                self.capture_main(["serve", "ax-qwen3.6-35b"])

            argv = execvp.call_args.args[1]
            self.assertIn("--preset", argv)
            preset_index = argv.index("--preset") + 1
            self.assertEqual(argv[preset_index], "qwen3.6-35b")
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

    def test_tui_forwards_dash_led_args_to_native_binary(self) -> None:
        with (
            mock.patch.object(_cli, "_native_bin", return_value="/opt/bin/ax-engine"),
            mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp,
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
            self.capture_main(["tui", "--help"])

        self.assertEqual(execvp.call_args.args[0], "/opt/bin/ax-engine")
        self.assertEqual(
            execvp.call_args.args[1],
            ["/opt/bin/ax-engine", "tui", "--help"],
        )

    def test_tui_strips_leading_separator_before_forwarding(self) -> None:
        with (
            mock.patch.object(_cli, "_native_bin", return_value="/opt/bin/ax-engine"),
            mock.patch.object(os, "execvp", side_effect=RuntimeError("stop")) as execvp,
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
            self.capture_main(["tui", "--", "--foo"])

        self.assertEqual(
            execvp.call_args.args[1],
            ["/opt/bin/ax-engine", "tui", "--foo"],
        )


class AxEngineInteractiveDownloadTests(unittest.TestCase):
    def capture_main(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = _cli.main(argv)
        return code, out.getvalue()

    def test_download_list_marks_bundled_mtp_without_packaging_target(self) -> None:
        code, stdout = self.capture_main(["download", "--list", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        targets = payload["targets"]
        self.assertEqual({target["repo_id"] for target in targets}, EXPECTED_AUTOMATOSX_REPOS)
        self.assertTrue(all(target["mtp_target"] is None for target in targets))
        self.assertEqual(sum(target["mtp_included"] for target in targets), 30)

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
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=False),
            self.assertRaises(SystemExit) as raised,
        ):
            self.capture_main(["ui-downloader"])

        self.assertIn("interactive terminal", str(raised.exception))

    def test_wizard_flow_invokes_download_with_progress(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "repo_id": "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
            "dest": "/tmp/model",
        }
        inputs = iter(["1", "", "y"])  # select first model, default path, confirm
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(_cli, "_download_summary", return_value=(0, summary, "")) as download,
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

    def test_wizard_mtp_snapshot_uses_standard_download(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "repo_id": ("AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP"),
            "dest": "/tmp/model",
        }
        idx = self._index_of("ax-gemma4-12b")
        # There is no Direct-vs-MTP prompt: select, accept cache, confirm.
        inputs = iter([str(idx), "", "y"])
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(_cli, "_download_summary", return_value=(0, summary, "")) as download,
        ):
            code, stdout = self.capture_main(["ui-downloader"])

        self.assertEqual(code, 0)
        download.assert_called_once_with("ax-gemma4-12b", dest=None, force=False, progress=True)
        self.assertIn("Status: ready", stdout)

    def test_wizard_non_mtp_snapshot_uses_standard_download(self) -> None:
        summary = {
            "schema_version": "ax.download_model.v1",
            "status": "ready",
            "repo_id": "AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit",
            "dest": "/tmp/model",
        }
        idx = self._index_of("ax-qwen3-coder-next")
        inputs = iter([str(idx), "", "y"])
        with (
            mock.patch.object(_cli, "_supports_interactive", return_value=True),
            mock.patch.object(_cli, "_wizard_input", side_effect=lambda _p: next(inputs)),
            mock.patch.object(_cli, "_download_summary", return_value=(0, summary, "")) as download,
        ):
            code, stdout = self.capture_main(["ui-downloader"])

        self.assertEqual(code, 0)
        download.assert_called_once_with(
            "ax-qwen3-coder-next", dest=None, force=False, progress=True
        )
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
