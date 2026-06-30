#!/usr/bin/env python3
"""Tests for the Rapid-MLX prompt-suite benchmark adapter."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_rapid_mlx_prompt_suites.py")
REPO_ROOT = SCRIPT_PATH.parent.parent
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_rapid_mlx_prompt_suites", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
rapid = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_rapid_mlx_prompt_suites"] = rapid
MODULE_SPEC.loader.exec_module(rapid)


class RapidMlxPromptSuiteTests(unittest.TestCase):
    def test_prepare_model_layout_symlinks_nested_mtp_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            (model / "mtp").mkdir(parents=True)
            (model / "config.json").write_text("{}\n")
            (model / "mtp" / "weights.safetensors").write_text("mtp\n")

            runtime_model, layout = rapid.prepare_model_layout(model=str(model), output_dir=root / "out")

            runtime = Path(runtime_model)
            self.assertEqual(layout["mode"], "symlink_view")
            self.assertTrue((runtime / "config.json").is_symlink())
            self.assertTrue((runtime / "mtp.safetensors").is_symlink())
            self.assertEqual(
                (runtime / "mtp.safetensors").resolve(),
                (model / "mtp" / "weights.safetensors").resolve(),
            )

    def test_prepare_model_layout_keeps_root_sidecar_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n")
            (model / "mtp.safetensors").write_text("mtp\n")

            runtime_model, layout = rapid.prepare_model_layout(model=str(model), output_dir=root / "out")

        self.assertEqual(Path(runtime_model), model.resolve())
        self.assertEqual(layout["mode"], "unchanged")

    def test_prepare_rapid_mtp_compat_site_writes_sitecustomize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lightning = root / "lightning-mlx"
            patch = lightning / "vllm_mlx" / "patches" / "qwen3_next_mtp.py"
            patch.parent.mkdir(parents=True)
            patch.write_text("# patch\n")

            compat = rapid.prepare_rapid_mtp_compat_site(
                output_dir=root / "out",
                lightning_source=lightning,
                mode="lightning",
            )

            sitecustomize = Path(compat["sitecustomize"])
            self.assertEqual(compat["mode"], "lightning")
            self.assertEqual(Path(compat["patch_path"]), patch)
            self.assertTrue(sitecustomize.is_file())
            text = sitecustomize.read_text()
            self.assertIn("AX_RAPID_MLX_QWEN3_NEXT_MTP_PATCH", text)
            self.assertIn("AX_RAPID_MLX_IGNORE_EOS", text)
            self.assertIn("_ax_empty_stop_tokens", text)
            self.assertIn("vllm_mlx.patches.qwen3_next_mtp", text)
            self.assertIn("vllm_mlx.share.cli", text)

    def test_prepare_rapid_mtp_compat_site_none_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compat = rapid.prepare_rapid_mtp_compat_site(
                output_dir=Path(tmp) / "out",
                lightning_source=Path(tmp) / "missing",
                mode="none",
            )

        self.assertEqual(compat, {"mode": "none", "ignore_eos": False})

    def test_lightning_mode_uses_benchmark_serve_preset_and_ngram_flags(self) -> None:
        class FakeProcess:
            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(rapid, "wait_until_ready"),
                patch.object(
                    rapid.subprocess, "Popen", return_value=FakeProcess()
                ) as popen,
            ):
                rapid.start_server(
                    model="/model",
                    rapid_python=Path("python"),
                    rapid_source=root,
                    lightning_source=root / "lightning",
                    rapid_mtp_patch="none",
                    port=18765,
                    depth=3,
                    startup_timeout=1.0,
                    output_dir=root / "out",
                    lightning_mode=True,
                    enable_ngram=True,
                    mtp_optimistic=False,
                    mtp_draft_temperature=0.5,
                    ignore_eos=True,
                )

        cmd = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "local")
        self.assertEqual(env["AX_RAPID_MLX_IGNORE_EOS"], "1")
        self.assertNotIn("--disable-prefix-cache", cmd)
        self.assertNotIn("--no-memory-aware-cache", cmd)
        self.assertNotIn("--prefill-step-size", cmd)
        self.assertIn("--no-thinking", cmd)
        self.assertNotIn("--mtp-optimistic", cmd)
        self.assertEqual(cmd[cmd.index("--mtp-draft-temperature") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "1")
        self.assertEqual(cmd[cmd.index("--prefill-batch-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--completion-batch-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--stream-interval") + 1], "1")
        self.assertIn("--enable-ngram", cmd)
        self.assertIn("--ngram-skip-tool-calls", cmd)
        self.assertEqual(
            cmd[cmd.index("--ngram-auto-disable-mtp-threshold") + 1], "0.85"
        )
        self.assertEqual(
            cmd[cmd.index("--ngram-auto-disable-min-ngram") + 1], "0.5"
        )


class RunCaseStreamHandlingTests(unittest.TestCase):
    """Verify run_case correctly separates content vs reasoning_content
    streams and surfaces the silent-thinking signal in the result dict."""

    def _make_handle(self) -> "rapid.ServerHandle":
        class _FakeProc:
            def poll(self) -> None:
                return None

            def send_signal(self, _sig) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                return 0

        return rapid.ServerHandle(
            proc=_FakeProc(),  # type: ignore[arg-type]
            base_url="http://test/v1",
            model="local",
            model_layout={"mode": "unchanged"},
            log_path=Path("/tmp/unused.log"),
            command=["fake"],
            compat_patch={"mode": "none"},
        )

    def _patch_stream(self, sse_lines: list[str]):
        """Patch httpx.stream to return our scripted SSE lines."""

        class _FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def iter_lines(self):
                yield from sse_lines

        class _FakeStreamCtx:
            def __enter__(self_inner):
                return _FakeResponse()

            def __exit__(self_inner, *args):
                return False

        return patch.object(rapid.httpx, "stream", return_value=_FakeStreamCtx())

    def _make_chunk(
        self,
        *,
        content: str | None = None,
        reasoning: str | None = None,
        usage: dict | None = None,
    ) -> str:
        import json as _json

        delta: dict = {}
        if content is not None:
            delta["content"] = content
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
        payload = {"choices": [{"delta": delta}]}
        if usage is not None:
            payload["usage"] = usage
        return f"data: {_json.dumps(payload)}"

    def test_separates_content_and_reasoning_streams(self) -> None:
        sse = [
            self._make_chunk(reasoning="thinking part 1 "),
            self._make_chunk(reasoning="thinking part 2"),
            self._make_chunk(content="visible response "),
            self._make_chunk(content="end."),
            self._make_chunk(usage={"completion_tokens": 50, "prompt_tokens": 10}),
            "data: [DONE]",
        ]
        case = rapid.PromptCase(
            id="t", category="c", prompt="hi", max_tokens=100
        )
        with self._patch_stream(sse):
            run = rapid.run_case(
                handle=self._make_handle(),
                case=case,
                max_tokens=50,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                seed=0,
                measured=True,
                repetition=0,
            )
        self.assertEqual(run["visible_text_chars"], len("visible response end."))
        self.assertEqual(run["reasoning_text_chars"], len("thinking part 1 thinking part 2"))
        self.assertEqual(run["text"], "visible response end.")
        self.assertEqual(run["content_head"], "visible response end.")
        self.assertEqual(run["reasoning_head"], "thinking part 1 thinking part 2")
        self.assertEqual(run["stream_chunk_stats"]["with_content"], 2)
        self.assertEqual(run["stream_chunk_stats"]["with_reasoning_content"], 2)
        self.assertFalse(run["silent_thinking_suspected"])

    def test_silent_thinking_flag_when_no_visible_output(self) -> None:
        # Server claims 500 completion_tokens but never streams any content
        # or reasoning chunks — classic silent-thinking pattern.
        sse = [
            self._make_chunk(usage={"completion_tokens": 500, "prompt_tokens": 10}),
            "data: [DONE]",
        ]
        case = rapid.PromptCase(
            id="t", category="c", prompt="hi", max_tokens=500
        )
        with self._patch_stream(sse):
            run = rapid.run_case(
                handle=self._make_handle(),
                case=case,
                max_tokens=500,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                seed=0,
                measured=True,
                repetition=0,
            )
        self.assertTrue(run["silent_thinking_suspected"])
        self.assertEqual(run["visible_text_chars"], 0)
        self.assertEqual(run["reasoning_text_chars"], 0)
        self.assertEqual(run["generated_tokens"], 500)

    def test_rejects_short_fixed_token_run_when_required(self) -> None:
        sse = [
            self._make_chunk(content="short"),
            self._make_chunk(usage={"completion_tokens": 42, "prompt_tokens": 10}),
            "data: [DONE]",
        ]
        case = rapid.PromptCase(
            id="t", category="c", prompt="hi", max_tokens=100
        )
        with self._patch_stream(sse):
            run = rapid.run_case(
                handle=self._make_handle(),
                case=case,
                max_tokens=100,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                seed=0,
                measured=True,
                repetition=0,
                require_full_output_tokens=True,
            )

        self.assertFalse(run["fixed_token_complete"])
        self.assertEqual(run["requested_tokens"], 100)
        self.assertEqual(run["generated_tokens"], 42)
        self.assertIsNone(run["decode_tok_s"])
        self.assertEqual(
            run["rejected_reason"], "generated_tokens_lt_requested_tokens"
        )


class CaptureLightningIdentityTests(unittest.TestCase):
    def test_captures_git_state_and_pyproject(self) -> None:
        # Smoke against the real lightning source — this is read-only.
        source = REPO_ROOT / ".internal" / "reference" / "lightning-mlx"
        if not source.is_dir():
            self.skipTest("lightning source not available")
        ident = rapid.capture_lightning_source_identity(source)
        self.assertEqual(ident["source_path"], str(source))
        # commit hash should be 40 hex chars
        self.assertIsNotNone(ident["git_commit"])
        self.assertEqual(len(ident["git_commit"] or ""), 40)
        # git_describe should resolve to v0.6.32 or similar tag-prefixed
        self.assertIsNotNone(ident["git_describe"])
        # modified_files should be a list (may be empty)
        self.assertIsInstance(ident["modified_files"], list)

    def test_handles_non_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ident = rapid.capture_lightning_source_identity(Path(tmp))
            self.assertIsNone(ident["git_commit"])
            self.assertIsNone(ident["git_describe"])
            self.assertFalse(ident["is_dirty"])
            self.assertEqual(ident["modified_files"], [])


class ParseServerHeaderTests(unittest.TestCase):
    def test_extracts_banner_lines(self) -> None:
        log = [
            "  Alias: qwen3.6-27b -> Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "  Model: Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "  Features: tools: qwen3_coder_xml, reasoning: qwen3",
            "MTP: enabled, draft_tokens=3, draft_temp=0.5",
            "N-gram: enabled, K=6, n=3, min_matches=2, accept=greedy",
            "WARNING:vllm_mlx.scheduler:[MTP] mtp_num_draft_tokens=3 requested "
            "but model has only 1 MTP layer(s). depth will be capped to 1 per verify cycle.",
        ]
        info = rapid.parse_server_header(log)
        self.assertIn("qwen3.6-27b", info["alias_line"])
        self.assertIn("Youssofal/Qwen3.6-27B-MTPLX", info["model_line"])
        self.assertIn("draft_tokens=3", info["mtp_line"])
        self.assertIn("K=6", info["ngram_line"])
        self.assertEqual(info["mtp_depth_cap_warnings"], 1)
        self.assertEqual(info["effective_mtp_depth"], 1)

    def test_handles_empty_log(self) -> None:
        info = rapid.parse_server_header([])
        self.assertIsNone(info["alias_line"])
        self.assertEqual(info["mtp_depth_cap_warnings"], 0)
        self.assertIsNone(info["effective_mtp_depth"])


class ReadLogTailTests(unittest.TestCase):
    def test_returns_last_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "x.log"
            log.write_text("\n".join(f"line {i}" for i in range(50)))
            tail = rapid.read_log_tail(log, max_lines=5, max_bytes=10_000)
            self.assertEqual(len(tail), 5)
            self.assertEqual(tail[-1], "line 49")

    def test_missing_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tail = rapid.read_log_tail(Path(tmp) / "does-not-exist.log")
            self.assertEqual(tail, [])


if __name__ == "__main__":
    unittest.main()
