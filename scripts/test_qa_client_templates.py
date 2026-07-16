#!/usr/bin/env python3
"""Regression tests: QA generate-path chat framing matches the server contract.

These exercise the shipped helpers in ``qa/client.py`` (not a reimplementation).
Fallbacks must stay aligned with ``crates/ax-engine-server/src/chat.rs``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_DIR = REPO_ROOT / "qa"
if str(QA_DIR) not in sys.path:
    sys.path.insert(0, str(QA_DIR))

from client import (  # noqa: E402
    _render_chat_prompt,
    strip_chat_output_text,
    strip_gemma4_channel_output,
    strip_gpt_oss_harmony_output,
)


class QaClientTemplateTests(unittest.TestCase):
    def test_gemma4_uses_server_turn_and_thought_prefill(self) -> None:
        prompt = _render_chat_prompt("gemma-4-12b-it", None, "hello")
        # Server uses <|turn> / <turn|>, not <|turn|>.
        self.assertIn("<|turn>user\nhello<turn|>\n", prompt)
        self.assertIn("<|turn>model\n<|channel>thought\n<channel|>", prompt)
        self.assertNotIn("<|turn|>", prompt)

    def test_gemma4_emits_separate_system_turn(self) -> None:
        prompt = _render_chat_prompt("gemma-4-12b-it", "Be brief.", "hello")
        self.assertIn("<|turn>system\nBe brief.<turn|>\n", prompt)
        self.assertIn("<|turn>user\nhello<turn|>\n", prompt)
        self.assertNotIn("<|turn>user\nBe brief.\nhello", prompt)

    def test_qwen_prefills_empty_think_skip(self) -> None:
        prompt = _render_chat_prompt("qwen3-8b", None, "hello")
        self.assertTrue(
            prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            prompt,
        )

    def test_qwen_coder_skips_think_block(self) -> None:
        prompt = _render_chat_prompt("qwen3-coder-30b", None, "hello")
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"), prompt)
        self.assertNotIn("<think>", prompt)

    def test_glm_prefills_empty_think(self) -> None:
        prompt = _render_chat_prompt("glm-4.7-flash", None, "hello")
        self.assertTrue(prompt.endswith("<|assistant|></think>"), prompt)
        self.assertIn("[gMASK]<sop><|user|>hello", prompt)

    def test_mistral_family_uses_inst_framing(self) -> None:
        for model in ("mistral-small", "ministral-8b", "devstral-small"):
            with self.subTest(model=model):
                prompt = _render_chat_prompt(model, None, "hello")
                self.assertIn("[INST]hello[/INST]", prompt)
                self.assertTrue(prompt.startswith("<s>"), prompt)

    def test_gpt_oss_uses_harmony_and_final_prefill(self) -> None:
        prompt = _render_chat_prompt("gpt-oss-20b", None, "hello")
        self.assertIn("<|start|>system<|message|>", prompt)
        self.assertIn("Valid channels: analysis, commentary, final", prompt)
        self.assertIn("<|start|>user<|message|>hello<|end|>", prompt)
        self.assertTrue(
            prompt.endswith("<|start|>assistant<|channel|>final<|message|>"),
            prompt,
        )
        self.assertNotIn("user: hello", prompt)

    def test_gpt_oss_developer_system_block(self) -> None:
        prompt = _render_chat_prompt("gpt-oss-20b", "Be brief.", "hello")
        self.assertIn(
            "<|start|>developer<|message|># Instructions\n\nBe brief.<|end|>",
            prompt,
        )

    def test_strip_gpt_oss_prefers_final_channel(self) -> None:
        raw = (
            "<|channel|>analysis<|message|>thinking…<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hi there!<|return|>"
        )
        self.assertEqual(strip_gpt_oss_harmony_output(raw), "Hi there!")
        self.assertEqual(strip_chat_output_text("gpt-oss-20b", raw), "Hi there!")

    def test_strip_gemma4_channel_body(self) -> None:
        raw = "<|channel>thought\nI am thinking.<channel|>The answer is 42"
        self.assertEqual(strip_gemma4_channel_output(raw), "The answer is 42")
        self.assertEqual(
            strip_chat_output_text("gemma-4-12b-it", raw), "The answer is 42"
        )


if __name__ == "__main__":
    unittest.main()
