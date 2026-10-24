"""Qwen3 pre-tokenizer regex restore in the Python OpenAI shim."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ax_engine import openai_server

try:
    import tokenizers
except ImportError:  # pragma: no cover - optional dependency
    tokenizers = None


def tokenizer_json(regex: str) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": regex},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": False,
                        "use_regex": False,
                    },
                ],
            },
            "post_processor": None,
            "decoder": None,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
        }
    )


def write_pack(root: Path, model_type: str) -> Path:
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    path = root / "tokenizer.json"
    path.write_text(tokenizer_json(openai_server._LEGACY_QWEN2_PRETOKENIZER_SPLIT))
    return path


def split_count(tokenizer, text: str) -> int:
    return len(tokenizer.pre_tokenizer.pre_tokenize_str(text))


class RestoreQwen3PretokenizerTests(unittest.TestCase):
    def test_restore_only_rewrites_the_legacy_regex(self) -> None:
        self.assertIsNone(openai_server.restore_qwen3_pretokenizer("{}"))
        already = tokenizer_json(openai_server._QWEN3_PRETOKENIZER_SPLIT)
        self.assertIsNone(openai_server.restore_qwen3_pretokenizer(already))
        fixed = openai_server.restore_qwen3_pretokenizer(
            tokenizer_json(openai_server._LEGACY_QWEN2_PRETOKENIZER_SPLIT)
        )
        self.assertIsNotNone(fixed)
        self.assertIn("\\\\p{M}", fixed)


@unittest.skipIf(tokenizers is None, "tokenizers not installed")
class LoadTokenizerTests(unittest.TestCase):
    def test_qwen35_pack_keeps_combining_marks_in_one_word(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pack(Path(tmp), "qwen3_5")
            tokenizer = openai_server.load_tokenizer(path, tokenizers.Tokenizer)
            self.assertEqual(split_count(tokenizer, "नमस्ते दुनिया"), 2)
            self.assertEqual(
                split_count(tokenizer, "Hello, world!"),
                split_count(tokenizers.Tokenizer.from_file(str(path)), "Hello, world!"),
            )

    def test_other_families_keep_the_shipped_regex(self) -> None:
        for model_type in ("qwen3", "qwen3_next", "gemma4"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                path = write_pack(Path(tmp), model_type)
                tokenizer = openai_server.load_tokenizer(path, tokenizers.Tokenizer)
                self.assertGreater(split_count(tokenizer, "नमस्ते दुनिया"), 2)


if __name__ == "__main__":
    unittest.main()
