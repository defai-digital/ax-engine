"""Qwen3 pre-tokenizer regex restore in the Python OpenAI shim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ax_engine import openai_server

tokenizers = pytest.importorskip("tokenizers")


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


def write_pack(tmp_path: Path, model_type: str) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    path = tmp_path / "tokenizer.json"
    path.write_text(tokenizer_json(openai_server._LEGACY_QWEN2_PRETOKENIZER_SPLIT))
    return path


def split_count(tokenizer, text: str) -> int:
    return len(tokenizer.pre_tokenizer.pre_tokenize_str(text))


def test_restore_only_rewrites_the_legacy_regex() -> None:
    assert openai_server.restore_qwen3_pretokenizer("{}") is None
    already = tokenizer_json(openai_server._QWEN3_PRETOKENIZER_SPLIT)
    assert openai_server.restore_qwen3_pretokenizer(already) is None
    fixed = openai_server.restore_qwen3_pretokenizer(
        tokenizer_json(openai_server._LEGACY_QWEN2_PRETOKENIZER_SPLIT)
    )
    assert fixed is not None and "\\\\p{M}" in fixed


def test_qwen35_pack_keeps_combining_marks_in_one_word(tmp_path: Path) -> None:
    path = write_pack(tmp_path, "qwen3_5")
    tokenizer = openai_server.load_tokenizer(path, tokenizers.Tokenizer)
    assert split_count(tokenizer, "नमस्ते दुनिया") == 2
    assert split_count(tokenizer, "Hello, world!") == split_count(
        tokenizers.Tokenizer.from_file(str(path)), "Hello, world!"
    )


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_next", "gemma4"])
def test_other_families_keep_the_shipped_regex(tmp_path: Path, model_type: str) -> None:
    path = write_pack(tmp_path, model_type)
    tokenizer = openai_server.load_tokenizer(path, tokenizers.Tokenizer)
    assert split_count(tokenizer, "नमस्ते दुनिया") > 2
