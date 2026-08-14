"""Frozen Wave-1 set and mlxcel log parse for PRD-M5-FLEET-AX-VS-MLXCEL."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "bench_ax_vs_mlxcel_fleet.py"


def _load():
    spec = importlib.util.spec_from_file_location("bench_ax_vs_mlxcel_fleet", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wave1_excludes_deepseek_and_covers_certified_text_families() -> None:
    mod = _load()
    ids = set(mod.WAVE1)
    families = {row["family"] for row in mod.WAVE1.values()}
    assert "deepseek_v3" not in families
    assert "deepseek_v32" not in families
    assert "deepseek_v4" not in families
    assert not any("deepseek" in key for key in ids)
    assert families == {
        "qwen3_5",
        "qwen3_next",
        "gemma4",
        "glm4_moe_lite",
        "gpt_oss",
    }
    assert "qwen3.6-27b" in ids
    assert "gemma4-12b" in ids
    assert "glm4.7-flash-4bit" in ids
    assert "gpt-oss-20b" in ids


def test_parse_mlxcel_log_reads_prefill_and_decode(tmp_path: Path) -> None:
    mod = _load()
    log = tmp_path / "mlxcel.log"
    log.write_text(
        "Prompt tokens: 128\n"
        "Prefill: 12.3 ms, 441.2 tok/s\n"
        "Decode:  3700 ms, 34.6 tok/s\n"
    )
    got = mod.parse_mlxcel_log(log)
    assert got["prefill_tok_s"] == 441.2
    assert got["decode_tok_s"] == 34.6


def test_parse_mlxcel_log_reads_parenthesized_profile_results(tmp_path: Path) -> None:
    mod = _load()
    log = tmp_path / "mlxcel.log"
    log.write_text(
        "[Profile Results]\n"
        "  Prompt tokens:    2048\n"
        "  Generated tokens: 128\n"
        "  Prefill:          695.32 ms (2945.42 tok/s)\n"
        "  Decode:           1236.48 ms (103.52 tok/s)\n"
    )
    got = mod.parse_mlxcel_log(log)
    assert got["prefill_tok_s"] == 2945.42
    assert got["decode_tok_s"] == 103.52
    assert got["prefill_tok_s"] != 695.32
