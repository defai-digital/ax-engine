"""Unit tests for scripts.update_readme_llama_cpp."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "update_readme_llama_cpp",
    _HERE / "update_readme_llama_cpp.py",
)
upd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upd)  # type: ignore[union-attr]
sys.modules["update_readme_llama_cpp"] = upd


def _make_sweep_doc(rows: list[dict]) -> dict:
    return {
        "schema_version": "ax.llama_cpp_metal_sweep.v1",
        "claim_boundary": "shape-only",
        "manifest_path": "benchmarks/manifests/llama_cpp_metal/inventory.json",
        "llama_bench": "/opt/homebrew/bin/llama-bench",
        "repetitions": 5,
        "n_gpu_layers": 99,
        "prompt_tokens": "128,512",
        "generation_tokens": 128,
        "rows": rows,
    }


def _ok_row(slug: str, readme_model: str, readme_quant: str, prefill_128=1000.0, prefill_512=2000.0, decode=50.0, ttft_128=20.0, ttft_512=80.0, repo="x/y-GGUF", quant_target="Q4_K_M") -> dict:
    return {
        "slug": slug,
        "readme_model": readme_model,
        "readme_quant": readme_quant,
        "gguf_quant_target": quant_target,
        "status": "ok",
        "resolved_repo": repo,
        "result_doc": {
            "results": [
                {
                    "engine": "llama_cpp_metal",
                    "prompt_tokens": 128,
                    "prefill_tok_s": {"median": prefill_128},
                    "decode_tok_s": {"median": decode},
                    "ttft_ms": {"median": ttft_128},
                },
                {
                    "engine": "llama_cpp_metal",
                    "prompt_tokens": 512,
                    "prefill_tok_s": {"median": prefill_512},
                    "decode_tok_s": {"median": decode + 1.0},
                    "ttft_ms": {"median": ttft_512},
                },
            ],
        },
    }


class UpdateReadmeLlamaCppTests(unittest.TestCase):
    def test_render_section_includes_prompt_hash_parity_disclaimer(self) -> None:
        doc = _make_sweep_doc([_ok_row("gemma-4-e2b-it-4bit", "Gemma 4 E2B", "4-bit")])
        out = upd.render_section(doc)
        self.assertIn("not prompt-hash parity", out.lower())
        self.assertIn("shape-compatible", out.lower())
        self.assertIn("Gemma 4 E2B", out)
        self.assertIn("Q4_K_M", out)
        self.assertIn("1,000.0", out)  # prefill 128
        self.assertIn("2,000.0", out)  # prefill 512

    def test_unresolved_row_renders_as_n_a_with_reason(self) -> None:
        doc = _make_sweep_doc([
            {
                "slug": "qwen3-coder-next-4bit",
                "readme_model": "Qwen Coder Next",
                "readme_quant": "4-bit",
                "gguf_quant_target": "Q4_K_M",
                "status": "unresolved",
            }
        ])
        out = upd.render_section(doc)
        self.assertIn("no GGUF found", out)
        self.assertIn("n/a", out)

    def test_splice_section_inserts_before_embedding_header(self) -> None:
        readme = (
            "# Title\n\n"
            "### Time to first token\n\n"
            "table content\n\n"
            "### Embedding throughput\n\n"
            "embedding content\n"
        )
        doc = _make_sweep_doc([_ok_row("gemma-4-e2b-it-4bit", "Gemma 4 E2B", "4-bit")])
        section_md = upd.render_section(doc)
        updated = upd.splice_section(readme, section_md)
        idx_section = updated.index(upd.SECTION_HEADER)
        idx_embedding = updated.index("### Embedding throughput")
        self.assertLess(idx_section, idx_embedding)

    def test_splice_section_is_idempotent(self) -> None:
        readme = (
            "# Title\n\n"
            "### Time to first token\n\nfoo\n\n"
            "### Embedding throughput\n\nbar\n"
        )
        doc = _make_sweep_doc([_ok_row("gemma-4-e2b-it-4bit", "Gemma 4 E2B", "4-bit")])
        section_md = upd.render_section(doc)
        once = upd.splice_section(readme, section_md)
        twice = upd.splice_section(once, section_md)
        self.assertEqual(once, twice)
        self.assertEqual(once.count(upd.SECTION_HEADER), 1)

    def test_splice_section_refuses_without_anchor(self) -> None:
        readme = "# Title\n\nno embedding section here\n"
        doc = _make_sweep_doc([_ok_row("gemma-4-e2b-it-4bit", "Gemma 4 E2B", "4-bit")])
        section_md = upd.render_section(doc)
        with self.assertRaisesRegex(RuntimeError, "insertion anchor"):
            upd.splice_section(readme, section_md)


if __name__ == "__main__":
    unittest.main()
