"""Unit tests for scripts.update_readme_inject_llama_cpp."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "update_readme_inject_llama_cpp",
    _HERE / "update_readme_inject_llama_cpp.py",
)
inj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inj)  # type: ignore[union-attr]
sys.modules["update_readme_inject_llama_cpp"] = inj


def _row(slug: str, prefill_128: float, prefill_512: float, decode: float, ttft_128: float, ttft_512: float) -> dict:
    return {
        "slug": slug,
        "status": "ok",
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


def _row_with_depth_decode(
    slug: str,
    prefill_128: float,
    prefill_512: float,
    decode: float,
    depth_decode: float,
    ttft_128: float,
    ttft_512: float,
) -> dict:
    row = _row(slug, prefill_128, prefill_512, decode, ttft_128, ttft_512)
    for cell in row["result_doc"]["results"]:
        offset = 0.0 if int(cell["prompt_tokens"]) == 128 else 1.0
        cell["decode_at_depth_tok_s"] = {"median": depth_decode + offset}
    return row


SAMPLE_README = """\
# Title

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |
|---|---|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 3,938.8 (+67.6%) |
|        |        | 512 | 8,378.7 | 8,510.1 (+13.4%) |
| Qwen 3.6 35B A3B | 4-bit | 128 | 968.2 | 2,295.7 (+147.8%) |
|        |        | 512 | 1,787.8 | 3,043.8 (+74.5%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | ax direct baseline |
|---|---|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 196.6 | 191.3 (-12.5%) |
|        |        | 512 | 189.6 | 184.1 (-12.4%) |

### Time to first token (ms) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |
|---|---|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 50.0 | **30.0 (-40.0%)** |
|        |        | 512 | 60.0 | **40.0 (-33.3%)** |

### Embedding throughput
foo
"""


class InjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep_doc = {
            "rows": [
                _row("gemma-4-e2b-it-4bit", 3500.0, 7000.0, 160.0, 36.0, 73.0),
                _row("qwen3_6-35b-a3b-4bit", 1800.0, 2500.0, 75.0, 71.0, 204.0),
            ],
        }

    def test_lookup_keys_by_readme_model_quant_pt(self) -> None:
        lookup = inj.build_llama_lookup(self.sweep_doc)
        self.assertEqual(lookup[("Gemma 4 E2B", "4-bit", 128)]["prefill"], 3500.0)
        self.assertEqual(lookup[("Qwen 3.6 35B A3B", "4-bit", 512)]["decode"], 76.0)
        self.assertEqual(lookup[("Qwen 3.6 35B A3B", "4-bit", 512)]["ttft"], 204.0)

    def test_lookup_accepts_qwen_gguf_quant_rows(self) -> None:
        doc = {
            "rows": [
                _row("qwen3_6-27b-6bit", 1400.0, 2400.0, 90.0, 90.0, 215.0),
            ],
        }
        lookup = inj.build_llama_lookup(doc)
        self.assertEqual(lookup[("Qwen 3.6 27B", "6-bit", 512)]["prefill"], 2400.0)
        self.assertNotIn(("Qwen 3.6 27B", "8-bit", 512), lookup)

    def test_lookup_prefers_llama_cpp_decode_at_depth_when_present(self) -> None:
        doc = {
            "rows": [
                _row_with_depth_decode(
                    "gemma-4-e2b-it-4bit",
                    3500.0,
                    7000.0,
                    160.0,
                    140.0,
                    36.0,
                    73.0,
                )
            ],
        }

        lookup = inj.build_llama_lookup(doc)

        self.assertEqual(lookup[("Gemma 4 E2B", "4-bit", 128)]["decode"], 140.0)
        self.assertEqual(lookup[("Gemma 4 E2B", "4-bit", 512)]["decode"], 141.0)

    def test_lookup_rejects_duplicate_model_quant_prompt_rows(self) -> None:
        doc = {
            "rows": [
                _row("gemma-4-e2b-it-4bit", 3500.0, 7000.0, 160.0, 36.0, 73.0),
                _row("gemma-4-e2b-it-4bit", 3600.0, 7100.0, 170.0, 35.0, 72.0),
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "duplicate llama.cpp lookup row"):
            inj.build_llama_lookup(doc)

    def test_apply_injects_column_in_all_three_tables(self) -> None:
        out, stats = inj.apply(SAMPLE_README, self.sweep_doc)
        self.assertTrue(stats["inserted_disclaimer"])
        self.assertEqual(stats["rows_in_lookup"], 4)
        # Disclaimer present
        self.assertIn(inj.DISCLAIMER_MARK, out)
        header_rows = [l for l in out.splitlines() if l.startswith("| Model |")]
        self.assertEqual(len(header_rows), 3)
        # llama.cpp column sits directly after 'Prompt tok' (canonical
        # position 3) so it is BEFORE mlx_lm.
        for row in header_rows:
            cells = row.split("|")[1:-1]
            stripped = [c.strip() for c in cells]
            self.assertEqual(stripped[0], "Model")
            self.assertEqual(stripped[1], "MLX quantization")
            self.assertEqual(stripped[2], "Prompt tok")
            self.assertEqual(stripped[3], inj.LLAMA_HEADER_CELL)
            self.assertEqual(stripped[4], "mlx_lm")
        # Sample data values appear in the expected tables.
        self.assertIn("3,500.0", out)
        self.assertIn("7,000.0", out)
        self.assertIn("1,800.0", out)
        self.assertIn("160.0", out)
        self.assertIn("161.0", out)
        self.assertIn("36.0", out)
        self.assertIn("73.0", out)

    def test_apply_is_idempotent(self) -> None:
        once, _ = inj.apply(SAMPLE_README, self.sweep_doc)
        twice, stats2 = inj.apply(once, self.sweep_doc)
        self.assertEqual(once, twice)
        self.assertFalse(stats2["inserted_disclaimer"])
        header_rows = [l for l in twice.splitlines() if l.startswith("| Model |")]
        self.assertEqual(len(header_rows), 3)
        # Each header should have exactly one llama.cpp column, at index 3.
        for row in header_rows:
            stripped = [c.strip() for c in row.split("|")[1:-1]]
            self.assertEqual(stripped.count(inj.LLAMA_HEADER_CELL), 1)
            self.assertEqual(stripped[3], inj.LLAMA_HEADER_CELL)

    def test_apply_migrates_legacy_trailing_column_to_canonical_position(self) -> None:
        """A README with the llama column at the end (legacy layout) should
        be migrated to the canonical pre-mlx_lm position on the next run."""
        legacy = SAMPLE_README.replace(
            "| Model | MLX quantization | Prompt tok | mlx_lm | ax engine |\n|---|---|---:|---:|---:|\n| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 3,938.8 (+67.6%) |",
            "| Model | MLX quantization | Prompt tok | mlx_lm | ax engine | llama.cpp Metal* |\n|---|---|---:|---:|---:| ---: |\n| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 3,938.8 (+67.6%) | 9999.9 |",
            1,
        )
        out, _ = inj.apply(legacy, self.sweep_doc)
        prefill_header = next(l for l in out.splitlines() if l.startswith("| Model |") and "ax engine" in l and "n-gram" not in l)
        stripped = [c.strip() for c in prefill_header.split("|")[1:-1]]
        self.assertEqual(stripped[3], inj.LLAMA_HEADER_CELL)
        # Old value (9999.9) must be gone; new value (3,500.0) must be present
        self.assertNotIn("9999.9", out)
        self.assertIn("3,500.0", out)

    def test_continuation_row_uses_carried_context(self) -> None:
        """The | 512 | row of Gemma E2B 4-bit has empty model+quant cells; the
        injected value must be the 512 entry for ('Gemma 4 E2B', '4-bit') and
        land at canonical position 3."""
        out, _ = inj.apply(SAMPLE_README, self.sweep_doc)
        prefill_512_line = next(l for l in out.splitlines() if "8,378.7" in l)
        stripped = [c.strip() for c in prefill_512_line.split("|")[1:-1]]
        self.assertEqual(stripped[3], "7,000.0")
        self.assertEqual(stripped[4], "8,378.7")  # mlx_lm column shifted right

    def test_apply_strips_standalone_section_if_present(self) -> None:
        readme_with_standalone = SAMPLE_README.replace(
            "### Embedding throughput\nfoo\n",
            "### External GGUF baseline — llama.cpp Metal (shape-compatible, not prompt-hash parity)\n\nold content\n\n### Embedding throughput\nfoo\n",
        )
        out, stats = inj.apply(readme_with_standalone, self.sweep_doc)
        self.assertTrue(stats["removed_standalone_section"])
        self.assertNotIn("### External GGUF baseline", out)

    def test_missing_row_renders_as_n_a(self) -> None:
        # Drop the Qwen 3.6 35B A3B row from the sweep doc — its README cells
        # should become 'n/a' rather than crash. With the new canonical
        # position the n/a sits at index 3, before mlx_lm.
        partial = {"rows": [_row("gemma-4-e2b-it-4bit", 3500.0, 7000.0, 160.0, 36.0, 73.0)]}
        out, _ = inj.apply(SAMPLE_README, partial)
        qwen_128_line = next(l for l in out.splitlines() if "968.2" in l)
        stripped = [c.strip() for c in qwen_128_line.split("|")[1:-1]]
        self.assertEqual(stripped[3], "n/a")
        self.assertEqual(stripped[4], "968.2")

    def test_partial_update_preserves_missing_llama_cells(self) -> None:
        full, _ = inj.apply(SAMPLE_README, self.sweep_doc)
        partial = {"rows": [_row("gemma-4-e2b-it-4bit", 3600.0, 7100.0, 170.0, 35.0, 72.0)]}

        out, _ = inj.apply(full, partial)

        gemma_128_line = next(l for l in out.splitlines() if "2,615.9" in l)
        gemma_128 = [c.strip() for c in gemma_128_line.split("|")[1:-1]]
        self.assertEqual(gemma_128[3], "3,600.0")

        qwen_128_line = next(l for l in out.splitlines() if "968.2" in l)
        qwen_128 = [c.strip() for c in qwen_128_line.split("|")[1:-1]]
        self.assertEqual(qwen_128[3], "1,800.0")


if __name__ == "__main__":
    unittest.main()
