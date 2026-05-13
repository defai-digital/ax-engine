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


SAMPLE_README = """\
# Title

### Prefill throughput (tok/s) — percentages vs mlx_lm

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 2,610.1 (-0.2%) | 3,938.8 (+67.6%) |
|        |        | 512 | 8,378.7 | 6,768.2 (-19.2%) | 8,510.1 (+13.4%) |
| Qwen 3.5 9B | 4-bit | 128 | 968.2 | 1,795.9 (+85.5%) | 2,295.7 (+147.8%) |
|        |        | 512 | 1,787.8 | 2,402.1 (+34.4%) | 3,043.8 (+74.5%) |

### Decode throughput (tok/s) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax direct baseline | ax default n-gram |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 196.6 | 194.6 (-1.0%) | 191.3 (-12.5%) | **591.1 (+170.5%)** |
|        |        | 512 | 189.6 | 189.8 (+0.1%) | 184.1 (-12.4%) | **581.5 (+176.6%)** |

### Time to first token (ms) — generation=128 tokens, temp=0

| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |
|---|---|---:|---:|---:|---:|
| Gemma 4 E2B | 4-bit | 128 | 50.0 | 50.0 (+0.0%) | **30.0 (-40.0%)** |
|        |        | 512 | 60.0 | 60.0 (+0.0%) | **40.0 (-33.3%)** |

### Embedding throughput
foo
"""


class InjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep_doc = {
            "rows": [
                _row("gemma-4-e2b-it-4bit", 3500.0, 7000.0, 160.0, 36.0, 73.0),
                _row("qwen3_5-9b-mlx-4bit", 1800.0, 2500.0, 75.0, 71.0, 204.0),
            ],
        }

    def test_lookup_keys_by_readme_model_quant_pt(self) -> None:
        lookup = inj.build_llama_lookup(self.sweep_doc)
        self.assertEqual(lookup[("Gemma 4 E2B", "4-bit", 128)]["prefill"], 3500.0)
        self.assertEqual(lookup[("Qwen 3.5 9B", "4-bit", 512)]["decode"], 76.0)
        self.assertEqual(lookup[("Qwen 3.5 9B", "4-bit", 512)]["ttft"], 204.0)

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
            "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine |\n|---|---|---:|---:|---:|---:|\n| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 2,610.1 (-0.2%) | 3,938.8 (+67.6%) |",
            "| Model | MLX quantization | Prompt tok | mlx_lm | mlx_swift_lm | ax engine | llama.cpp Metal* |\n|---|---|---:|---:|---:|---:| ---: |\n| Gemma 4 E2B | 4-bit | 128 | 2,615.9 | 2,610.1 (-0.2%) | 3,938.8 (+67.6%) | 9999.9 |",
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
        # Drop the Qwen 3.5 9B row from the sweep doc — its README cells
        # should become 'n/a' rather than crash. With the new canonical
        # position the n/a sits at index 3, before mlx_lm.
        partial = {"rows": [_row("gemma-4-e2b-it-4bit", 3500.0, 7000.0, 160.0, 36.0, 73.0)]}
        out, _ = inj.apply(SAMPLE_README, partial)
        qwen_128_line = next(l for l in out.splitlines() if "968.2" in l)
        stripped = [c.strip() for c in qwen_128_line.split("|")[1:-1]]
        self.assertEqual(stripped[3], "n/a")
        self.assertEqual(stripped[4], "968.2")


if __name__ == "__main__":
    unittest.main()
