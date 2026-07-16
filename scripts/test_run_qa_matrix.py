#!/usr/bin/env python3
"""Unit tests for scripts/run_qa_matrix pure helpers (no GPU)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    import sys

    path = ROOT / "scripts" / "run_qa_matrix.py"
    spec = importlib.util.spec_from_file_location("run_qa_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Required so @dataclass can resolve annotations under Python 3.14+.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class RunQaMatrixTests(unittest.TestCase):
    def test_classify_engine_fail_detects_panic_and_null(self) -> None:
        m = _load()
        self.assertEqual(m.classify_engine_fail("thread panicked", ""), "panic")
        self.assertEqual(
            m.classify_engine_fail("", 'choices":[{"message":{"content": null}}]'),
            'content": null',
        )
        self.assertIsNone(
            m.classify_engine_fail(
                "listening", "Results: 8/8 passed (100.0%)\n[1/8] math_modulo ... PASS"
            )
        )

    def test_load_matrix_parses_ok_lines(self) -> None:
        m = _load()
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "qa-matrix.txt"
            p.write_text(
                "\n".join(
                    [
                        "# comment",
                        "SKIP|direct|missing|nope",
                        "OK|direct|llama3.1-8b|/tmp/fake",
                        "OK|mtp|ax-local/x|/tmp/mtp",
                    ]
                )
                + "\n"
            )
            cells = m.load_matrix(p)
            self.assertEqual(len(cells), 2)
            self.assertEqual(cells[0].mode, "direct")
            self.assertEqual(cells[0].model_id, "llama3.1-8b")
            self.assertEqual(cells[1].mode, "mtp")


if __name__ == "__main__":
    unittest.main()
