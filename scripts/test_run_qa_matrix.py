#!/usr/bin/env python3
"""Unit tests for scripts/run_qa_matrix pure helpers (no GPU)."""

from __future__ import annotations

import importlib.util
import tempfile
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
        self.assertEqual(
            m.classify_engine_fail("", "surface_hard_fail"), "surface_hard_fail"
        )
        self.assertIsNone(
            m.classify_engine_fail(
                "listening", "Results: 8/8 passed (100.0%)\n[1/8] math_modulo ... PASS"
            )
        )

    def test_load_matrix_parses_ok_lines_and_modes(self) -> None:
        m = _load()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "qa-matrix.txt"
            p.write_text(
                "\n".join(
                    [
                        "# comment",
                        "SKIP|direct|missing|nope",
                        "OK|direct|llama3.1-8b|/tmp/fake",
                        "OK|ngram|llama3.1-8b|/tmp/fake",
                        "OK|mtp|ax-local/x|/tmp/mtp",
                        "OK|embed|qwen3-embedding|/tmp/emb",
                        "OK|bogus|x|/tmp/x",
                    ]
                )
                + "\n"
            )
            cells = m.load_matrix(p)
            self.assertEqual(len(cells), 4)
            self.assertEqual(cells[0].mode, "direct")
            self.assertEqual(cells[1].mode, "ngram")
            self.assertEqual(cells[2].mode, "mtp")
            self.assertEqual(cells[3].mode, "embed")

    def test_build_server_cmd_embed_disables_ngram(self) -> None:
        m = _load()
        from pathlib import Path as P

        server = P("/tmp/ax-engine-server")
        embed = m.Cell(mode="embed", model_id="e", artifacts=P("/tmp/e"))
        cmd = m.build_server_cmd(embed, "e", server_bin=server, host="127.0.0.1", port=1)
        self.assertIn("--disable-ngram-acceleration", cmd)

    def test_write_inventory(self) -> None:
        m = _load()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "inv.txt"
            m.write_inventory(
                p,
                [
                    ("direct", "m1", "/a"),
                    ("ngram", "m1", "/a"),
                ],
            )
            text = p.read_text()
            self.assertIn("OK|direct|m1|/a", text)
            self.assertIn("OK|ngram|m1|/a", text)

    def test_build_server_cmd_modes(self) -> None:
        """Regression: MTP must not disable ngram; direct must; ngram neither MTP flag."""
        m = _load()
        from pathlib import Path as P

        server = P("/tmp/ax-engine-server")
        direct = m.Cell(mode="direct", model_id="x", artifacts=P("/tmp/a"))
        ngram = m.Cell(mode="ngram", model_id="x", artifacts=P("/tmp/a"))
        mtp = m.Cell(mode="mtp", model_id="y", artifacts=P("/tmp/b"))
        d_cmd = m.build_server_cmd(direct, "x", server_bin=server, host="127.0.0.1", port=1)
        n_cmd = m.build_server_cmd(ngram, "x", server_bin=server, host="127.0.0.1", port=1)
        m_cmd = m.build_server_cmd(mtp, "mtp-y", server_bin=server, host="127.0.0.1", port=1)
        self.assertIn("--disable-ngram-acceleration", d_cmd)
        self.assertNotIn("--disable-ngram-acceleration", n_cmd)
        self.assertNotIn("--mlx-mtp-disable-ngram-stacking", n_cmd)
        self.assertNotIn("--disable-ngram-acceleration", m_cmd)
        self.assertIn("--mlx-mtp-disable-ngram-stacking", m_cmd)

    def test_mtp_telemetry_active_requires_positive_signal(self) -> None:
        m = _load()
        self.assertFalse(m.mtp_telemetry_active({}))
        self.assertFalse(
            m.mtp_telemetry_active(
                {
                    "ax_mtp_source_mtp_proposer_wall_us": 0,
                    "ax_mtp_verify_tokens": 0,
                    "ax_mtp_draft_tokens": 0,
                }
            )
        )
        self.assertTrue(
            m.mtp_telemetry_active({"ax_mtp_source_mtp_proposer_wall_us": 12})
        )
        self.assertTrue(
            m.mtp_telemetry_active({"ax_mtp_source_assistant_proposer_wall_us": 9})
        )
        self.assertTrue(m.mtp_telemetry_active({"ax_mtp_verify_tokens": 3}))
        self.assertTrue(
            m.mtp_telemetry_active(
                {
                    "ax_mlx_gemma4_assistant_mtp_enabled": 1,
                    "ax_mlx_gemma4_assistant_mtp_draft_tokens": 4,
                }
            )
        )

    def test_package_looks_like_mtp_sidecar_and_assistant(self) -> None:
        m = _load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plain = root / "plain"
            plain.mkdir()
            (plain / "config.json").write_text("{}")
            self.assertFalse(m.package_looks_like_mtp(plain))

            fused = root / "fused"
            fused.mkdir()
            (fused / "mtp.safetensors").write_bytes(b"x")
            self.assertTrue(m.package_looks_like_mtp(fused))

            glm = root / "glm"
            glm.mkdir()
            (glm / "glm_mtp.safetensors").write_bytes(b"x")
            self.assertTrue(m.package_looks_like_mtp(glm))

            asst = root / "asst"
            asst.mkdir()
            (asst / "ax_gemma4_assistant_mtp.json").write_text("{}")
            self.assertTrue(m.package_looks_like_mtp(asst))


if __name__ == "__main__":
    unittest.main()
