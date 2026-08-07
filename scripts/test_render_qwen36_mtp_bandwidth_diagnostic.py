from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import render_qwen36_mtp_bandwidth_diagnostic as renderer


class Qwen36MtpBandwidthDiagnosticTests(unittest.TestCase):
    def test_build_diagnostic_accepts_repo_relative_summary_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=renderer.REPO_ROOT) as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(json.dumps({"rows": []}))
            relative_path = summary_path.relative_to(renderer.REPO_ROOT)

            diagnostic = renderer.build_diagnostic(relative_path)

        self.assertEqual(
            diagnostic["source_summary"],
            relative_path.as_posix(),
        )

    def test_engine_labels_use_measured_summary_identities(self) -> None:
        labels = renderer.measured_engine_labels(
            {
                "engine_identities": {
                    "ax_engine": {
                        "name": "AX Engine",
                        "version": "6.13.3",
                    },
                    "mtplx": {
                        "name": "MTPLX",
                        "version": "2.1.0",
                    },
                    "lightning_mlx": {
                        "name": "lightning-mlx",
                        "version": "0.6.10",
                    },
                }
            }
        )

        self.assertEqual(labels["ax_engine"], "AX Engine v6.13.3")
        self.assertEqual(labels["mtplx"], "MTPLX v2.1.0")
        self.assertEqual(labels["lightning_mlx"], "lightning-mlx v0.6.10")

    def test_engine_labels_preserve_legacy_fallback(self) -> None:
        self.assertEqual(
            renderer.measured_engine_labels({}),
            renderer.LEGACY_ENGINE_LABELS,
        )

    def test_mtplx_same_sidecar_estimate_does_not_require_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "mtplx.json"
            artifact.write_text(
                json.dumps(
                    {
                        "model_inspection": {
                            "model_dir": "/missing/private/model/cache"
                        },
                        "results": [
                            {
                                "runs": [
                                    {
                                        "measured": True,
                                        "generated_tokens": 100,
                                        "accepted_drafts": 75,
                                        "decode_elapsed_s": 2.0,
                                    }
                                ]
                            }
                        ],
                    }
                )
            )
            with patch.object(renderer, "REPO_ROOT", root):
                bytes_used, source, cycle_summary = renderer.mtplx_artifact_estimate(
                    {
                        "artifact": "mtplx.json",
                        "model_label": "Qwen3.6 27B 4-bit",
                    },
                    same_sidecar_bytes=16_900_000_000,
                )

        self.assertEqual(bytes_used, 16_900_000_000)
        self.assertEqual(
            source,
            "same_ax_sidecar_bytes_from_committed_ax_artifact",
        )
        self.assertEqual(cycle_summary["target_cycles_per_s"], 12.5)


if __name__ == "__main__":
    unittest.main()
