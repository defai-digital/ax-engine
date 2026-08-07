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

    def test_build_diagnostic_accepts_summary_path_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(json.dumps({"rows": []}))

            diagnostic = renderer.build_diagnostic(summary_path)

        self.assertEqual(
            diagnostic["source_summary"],
            summary_path.resolve().as_posix(),
        )

    def test_lightning_row_uses_fresh_matching_package_proxy_label(self) -> None:
        rows = [
            {
                "model_label": "Qwen3.6 35B-A3B 4-bit",
                "engine": engine,
                "artifact": f"{engine}.json",
                "metrics": {"decode_tok_s": decode},
            }
            for engine, decode in (
                ("ax_engine", 140.0),
                ("mtplx", 145.0),
                ("lightning_mlx", 124.0),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.json"
            summary_path.write_text(json.dumps({"rows": []}))
            with (
                patch.object(renderer, "peer_rows", return_value=rows),
                patch.object(
                    renderer,
                    "ax_artifact_estimate",
                    return_value=(1_700_000_000, "ax_estimate", {}),
                ),
                patch.object(
                    renderer,
                    "mtplx_artifact_estimate",
                    return_value=(2_900_000_000, "mtplx_estimate", {}),
                ),
            ):
                diagnostic = renderer.build_diagnostic(summary_path)

        lightning_row = next(
            row
            for row in diagnostic["rows"]
            if row["engine"] == "lightning_mlx"
        )
        self.assertEqual(
            lightning_row["byte_estimate_source"],
            "matching_peer_package_proxy_from_mtplx_estimate",
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
