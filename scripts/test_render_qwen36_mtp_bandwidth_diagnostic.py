from __future__ import annotations

import unittest

from scripts import render_qwen36_mtp_bandwidth_diagnostic as renderer


class Qwen36MtpBandwidthDiagnosticTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
