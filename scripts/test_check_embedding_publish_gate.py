#!/usr/bin/env python3
"""Unit tests for embedding publication gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = load_script("check_embedding_publish_gate", "check_embedding_publish_gate.py")


def _paired_fair_artifact(**overrides):
    payload = {
        "schema_version": "ax.embedding_fair.v2",
        "output_contract": "contiguous_cpu_f32_batch_hidden",
        "ax_only": False,
        "publication_claim": "paired_delta",
        "reference": "mlx_lm",
        "warmup": 2,
        "trials": 5,
        "git_commit": "abc123",
        "build": {"commit": "abc123", "git_tracked_dirty": False},
        "host": {"chip": "Apple M5 Max", "memory_gb": 128, "platform": "darwin"},
        "runtime_identity": {
            "benchmark_surface": "embedding_in_process",
            "ax_engine_native": {
                "path": "/tmp/_ax_engine.abi3.so",
                "sha256": "deadbeef",
                "linked_mlx": [
                    {
                        "install_name": "/venv/lib/libmlx.dylib",
                        "source_class": "pip_or_venv",
                        "sha256": "111",
                    }
                ],
            },
            "reference_runtime": {
                "module": "mlx_lm",
                "linked_mlx": [
                    {
                        "install_name": "/venv/lib/libmlx.dylib",
                        "source_class": "pip_or_venv",
                        "sha256": "111",
                    }
                ],
            },
        },
        "models": [
            {
                "model_label": "qwen-test",
                "rows": [
                    {
                        "workload": "short_query_b1",
                        "primary_metric": "median_ms_per_item",
                        "results": {
                            "mlx_lm": {
                                "median_tokens_per_sec": 100.0,
                                "median_ms_per_item": 10.0,
                            },
                            "ax_engine_py": {
                                "median_tokens_per_sec": 110.0,
                                "median_ms_per_item": 9.0,
                            },
                        },
                        "comparison": {
                            "ax_vs_reference_tokens_pct": 10.0,
                            "ax_vs_reference_ms_per_item_pct": -10.0,
                        },
                    },
                    {
                        "workload": "fixed_16_b8",
                        "primary_metric": "median_tokens_per_sec",
                        "results": {
                            "mlx_lm": {"median_tokens_per_sec": 1000.0},
                            "ax_engine_py": {"median_tokens_per_sec": 1100.0},
                        },
                        "comparison": {"ax_vs_reference_tokens_pct": 10.0},
                    },
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


class EmbeddingPublishGateTests(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        path = Path(tmp.name)
        path.write_text(json.dumps(payload) + "\n")
        tmp.close()
        return path

    def test_paired_v2_artifact_passes(self) -> None:
        path = self._write(_paired_fair_artifact())
        report = gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)
        self.assertTrue(report["ok"])

    def test_paired_rejects_ax_only(self) -> None:
        path = self._write(
            _paired_fair_artifact(ax_only=True, publication_claim="paired_delta")
        )
        with self.assertRaisesRegex(gate.PublishGateError, "ax_only=false"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_ax_absolute_rejects_reference_results_without_ax_only(self) -> None:
        path = self._write(_paired_fair_artifact(ax_only=False))
        with self.assertRaisesRegex(gate.PublishGateError, "ax_absolute_trend"):
            gate.validate_artifact(path, claim=gate.CLAIM_AX_ONLY)

    def test_homebrew_vs_pip_rejects_paired(self) -> None:
        payload = _paired_fair_artifact()
        payload["runtime_identity"]["ax_engine_native"]["linked_mlx"] = [
            {
                "install_name": "/opt/homebrew/opt/mlx/lib/libmlx.dylib",
                "source_class": "homebrew",
                "sha256": "aaa",
            }
        ]
        payload["runtime_identity"]["reference_runtime"]["linked_mlx"] = [
            {
                "install_name": "/venv/lib/python3.14/site-packages/mlx/lib/libmlx.dylib",
                "source_class": "pip_or_venv",
                "sha256": "bbb",
            }
        ]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "Homebrew libmlx"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_legacy_requires_flag(self) -> None:
        payload = {
            "schema_version": "ax.embedding_fair.v1",
            "output_contract": "contiguous_cpu_f32_batch_hidden",
            "ax_only": False,
            "reference": "mlx_lm",
            "git_commit": "old",
            "models": [
                {
                    "model_label": "qwen",
                    "rows": [
                        {
                            "workload": "fixed_16_b1",
                            "results": {
                                "mlx_lm": {"median_tokens_per_sec": 1.0},
                                "ax_engine_py": {"median_tokens_per_sec": 1.0},
                            },
                            "comparison": {"ax_vs_reference_tokens_pct": 0.0},
                        }
                    ],
                }
            ],
        }
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "legacy"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)
        report = gate.validate_artifact(
            path, claim=gate.CLAIM_PAIRED, allow_legacy=True
        )
        self.assertTrue(report["ok"])
        self.assertTrue(any("legacy" in w for w in report["warnings"]))

    def test_missing_runtime_identity_fails_v2(self) -> None:
        payload = _paired_fair_artifact()
        del payload["runtime_identity"]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "runtime_identity"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_scale_requires_p95(self) -> None:
        payload = {
            "schema_version": "ax.embedding_ingest_scale.v2",
            "output_contract": "contiguous_cpu_f32_batch_hidden",
            "ax_only": False,
            "publication_claim": "paired_delta",
            "reference": "mlx_lm",
            "warmup": 2,
            "trials": 5,
            "git_commit": "abc",
            "build": {"commit": "abc", "git_tracked_dirty": False},
            "host": {"chip": "Apple M5 Max"},
            "runtime_identity": {
                "ax_engine_native": {
                    "path": "/tmp/x.so",
                    "linked_mlx": [
                        {"install_name": "/venv/libmlx.dylib", "source_class": "pip_or_venv"}
                    ],
                },
                "reference_runtime": {
                    "module": "mlx_lm",
                    "linked_mlx": [
                        {"install_name": "/venv/libmlx.dylib", "source_class": "pip_or_venv"}
                    ],
                },
            },
            "models": [
                {
                    "model_label": "qwen",
                    "rows": [
                        {
                            "workload": "scale_512x256_b8",
                            "results": {
                                "mlx_lm": {
                                    "median_tokens_per_sec": 1.0,
                                    "median_batch_p95_ms": 10.0,
                                },
                                "ax_engine_py": {
                                    "median_tokens_per_sec": 1.0,
                                    # missing p95
                                },
                            },
                            "comparison": {"ax_vs_reference_tokens_pct": 0.0},
                        }
                    ],
                }
            ],
        }
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "median_batch_p95_ms"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)


if __name__ == "__main__":
    unittest.main()
