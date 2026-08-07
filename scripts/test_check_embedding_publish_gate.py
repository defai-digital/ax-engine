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


def _performance_conditions() -> dict:
    return {
        "load_average": {"one_minute": 0.5},
        "power_source": "AC Power",
        "thermal_warning_recorded": False,
        "performance_warning_recorded": False,
        "cpu_power_status_recorded": False,
        "top_processes_cpu": [{"cpu_percent": 0.1, "command": "test"}],
    }


def _trials(tokens_per_sec: float, **metrics: float) -> list[dict[str, float]]:
    return [{"tokens_per_sec": tokens_per_sec, **metrics} for _ in range(5)]


def _paired_fair_artifact(**overrides):
    payload = {
        "schema_version": "ax.embedding_fair.v2",
        "output_contract": "contiguous_cpu_f32_batch_hidden",
        "ax_only": False,
        "publication_claim": "paired_delta",
        "reference": "mlx_lm",
        "warmup": 2,
        "trials": 5,
        "trial_order": "interleaved_alternating",
        "git_commit": "a" * 40,
        "build": {
            "commit": "a" * 40,
            "engine_version": "6.13.2",
            "git_tracked_dirty": False,
        },
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
                                "trials": _trials(100.0, ms_per_item=10.0),
                            },
                            "ax_engine_py": {
                                "median_tokens_per_sec": 110.0,
                                "median_ms_per_item": 9.0,
                                "trials": _trials(110.0, ms_per_item=9.0),
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
                            "mlx_lm": {
                                "median_tokens_per_sec": 1000.0,
                                "trials": _trials(1000.0),
                            },
                            "ax_engine_py": {
                                "median_tokens_per_sec": 1100.0,
                                "trials": _trials(1100.0),
                            },
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
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)
        path.write_text(json.dumps(payload) + "\n")
        return path

    def test_paired_v2_artifact_passes(self) -> None:
        path = self._write(_paired_fair_artifact())
        report = gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)
        self.assertTrue(report["ok"])

    def test_paired_rejects_ax_only(self) -> None:
        path = self._write(_paired_fair_artifact(ax_only=True, publication_claim="paired_delta"))
        with self.assertRaisesRegex(gate.PublishGateError, "ax_only=false"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_requires_boolean_ax_only(self) -> None:
        path = self._write(_paired_fair_artifact(ax_only="false"))
        with self.assertRaisesRegex(gate.PublishGateError, "boolean ax_only"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_requires_recognized_publication_claim(self) -> None:
        path = self._write(_paired_fair_artifact(publication_claim="directional"))
        with self.assertRaisesRegex(gate.PublishGateError, "recognized publication_claim"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_unknown_reference_backend_fails_closed(self) -> None:
        path = self._write(_paired_fair_artifact(reference="unknown"))
        with self.assertRaisesRegex(gate.PublishGateError, "unsupported embedding reference"):
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
        with self.assertRaisesRegex(gate.PublishGateError, "Homebrew / pip"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_pip_vs_homebrew_rejects_paired_in_reverse_direction(self) -> None:
        payload = _paired_fair_artifact()
        payload["runtime_identity"]["reference_runtime"]["linked_mlx"] = [
            {
                "install_name": "/opt/homebrew/opt/mlx/lib/libmlx.dylib",
                "source_class": "homebrew",
                "sha256": "aaa",
            }
        ]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "Homebrew / pip"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_different_pip_mlx_hashes_reject_paired(self) -> None:
        payload = _paired_fair_artifact()
        payload["runtime_identity"]["reference_runtime"]["linked_mlx"][0]["sha256"] = "different"
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "different linked MLX binaries"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_missing_linked_mlx_fingerprint_rejects_paired(self) -> None:
        payload = _paired_fair_artifact()
        payload["runtime_identity"]["reference_runtime"]["linked_mlx"] = []
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "requires linked MLX fingerprints"):
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
        report = gate.validate_artifact(path, claim=gate.CLAIM_PAIRED, allow_legacy=True)
        self.assertTrue(report["ok"])
        self.assertTrue(any("legacy" in w for w in report["warnings"]))

    def test_missing_runtime_identity_fails_v2(self) -> None:
        payload = _paired_fair_artifact()
        del payload["runtime_identity"]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "runtime_identity"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_missing_v2_build_metadata_fails_cleanly(self) -> None:
        payload = _paired_fair_artifact()
        del payload["build"]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "requires build metadata"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_requires_tracked_tree_status(self) -> None:
        payload = _paired_fair_artifact()
        del payload["build"]["git_tracked_dirty"]
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "build.git_tracked_dirty"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_requires_full_build_identity_and_interleaved_trials(self) -> None:
        short_commit = _paired_fair_artifact(
            build={
                "commit": "abc123",
                "engine_version": "6.13.2",
                "git_tracked_dirty": False,
            }
        )
        path = self._write(short_commit)
        with self.assertRaisesRegex(gate.PublishGateError, "full measured"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

        blocked_order = _paired_fair_artifact(trial_order="blocked")
        path = self._write(blocked_order)
        with self.assertRaisesRegex(gate.PublishGateError, "interleaved_alternating"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_rejects_dirty_or_under_sampled_artifact(self) -> None:
        dirty = _paired_fair_artifact(
            build={
                "commit": "a" * 40,
                "engine_version": "6.13.2",
                "git_tracked_dirty": True,
            }
        )
        path = self._write(dirty)
        with self.assertRaisesRegex(gate.PublishGateError, "require-clean-tree"):
            gate.validate_artifact(
                path,
                claim=gate.CLAIM_PAIRED,
                require_clean_tree=True,
            )

        under_sampled = _paired_fair_artifact(warmup=1)
        path = self._write(under_sampled)
        with self.assertRaisesRegex(gate.PublishGateError, "warmup >= 2"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_scale_requires_p95(self) -> None:
        payload = {
            "schema_version": "ax.embedding_ingest_scale.v2",
            "status": "complete",
            "output_contract": "contiguous_cpu_f32_batch_hidden",
            "ax_only": False,
            "publication_claim": "paired_delta",
            "reference": "mlx_lm",
            "warmup": 2,
            "trials": 5,
            "cooldown_s": 15.0,
            "trial_order": "interleaved_alternating",
            "max_load_average": 2.0,
            "max_top_process_cpu_percent": 50.0,
            "benchmark_window": {
                "performance_conditions_start": _performance_conditions(),
                "performance_conditions_end": _performance_conditions(),
            },
            "git_commit": "a" * 40,
            "build": {
                "commit": "a" * 40,
                "engine_version": "6.13.2",
                "git_tracked_dirty": False,
            },
            "host": {"chip": "Apple M5 Max"},
            "runtime_identity": {
                "ax_engine_native": {
                    "path": "/tmp/x.so",
                    "linked_mlx": [
                        {
                            "install_name": "/venv/libmlx.dylib",
                            "source_class": "pip_or_venv",
                            "sha256": "same",
                        }
                    ],
                },
                "reference_runtime": {
                    "module": "mlx_lm",
                    "linked_mlx": [
                        {
                            "install_name": "/venv/libmlx.dylib",
                            "source_class": "pip_or_venv",
                            "sha256": "same",
                        }
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
                                    "trials": _trials(1.0, batch_p95_ms=10.0),
                                },
                                "ax_engine_py": {
                                    "median_tokens_per_sec": 1.0,
                                    "trials": _trials(1.0, batch_p95_ms=10.0),
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

    def test_paired_delta_must_match_recorded_medians(self) -> None:
        payload = _paired_fair_artifact()
        payload["models"][0]["rows"][0]["comparison"]["ax_vs_reference_tokens_pct"] = 99.0
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "is inconsistent"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_throughput_medians_must_be_finite_and_positive(self) -> None:
        payload = _paired_fair_artifact()
        payload["models"][0]["rows"][0]["results"]["ax_engine_py"]["median_tokens_per_sec"] = float(
            "nan"
        )
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "must be finite and positive"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_requires_all_declared_trial_rows(self) -> None:
        payload = _paired_fair_artifact()
        payload["models"][0]["rows"][0]["results"]["ax_engine_py"]["trials"].pop()
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "exactly 5 trial rows"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_v2_summary_must_match_trial_median(self) -> None:
        payload = _paired_fair_artifact()
        trials = payload["models"][0]["rows"][0]["results"]["ax_engine_py"]["trials"]
        for trial in trials[:3]:
            trial["tokens_per_sec"] = 1.0
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "inconsistent with trial rows"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_scale_requires_publication_cooldown(self) -> None:
        payload = {
            "schema_version": "ax.embedding_ingest_scale.v2",
            "status": "complete",
            "output_contract": "contiguous_cpu_f32_batch_hidden",
            "ax_only": False,
            "publication_claim": "paired_delta",
            "reference": "mlx_lm",
            "warmup": 2,
            "trials": 5,
            "cooldown_s": 0.0,
            "trial_order": "interleaved_alternating",
            "max_load_average": 2.0,
            "max_top_process_cpu_percent": 50.0,
            "benchmark_window": {
                "performance_conditions_start": _performance_conditions(),
                "performance_conditions_end": _performance_conditions(),
            },
            "build": {
                "commit": "a" * 40,
                "engine_version": "6.13.2",
                "git_tracked_dirty": False,
            },
            "host": {"chip": "Apple M5 Max"},
            "runtime_identity": {
                "ax_engine_native": {
                    "linked_mlx": [
                        {
                            "source_class": "pip_or_venv",
                            "sha256": "same",
                        }
                    ]
                },
                "reference_runtime": {
                    "linked_mlx": [
                        {
                            "source_class": "pip_or_venv",
                            "sha256": "same",
                        }
                    ]
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
                                    "median_batch_p95_ms": 1.0,
                                    "trials": _trials(1.0, batch_p95_ms=1.0),
                                },
                                "ax_engine_py": {
                                    "median_tokens_per_sec": 1.0,
                                    "median_batch_p95_ms": 1.0,
                                    "trials": _trials(1.0, batch_p95_ms=1.0),
                                },
                            },
                            "comparison": {"ax_vs_reference_tokens_pct": 0.0},
                        }
                    ],
                }
            ],
        }
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "cooldown_s >= 15"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)

    def test_scale_requires_clean_benchmark_conditions(self) -> None:
        payload = _paired_fair_artifact(
            schema_version="ax.embedding_ingest_scale.v2",
            status="complete",
            cooldown_s=15.0,
            max_load_average=2.0,
            max_top_process_cpu_percent=50.0,
            benchmark_window={
                "performance_conditions_start": _performance_conditions(),
                "performance_conditions_end": _performance_conditions(),
            },
        )
        for model in payload["models"]:
            for row in model["rows"]:
                for result in row["results"].values():
                    result["median_batch_p95_ms"] = 1.0
                    for trial in result["trials"]:
                        trial["batch_p95_ms"] = 1.0
        path = self._write(payload)
        report = gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)
        self.assertTrue(report["ok"])

        payload["benchmark_window"]["performance_conditions_end"]["thermal_warning_recorded"] = True
        path = self._write(payload)
        with self.assertRaisesRegex(gate.PublishGateError, "thermal_warning"):
            gate.validate_artifact(path, claim=gate.CLAIM_PAIRED)


if __name__ == "__main__":
    unittest.main()
