#!/usr/bin/env python3
"""Unit tests for direct/ngram outperformance gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("check_direct_ngram_outperformance.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_direct_ngram_outperformance", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
checker = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = checker
MODULE_SPEC.loader.exec_module(checker)


def metric(value: float) -> dict[str, float]:
    return {"mean": value, "median": value, "min": value, "max": value}


def row(
    engine: str,
    decode_tok_s: float,
    *,
    status: str | None = None,
    route: str | None = None,
    prompt_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine": engine,
        "prompt_tokens": 128,
        "generation_tokens": 128,
        "decode_tok_s": metric(decode_tok_s),
    }
    if status is not None:
        payload["ax_decode_claim_status"] = status
    if route is not None:
        payload["ax_decode_effective_route"] = route
    if prompt_source is not None:
        payload["prompt_source"] = prompt_source
    return payload


def write_artifact(root: Path, *, rows: list[dict[str, object]]) -> Path:
    path = root / "model.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": checker.SCHEMA_VERSION,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def write_named_artifact(root: Path, name: str, *, rows: list[dict[str, object]]) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            {
                "schema_version": checker.SCHEMA_VERSION,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return path


class DirectNgramOutperformanceTests(unittest.TestCase):
    def test_passes_when_direct_and_effective_ngram_beat_mlx_lm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=True,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)
        self.assertAlmostEqual(checked[0].direct_delta_pct, 5.0)
        self.assertAlmostEqual(checked[0].ngram_delta_pct, 40.0)

    def test_skips_direct_only_probe_artifacts_in_gate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )
            write_named_artifact(
                root,
                "model-dense-fusions-probe.json",
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=True,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)

    def test_fails_when_direct_does_not_beat_mlx_lm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 99.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )

            with self.assertRaisesRegex(checker.GateError, "direct decode"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_fails_when_ngram_is_only_fallback_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route="linear_no_draft_direct_pipeline_fallback",
                    ),
                ],
            )

            with self.assertRaisesRegex(checker.GateError, "not effective throughput"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_allows_random_prompt_ngram_fallback_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route="linear_no_draft_direct_pipeline_fallback",
                        prompt_source="random",
                    ),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=True,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)
        self.assertFalse(checked[0].ngram_effective_required)

    def test_allows_random_prompt_request_disabled_direct_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route="no_draft_fallback",
                        prompt_source="random",
                    ),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=True,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)
        self.assertFalse(checked[0].ngram_effective_required)

    def test_allows_ngram_fallback_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route="linear_no_draft_direct_pipeline_fallback",
                    ),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=False,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)

    def test_rejects_unknown_ngram_claim_status_even_for_random_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="unknown_status",
                        route="no_draft_fallback",
                        prompt_source="random",
                    ),
                ],
            )

            with self.assertRaisesRegex(checker.GateError, "unknown claim status"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_rejects_unknown_ngram_effective_route_even_for_random_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route="unknown_route",
                        prompt_source="random",
                    ),
                ],
            )

            with self.assertRaisesRegex(checker.GateError, "unknown effective route"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_rejects_inconsistent_ngram_status_route_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_draft_direct_fallback",
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                        prompt_source="random",
                    ),
                ],
            )

            with self.assertRaisesRegex(checker.GateError, "inconsistent status/route"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_allows_observed_no_draft_route_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0, prompt_source="random"),
                    row("ax_engine_mlx", 105.0, prompt_source="random"),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        106.0,
                        status="ngram_no_observed_draft_path",
                        route="ngram_accepted_without_decode_route",
                        prompt_source="random",
                    ),
                ],
            )

            checked = checker.check_artifact_dir(
                root,
                min_delta_pct=0.0,
                require_effective_ngram=True,
                require_sweep_ok=True,
            )

        self.assertEqual(len(checked), 1)

    def test_sweep_skips_fail_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )
            (root / "sweep_results.json").write_text(
                json.dumps({"rows": [{"slug": "missing-model", "status": "model_dir_missing"}]})
            )

            with self.assertRaisesRegex(checker.GateError, "non-ok rows"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_sweep_publication_candidate_drift_fails_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )
            (root / "sweep_results.json").write_text(
                json.dumps(
                    {
                        "publication_candidate": False,
                        "failed_row_count": 0,
                        "status_counts": {"ok": 1},
                        "rows": [{"slug": "ok-row", "status": "ok"}],
                    }
                )
            )

            with self.assertRaisesRegex(
                checker.GateError, "publication_candidate=false"
            ):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_sweep_failed_row_count_drift_fails_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )
            (root / "sweep_results.json").write_text(
                json.dumps(
                    {
                        "publication_candidate": True,
                        "failed_row_count": 1,
                        "status_counts": {"ok": 1},
                        "rows": [{"slug": "ok-row", "status": "ok"}],
                    }
                )
            )

            with self.assertRaisesRegex(checker.GateError, "failed_row_count=1"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )

    def test_sweep_status_counts_drift_fails_completion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_artifact(
                root,
                rows=[
                    row("mlx_lm", 100.0),
                    row("ax_engine_mlx", 105.0),
                    row(
                        "ax_engine_mlx_ngram_accel",
                        140.0,
                        status=checker.NGRAM_EFFECTIVE_STATUS,
                        route=checker.NGRAM_EFFECTIVE_ROUTE,
                    ),
                ],
            )
            (root / "sweep_results.json").write_text(
                json.dumps(
                    {
                        "publication_candidate": True,
                        "failed_row_count": 0,
                        "status_counts": {"bench_failed": 1},
                        "rows": [{"slug": "ok-row", "status": "ok"}],
                    }
                )
            )

            with self.assertRaisesRegex(checker.GateError, "status_counts"):
                checker.check_artifact_dir(
                    root,
                    min_delta_pct=0.0,
                    require_effective_ngram=True,
                    require_sweep_ok=True,
                )


if __name__ == "__main__":
    unittest.main()
