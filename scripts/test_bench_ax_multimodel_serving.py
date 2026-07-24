#!/usr/bin/env python3
"""Tests for the timed multi-model serving benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_ax_multimodel_serving.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
MODULE_SPEC = importlib.util.spec_from_file_location("bench_ax_multimodel_serving", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_ax_multimodel_serving"] = benchmark
MODULE_SPEC.loader.exec_module(benchmark)


class MultiModelServingBenchmarkTests(unittest.TestCase):
    def test_replay_separates_request_models_and_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario = Path(temp_dir) / "scenario.jsonl"
            rows = [
                {
                    "id": "qwen",
                    "kind": "request",
                    "at_ms": 0,
                    "model": "qwen",
                    "input_tokens": [1, 2],
                    "max_output_tokens": 2,
                },
                {
                    "id": "gemma",
                    "kind": "request",
                    "at_ms": 1,
                    "model": "gemma",
                    "input_text": "hello",
                    "max_output_tokens": 2,
                },
                {
                    "id": "remove",
                    "kind": "unload",
                    "at_ms": 2,
                    "model": "gemma",
                },
            ]
            scenario.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = argparse.Namespace(
                scenario=scenario,
                base_url="http://127.0.0.1:1",
                workers=3,
                input_kind="auto",
                timeout=1.0,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=0,
                slo_ttft_ms=100.0,
                slo_tpot_ms=100.0,
                slo_e2e_ms=1000.0,
            )

            def fake_request_runner(**kwargs: object) -> dict[str, object]:
                prompt = kwargs["prompt"]
                return {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "phase": "measured",
                    "status": 200,
                    "ok": True,
                    "error": None,
                    "scheduled_at_s": kwargs["scheduled_offset_s"],
                    "started_at_s": kwargs["scheduled_offset_s"],
                    "queue_delay_ms": 0.0,
                    "e2e_latency_ms": 10.0,
                    "ttft_ms": 2.0,
                    "client_tpot_ms": 3.0,
                    "stream_step_interval_ms": [3.0],
                    "input_tokens": prompt.input_tokens_count,
                    "max_output_tokens": prompt.max_output_tokens,
                    "output_tokens": 2,
                    "output_chunks": 2,
                    "events": 3,
                    "route_decisions": {},
                    "metadata": {},
                }

            def fake_control_runner(
                event: benchmark.ScenarioEvent, **_: object
            ) -> dict[str, object]:
                return {
                    "event_id": event.id,
                    "kind": event.kind,
                    "model_id": event.model_id,
                    "category": event.category,
                    "scheduled_at_s": event.at_s,
                    "started_at_s": event.at_s,
                    "latency_ms": 4.0,
                    "status": 200,
                    "ok": True,
                    "error": None,
                    "response": {},
                }

            artifact = benchmark.run_benchmark(
                args,
                request_runner=fake_request_runner,
                control_runner=fake_control_runner,
            )

        self.assertEqual(artifact["schema_version"], "ax.multimodel_serving_benchmark.v1")
        self.assertEqual(sorted(artifact["by_model"]), ["gemma", "qwen"])
        self.assertEqual(artifact["summary"]["requests"], 2)
        self.assertEqual(artifact["lifecycle"]["events"], 1)
        self.assertEqual(artifact["lifecycle"]["error_events"], 0)
        self.assertEqual(artifact["focus"]["policy"], "qwen3_gemma4_primary")
        self.assertEqual(artifact["availability"]["request_http_503"], 0)
        self.assertIn("request_error_rate", artifact["availability"])
        self.assertTrue(artifact["route_contract"]["passed"])

    def test_route_contract_fails_closed(self) -> None:
        contract = benchmark.route_contract(
            {"route_decisions": {"used": 2}},
            [("used", 2), ("missing", 1)],
        )
        self.assertFalse(contract["passed"])
        self.assertEqual(contract["observed"]["used"], 2)
        self.assertIsNone(contract["observed"]["missing"])
        self.assertIn("missing", contract["failures"][0])

    def test_route_requirement_parser(self) -> None:
        self.assertEqual(benchmark.parse_route_requirement("route"), ("route", 1))
        self.assertEqual(benchmark.parse_route_requirement("route=3"), ("route", 3))
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.parse_route_requirement("route=-1")

    def test_load_requires_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario = Path(temp_dir) / "scenario.jsonl"
            scenario.write_text(
                json.dumps(
                    {
                        "id": "load",
                        "kind": "load",
                        "model": "gemma",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(SystemExit, "requires model_path"):
                benchmark.load_scenario(scenario)

    def test_focus_family_classification(self) -> None:
        self.assertEqual(benchmark.classify_focus_family("qwen3.5-9b"), "qwen3")
        self.assertEqual(benchmark.classify_focus_family("gemma-4-12b-it"), "gemma4")
        self.assertIsNone(benchmark.classify_focus_family("llama-3.1-8b"))

    def test_agent_coexist_manifest_loads(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        scenario = repo / "benchmarks/manifests/replay/qwen_gemma_agent_coexist.jsonl"
        events = benchmark.load_scenario(scenario)
        self.assertEqual(len(events), 4)
        models = {event.model_id for event in events}
        self.assertEqual(models, {"qwen3.5-9b", "gemma-4-12b-it"})
        self.assertTrue(all(event.kind == "request" for event in events))

    def test_compact_token_pattern_expands_deterministically(self) -> None:
        event = benchmark.ScenarioEvent(
            id="long",
            kind="request",
            at_s=0.0,
            model_id="gemma",
            category="long_prefill",
            raw={
                "input_token_pattern": [3, 7, 11],
                "input_tokens_count": 8,
                "max_output_tokens": 1,
            },
        )
        prompt = benchmark.prompt_for_event(event)
        self.assertEqual(prompt.input_tokens, [3, 7, 11, 3, 7, 11, 3, 7])
        self.assertEqual(prompt.input_tokens_count, 8)

    def test_compact_text_pattern_expands_deterministically(self) -> None:
        event = benchmark.ScenarioEvent(
            id="long-text",
            kind="request",
            at_s=0.0,
            model_id="gemma",
            category="long_prefill",
            raw={
                "input_text_pattern": "alpha beta ",
                "input_text_repeats": 3,
                "input_tokens_count": 6,
                "max_output_tokens": 1,
            },
        )

        prompt = benchmark.prompt_for_event(event)

        self.assertEqual(prompt.input_text, "alpha beta alpha beta alpha beta ")
        self.assertEqual(prompt.input_tokens_count, 6)

    def test_text_prefix_precedes_expanded_pattern(self) -> None:
        event = benchmark.ScenarioEvent(
            id="gemma",
            kind="request",
            at_s=0.0,
            model_id="gemma-4-12b-it",
            category="long_prefill",
            raw={
                "input_text_prefix": "<bos>",
                "input_text_pattern": "alpha ",
                "input_text_repeats": 2,
                "max_output_tokens": 1,
            },
        )

        prompt = benchmark.prompt_for_event(event)

        self.assertEqual(prompt.input_text, "<bos>alpha alpha ")

    def test_token_file_loads_relative_to_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scenario_dir = Path(temp_dir)
            tokens = scenario_dir / "tokens.json"
            tokens.write_text(json.dumps({"token_ids": [9, 8, 7]}))
            event = benchmark.ScenarioEvent(
                id="file",
                kind="request",
                at_s=0.0,
                model_id="qwen",
                category="mtp",
                raw={"input_tokens_path": "tokens.json", "max_output_tokens": 2},
            )

            prompt = benchmark.prompt_for_event(event, scenario_dir=scenario_dir)

        self.assertEqual(prompt.input_tokens, [9, 8, 7])
        self.assertEqual(prompt.input_tokens_count, 3)


if __name__ == "__main__":
    unittest.main()
