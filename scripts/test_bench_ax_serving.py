#!/usr/bin/env python3
"""Unit tests for the AX serving benchmark harness."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("bench_ax_serving.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_ax_serving", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
bench = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = bench
MODULE_SPEC.loader.exec_module(bench)


def prompt() -> object:
    return bench.PromptItem(
        id="p1",
        category="chat_short",
        input_text="Hello",
        input_tokens=[1, 2, 3],
        input_tokens_count=3,
        max_output_tokens=3,
        metadata={"suite": "unit"},
    )


def fake_stream(url: str, payload: dict[str, object], timeout: float):
    del url, timeout
    assert payload["input_tokens"] == [1, 2, 3]
    yield "__http_status__", {"status": 200}, 0.0
    yield "request", {"request": {"request_id": 1}}, 0.01
    yield "step", {"delta_tokens": [10]}, 0.10
    yield "step", {"delta_tokens": [11]}, 0.15
    yield "step", {"delta_tokens": [12]}, 0.21
    yield "response", {"response": {"output_token_count": 3}}, 0.24
    yield None, {"done": True}, 0.25


class AxServingBenchTests(unittest.TestCase):
    def test_parse_sse_text_supports_named_and_done_frames(self) -> None:
        frames = bench.parse_sse_text(
            'event: step\n'
            'data: {"delta_tokens":[1]}\n'
            "\n"
            "data: [DONE]\n"
            "\n"
        )

        self.assertEqual(frames[0], ("step", '{"delta_tokens":[1]}'))
        self.assertEqual(frames[1], (None, "[DONE]"))

    def test_parse_sse_text_accepts_no_space_after_colon(self) -> None:
        frames = bench.parse_sse_text(
            'event:step\n'
            'data:{"delta_tokens":[1]}\n'
            "\n"
            "data:[DONE]\n"
            "\n"
        )

        self.assertEqual(frames[0], ("step", '{"delta_tokens":[1]}'))
        self.assertEqual(frames[1], (None, "[DONE]"))

    def test_load_corpus_rejects_empty_token_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "category": "chat_short",
                        "input_tokens": [],
                        "max_output_tokens": 3,
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(SystemExit, "input_tokens must not be empty"):
                bench.load_corpus(corpus)

    def test_load_corpus_rejects_empty_text_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "category": "chat_short",
                        "input_text": "",
                        "max_output_tokens": 3,
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(SystemExit, "input_text must not be empty"):
                bench.load_corpus(corpus)

    def test_observe_stream_computes_ttft_tpot_and_intervals(self) -> None:
        observation = bench.observe_stream(
            fake_stream("http://unused", {"input_tokens": [1, 2, 3]}, 1.0),
            prompt=prompt(),
            scheduled_at_s=0.0,
            started_at_s=0.0,
            completed_at_s=0.25,
        )

        self.assertTrue(observation["ok"])
        self.assertEqual(observation["output_tokens"], 3)
        self.assertAlmostEqual(observation["ttft_ms"], 100.0)
        self.assertAlmostEqual(observation["client_tpot_ms"], 75.0)
        self.assertAlmostEqual(observation["stream_step_interval_ms"][0], 50.0)
        self.assertAlmostEqual(observation["stream_step_interval_ms"][1], 60.0)

    def test_summary_computes_percentiles_and_goodput(self) -> None:
        observations = [
            {
                "phase": "measured",
                "ok": True,
                "category": "chat_short",
                "ttft_ms": 100.0,
                "client_tpot_ms": 20.0,
                "e2e_latency_ms": 200.0,
                "queue_delay_ms": 0.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "stream_step_interval_ms": [20.0, 25.0],
            },
            {
                "phase": "measured",
                "ok": True,
                "category": "chat_short",
                "ttft_ms": 300.0,
                "client_tpot_ms": 40.0,
                "e2e_latency_ms": 500.0,
                "queue_delay_ms": 10.0,
                "input_tokens": 20,
                "output_tokens": 10,
                "stream_step_interval_ms": [30.0],
            },
        ]

        summary = bench.summarize_observations(
            observations,
            wall_duration_s=1.0,
            ttft_slo_ms=200.0,
            tpot_slo_ms=50.0,
            e2e_slo_ms=600.0,
        )

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["goodput"]["requests"], 1)
        self.assertEqual(summary["output_token_throughput_tok_s"], 15.0)
        self.assertEqual(summary["ttft_ms"]["p50"], 200.0)

    def test_main_writes_serving_artifact_from_fake_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "category": "chat_short",
                        "input_text": "Hello",
                        "input_tokens": [1, 2, 3],
                        "max_output_tokens": 3,
                    }
                )
                + "\n"
            )
            output = root / "result.json"

            with contextlib.redirect_stdout(io.StringIO()):
                code = bench.main_with_args_for_test(
                    [
                        "--base-url",
                        "http://127.0.0.1:8080",
                        "--model-id",
                        "qwen3_dense",
                        "--corpus",
                        str(corpus),
                        "--requests",
                        "1",
                        "--warmup-requests",
                        "0",
                        "--input-kind",
                        "tokens",
                        "--output",
                        str(output),
                    ],
                    stream_func=fake_stream,
                )

            result = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(result["schema_version"], "ax.serving_benchmark.v1")
            self.assertEqual(result["summary"]["ok_requests"], 1)
            self.assertEqual(result["corpus"]["prompt_count"], 1)


if __name__ == "__main__":
    unittest.main()
