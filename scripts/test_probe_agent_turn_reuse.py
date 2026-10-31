#!/usr/bin/env python3
"""Unit tests for the agent turn-reuse probe helpers (no server needed)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("probe_agent_turn_reuse.py")
MODULE_SPEC = importlib.util.spec_from_file_location("probe_agent_turn_reuse", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
probe = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = probe
MODULE_SPEC.loader.exec_module(probe)


def sse(*chunks: object, error: dict | None = None) -> list[bytes]:
    lines: list[bytes] = []
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}\n".encode(), b"\n"]
    if error is not None:
        lines += [b"event: error\n", f"data: {json.dumps(error)}\n".encode(), b"\n"]
    return lines + [b"data: [DONE]\n", b"\n"]


def delta(**fields: object) -> dict:
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


class AccumulateStreamTests(unittest.TestCase):
    def test_folds_reasoning_content_tool_calls_and_usage(self):
        ticks = iter([10.0, 11.0, 12.0])
        lines = sse(
            delta(role="assistant"),
            delta(reasoning_content="plan "),
            delta(reasoning_content="it"),
            delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":'},
                    }
                ]
            ),
            delta(tool_calls=[{"index": 0, "function": {"arguments": '"a.rs"}'}}]),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 9,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
        )
        result = probe.accumulate_stream(probe.iter_sse_data(lines), clock=lambda: next(ticks))
        self.assertEqual(result["first_delta_at"], 10.0)
        self.assertEqual(result["finish_reason"], "tool_calls")
        message = result["message"]
        self.assertEqual(message["reasoning_content"], "plan it")
        self.assertEqual(message["content"], "")
        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.rs"}'},
                }
            ],
        )
        row = probe.turn_row(1, 9.5, result)
        self.assertEqual(row["ttft_ms"], 500.0)
        self.assertEqual((row["prompt_tokens"], row["cached_tokens"]), (100, 64))
        self.assertEqual(row["recomputed_tokens"], 36)
        self.assertEqual(row["tool_calls"], 1)

    def test_records_stream_errors(self):
        lines = sse(delta(content="x"), error={"error": {"code": "incomplete_tool_call"}})
        result = probe.accumulate_stream(probe.iter_sse_data(lines))
        self.assertEqual(result["error"], {"code": "incomplete_tool_call"})
        self.assertNotIn("reasoning_content", result["message"])


class NextMessagesTests(unittest.TestCase):
    def test_tool_calls_get_deterministic_results_and_plain_turns_continue(self):
        reply = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{}"},
                }
            ],
        }
        first = probe.next_messages([], reply, tool_result_lines=3)
        self.assertEqual(first, probe.next_messages([], reply, tool_result_lines=3))
        self.assertEqual([m["role"] for m in first], ["assistant", "tool"])
        self.assertEqual(first[1]["tool_call_id"], "c1")
        self.assertEqual(len(first[1]["content"].splitlines()), 3)

        plain = probe.next_messages([], {"role": "assistant", "content": "done"}, 3)
        self.assertEqual(plain[-1], {"role": "user", "content": "Continue."})


class SummaryTests(unittest.TestCase):
    def test_summary_skips_the_cold_turn_and_errors(self):
        rows = [
            {
                "turn": 0,
                "ttft_ms": 900.0,
                "recomputed_tokens": 800,
                "cached_tokens": 0,
                "error": None,
            },
            {
                "turn": 1,
                "ttft_ms": 300.0,
                "recomputed_tokens": 200,
                "cached_tokens": 700,
                "error": None,
            },
            {
                "turn": 2,
                "ttft_ms": 100.0,
                "recomputed_tokens": 50,
                "cached_tokens": 950,
                "error": None,
            },
            {"turn": 3, "ttft_ms": None, "recomputed_tokens": 9, "cached_tokens": 0, "error": {}},
        ]
        self.assertEqual(
            probe.summarize(rows),
            {
                "followup_turns": 2,
                "recomputed_tokens": 250,
                "cached_tokens": 1650,
                "ttft_ms_total": 400.0,
                "ttft_ms_mean": 200.0,
            },
        )


class RequestErrorTests(unittest.TestCase):
    def test_http_error_becomes_an_error_row_that_summary_skips(self):
        import io
        import urllib.error

        exc = urllib.error.HTTPError(
            "http://x/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"code":"context_length_exceeded"}}'),
        )
        row = probe.error_row(7, probe.request_error(exc))
        self.assertEqual(row["error"]["http_status"], 400)
        self.assertIn("context_length_exceeded", row["error"]["body"])
        self.assertIsNone(row["reply_sha256"])
        ok = {
            "turn": 1,
            "ttft_ms": 10.0,
            "recomputed_tokens": 5,
            "cached_tokens": 95,
            "error": None,
        }
        self.assertEqual(probe.summarize([ok, row])["followup_turns"], 1)

    def test_other_failures_keep_their_type(self):
        self.assertEqual(
            probe.request_error(ConnectionResetError("peer"))["exception"],
            "ConnectionResetError",
        )


class ArgumentTests(unittest.TestCase):
    def test_settings_parse_and_reject_unknown_values(self):
        self.assertEqual(probe.parse_settings("off, on"), ["off", "on"])
        with self.assertRaises(argparse.ArgumentTypeError):
            probe.parse_settings("maybe")
        with self.assertRaises(argparse.ArgumentTypeError):
            probe.positive_int("0")


if __name__ == "__main__":
    unittest.main()
