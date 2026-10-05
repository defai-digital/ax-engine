#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next lifecycle controls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run_flash_next_lifecycle.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_lifecycle", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def sse(*events: object, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def chunk(text: str | None = None, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


USAGE = {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}, "choices": []}


class LifecycleTests(unittest.TestCase):
    def test_sse_is_reassembled_with_finish_usage_and_done(self):
        out = mod.parse_sse(sse(chunk("1,"), chunk(" 2"), chunk(None, "stop"), USAGE))
        self.assertEqual(out["text"], "1, 2")
        self.assertEqual(out["finish_reason"], "stop")
        self.assertEqual(out["usage"]["completion_tokens"], 2)

    def test_sse_rejects_missing_done_usage_finish_or_error_events(self):
        for raw in (sse(chunk("a"), chunk(None, "stop"), USAGE, done=False),
                    sse(chunk("a"), chunk(None, "stop")),
                    sse(chunk("a"), USAGE),
                    sse(chunk("a"), chunk(None, "stop"), chunk(None, "stop"), USAGE),
                    sse({"error": {"message": "boom"}}, chunk(None, "stop"), USAGE)):
            with self.assertRaises(ValueError):
                mod.parse_sse(raw)

    def test_counters_require_every_lifecycle_metric(self):
        full = "".join(f"{name} 0\n" for name in mod.COUNTERS)
        with patch.object(mod.native, "request", return_value=full.encode()):
            self.assertEqual(set(mod.counters("http://x")), set(mod.COUNTERS))
        partial = "".join(f"{name} 0\n" for name in mod.COUNTERS[1:])
        with patch.object(mod.native, "request", return_value=partial.encode()):
            with self.assertRaisesRegex(ValueError, "missing lifecycle counters"):
                mod.counters("http://x")

    def test_wait_drained_returns_on_zero_and_times_out_otherwise(self):
        busy = {name: 0.0 for name in mod.COUNTERS}
        busy["ax_engine_generation_active_streams"] = 1.0
        idle = {name: 0.0 for name in mod.COUNTERS}
        with patch.object(mod, "counters", side_effect=[busy, busy, idle]), \
                patch.object(mod.time, "sleep"):
            self.assertEqual(mod.wait_drained("http://x"), idle)
        with patch.object(mod, "counters", return_value=busy), patch.object(mod.time, "sleep"):
            with self.assertRaises(TimeoutError):
                mod.wait_drained("http://x", timeout=0)

    def test_request_body_disables_thinking_and_is_greedy(self):
        body = mod.chat_body("hi", 3, stop=["4"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["stop"], ["4"])

    def test_dry_run_declares_fourteen_actions_and_cannot_claim_qualification(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True,
                             text=True, check=True)
        contract = json.loads(out.stdout)
        self.assertEqual(len(contract["actions"]) * len(contract["modes"]), 14)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])
        self.assertGreater(contract["total_blocks"] * 16, 16384)

    def test_live_run_requires_all_inputs(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
