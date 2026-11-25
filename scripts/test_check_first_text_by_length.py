#!/usr/bin/env python3
"""Unit tests for check_first_text_by_length.py. No model and no network beyond loopback."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_first_text_by_length as gate  # noqa: E402


def document(rows, api="completions"):
    return {
        "schema": gate.SCHEMA,
        "api": api,
        "rows": [
            {"sentences": sentences, "input_tokens": 13 * sentences + 23, "median_seconds": median}
            for sentences, median in rows
        ],
    }


class PromptTests(unittest.TestCase):
    def test_marker_is_the_first_line_and_length_scales_with_sentences(self):
        one = gate.build_prompt("m1", 1)
        many = gate.build_prompt("m1", 5)
        self.assertTrue(one.startswith("Test record m1\n"))
        self.assertEqual(many.count(gate.SENTENCE), 5)
        self.assertTrue(many.endswith(gate.QUESTION))
        with self.assertRaises(gate.GateError):
            gate.build_prompt("m1", 0)

    def test_completions_prompt_keeps_the_marker_before_the_chat_wrapper(self):
        path, body = gate.request_body("completions", "model", "m2", 2)
        self.assertEqual(path, "/v1/completions")
        self.assertTrue(body["prompt"].startswith("Test record m2\n" + gate.CHATML_PREFIX))
        self.assertTrue(body["prompt"].endswith(gate.CHATML_SUFFIX))
        self.assertEqual((body["max_tokens"], body["temperature"], body["stream"]), (1, 0, True))

    def test_chat_request_carries_the_same_user_text(self):
        path, body = gate.request_body("chat", "model", "m3", 2)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["messages"], [{"role": "user", "content": gate.build_prompt("m3", 2)}])
        with self.assertRaises(gate.GateError):
            gate.request_body("other", "model", "m3", 2)

    def test_lengths_must_be_distinct_positive_integers(self):
        self.assertEqual(gate.parse_lengths("1, 8,16"), [1, 8, 16])
        for bad in ("", "0", "4,4", "a"):
            with self.assertRaises(gate.GateError):
                gate.parse_lengths(bad)


class StreamTests(unittest.TestCase):
    def clock(self, ticks):
        values = iter(ticks)
        return lambda: next(values)

    def test_first_text_is_the_first_nonempty_chunk_of_either_endpoint(self):
        lines = [
            b": keep-alive\n",
            b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n',
            b'data: {"choices": [{"delta": {"content": ""}}]}\n',
            b'data: {"choices": [{"delta": {"content": "Yes"}}]}\n',
            b'data: {"choices": [], "usage": {"prompt_tokens": 127}}\n',
            b"data: [DONE]\n",
        ]
        first, tokens = gate.read_stream(lines, self.clock([10.5]), 10.0)
        self.assertEqual((first, tokens), (0.5, 127))
        first, tokens = gate.read_stream([b'data: {"choices": [{"text": "Yes"}]}\n'], self.clock([3.0]), 1.0)
        self.assertEqual((first, tokens), (2.0, None))

    def test_a_reply_without_text_or_with_a_broken_event_is_an_error(self):
        with self.assertRaises(gate.GateError):
            gate.read_stream([b'data: {"choices": [{"text": ""}]}\n', b"data: [DONE]\n"], self.clock([]), 0.0)
        with self.assertRaises(gate.GateError):
            gate.read_stream([b"data: {not json\n"], self.clock([]), 0.0)


class CompareTests(unittest.TestCase):
    def test_a_slowdown_must_exceed_both_margins(self):
        baseline = document([(8, 0.200), (113, 0.800), (340, 3.000)])
        # 8: +15% but only 30 ms over a 50 ms floor; 113: +60 ms but only 7.5%; 340: +20% and 600 ms.
        candidate = document([(8, 0.230), (113, 0.860), (340, 3.600)])
        verdicts = gate.compare(baseline, candidate, 0.10, 0.050)
        self.assertEqual([verdict["status"] for verdict in verdicts], ["ok", "ok", "regressed"])
        self.assertEqual(verdicts[2]["change"], 0.2)

    def test_faster_candidates_and_extra_lengths_pass(self):
        verdicts = gate.compare(document([(8, 0.3)]), document([(8, 0.2), (16, 9.0)]), 0.10, 0.020)
        self.assertEqual([verdict["status"] for verdict in verdicts], ["ok"])

    def test_a_length_missing_from_the_candidate_is_reported(self):
        verdicts = gate.compare(document([(8, 0.3), (16, 0.4)]), document([(8, 0.3)]), 0.10, 0.020)
        self.assertEqual([verdict["status"] for verdict in verdicts], ["ok", "missing"])
        self.assertIn("missing", gate.render(verdicts))

    def test_documents_from_different_endpoints_do_not_compare(self):
        with self.assertRaises(gate.GateError):
            gate.compare(document([(8, 0.3)]), document([(8, 0.3)], api="chat"), 0.10, 0.020)


class FakeServer(BaseHTTPRequestHandler):
    """Streams one text chunk and a usage event for every POST."""

    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        tokens = body["prompt"].count(gate.SENTENCE) * 13 + 23
        for event in ({"choices": [{"text": "Yes"}]}, {"choices": [], "usage": {"prompt_tokens": tokens}}):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        FakeServer.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_measure_writes_a_document_and_compare_gates_it(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "candidate.json"
            status = gate.main(
                [
                    "measure", "--base-url", self.base_url, "--model", "fake", "--sentences", "1,4",
                    "--repeats", "2", "--warmups", "1", "--run-id", "unit", "--output", str(output),
                ]
            )
            self.assertEqual(status, 0)
            measured = gate.load_document(output)
            self.assertEqual([row["sentences"] for row in measured["rows"]], [1, 4])
            self.assertEqual([row["input_tokens"] for row in measured["rows"]], [36, 75])
            self.assertTrue(all(len(row["first_text_seconds"]) == 2 for row in measured["rows"]))
            # One warmup, then two repeats per length, each with its own marker.
            prompts = [request["body"]["prompt"] for request in FakeServer.requests]
            self.assertEqual(len(prompts), 5)
            self.assertEqual(len({prompt.split("\n", 1)[0] for prompt in prompts}), 5)

            slow = json.loads(output.read_text())
            for row in slow["rows"]:
                row["median_seconds"] = row["median_seconds"] + 1.0
            slower = Path(scratch) / "slow.json"
            slower.write_text(json.dumps(slow))
            self.assertEqual(gate.main(["compare", "--baseline", str(output), "--candidate", str(output)]), 0)
            self.assertEqual(gate.main(["compare", "--baseline", str(output), "--candidate", str(slower)]), 1)
            self.assertEqual(gate.main(["compare", "--baseline", str(slower), "--candidate", str(output)]), 0)

    def test_bad_inputs_exit_two(self):
        with tempfile.TemporaryDirectory() as scratch:
            wrong = Path(scratch) / "wrong.json"
            wrong.write_text(json.dumps({"schema": "something.else", "rows": []}))
            self.assertEqual(gate.main(["compare", "--baseline", str(wrong), "--candidate", str(wrong)]), 2)
            self.assertEqual(
                gate.main(
                    [
                        "measure", "--base-url", "ftp://127.0.0.1", "--model", "fake",
                        "--warmups", "0", "--output", str(Path(scratch) / "out.json"),
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
