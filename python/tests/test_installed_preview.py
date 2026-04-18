from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
from typing import Any, Iterator


def _allocate_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _compatibility_upstream() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    port = _allocate_port()

    class LocalThreadingHTTPServer(ThreadingHTTPServer):
        def server_bind(self) -> None:
            # Avoid getfqdn() during bind, which can hang in packaging smoke environments.
            TCPServer.server_bind(self)
            host, assigned_port = self.server_address[:2]
            self.server_name = host
            self.server_port = assigned_port

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/completion":
                self.send_error(404)
                return

            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append(payload)

            if payload.get("stream"):
                body = (
                    'data: {"content":"compat","tokens":[41],"stop":false}\n\n'
                    'data: {"content":" stream","tokens":[42],"stop":true,"stop_type":"limit"}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                content = "compat tokens"
                tokens = [31, 32]
            else:
                content = f"compat::{prompt}"
                tokens = [21, 22]

            body = json.dumps(
                {
                    "content": content,
                    "tokens": tokens,
                    "stop": True,
                    "stop_type": "limit",
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = LocalThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield (f"http://127.0.0.1:{port}", requests)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class InstalledPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("AX_ENGINE_RUN_INSTALLED_TESTS") != "1":
            raise unittest.SkipTest(
                "installed preview tests run only during packaging smoke checks"
            )
        try:
            import ax_engine as module
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                "installed ax_engine preview package is not available"
            ) from exc

        cls.ax_engine = module

    def test_installed_package_reports_runtime_and_generate_result(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_dense") as session:
            runtime = session.runtime()
            result = session.generate([1, 2, 3], max_output_tokens=2)

        self.assertEqual(runtime.selected_backend, "ax_native")
        self.assertEqual(runtime.support_tier, "native_preview")
        self.assertEqual(runtime.resolution_policy, "strict_native")
        self.assertTrue(runtime.host.os)
        self.assertTrue(runtime.host.arch)
        self.assertIsInstance(runtime.host.supported_native_runtime, bool)
        self.assertIsInstance(runtime.metal_toolchain.fully_available, bool)
        self.assertEqual(result.output_tokens, [4, 5])
        self.assertEqual(len(result.output_token_logprobs), len(result.output_tokens))
        self.assertEqual(result.finish_reason, "max_output_tokens")
        self.assertEqual(result.runtime.support_tier, "native_preview")

    def test_installed_package_stream_blocks_reuse_until_iterator_finishes(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_dense") as session:
            stream = session.stream_generate([1, 2, 3], max_output_tokens=2)
            first_event = next(stream)

            self.assertEqual(first_event.event, "request")
            with self.assertRaisesRegex(RuntimeError, "active stream"):
                session.step()

            remaining_events = list(stream)
            self.assertEqual(
                [event.event for event in [first_event, *remaining_events]],
                ["request", "step", "step", "step", "response"],
            )

            result = session.generate([8, 9], max_output_tokens=1)

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.output_tokens, [10])

    def test_installed_package_supports_compatibility_server_generate_and_stream(
        self,
    ) -> None:
        with _compatibility_upstream() as (server_url, requests):
            with self.ax_engine.Session(
                model_id="qwen3_dense",
                support_tier="compatibility",
                compat_server_url=server_url,
            ) as session:
                runtime = session.runtime()
                text_result = session.generate(
                    input_text="hello compatibility",
                    max_output_tokens=2,
                )
                token_result = session.generate([1, 2, 3], max_output_tokens=2)
                events = list(session.stream_generate([1, 2, 3], max_output_tokens=2))

        self.assertEqual(runtime.selected_backend, "llama_cpp")
        self.assertEqual(runtime.support_tier, "compatibility")
        self.assertEqual(runtime.resolution_policy, "allow_compat")
        self.assertTrue(runtime.host.os)
        self.assertIsInstance(runtime.metal_toolchain.fully_available, bool)

        self.assertEqual(text_result.prompt_text, "hello compatibility")
        self.assertEqual(text_result.output_text, "compat::hello compatibility")
        self.assertEqual(
            text_result.route.execution_plan,
            "compatibility.llama_cpp.server_completion",
        )

        self.assertEqual(token_result.prompt_tokens, [1, 2, 3])
        self.assertEqual(token_result.output_tokens, [31, 32])
        self.assertEqual(token_result.output_text, "compat tokens")
        self.assertEqual(token_result.runtime.support_tier, "compatibility")

        self.assertEqual(
            [event.event for event in events],
            ["request", "step", "step", "response"],
        )
        self.assertEqual(events[0].runtime.support_tier, "compatibility")
        self.assertEqual(events[1].request.state, "running")
        self.assertEqual(events[1].delta_tokens, [41])
        self.assertEqual(events[1].delta_token_logprobs, [None])
        self.assertEqual(events[2].request.state, "finished")
        self.assertEqual(events[2].request.finish_reason, "max_output_tokens")
        self.assertEqual(events[2].request.terminal_stop_reason, "max_output_tokens")
        self.assertEqual(events[2].delta_tokens, [42])
        self.assertEqual(events[2].delta_token_logprobs, [None])
        self.assertEqual(events[3].response.output_tokens, [41, 42])
        self.assertEqual(events[3].response.output_token_logprobs, [None, None])
        self.assertEqual(events[3].response.output_text, "compat stream")
        self.assertEqual(
            events[3].response.route.execution_plan,
            "compatibility.llama_cpp.server_completion_stream",
        )

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]["prompt"], "hello compatibility")
        self.assertIs(requests[0]["stream"], False)
        self.assertEqual(requests[1]["prompt"], [1, 2, 3])
        self.assertIs(requests[1]["stream"], False)
        self.assertEqual(requests[2]["prompt"], [1, 2, 3])
        self.assertIs(requests[2]["stream"], True)
        for payload in requests:
            self.assertIs(payload["return_tokens"], True)

    def test_installed_package_supports_compatibility_server_stepwise_lifecycle(
        self,
    ) -> None:
        with _compatibility_upstream() as (server_url, requests):
            with self.ax_engine.Session(
                model_id="qwen3_dense",
                support_tier="compatibility",
                compat_server_url=server_url,
            ) as session:
                request_id = session.submit([1, 2, 3], max_output_tokens=2)
                initial = session.snapshot(request_id)

                self.assertIsNotNone(initial)
                self.assertEqual(initial.state, "waiting")

                first_step = session.step()
                running = session.snapshot(request_id)

                self.assertEqual(first_step.scheduled_requests, 1)
                self.assertEqual(first_step.scheduled_tokens, 1)
                self.assertEqual(first_step.ttft_events, 1)
                self.assertEqual(
                    first_step.route.execution_plan,
                    "compatibility.llama_cpp.server_completion_stream",
                )
                self.assertIsNotNone(running)
                self.assertEqual(running.state, "running")
                self.assertEqual(running.output_tokens, [41])
                self.assertEqual(running.output_token_logprobs, [None])

                second_step = session.step()
                terminal = session.snapshot(request_id)

                self.assertEqual(second_step.scheduled_requests, 1)
                self.assertEqual(second_step.scheduled_tokens, 1)
                self.assertEqual(
                    second_step.route.execution_plan,
                    "compatibility.llama_cpp.server_completion_stream",
                )
                self.assertIsNotNone(terminal)
                self.assertEqual(terminal.state, "finished")
                self.assertEqual(terminal.output_tokens, [41, 42])
                self.assertEqual(terminal.output_token_logprobs, [None, None])
                self.assertEqual(terminal.finish_reason, "max_output_tokens")
                self.assertEqual(terminal.terminal_stop_reason, "max_output_tokens")
                self.assertEqual(
                    terminal.route.execution_plan,
                    "compatibility.llama_cpp.server_completion_stream",
                )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["prompt"], [1, 2, 3])
        self.assertIs(requests[0]["stream"], True)
        self.assertIs(requests[0]["return_tokens"], True)

    def test_installed_package_supports_multiple_compatibility_stepwise_requests(
        self,
    ) -> None:
        with _compatibility_upstream() as (server_url, requests):
            with self.ax_engine.Session(
                model_id="qwen3_dense",
                support_tier="compatibility",
                compat_server_url=server_url,
            ) as session:
                first_request_id = session.submit([1, 2, 3], max_output_tokens=2)
                second_request_id = session.submit([7, 8, 9], max_output_tokens=2)

                first_step = session.step()
                first_running = session.snapshot(first_request_id)
                second_running = session.snapshot(second_request_id)

                self.assertEqual(first_step.scheduled_requests, 2)
                self.assertEqual(first_step.scheduled_tokens, 2)
                self.assertEqual(first_step.ttft_events, 2)
                self.assertIsNotNone(first_running)
                self.assertIsNotNone(second_running)
                self.assertEqual(first_running.state, "running")
                self.assertEqual(second_running.state, "running")
                self.assertEqual(first_running.output_tokens, [41])
                self.assertEqual(second_running.output_tokens, [41])

                second_step = session.step()
                first_terminal = session.snapshot(first_request_id)
                second_terminal = session.snapshot(second_request_id)

                self.assertEqual(second_step.scheduled_requests, 2)
                self.assertEqual(second_step.scheduled_tokens, 2)
                self.assertIsNotNone(first_terminal)
                self.assertIsNotNone(second_terminal)
                self.assertEqual(first_terminal.state, "finished")
                self.assertEqual(second_terminal.state, "finished")
                self.assertEqual(first_terminal.output_tokens, [41, 42])
                self.assertEqual(second_terminal.output_tokens, [41, 42])

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["prompt"], [1, 2, 3])
        self.assertEqual(requests[1]["prompt"], [7, 8, 9])
        for payload in requests:
            self.assertIs(payload["stream"], True)
            self.assertIs(payload["return_tokens"], True)


if __name__ == "__main__":
    unittest.main()
