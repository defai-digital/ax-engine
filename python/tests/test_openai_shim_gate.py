from __future__ import annotations

import unittest
from collections.abc import Iterator

from ax_engine import openai_server

try:
    import anyio
    import starlette  # noqa: F401
except ImportError:  # pragma: no cover - optional shim dependencies
    anyio = None


@unittest.skipIf(anyio is None, "shim dependencies (anyio, starlette) are not installed")
class GatedStreamTests(unittest.TestCase):
    def test_abandoned_stream_closes_its_native_iterator_before_releasing_the_gate(
        self,
    ) -> None:
        events: list[str] = []

        async def scenario() -> None:
            gate = anyio.Semaphore(1)

            def chunks() -> Iterator[str]:
                try:
                    yield "first"
                    yield "second"
                finally:
                    # The native stream must be released while the gate is
                    # still held, or the next admitted request finds the
                    # session busy.
                    events.append(f"closed with gate value {gate.value}")

            stream = openai_server.gated_stream(gate, chunks())
            self.assertEqual(await stream.__anext__(), "first")
            # A client disconnect closes only the async wrapper.
            await stream.aclose()
            events.append(f"after close gate value {gate.value}")

        anyio.run(scenario)
        self.assertEqual(events, ["closed with gate value 0", "after close gate value 1"])


if __name__ == "__main__":
    unittest.main()
