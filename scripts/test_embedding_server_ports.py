#!/usr/bin/env python3
"""Unit tests for embedding benchmark server port preflight helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench_embedding = load_script("bench_embedding_models", "bench_embedding_models.py")
verify_embedding = load_script("verify_embedding_models", "verify_embedding_models.py")


class ConnectedSocket:
    def __enter__(self) -> "ConnectedSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def connect_ex(self, _address: tuple[str, int]) -> int:
        return 0


class DisconnectedSocket:
    def __enter__(self) -> "DisconnectedSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def connect_ex(self, _address: tuple[str, int]) -> int:
        return 61


class EmbeddingServerPortTests(unittest.TestCase):
    def test_embedding_benchmark_rejects_occupied_port(self) -> None:
        with patch.object(bench_embedding.socket, "socket", return_value=ConnectedSocket()):
            with self.assertRaisesRegex(RuntimeError, "--skip-ax-server"):
                bench_embedding.ensure_port_available(8083)

    def test_embedding_benchmark_allows_free_port(self) -> None:
        with patch.object(
            bench_embedding.socket,
            "socket",
            return_value=DisconnectedSocket(),
        ):
            bench_embedding.ensure_port_available(8083)

    def test_embedding_verifier_rejects_occupied_port(self) -> None:
        with patch.object(verify_embedding.socket, "socket", return_value=ConnectedSocket()):
            with self.assertRaisesRegex(RuntimeError, "--skip-server"):
                verify_embedding.ensure_port_available(8082)

    def test_embedding_verifier_allows_free_port(self) -> None:
        with patch.object(
            verify_embedding.socket,
            "socket",
            return_value=DisconnectedSocket(),
        ):
            verify_embedding.ensure_port_available(8082)


if __name__ == "__main__":
    unittest.main()
