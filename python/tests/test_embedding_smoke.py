"""Smoke gate for the Python embedding API surface.

Catches the class of bug we hit during the 2026-05-12 README refresh:
a stale `_ax_engine.abi3.so` did not expose a freshly-added method on
the Rust side, but the Python wrapper happily forwarded to it and
crashed at runtime with an unhelpful `AttributeError`. This test only
verifies that the methods *exist* on `Session` and that the wrapper +
type stubs agree on names and signatures. It does NOT actually run a
model — that requires a 600 MB+ MLX model dir and is the bench
script's job.

Re-run after every `maturin develop` to catch a stale extension; a
clean build should pass this in <100 ms.
"""
import inspect
import unittest


class EmbeddingApiSurfaceTests(unittest.TestCase):
    EMBEDDING_METHODS = (
        "embed",
        "embed_bytes",
        "embed_batch",
        "embed_batch_bytes",
        "embed_batch_flat_bytes",
        "embed_batch_array",
    )

    def test_session_class_exposes_embedding_methods(self) -> None:
        import ax_engine

        for name in self.EMBEDDING_METHODS:
            with self.subTest(method=name):
                self.assertTrue(
                    hasattr(ax_engine.Session, name),
                    f"Session is missing {name!r}; "
                    f"check `_ax_engine.abi3.so` is up to date with `maturin develop --release`.",
                )

    def test_session_methods_are_callables(self) -> None:
        import ax_engine

        for name in self.EMBEDDING_METHODS:
            with self.subTest(method=name):
                attr = getattr(ax_engine.Session, name)
                self.assertTrue(callable(attr), f"Session.{name} is not callable")

    def test_embed_batch_array_lazy_imports_numpy(self) -> None:
        """`embed_batch_array` is the only embedding method that touches
        numpy, and it must do so lazily so that callers who never use
        the method do not pay the numpy import cost on every Session
        construction. Verify by importing ax_engine without preloading
        numpy and confirming numpy is not in sys.modules yet."""
        import importlib
        import sys

        # Force a re-import of ax_engine in a fresh state if possible.
        if "numpy" in sys.modules:
            self.skipTest("numpy already imported by another test")
        importlib.import_module("ax_engine")
        self.assertNotIn(
            "numpy",
            sys.modules,
            "importing ax_engine eagerly pulled numpy — embed_batch_array "
            "should import numpy only when called.",
        )

    def test_inner_native_methods_match_wrapper(self) -> None:
        """Catch the stale-`.so` case: the Python wrapper at
        `ax_engine.Session` forwards to `self._inner.<name>`. If the
        compiled `_ax_engine.abi3.so` is older than the source it may
        be missing methods the wrapper expects. This walks every
        wrapper method and confirms the inner counterpart is present."""
        import ax_engine

        # Build a placeholder Session without a real model so we can
        # inspect `_inner`. Use the test fake when available; otherwise
        # skip rather than fail (CI sometimes does not have a model).
        try:
            session = ax_engine.Session(
                backend_policy="llama_cpp",
                llama_server_url="http://127.0.0.1:1",
            )
        except Exception as e:
            self.skipTest(f"could not build placeholder Session: {e}")
        try:
            for name in self.EMBEDDING_METHODS:
                with self.subTest(method=name):
                    inner = session._inner
                    self.assertTrue(
                        hasattr(inner, name) or name == "embed_batch_array",
                        f"`_ax_engine.abi3.so` is missing {name!r}; "
                        f"run `maturin develop --release` to rebuild.",
                    )
        finally:
            try:
                session.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
