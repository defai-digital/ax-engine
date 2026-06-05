#!/usr/bin/env python3
"""Tests for prepare_gemma4_assistant_mtp.py.

Pure-Python (no MLX / no network) so it runs in the script hygiene gate. The
pair-detection and architecture-validation logic mirrors the runtime
(crates/ax-engine-mlx/src/gemma4_assistant_mtp.rs); these tests pin that parity.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load("prepare_gemma4_assistant_mtp")


def _good_assistant(target_hidden: int = 2048, layers: int = 4, vocab: int = 262144) -> dict:
    return {
        "model_type": "gemma4_assistant",
        "hidden_size": 1024,
        "backbone_hidden_size": target_hidden,
        "num_hidden_layers": layers,
        "num_kv_shared_layers": layers,
        "vocab_size": vocab,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": 0,
        "enable_moe_block": False,
        "use_double_wide_mlp": False,
    }


def _target(hidden: int = 2048, vocab: int = 262144) -> dict:
    return {"hidden_size": hidden, "vocab_size": vocab, "num_hidden_layers": 26}


class PairDetectionTests(unittest.TestCase):
    def test_known_pairs_accepted(self) -> None:
        self.assertTrue(prep.is_known_pair("gemma-4-e2b-it-assistant", "gemma-4-e2b-it"))
        self.assertTrue(
            prep.is_known_pair("google/gemma-4-31b-it-assistant", "google/gemma-4-31b-it")
        )

    def test_unknown_target_rejected(self) -> None:
        self.assertFalse(prep.is_known_pair("gemma-4-9b-it-assistant", "gemma-4-9b-it"))

    def test_assistant_must_be_target_plus_suffix(self) -> None:
        # Right target, but assistant is a different model.
        self.assertFalse(prep.is_known_pair("gemma-4-e4b-it-assistant", "gemma-4-e2b-it"))
        # Missing -assistant suffix.
        self.assertFalse(prep.is_known_pair("gemma-4-e2b-it", "gemma-4-e2b-it"))

    def test_derive_canonical_target_strips_quant_suffix(self) -> None:
        self.assertEqual(prep._derive_canonical_target_id("mlx-community/gemma-4-e2b-it-4bit"), "gemma-4-e2b-it")
        self.assertEqual(prep._derive_canonical_target_id("/x/gemma-4-31b-it-bf16"), "gemma-4-31b-it")
        self.assertEqual(prep._derive_canonical_target_id("google/gemma-4-e4b-it"), "gemma-4-e4b-it")


class ArchValidationTests(unittest.TestCase):
    def test_compatible_assistant_has_no_problems(self) -> None:
        self.assertEqual(prep.validate_assistant_arch(_good_assistant(), _target()), [])

    def test_vocab_mismatch_flagged(self) -> None:
        a = _good_assistant(vocab=1000)
        problems = prep.validate_assistant_arch(a, _target(vocab=262144))
        self.assertTrue(any("VocabMismatch" in p for p in problems))

    def test_backbone_hidden_must_match_target(self) -> None:
        a = _good_assistant(target_hidden=4096)
        problems = prep.validate_assistant_arch(a, _target(hidden=2048))
        self.assertTrue(any("backbone_hidden_size" in p for p in problems))

    def test_kv_sharing_must_be_full(self) -> None:
        a = _good_assistant(layers=4)
        a["num_kv_shared_layers"] = 2
        problems = prep.validate_assistant_arch(a, _target())
        self.assertTrue(any("num_kv_shared_layers" in p for p in problems))

    def test_unsupported_blocks_flagged(self) -> None:
        a = _good_assistant()
        a["enable_moe_block"] = True
        a["use_double_wide_mlp"] = True
        a["hidden_size_per_layer_input"] = 256
        problems = prep.validate_assistant_arch(a, _target())
        self.assertTrue(any("enable_moe_block" in p for p in problems))
        self.assertTrue(any("use_double_wide_mlp" in p for p in problems))
        self.assertTrue(any("hidden_size_per_layer_input" in p for p in problems))

    def test_text_config_nesting_is_read(self) -> None:
        # Values nested under text_config must be honored, like the loader does.
        a = {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 2048,
            "text_config": {
                "hidden_size": 1024,
                "vocab_size": 262144,
                "num_hidden_layers": 4,
                "num_kv_shared_layers": 4,
            },
        }
        self.assertEqual(prep.validate_assistant_arch(a, _target()), [])


class ContractTests(unittest.TestCase):
    def test_contract_shape_matches_schema(self) -> None:
        contract = prep.build_contract(
            target_model_id="gemma-4-e2b-it",
            assistant_model_id="gemma-4-e2b-it-assistant",
            assistant_rel_path="assistant",
            max_depth=1,
        )
        self.assertEqual(contract["schema_version"], "ax.gemma4_assistant_mtp.v1")
        self.assertEqual(contract["backend"], "gemma4_assistant")
        self.assertEqual(contract["assistant_path"], "assistant")
        self.assertEqual(contract["pairing"], "exact")
        self.assertEqual(contract["max_depth"], 1)
        # The pair the contract names must itself pass the known-pair check.
        self.assertTrue(
            prep.is_known_pair(contract["assistant_model_id"], contract["target_model_id"])
        )


class DriverTests(unittest.TestCase):
    """Integration coverage of prepare() and its fail-closed guards."""

    def _make_pair(
        self,
        root: Path,
        *,
        target_tokenizer: bool = True,
        target_weights: bool = True,
        assistant_weights: bool = True,
        target_model_type: str = "gemma4",
        assistant_backbone: int | None = None,
    ) -> tuple[Path, Path]:
        tdir = root / "gemma-4-e2b-it-4bit"
        tdir.mkdir()
        (tdir / "config.json").write_text(
            json.dumps({"model_type": target_model_type, "hidden_size": 2048, "vocab_size": 262144})
        )
        if target_tokenizer:
            (tdir / "tokenizer.json").write_text('{"tok":"v1"}')
        if target_weights:
            (tdir / "model.safetensors").write_text("W")

        adir = root / "gemma-4-e2b-it-assistant"
        adir.mkdir()
        acfg: dict = {
            "model_type": "gemma4",
            "hidden_size": 1024,
            "vocab_size": 262144,
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 4,
        }
        if assistant_backbone is not None:
            acfg["backbone_hidden_size"] = assistant_backbone
        (adir / "config.json").write_text(json.dumps(acfg))
        (adir / "tokenizer.json").write_text('{"tok":"DIFFERENT"}')
        if assistant_weights:
            (adir / "model.safetensors").write_text("A")
        return tdir, adir

    def _run(self, root: Path, tdir: Path, adir: Path, **kw):
        # Suppress the driver's progress prints so they don't clutter the gate log.
        with contextlib.redirect_stdout(io.StringIO()):
            return prep.prepare(
                target=str(tdir),
                assistant=str(adir),
                target_model_id=None,
                assistant_model_id=None,
                output=root / "out",
                max_depth=kw.pop("max_depth", 1),
            )

    def test_happy_path_writes_valid_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root)
            out = self._run(root, tdir, adir)
            contract = json.loads((out / prep.CONTRACT_FILE).read_text())
            self.assertTrue(prep.is_known_pair(contract["assistant_model_id"], contract["target_model_id"]))
            tok_t = (out / "tokenizer.json").read_bytes()
            tok_a = (out / "assistant" / "tokenizer.json").read_bytes()
            self.assertEqual(tok_t, tok_a)

    def test_missing_target_tokenizer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root, target_tokenizer=False)
            with self.assertRaises(SystemExit):
                self._run(root, tdir, adir)

    def test_missing_assistant_weights_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root, assistant_weights=False)
            with self.assertRaises(SystemExit):
                self._run(root, tdir, adir)

    def test_non_gemma_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root, target_model_type="qwen3")
            with self.assertRaises(SystemExit):
                self._run(root, tdir, adir)

    def test_conflicting_backbone_is_not_overridden(self) -> None:
        # Assistant declares backbone 4096 but target hidden is 2048 — a genuine
        # mismatch that must be reported, not silently rewritten to 2048.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root, assistant_backbone=4096)
            with self.assertRaises(SystemExit):
                self._run(root, tdir, adir)

    def test_max_depth_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tdir, adir = self._make_pair(root)
            with self.assertRaises(SystemExit):
                self._run(root, tdir, adir, max_depth=0)


if __name__ == "__main__":
    unittest.main()
