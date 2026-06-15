#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_direct_model_compat_smoke.py")
SPEC = importlib.util.spec_from_file_location("check_direct_model_compat_smoke", SCRIPT)
assert SPEC is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "qwen_artifacts": None,
        "qwen36_artifacts": None,
        "gemma4_artifacts": None,
        "qwen_model_id": "ax-engine/qwen3-coder-next",
        "qwen36_model_id": "Qwen3.6-35B-A3B-4bit",
        "gemma4_model_id": "gemma4-e2b-it",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DirectModelCompatSmokeTests(unittest.TestCase):
    def test_resolve_targets_skips_without_artifacts(self) -> None:
        self.assertEqual(smoke.resolve_smoke_targets(args(), env={}), [])

    def test_resolve_targets_uses_dedicated_qwen_env(self) -> None:
        targets = smoke.resolve_smoke_targets(
            args(),
            env={smoke.QWEN_ARTIFACTS_ENV: "/models/qwen"},
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].kind, "qwen3-coder-next")
        self.assertEqual(targets[0].model_id, "ax-engine/qwen3-coder-next")
        self.assertEqual(targets[0].artifacts_dir, Path("/models/qwen"))

    def test_resolve_targets_uses_legacy_mlx_env_as_qwen(self) -> None:
        targets = smoke.resolve_smoke_targets(
            args(),
            env={smoke.LEGACY_MLX_ARTIFACTS_ENV: "/models/default"},
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].kind, "qwen3-coder-next")
        self.assertEqual(targets[0].artifacts_dir, Path("/models/default"))

    def test_resolve_targets_can_include_qwen_and_gemma4(self) -> None:
        targets = smoke.resolve_smoke_targets(
            args(
                qwen_artifacts=Path("/models/qwen"),
                qwen36_artifacts=Path("/models/qwen36"),
                gemma4_artifacts=Path("/models/gemma4"),
            ),
            env={},
        )

        self.assertEqual(
            [target.kind for target in targets],
            ["qwen3-coder-next", "qwen3.6-35b-a3b", "gemma4"],
        )
        self.assertEqual(
            [target.model_id for target in targets],
            ["ax-engine/qwen3-coder-next", "Qwen3.6-35B-A3B-4bit", "gemma4-e2b-it"],
        )

    def test_resolve_targets_uses_qwen36_env(self) -> None:
        targets = smoke.resolve_smoke_targets(
            args(),
            env={smoke.QWEN36_ARTIFACTS_ENV: "/models/qwen36"},
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].kind, "qwen3.6-35b-a3b")
        self.assertEqual(targets[0].model_id, "Qwen3.6-35B-A3B-4bit")
        self.assertEqual(targets[0].artifacts_dir, Path("/models/qwen36"))

    def test_openai_tool_request_exercises_systemless_tool_prompt(self) -> None:
        request = smoke.build_openai_tool_request("ax-engine/qwen3-coder-next")

        self.assertEqual(request["model"], "ax-engine/qwen3-coder-next")
        self.assertEqual(request["stream"], False)
        self.assertNotIn("system", {message["role"] for message in request["messages"]})
        self.assertEqual(request["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(request["tool_choice"], "auto")

    def test_ollama_tool_request_uses_ollama_non_streaming_envelope(self) -> None:
        request = smoke.build_ollama_chat_request("gemma4-e2b-it")

        self.assertEqual(request["model"], "gemma4-e2b-it")
        self.assertEqual(request["stream"], False)
        self.assertIn("options", request)
        self.assertEqual(request["tools"][0]["function"]["parameters"]["type"], "object")

    def test_model_metadata_matches_target_when_target_is_not_first(self) -> None:
        card = smoke.assert_model_metadata(
            {
                "data": [
                    {
                        "id": "other-model",
                        "capabilities": {"toolcall": False},
                        "ax_engine": {"openai_tool_calling_supported": False},
                    },
                    {
                        "id": "gemma4-e2b-it",
                        "capabilities": {"toolcall": True},
                        "ax_engine": {"openai_tool_calling_supported": True},
                    },
                ]
            },
            "gemma4-e2b-it",
        )

        self.assertEqual(card["id"], "gemma4-e2b-it")

    def test_model_metadata_fails_when_target_missing(self) -> None:
        with self.assertRaisesRegex(smoke.SmokeFailure, "did not include expected id"):
            smoke.assert_model_metadata(
                {
                    "data": [
                        {
                            "id": "other-model",
                            "capabilities": {"toolcall": True},
                            "ax_engine": {"openai_tool_calling_supported": True},
                        }
                    ]
                },
                "gemma4-e2b-it",
            )

    def test_raw_tool_marker_detection_fails_closed(self) -> None:
        with self.assertRaises(smoke.SmokeFailure):
            smoke.assert_no_raw_tool_markers("<tool_call>{}</tool_call>", "content")


if __name__ == "__main__":
    unittest.main()
