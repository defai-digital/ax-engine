#!/usr/bin/env python3
"""Unit tests for building MLX prefix warmup artifacts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("build_mlx_prefix_warmup_artifact.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_mlx_prefix_warmup_artifact", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


class PrefixWarmupBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: self.cleanup_root(self.root))
        self.result_dir = self.root / "result"
        self.manifest_root = self.root / "manifest"
        self.prompt_dir = self.manifest_root / "prompts"
        self.result_dir.mkdir()
        self.prompt_dir.mkdir(parents=True)
        (self.prompt_dir / "prefix_a_variant.txt").write_text("shared prefix prompt\n")
        self.write_result_fixture()

    @staticmethod
    def cleanup_root(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        root.rmdir()

    def write_result_fixture(self) -> None:
        write_json(
            self.result_dir / "manifest.json",
            {
                "schema_version": "ax.engine_bench.manifest.v1",
                "id": "shared_prefix_long_churn",
                "class": "replay",
                "model": {"family": "qwen3_dense", "quant": "q4_k_m"},
                "runtime": {
                    "selected_backend": "mlx",
                    "deterministic": True,
                    "flags": {"prefix_cache": True},
                },
                "events": [
                    {
                        "type": "submit",
                        "request_id": "req-2",
                        "prompt_ref": "prompts/prefix_a_variant.txt",
                        "output_tokens_target": 16,
                    }
                ],
                "checks": {
                    "expect_deterministic": True,
                    "require_prefix_reuse": True,
                },
            },
        )
        write_json(
            self.result_dir / "metrics.json",
            {
                "schema_version": "ax.engine_bench.metrics.v1",
                "correctness": {"passed": True, "reason": "ok"},
                "determinism": {"passed": True, "reason": "ok"},
            },
        )
        write_json(
            self.result_dir / "routes.json",
            {
                "schema_version": "ax.engine_bench.routes.v1",
                "runtime": {"selected_backend": "mlx"},
                "route": {
                    "ax_mlx_prefix_cache_hits": 0,
                    "ax_mlx_prefix_cache_misses": 1,
                    "ax_mlx_prefix_cache_warmup_tokens": 256,
                    "ax_mlx_prefix_cache_reused_tokens": 0,
                    "ax_mlx_prefix_cache_blocked": 0,
                },
            },
        )
        write_json(
            self.result_dir / "trace.json",
            {
                "schema_version": "ax.engine_bench.trace.v1",
                "steps": [
                    {
                        "items": [
                            {
                                "request_id": 2,
                                "prefix_tokens_reused": 256,
                                "prefix_blocks_reused": 4,
                            }
                        ]
                    }
                ],
                "observation": {
                    "requests": [
                        {
                            "request_id": 2,
                            "external_id": "req-2",
                            "generated_tokens": [1, 2, 3],
                        }
                    ]
                },
            },
        )

    def test_builds_valid_prefix_warmup_artifact(self) -> None:
        artifact = builder.build_prefix_warmup_artifact(
            result_dir=self.result_dir,
            manifest_root=self.manifest_root,
        )

        self.assertEqual(artifact["schema_version"], "ax.mlx_prefix_warmup.v1")
        observation = artifact["observations"][0]
        self.assertEqual(observation["request_id"], "req-2")
        self.assertEqual(observation["prompt_digest_kind"], "prompt_ref_bytes")
        self.assertEqual(
            observation["physical_prefix_snapshot"]["physical_snapshot_coverage"],
            "miss_warmup_only",
        )

    def test_missing_logical_prefix_reuse_fails(self) -> None:
        trace_path = self.result_dir / "trace.json"
        trace = json.loads(trace_path.read_text())
        trace["steps"][0]["items"][0]["prefix_tokens_reused"] = 0
        write_json(trace_path, trace)

        with self.assertRaisesRegex(builder.PrefixWarmupBuildError, "logical prefix reuse"):
            builder.build_prefix_warmup_artifact(
                result_dir=self.result_dir,
                manifest_root=self.manifest_root,
            )

    def test_failed_correctness_fails(self) -> None:
        metrics_path = self.result_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["correctness"]["passed"] = False
        write_json(metrics_path, metrics)

        with self.assertRaisesRegex(builder.PrefixWarmupBuildError, "correctness"):
            builder.build_prefix_warmup_artifact(
                result_dir=self.result_dir,
                manifest_root=self.manifest_root,
            )

    def test_non_replay_manifest_fails(self) -> None:
        manifest_path = self.result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["class"] = "scenario"
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(builder.PrefixWarmupBuildError, "class"):
            builder.build_prefix_warmup_artifact(
                result_dir=self.result_dir,
                manifest_root=self.manifest_root,
            )

    def test_prefix_cache_disabled_manifest_fails(self) -> None:
        manifest_path = self.result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime"]["flags"]["prefix_cache"] = False
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(builder.PrefixWarmupBuildError, "prefix_cache"):
            builder.build_prefix_warmup_artifact(
                result_dir=self.result_dir,
                manifest_root=self.manifest_root,
            )

    def test_manifest_must_require_prefix_reuse(self) -> None:
        manifest_path = self.result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checks"]["require_prefix_reuse"] = False
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(builder.PrefixWarmupBuildError, "require_prefix_reuse"):
            builder.build_prefix_warmup_artifact(
                result_dir=self.result_dir,
                manifest_root=self.manifest_root,
            )

    def test_cli_writes_checked_artifact(self) -> None:
        output = self.root / "prefix-warmup.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--result-dir",
                str(self.result_dir),
                "--manifest-root",
                str(self.manifest_root),
                "--output",
                str(output),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertIn("MLX prefix warmup artifact written", completed.stdout)
        artifact = json.loads(output.read_text())
        self.assertEqual(artifact["schema_version"], "ax.mlx_prefix_warmup.v1")

    def test_cli_does_not_leave_output_when_validation_fails(self) -> None:
        routes_path = self.result_dir / "routes.json"
        routes = json.loads(routes_path.read_text())
        routes["route"]["ax_mlx_prefix_cache_hits"] = 1
        write_json(routes_path, routes)
        output = self.root / "prefix-warmup.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--result-dir",
                str(self.result_dir),
                "--manifest-root",
                str(self.manifest_root),
                "--output",
                str(output),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".prefix-warmup.json.*.tmp")), [])

    def test_cli_preserves_existing_output_when_validation_fails(self) -> None:
        routes_path = self.result_dir / "routes.json"
        routes = json.loads(routes_path.read_text())
        routes["route"]["ax_mlx_prefix_cache_hits"] = 1
        write_json(routes_path, routes)
        output = self.root / "prefix-warmup.json"
        output.write_text('{"schema_version":"existing"}\n')

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--result-dir",
                str(self.result_dir),
                "--manifest-root",
                str(self.manifest_root),
                "--output",
                str(output),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(output.read_text(), '{"schema_version":"existing"}\n')
        self.assertEqual(list(self.root.glob(".prefix-warmup.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
