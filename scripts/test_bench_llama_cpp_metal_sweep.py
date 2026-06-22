#!/usr/bin/env python3
"""Unit tests for scripts.bench_llama_cpp_metal_sweep."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "bench_llama_cpp_metal_sweep",
    _HERE / "bench_llama_cpp_metal_sweep.py",
)
sweep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sweep)  # type: ignore[union-attr]
sys.modules["bench_llama_cpp_metal_sweep"] = sweep


class BenchLlamaCppMetalSweepTests(unittest.TestCase):
    def _run_one(
        self,
        *,
        full_stack: bool,
        flash_attn: bool = False,
        decode_at_depth: bool = False,
        extra_args: str | None = None,
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = tmp_path / "model"
            model_dir.mkdir()
            gguf = tmp_path / "model.gguf"
            gguf.write_text("fake")
            bench_script = tmp_path / "bench_mlx_inference_stack.py"
            bench_script.write_text("# fake")
            llama_bench = tmp_path / "llama-bench"
            llama_bench.write_text("# fake")

            captured: list[str] = []

            def fake_run(cmd, stdout=None, stderr=None):
                captured.extend(str(part) for part in cmd)
                output_path = Path(cmd[cmd.index("--output") + 1])
                output_path.write_text(json.dumps({"results": []}) + "\n")
                return subprocess.CompletedProcess(cmd, 0)

            with patch.object(sweep.subprocess, "run", side_effect=fake_run):
                result = sweep.run_bench_for_row(
                    {"slug": "gemma-4-e2b-it-4bit"},
                    gguf,
                    output_dir=tmp_path,
                    bench_script=bench_script,
                    llama_bench=llama_bench,
                    prompt_tokens="128,512",
                    generation_tokens=128,
                    repetitions=3,
                    cooldown=0.0,
                    n_gpu_layers=99,
                    extra_args=extra_args,
                    flash_attn=flash_attn,
                    decode_at_depth=decode_at_depth,
                    model_args=["--model-dir", str(model_dir)],
                    full_stack=full_stack,
                    build_ax_engine=False,
                )

            self.assertEqual(result["status"], "ok")
            return captured

    def test_default_mode_runs_llama_cpp_only(self) -> None:
        cmd = self._run_one(full_stack=False)
        self.assertIn("--skip-mlx-lm", cmd)
        self.assertIn("--skip-ax-engine", cmd)
        self.assertIn("--no-build-ax-engine", cmd)
        self.assertNotIn("--ax-compare-policies", cmd)

    def test_full_stack_runs_mlx_lm_and_both_ax_modes(self) -> None:
        cmd = self._run_one(full_stack=True)
        self.assertIn("--ax-compare-policies", cmd)
        self.assertIn("--no-build-ax-engine", cmd)
        self.assertNotIn("--skip-mlx-lm", cmd)
        self.assertNotIn("--skip-ax-engine", cmd)

    def test_forwards_llama_cpp_depth_and_flash_attention(self) -> None:
        cmd = self._run_one(
            full_stack=False,
            flash_attn=True,
            decode_at_depth=True,
            extra_args="-ctk q8_0",
        )
        self.assertIn("--llama-cpp-decode-at-depth", cmd)
        self.assertIn("--llama-cpp-extra-args", cmd)
        forwarded = cmd[cmd.index("--llama-cpp-extra-args") + 1]
        self.assertEqual(forwarded, "-fa 1 -ctk q8_0")

    def test_filter_manifest_rows_returns_all_rows_without_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        self.assertEqual(sweep.filter_manifest_rows(rows, None), rows)

    def test_filter_manifest_rows_keeps_manifest_order(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]

        selected = sweep.filter_manifest_rows(rows, ["c", "a"])

        self.assertEqual([row["slug"] for row in selected], ["a", "c"])

    def test_filter_manifest_rows_rejects_unknown_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.LlamaCppMetalSweepError,
            "--rows-filter references unknown slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "missing"])

    def test_filter_manifest_rows_rejects_duplicate_manifest_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "a"}]

        with self.assertRaisesRegex(
            sweep.LlamaCppMetalSweepError,
            "manifest contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, None)

    def test_filter_manifest_rows_rejects_duplicate_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.LlamaCppMetalSweepError,
            "--rows-filter contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "a"])

    def test_filter_manifest_rows_rejects_empty_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(sweep.LlamaCppMetalSweepError, "requires at least one"):
            sweep.filter_manifest_rows(rows, [])

    def test_filter_manifest_rows_rejects_empty_manifest(self) -> None:
        with self.assertRaisesRegex(sweep.LlamaCppMetalSweepError, "manifest contains no rows"):
            sweep.filter_manifest_rows([], None)

    def test_filter_manifest_rows_rejects_missing_row_slug(self) -> None:
        with self.assertRaisesRegex(sweep.LlamaCppMetalSweepError, "non-empty slug"):
            sweep.filter_manifest_rows([{"readme_model": "Gemma"}], None)

    def test_filter_manifest_rows_rejects_non_object_row(self) -> None:
        with self.assertRaisesRegex(sweep.LlamaCppMetalSweepError, "row must be an object"):
            sweep.filter_manifest_rows(["not-a-row"], None)

    def test_main_rejects_unknown_filter_before_creating_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"slug": "a"}]}) + "\n")
            out_dir = root / "out"

            argv = [
                "bench_llama_cpp_metal_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--rows-filter",
                "missing",
            ]
            with patch.object(sys, "argv", argv), patch("sys.stdout", io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(out_dir.exists())

    def test_resolve_mlx_model_args_falls_back_to_ready_hf_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            snapshot = (
                cache
                / "models--mlx-community--Qwen3.6-35B-A3B-4bit"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model-manifest.json").write_text("{}")
            (snapshot / "weights.safetensors").write_text("fake")

            args, note = sweep.resolve_mlx_model_args(
                {
                    "mlx_local_dir": "missing",
                    "mlx_repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                },
                cache_dir=cache,
            )

        self.assertIsNone(note)
        self.assertEqual(
            args,
            [
                "--model-repo-id",
                "mlx-community/Qwen3.6-35B-A3B-4bit",
                "--hf-cache-root",
                str(cache),
            ],
        )

    def test_resolve_gguf_candidate_uses_cache_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            snapshot = (
                cache
                / "models--unsloth--Qwen3.6-27B-GGUF"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "Qwen3.6-27B-Q6_K.gguf").write_text("fake")

            resolved = sweep.resolve_gguf_candidate(
                [
                    {
                        "repo": "unsloth/Qwen3.6-27B-GGUF",
                        "filename_pattern": "*Q6_K*.gguf",
                    }
                ],
                cache_dir=cache,
                hf_token=None,
                cache_only=True,
            )

        self.assertIsNotNone(resolved)
        repo, filename, probe_log = resolved
        self.assertEqual(repo, "unsloth/Qwen3.6-27B-GGUF")
        self.assertEqual(filename, "Qwen3.6-27B-Q6_K.gguf")
        self.assertEqual(probe_log[0]["result"], "resolved_from_cache")

    def test_gguf_candidate_sort_prefers_standard_root_quant(self) -> None:
        candidates = [
            "MTP/gemma-4-E2B-it-Q8_0-MTP.gguf",
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        ]

        self.assertEqual(
            sorted(candidates, key=sweep.gguf_candidate_sort_key)[0],
            "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        )

    def test_resolve_gguf_candidate_allows_explicit_dynamic_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            snapshot = (
                cache
                / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
                / "snapshots"
                / "abc"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").write_text("fake")

            resolved = sweep.resolve_gguf_candidate(
                [
                    {
                        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
                        "filename_pattern": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    }
                ],
                cache_dir=cache,
                hf_token=None,
                cache_only=True,
            )

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved[1], "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

    def test_resolve_gguf_candidate_rejects_mtp_cache_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            snapshot = (
                cache
                / "models--unsloth--gemma-4-E2B-it-GGUF"
                / "snapshots"
                / "abc"
                / "MTP"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "gemma-4-E2B-it-Q8_0-MTP.gguf").write_text("fake")

            resolved = sweep.resolve_gguf_candidate(
                [
                    {
                        "repo": "unsloth/gemma-4-E2B-it-GGUF",
                        "filename_pattern": "*Q8_0*.gguf",
                    }
                ],
                cache_dir=cache,
                hf_token=None,
                cache_only=True,
            )

        self.assertIsNone(resolved)

    def test_download_gguf_cache_only_refuses_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sweep.download_gguf(
                    "unsloth/Qwen3.6-27B-GGUF",
                    "Qwen3.6-27B-Q6_K.gguf",
                    cache_dir=Path(tmp),
                    hf_token=None,
                    cache_only=True,
                )

    def test_full_stack_readme_update_refuses_partial_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            readme = tmp_path / "README.md"
            readme.write_text("stub")
            with self.assertRaisesRegex(RuntimeError, "incomplete full-stack sweep"):
                sweep.update_readme_from_sweep(
                    readme=readme,
                    sweep_path=tmp_path / "sweep_results.json",
                    sweep_doc={"rows": [{"slug": "a", "status": "bench_failed"}]},
                    full_stack=True,
                    output_root=tmp_path,
                    allow_partial=False,
                )

    def test_readme_source_marker_uses_scoped_sources_not_legacy_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            readme = tmp_path / "README.md"
            readme.write_text(
                "<!-- readme-performance-artifacts: "
                "reference=old-ref/; ax-overlay=old-ax/ -->\n"
            )

            sweep.update_readme_source_marker(
                readme,
                sweep.REPO_ROOT / "benchmarks/results/new-run",
            )

            text = readme.read_text()
            self.assertIn(
                "reference=benchmarks/results/new-run/; "
                "ax-base=benchmarks/results/new-run/",
                text,
            )
            self.assertNotRegex(text, r"readme-performance-artifacts:[^\n]*(?:^|[; ])base=")


if __name__ == "__main__":
    unittest.main()
