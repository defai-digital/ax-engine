#!/usr/bin/env python3
"""Unit tests for scripts.bench_ax_only_sweep."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT_PATH = Path(__file__).with_name("bench_ax_only_sweep.py")
MODULE_SPEC = importlib.util.spec_from_file_location("bench_ax_only_sweep", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
sweep = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_ax_only_sweep"] = sweep
MODULE_SPEC.loader.exec_module(sweep)


class BenchAxOnlySweepTests(unittest.TestCase):
    def write_ax_snapshot(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}\n")
        (path / "model-manifest.json").write_text("{}\n")
        (path / "weights.safetensors").write_bytes(b"")

    def test_filter_manifest_rows_returns_all_rows_without_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        self.assertEqual(sweep.filter_manifest_rows(rows, None), rows)

    def test_default_manifest_uses_real_e4b_6bit_mlx_repo(self) -> None:
        manifest = json.loads(sweep.DEFAULT_MANIFEST.read_text())
        row = next(
            row
            for row in manifest["rows"]
            if row.get("slug") == "gemma-4-e4b-it-6bit"
        )

        self.assertEqual(
            row["mlx_repo_id"],
            "mlx-community/gemma-4-e4b-it-6bit",
        )
        self.assertEqual(row["mlx_local_dir"], ".internal/models/gemma-4-e4b-it-6bit")
        self.assertIn("AX manifest is present", row["prompt_source_note"])

    def test_filter_manifest_rows_keeps_manifest_order(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]

        selected = sweep.filter_manifest_rows(rows, ["c", "a"])

        self.assertEqual([row["slug"] for row in selected], ["a", "c"])

    def test_filter_manifest_rows_rejects_unknown_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "--rows-filter references unknown slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "missing"])

    def test_filter_manifest_rows_rejects_duplicate_manifest_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "a"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "manifest contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, None)

    def test_filter_manifest_rows_rejects_duplicate_filter_slug(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(
            sweep.AxOnlySweepError,
            "--rows-filter contains duplicate slug",
        ):
            sweep.filter_manifest_rows(rows, ["a", "a"])

    def test_filter_manifest_rows_rejects_empty_filter(self) -> None:
        rows = [{"slug": "a"}, {"slug": "b"}]

        with self.assertRaisesRegex(sweep.AxOnlySweepError, "requires at least one"):
            sweep.filter_manifest_rows(rows, [])

    def test_filter_manifest_rows_rejects_empty_manifest(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "manifest contains no rows"):
            sweep.filter_manifest_rows([], None)

    def test_filter_manifest_rows_rejects_missing_row_slug(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "non-empty slug"):
            sweep.filter_manifest_rows([{"readme_model": "Gemma"}], None)

    def test_filter_manifest_rows_rejects_non_object_row(self) -> None:
        with self.assertRaisesRegex(sweep.AxOnlySweepError, "row must be an object"):
            sweep.filter_manifest_rows(["not-a-row"], None)

    def test_main_rejects_unknown_filter_before_creating_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"slug": "a"}]}) + "\n")
            out_dir = root / "out"

            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
                "--rows-filter",
                "missing",
            ]
            with patch.object(sys, "argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 1)
            self.assertFalse(out_dir.exists())

    def test_fail_if_sweep_incomplete_reports_status_counts(self) -> None:
        stderr = io.StringIO()
        rows = [
            {"slug": "a", "status": "ok"},
            {"slug": "b", "status": "bench_failed", "exit_code": 2},
            {"slug": "c", "status": "model_dir_missing"},
        ]

        self.assertEqual(
            sweep.status_counts(rows),
            {"bench_failed": 1, "model_dir_missing": 1, "ok": 1},
        )
        self.assertEqual(
            sweep.status_counts_text(sweep.status_counts(rows)),
            "bench_failed=1, model_dir_missing=1, ok=1",
        )
        with patch.object(sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as caught:
                sweep.fail_if_sweep_incomplete(rows)

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("bench_failed=1", stderr.getvalue())
        self.assertIn("model_dir_missing=1", stderr.getvalue())

    def test_sweep_row_note_includes_failure_diagnostics(self) -> None:
        note = sweep.sweep_row_note(
            {
                "status": "bench_failed",
                "exit_code": 2,
                "log_path": "/tmp/row.log",
                "output_path": "/tmp/row.json",
            }
        )

        self.assertIn("exit_code=2", note)
        self.assertIn("log=/tmp/row.log", note)
        self.assertIn("output=/tmp/row.json", note)

    def test_run_row_terminates_child_when_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference.mkdir()
            (reference / "a.json").write_text("{}\n")
            proc = Mock()
            proc.pid = 12345
            proc.poll.return_value = None
            proc.wait.side_effect = [
                sweep.SweepInterrupted("received SIGTERM"),
                0,
            ]

            with (
                patch.object(sweep.subprocess, "Popen", return_value=proc) as popen,
                patch.object(sweep.os, "killpg") as killpg,
                self.assertRaises(sweep.SweepInterrupted),
            ):
                sweep.run_row(
                    {
                        "slug": "a",
                    },
                    output_dir=root / "out",
                    bench_script=root / "bench.py",
                    prompt_tokens="128",
                    generation_tokens=128,
                    repetitions=5,
                    cooldown=15.0,
                    model_args=["--model-dir", str(root / "model")],
                    reuse_ref_root=reference,
                )

            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            killpg.assert_called_once_with(12345, sweep.signal.SIGTERM)
            self.assertEqual(proc.send_signal.call_count, 0)
            self.assertEqual(proc.kill.call_count, 0)

    def test_run_row_forwards_load_gate_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference.mkdir()
            (reference / "a.json").write_text("{}\n")
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "a.json").write_text(json.dumps({"ok": True}) + "\n")
            proc = Mock()
            proc.wait.return_value = 0

            with patch.object(sweep.subprocess, "Popen", return_value=proc) as popen:
                result = sweep.run_row(
                    {
                        "slug": "a",
                    },
                    output_dir=output_dir,
                    bench_script=root / "bench.py",
                    prompt_tokens="128",
                    generation_tokens=128,
                    repetitions=5,
                    cooldown=15.0,
                    model_args=["--model-dir", str(root / "model")],
                    reuse_ref_root=reference,
                    max_load_average=1.5,
                    max_top_process_cpu_percent=50.0,
                    load_average_wait_timeout=600.0,
                    load_average_poll_interval=10.0,
                )

            cmd = popen.call_args.args[0]
            self.assertEqual(result["status"], "ok")
            self.assertIn("--max-load-average", cmd)
            self.assertIn("1.5", cmd)
            self.assertIn("--max-top-process-cpu-percent", cmd)
            self.assertIn("50.0", cmd)
            self.assertIn("--load-average-wait-timeout", cmd)
            self.assertIn("600.0", cmd)
            self.assertIn("--load-average-poll-interval", cmd)
            self.assertIn("10.0", cmd)

    def test_run_row_ax_direct_only_skips_reference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "a.json").write_text(json.dumps({"ok": True}) + "\n")
            proc = Mock()
            proc.wait.return_value = 0

            with patch.object(sweep.subprocess, "Popen", return_value=proc) as popen:
                result = sweep.run_row(
                    {
                        "slug": "a",
                    },
                    output_dir=output_dir,
                    bench_script=root / "bench.py",
                    prompt_tokens="128",
                    generation_tokens=128,
                    repetitions=5,
                    cooldown=15.0,
                    model_args=["--model-dir", str(root / "model")],
                    reuse_ref_root=None,
                    ax_direct_only=True,
                )

            cmd = popen.call_args.args[0]
            self.assertEqual(result["status"], "ok")
            self.assertIn("--skip-mlx-lm", cmd)
            self.assertIn("--ax-direct", cmd)
            self.assertNotIn("--reuse-reference-results-from", cmd)
            self.assertNotIn("--ax-compare-policies", cmd)

    def test_collect_performance_condition_metadata_parses_pmset(self) -> None:
        def fake_check_output(
            cmd: list[str],
            *,
            text: bool,
            stderr: int | None = None,
        ) -> str:
            del text, stderr
            if cmd == ["pmset", "-g", "batt"]:
                return (
                    "Now drawing from 'AC Power'\n"
                    " -InternalBattery-0\t80%; AC attached; not charging present: true\n"
                )
            if cmd == ["pmset", "-g", "therm"]:
                return (
                    "Note: No thermal warning level has been recorded\n"
                    "Note: Performance warning level: 1\n"
                    "Note: No CPU power status has been recorded\n"
                )
            raise AssertionError(cmd)

        with (
            patch.object(sweep.subprocess, "check_output", side_effect=fake_check_output),
            patch.object(sweep.os, "getloadavg", return_value=(1.0, 2.0, 3.0)),
        ):
            metadata = sweep.collect_performance_condition_metadata()

        self.assertEqual(
            metadata["load_average"],
            {"one_minute": 1.0, "five_minutes": 2.0, "fifteen_minutes": 3.0},
        )
        self.assertEqual(metadata["power_source"], "AC Power")
        self.assertIn("80%", metadata["battery_status"])
        self.assertFalse(metadata["thermal_warning_recorded"])
        self.assertTrue(metadata["performance_warning_recorded"])
        self.assertFalse(metadata["cpu_power_status_recorded"])

    def test_reference_model_dir_wins_over_moving_hf_cache_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "hub"
            repo_cache = cache_dir / "models--org--model"
            old_snapshot = repo_cache / "snapshots" / "oldrev"
            new_snapshot = repo_cache / "snapshots" / "newrev"
            self.write_ax_snapshot(old_snapshot)
            self.write_ax_snapshot(new_snapshot)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("newrev\n")
            reference = root / "reference.json"
            reference.write_text(
                json.dumps({"model_dir": str(old_snapshot)}) + "\n"
            )

            args, error = sweep.resolve_model_args(
                {"slug": "row", "mlx_repo_id": "org/model"},
                cache_dir,
                reference_artifact=reference,
            )

            self.assertIsNone(error)
            self.assertEqual(
                args,
                [
                    "--model-repo-id",
                    "org/model",
                    "--model-dir",
                    str(old_snapshot),
                ],
            )

    def test_reference_home_cache_path_remaps_to_configured_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "external-hub"
            remapped_snapshot = (
                cache_dir
                / "models--org--model"
                / "snapshots"
                / "same-revision"
            )
            self.write_ax_snapshot(remapped_snapshot)
            stale_reference_snapshot = (
                root
                / "stale-home"
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--org--model"
                / "snapshots"
                / "same-revision"
            )
            reference = root / "reference.json"
            reference.write_text(
                json.dumps({"model_dir": str(stale_reference_snapshot)}) + "\n"
            )

            args, error = sweep.resolve_model_args(
                {"slug": "row", "mlx_repo_id": "org/model"},
                cache_dir,
                reference_artifact=reference,
            )

            self.assertIsNone(error)
            self.assertEqual(
                args,
                [
                    "--model-repo-id",
                    "org/model",
                    "--model-dir",
                    str(remapped_snapshot),
                ],
            )

    def test_unusable_reference_model_dir_falls_back_to_latest_cache_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "hub"
            repo_cache = cache_dir / "models--org--model"
            latest_snapshot = repo_cache / "snapshots" / "latest"
            self.write_ax_snapshot(latest_snapshot)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("latest\n")
            reference = root / "reference.json"
            reference.write_text(
                json.dumps({"model_dir": str(root / "missing-snapshot")}) + "\n"
            )

            args, error = sweep.resolve_model_args(
                {"slug": "row", "mlx_repo_id": "org/model"},
                cache_dir,
                reference_artifact=reference,
            )

            self.assertIsNone(error)
            self.assertEqual(
                args,
                ["--model-repo-id", "org/model", "--hf-cache-root", str(cache_dir)],
            )

    def test_main_passes_row_reference_artifact_to_model_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "4-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            reference = root / "reference"
            reference.mkdir()
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(reference),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ) as resolve_model_args,
                patch.object(
                    sweep,
                    "run_row",
                    return_value={
                        "status": "ok",
                        "output_path": str(out_dir / "a.json"),
                    },
                ),
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
            ):
                sweep.main()

            self.assertEqual(
                resolve_model_args.call_args.args[1],
                Path.home() / ".cache" / "huggingface" / "hub",
            )
            self.assertEqual(
                resolve_model_args.call_args.kwargs["reference_artifact"],
                reference / "a.json",
            )

    def test_main_requires_reference_root_outside_direct_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"rows": [{"slug": "a"}]}) + "\n")
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
            ]

            with patch.object(sys, "argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(out_dir.exists())

    def test_main_direct_only_uses_model_snapshot_reference_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "4-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            snapshot_reference = root / "snapshot-reference"
            snapshot_reference.mkdir()
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--ax-direct-only",
                "--model-snapshot-reference-root",
                str(snapshot_reference),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ) as resolve_model_args,
                patch.object(
                    sweep,
                    "run_row",
                    return_value={
                        "status": "ok",
                        "output_path": str(out_dir / "a.json"),
                    },
                ) as run_row,
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
            ):
                sweep.main()

            self.assertEqual(
                resolve_model_args.call_args.kwargs["reference_artifact"],
                snapshot_reference / "a.json",
            )
            self.assertIsNone(run_row.call_args.kwargs["reuse_ref_root"])
            self.assertTrue(run_row.call_args.kwargs["ax_direct_only"])
            sweep_results = json.loads((out_dir / "sweep_results.json").read_text())
            self.assertTrue(sweep_results["ax_direct_only"])
            self.assertIsNone(sweep_results["reuse_reference_root"])
            self.assertEqual(
                sweep_results["model_snapshot_reference_root"],
                str(snapshot_reference),
            )

    def test_main_writes_summary_then_fails_on_failed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "4-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ),
                patch.object(
                    sweep,
                    "run_row",
                    return_value={
                        "status": "bench_failed",
                        "exit_code": 2,
                        "log_path": str(out_dir / "logs" / "a.log"),
                        "output_path": str(out_dir / "a.json"),
                    },
                ),
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 2)
            sweep_results = json.loads((out_dir / "sweep_results.json").read_text())
            self.assertFalse(sweep_results["publication_candidate"])
            self.assertEqual(sweep_results["failed_row_count"], 1)
            self.assertEqual(sweep_results["status_counts"], {"bench_failed": 1})
            self.assertEqual(sweep_results["rows"][0]["status"], "bench_failed")
            self.assertIn("benchmark_window", sweep_results)
            self.assertEqual(
                sweep_results["benchmark_window"]["performance_conditions_start"],
                {"load_average": {"one_minute": 1.0}},
            )
            markdown = (out_dir / "sweep_summary.md").read_text()
            self.assertIn("publication_candidate: false", markdown)
            self.assertIn("failed_row_count: 1", markdown)
            self.assertIn("status_counts: bench_failed=1", markdown)
            self.assertIn("exit_code=2", markdown)
            self.assertIn("logs/a.log", markdown)
            self.assertIn("a.json", markdown)

    def test_main_forwards_default_publication_load_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "4-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ),
                patch.object(
                    sweep,
                    "run_row",
                    return_value={
                        "status": "ok",
                        "output_path": str(out_dir / "a.json"),
                    },
                ) as run_row,
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
            ):
                sweep.main()

            self.assertEqual(
                run_row.call_args.kwargs["max_load_average"],
                sweep.DEFAULT_MAX_LOAD_AVERAGE,
            )
            self.assertEqual(
                run_row.call_args.kwargs["max_top_process_cpu_percent"],
                sweep.DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
            )
            sweep_results = json.loads((out_dir / "sweep_results.json").read_text())
            self.assertEqual(
                sweep_results["max_load_average"],
                sweep.DEFAULT_MAX_LOAD_AVERAGE,
            )
            self.assertEqual(
                sweep_results["max_top_process_cpu_percent"],
                sweep.DEFAULT_MAX_TOP_PROCESS_CPU_PERCENT,
            )

    def test_main_can_disable_default_publication_load_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "4-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
                "--no-load-gate",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ),
                patch.object(
                    sweep,
                    "run_row",
                    return_value={
                        "status": "ok",
                        "output_path": str(out_dir / "a.json"),
                    },
                ) as run_row,
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
            ):
                sweep.main()

            self.assertIsNone(run_row.call_args.kwargs["max_load_average"])
            self.assertIsNone(run_row.call_args.kwargs["max_top_process_cpu_percent"])
            sweep_results = json.loads((out_dir / "sweep_results.json").read_text())
            self.assertIsNone(sweep_results["max_load_average"])
            self.assertIsNone(sweep_results["max_top_process_cpu_percent"])

    def test_main_writes_non_candidate_summary_when_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "slug": "a",
                                "readme_model": "Gemma",
                                "readme_quant": "6-bit",
                            }
                        ]
                    }
                )
                + "\n"
            )
            out_dir = root / "out"
            argv = [
                "bench_ax_only_sweep.py",
                "--manifest",
                str(manifest),
                "--output-root",
                str(out_dir),
                "--reuse-reference-root",
                str(root / "reference"),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    sweep,
                    "resolve_model_args",
                    return_value=(["--model-dir", str(root / "model")], None),
                ),
                patch.object(
                    sweep,
                    "run_row",
                    side_effect=sweep.SweepInterrupted("received SIGTERM"),
                ),
                patch.object(
                    sweep,
                    "collect_performance_condition_metadata",
                    return_value={"load_average": {"one_minute": 1.0}},
                ),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    sweep.main()

            self.assertEqual(caught.exception.code, 2)
            sweep_results = json.loads((out_dir / "sweep_results.json").read_text())
            self.assertFalse(sweep_results["publication_candidate"])
            self.assertEqual(sweep_results["failed_row_count"], 1)
            self.assertEqual(sweep_results["status_counts"], {"interrupted": 1})
            self.assertEqual(sweep_results["planned_row_count"], 1)
            self.assertEqual(sweep_results["completed_row_count"], 0)
            self.assertEqual(sweep_results["rows"][0]["status"], "interrupted")
            self.assertIn("received SIGTERM", sweep_results["rows"][0]["note"])
            self.assertIn("benchmark_window", sweep_results)
            markdown = (out_dir / "sweep_summary.md").read_text()
            self.assertIn("publication_candidate: false", markdown)
            self.assertIn("completed_row_count: 0/1", markdown)
            self.assertIn("interrupted", markdown)


if __name__ == "__main__":
    unittest.main()
