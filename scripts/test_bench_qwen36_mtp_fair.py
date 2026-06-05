#!/usr/bin/env python3
"""Tests for the fair Qwen3.6 MTP benchmark summary builder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("bench_qwen36_mtp_fair.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_qwen36_mtp_fair", SCRIPT_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
fair = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["bench_qwen36_mtp_fair"] = fair
MODULE_SPEC.loader.exec_module(fair)

PREFILL_SCRIPT_PATH = Path(__file__).with_name("bench_mtp_prefill_ttft_report.py")
PREFILL_MODULE_SPEC = importlib.util.spec_from_file_location(
    "bench_mtp_prefill_ttft_report", PREFILL_SCRIPT_PATH
)
assert PREFILL_MODULE_SPEC and PREFILL_MODULE_SPEC.loader
prefill = importlib.util.module_from_spec(PREFILL_MODULE_SPEC)
sys.modules["bench_mtp_prefill_ttft_report"] = prefill
PREFILL_MODULE_SPEC.loader.exec_module(prefill)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


class Qwen36MtpFairTests(unittest.TestCase):
    def test_resolve_engines_supports_tuned_mode(self) -> None:
        self.assertEqual(
            fair.resolve_engines(["mtplx", "ax"], ["tuned"]),
            ["mtplx_tuned", "ax_engine_tuned"],
        )

    def test_depth_policy_native_uses_profile_depth(self) -> None:
        profile = fair.QWEN36_PROFILES["27b-4bit"]
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "native", None), 3
        )
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "fair-shared", None),
            3,
        )
        self.assertEqual(
            fair.effective_depth(profile, ["mtplx", "ax_engine"], "native", 2), 2
        )

    def test_validate_sidecar_for_profile_accepts_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_cache = Path(tmp) / "hf"
            make_sidecar_manifest(hf_cache)
            summary = fair.validate_sidecar_for_profile(
                fair.QWEN36_PROFILES["27b-4bit"], hf_cache
            )

        self.assertEqual(summary["model_key"], "27b")
        self.assertEqual(summary["norm_policy"], "shift_mtp_norm_weights_by_1")

    def test_validate_sidecar_for_profile_rejects_old_norm_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_cache = Path(tmp) / "hf"
            make_sidecar_manifest(
                hf_cache,
                norm_policy="scale_selected_mtp_norm_weights_by_2",
            )

            with self.assertRaisesRegex(ValueError, "norm_policy"):
                fair.validate_sidecar_for_profile(
                    fair.QWEN36_PROFILES["27b-4bit"], hf_cache
                )

    def test_build_summary_includes_two_engines_and_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", fake_ax_artifact()
                ),
                ("27b-4bit", "flappy", "mtplx"): write_json(
                    root / "mtplx.json", fake_mtplx_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["mtplx", "ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=False,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertEqual(summary["schema"], "ax.qwen36_mtp_fair.v1")
        self.assertTrue(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(summary["contract"]["benchmark_contract"], "fixed-depth")
        self.assertEqual(
            summary["contract"]["ax_engine_modes"], {"ax_engine": "pure_mtp"}
        )
        row = summary["rows"][0]
        self.assertEqual(row["depth"], 3)
        self.assertAlmostEqual(row["engines"]["ax_engine"]["decode_tok_s"], 10.0)
        self.assertAlmostEqual(row["engines"]["mtplx"]["decode_tok_s"], 8.0)
        self.assertAlmostEqual(row["ratios"]["ax_engine_vs_mtplx"], 1.25)
        self.assertEqual(row["provenance"]["kind"], "ax_mtp_sidecar_manifest")

    def test_build_summary_records_pure_mtp_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", fake_ax_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=True,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertTrue(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"], {"ax_engine": "pure_mtp"}
        )

    def test_build_summary_rejects_pure_mtp_with_ngram_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifact = fake_ax_artifact()
            artifact["results"][0]["ngram_acceleration_telemetry"][
                "ax_mtp_ngram_hit_steps"
            ] = 3
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine"): write_json(
                    root / "ax.json", artifact
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=True,
            )

            with self.assertRaisesRegex(RuntimeError, "pure-MTP"):
                fair.build_summary(args, artifacts)

    def test_build_summary_records_stacked_ax_engine_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine_ngram"): write_json(
                    root / "ax-ngram.json", fake_ax_artifact()
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["ax_engine_ngram"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                pure_mtp=False,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertFalse(summary["contract"]["ax_pure_mtp"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"],
            {"ax_engine_ngram": "mtp_ngram_stacked"},
        )

    def test_build_summary_records_tuned_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            make_sidecar_manifest(hf_cache)
            ax_artifact = fake_ax_artifact()
            ax_artifact["tuning"] = {
                "schema": "ax.qwen36_mtp_fair.tune.ax.v1",
                "selected": {"key": "ngram", "policy": "ngram", "depth": 0},
            }
            mtplx_artifact = fake_mtplx_artifact()
            mtplx_artifact["tuning"] = {
                "schema": "ax.qwen36_mtp_fair.tune.mtplx.v1",
                "selected_depth": 2,
            }
            artifacts = {
                ("27b-4bit", "flappy", "ax_engine_tuned"): write_json(
                    root / "ax-tuned.json", ax_artifact
                ),
                ("27b-4bit", "flappy", "mtplx_tuned"): write_json(
                    root / "mtplx-tuned.json", mtplx_artifact
                ),
            }
            args = Namespace(
                models=["27b-4bit"],
                engines=["mtplx_tuned", "ax_engine_tuned"],
                suites=["flappy"],
                depth_policy="native",
                depth=None,
                mode="sampled",
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=128,
                repetitions=1,
                warmup_repetitions=1,
                cooldown=0.0,
                hf_cache=hf_cache,
                retune=True,
                ax_tune_policies=["direct", "ngram", "mtp", "mtp-ngram"],
                tune_max_tokens=192,
                tune_repetitions=1,
                tune_warmup_repetitions=0,
                tune_cooldown=0.0,
                tune_limit=1,
            )

            summary = fair.build_summary(args, artifacts)

        self.assertEqual(summary["contract"]["benchmark_contract"], "tuned-best-of")
        self.assertTrue(summary["contract"]["tuning"]["enabled"])
        self.assertEqual(
            summary["contract"]["ax_engine_modes"],
            {"ax_engine_tuned": "tuned_best_of"},
        )
        row = summary["rows"][0]
        self.assertEqual(
            row["engines"]["ax_engine_tuned"]["tuning"]["selected"]["policy"],
            "ngram",
        )
        self.assertAlmostEqual(
            row["ratios"]["ax_engine_tuned_vs_mtplx_tuned"], 1.25
        )

    def test_run_ax_suite_maps_policy_flags(self) -> None:
        config = fair.diff.RunConfig(
            mode="sampled",
            depth=3,
            max_tokens=64,
            repetitions=1,
            warmup_repetitions=1,
            cooldown_s=0.0,
            sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            enable_thinking=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(fair, "run_subprocess") as run_subprocess:
                fair.run_ax_suite(
                    python=Path("python"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=root / "direct.json",
                    model_dir=root / "model",
                    config=config,
                    no_build=True,
                    ax_policy="direct",
                )
                fair.run_ax_suite(
                    python=Path("python"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=root / "ngram.json",
                    model_dir=root / "model",
                    config=config,
                    no_build=True,
                    ax_policy="ngram",
                )
                fair.run_ax_suite(
                    python=Path("python"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=root / "mtp.json",
                    model_dir=root / "model",
                    config=config,
                    no_build=True,
                    ax_policy="mtp",
                )

        direct_cmd = run_subprocess.call_args_list[0].args[0]
        ngram_cmd = run_subprocess.call_args_list[1].args[0]
        mtp_cmd = run_subprocess.call_args_list[2].args[0]
        self.assertIn("--ax-direct", direct_cmd)
        self.assertNotIn("--ax-ngram-accel", direct_cmd)
        self.assertIn("--ax-ngram-accel", ngram_cmd)
        self.assertEqual(ngram_cmd[ngram_cmd.index("--ax-mtp-max-depth") + 1], "0")
        self.assertIn("--ax-mtp-disable-ngram-stacking", mtp_cmd)
        self.assertEqual(mtp_cmd[mtp_cmd.index("--ax-mtp-max-depth") + 1], "3")

    def test_mtplx_tune_selects_best_depth_or_fallback(self) -> None:
        self.assertEqual(
            fair.mtplx_best_depth_from_tune({"best": {"depth": 2}}, 3),
            2,
        )
        self.assertEqual(fair.mtplx_best_depth_from_tune({"best": None}, 3), 3)

    def test_selected_ax_tune_candidate_chooses_fastest_valid_row(self) -> None:
        selected = fair.selected_ax_tune_candidate(
            [
                {"key": "direct", "status": "ok", "decode_tok_s": 10.0},
                {"key": "mtp_d1", "status": "ok", "decode_tok_s": 12.0},
                {"key": "mtp_d2", "status": "error", "decode_tok_s": 99.0},
            ]
        )

        self.assertEqual(selected["key"], "mtp_d1")

    def test_run_mtplx_tune_uses_official_cli_and_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                tune_max_tokens=192,
                tune_limit=1,
                tune_seed=0,
                retune=True,
            )

            def fake_run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
                output = Path(cmd[cmd.index("--output") + 1])
                write_json(output, {"best": {"depth": 2}, "saved": True})
                self.assertIn("MTPLX_TUNE_STATE", env or {})

            with patch.object(fair, "run_subprocess", side_effect=fake_run) as run:
                depth, tuning = fair.run_mtplx_tune(
                    python=Path("python"),
                    output_path=root / "mtplx_tuned.json",
                    model_dir=root / "model",
                    profile=fair.QWEN36_PROFILES["27b-4bit"],
                    args=args,
                )

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:4], ["python", "-m", "mtplx.cli", "tune"])
        self.assertIn("--retune", cmd)
        self.assertEqual(cmd[cmd.index("--depths") + 1], "1,2,3")
        self.assertEqual(depth, 2)
        self.assertEqual(tuning["selected_depth"], 2)

    def test_run_rapid_mlx_suite_forwards_mtp_tuning_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "lightning.json"
            config = fair.diff.RunConfig(
                mode="sampled",
                depth=3,
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=1,
                cooldown_s=0.0,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                enable_thinking=False,
            )
            with patch.object(fair, "run_subprocess") as run_subprocess:
                fair.run_rapid_mlx_suite(
                    python=Path("python"),
                    lightning_source=Path("lightning"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=output,
                    model_dir=root / "model",
                    config=config,
                    mtp_optimistic=False,
                    mtp_draft_temperature=0.5,
                )

        cmd = run_subprocess.call_args.args[0]
        self.assertNotIn("--mtp-optimistic", cmd)
        self.assertEqual(cmd[cmd.index("--mtp-draft-temperature") + 1], "0.5")
        self.assertIn("--disable-thinking", cmd)

    def test_run_rapid_mlx_suite_defaults_to_source_serve_non_optimistic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fair.diff.RunConfig(
                mode="sampled",
                depth=3,
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=1,
                cooldown_s=0.0,
                sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                enable_thinking=False,
            )
            with patch.object(fair, "run_subprocess") as run_subprocess:
                fair.run_rapid_mlx_suite(
                    python=Path("python"),
                    lightning_source=Path("lightning"),
                    suite="flappy",
                    suite_file=root / "suite.jsonl",
                    output_path=root / "lightning.json",
                    model_dir=root / "model",
                    config=config,
                )

        cmd = run_subprocess.call_args.args[0]
        self.assertNotIn("--mtp-optimistic", cmd)
        self.assertIn("--disable-thinking", cmd)

    def test_lightning_optimized_alias_source_passes_alias_to_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                hf_cache=root / "hf",
                suites_dir=root / "suites",
                output_dir=root / "out",
                mode="sampled",
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=0,
                cooldown=0.0,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                enable_thinking=False,
                lightning_enable_thinking=True,
                skip_existing=False,
                rapid_python=Path("python"),
                lightning_source=Path("lightning"),
                base_port=18765,
                lightning_mtp_optimistic=False,
                lightning_mtp_draft_temperature=0.5,
                lightning_ngram_auto_disable_mtp_threshold=0.85,
                lightning_ngram_auto_disable_min_ngram=0.50,
                inter_case_cooldown=0.0,
                lightning_model_source="optimized-alias",
            )
            with patch.object(fair, "run_rapid_mlx_suite", return_value=root / "out.json") as run:
                fair.run_engine_suite(
                    args,
                    engine="lightning_mlx",
                    profile=fair.QWEN36_PROFILES["27b-4bit"],
                    suite="flappy",
                    depth=3,
                    port=18765,
                )

        self.assertEqual(run.call_args.kwargs["model_dir"], Path("qwen3.6-27b"))

    def test_lightning_ngram_optimized_alias_source_passes_alias_to_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                hf_cache=root / "hf",
                suites_dir=root / "suites",
                output_dir=root / "out",
                mode="sampled",
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=0,
                cooldown=0.0,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                enable_thinking=False,
                lightning_enable_thinking=False,
                skip_existing=False,
                rapid_python=Path("python"),
                lightning_source=Path("lightning"),
                base_port=18765,
                lightning_mtp_optimistic=False,
                lightning_mtp_draft_temperature=0.5,
                lightning_ngram_auto_disable_mtp_threshold=0.85,
                lightning_ngram_auto_disable_min_ngram=0.50,
                inter_case_cooldown=0.0,
                lightning_model_source="optimized-alias",
            )
            with patch.object(fair, "run_rapid_mlx_suite", return_value=root / "out.json") as run:
                fair.run_engine_suite(
                    args,
                    engine="lightning_mtp_ngram",
                    profile=fair.QWEN36_PROFILES["35b-a3b-4bit"],
                    suite="flappy",
                    depth=1,
                    port=18765,
                )

        self.assertEqual(run.call_args.kwargs["model_dir"], Path("qwen3.6-35b"))
        self.assertTrue(run.call_args.kwargs["enable_ngram"])

    def test_mtplx_ignores_lightning_optimized_alias_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_cache = root / "hf"
            sidecar = hf_cache / "models--ax-local--Qwen3.6-27B-MTP" / "snapshots" / "v1"
            args = Namespace(
                hf_cache=hf_cache,
                suites_dir=root / "suites",
                output_dir=root / "out",
                mode="sampled",
                max_tokens=64,
                repetitions=1,
                warmup_repetitions=0,
                cooldown=0.0,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                enable_thinking=False,
                skip_existing=False,
                mtplx_python=Path("python"),
                mtplx_profile="stable",
                inter_case_cooldown=0.0,
                lightning_model_source="optimized-alias",
            )
            with patch.object(fair, "run_mtplx_suite", return_value=root / "out.json") as run:
                fair.run_engine_suite(
                    args,
                    engine="mtplx",
                    profile=fair.QWEN36_PROFILES["27b-4bit"],
                    suite="flappy",
                    depth=3,
                    port=18765,
                )

        self.assertEqual(run.call_args.kwargs["model_dir"], sidecar)

    def test_error_artifact_keeps_table_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error_path = fair.write_error_artifact(
                root / "mtplx.json",
                engine="mtplx",
                error=RuntimeError("server failed"),
                command=["mtplx"],
            )
            summary = fair.summarize_engine_artifact("mtplx", error_path)
            artifact = json.loads(error_path.read_text())

        self.assertEqual(summary["status"], "error")
        self.assertIn("server failed", summary["error"])
        self.assertEqual(artifact["command"], ["mtplx"])

    def test_error_artifact_uses_called_process_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error_path = fair.write_error_artifact(
                Path(tmp) / "ax.json",
                engine="ax_engine",
                error=fair.subprocess.CalledProcessError(7, ["ax", "bench"]),
                command=None,
            )
            artifact = json.loads(error_path.read_text())

        self.assertEqual(artifact["command"], ["ax", "bench"])
        self.assertEqual(artifact["returncode"], 7)

    def test_validation_warnings_change_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = fake_mtplx_artifact()
            artifact["validations_passed"] = 1
            artifact["validations_total"] = 2
            path = write_json(Path(tmp) / "mtplx.json", artifact)
            summary = fair.summarize_engine_artifact("mtplx", path)

        self.assertEqual(summary["status"], "ok_validation_warnings")
        self.assertEqual(summary["validations_passed"], 1)
        self.assertEqual(summary["validations_total"], 2)

    def test_summarize_engine_records_box_plot_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = fake_mtplx_artifact()
            artifact["summary"] = {
                "decode_tok_s": {"median": 9.0, "values": [8.0, 9.0, 10.0]}
            }
            artifact["results"][0]["runs"] = [
                {"accepted_by_depth": [2], "drafted_by_depth": [4]},
                {"accepted_by_depth": [3], "drafted_by_depth": [4]},
            ]
            path = write_json(Path(tmp) / "mtplx.json", artifact)
            summary = fair.summarize_engine_artifact("mtplx", path)

        self.assertEqual(summary["decode_tok_s_samples"], [8.0, 9.0, 10.0])
        self.assertEqual(summary["accept_rate_samples"], [0.5, 0.75])

    def test_mtp_decode_svg_is_grouped_box_whisker(self) -> None:
        summary = fake_chart_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decode.svg"
            fair.write_decode_model_svg(path, summary, "27b-4bit")
            text = path.read_text()

        self.assertIn("Grouped box-and-whisker plot", text)
        self.assertIn("stroke-dasharray", text)
        self.assertIn("<circle", text)
        self.assertIn("#f8fafc", text)
        self.assertIn("highest median", text)
        self.assertIn('paint-order="stroke"', text)
        self.assertIn(">all suites</text>", text)
        self.assertNotIn(">flappy</text>", text)
        self.assertNotIn(">long_code</text>", text)

    def test_mtp_combined_suite_chart_group_merges_rows(self) -> None:
        summary = fake_chart_summary()
        groups = fair.combined_suite_chart_group(
            summary["rows"], ["mtplx", "ax_engine"], "decode_tok_s"
        )

        self.assertEqual([group["label"] for group in groups], ["all suites"])
        self.assertEqual(
            groups[0]["values"]["mtplx"],
            [7.5, 8.0, 8.5, 8.8, 9.0, 9.2],
        )
        self.assertEqual(
            groups[0]["values"]["ax_engine"],
            [9.5, 10.0, 10.5, 10.8, 11.0, 11.2],
        )

    def test_prefill_svg_is_grouped_box_whisker(self) -> None:
        report = fake_prefill_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefill.svg"
            prefill.write_prefill_model_svg(path, report, "27b-4bit")
            text = path.read_text()

        self.assertIn("Grouped box-and-whisker plot", text)
        self.assertIn("<circle", text)
        self.assertIn("#f8fafc", text)
        self.assertIn("highest median", text)
        self.assertIn('paint-order="stroke"', text)
        self.assertIn(">all suites</text>", text)
        self.assertNotIn(">flappy</text>", text)
        self.assertNotIn(">long_code</text>", text)


def sidecar_record(path: Path, content: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": fair.provenance_check.sha256_file(path),
    }


def make_sidecar_manifest(
    hf_cache: Path,
    *,
    norm_policy: str = "shift_mtp_norm_weights_by_1",
) -> None:
    sidecar = hf_cache / "models--ax-local--Qwen3.6-27B-MTP" / "snapshots" / "v1"
    sidecar.mkdir(parents=True)
    manifest = {
        "schema_version": "ax.mtp_sidecar_provenance.v1",
        "generated_by": "scripts/prepare_qwen36_mtp_sidecar.py",
        "model_key": "27b",
        "base": {
            "model_id": "mlx-community/Qwen3.6-27B-4bit",
            "snapshot": "abc",
            "config": sidecar_record(sidecar / "base-config.json", "{}"),
        },
        "source": {
            "model_id": "Qwen/Qwen3.6-27B",
            "mtp_shards": [
                {
                    "name": "model-00013-of-00015.safetensors",
                    **sidecar_record(sidecar / "source-shard-a.safetensors", "a"),
                },
                {
                    "name": "model-00015-of-00015.safetensors",
                    **sidecar_record(sidecar / "source-shard-b.safetensors", "b"),
                },
            ],
        },
        "output": {
            "mtp": sidecar_record(sidecar / "model.safetensors", "mtp"),
            "runtime": sidecar_record(sidecar / "mtplx_runtime.json", "{}"),
            "config": sidecar_record(sidecar / "config.json", "{}"),
        },
        "transform": {"norm_policy": norm_policy},
        "runtime": {
            "arch_id": "qwen3-next-mtp",
            "mtplx_version": "0.3.7",
            "mtp_depth_max": 3,
            "mtp_tensor_count": 15,
            "recommended_draft_sampler": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
            },
            "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "exactness_baseline": {"context": 2048, "max_abs_diff": 0.0},
            "verified_on": {"system": "Darwin", "machine": "arm64"},
        },
    }
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(json.dumps(manifest) + "\n")


def fake_ax_artifact() -> dict:
    return {
        "schema_version": "ax.mlx_inference_stack.v2",
        "build": {"git_tracked_dirty": False},
        "repetitions": 1,
        "ax_mtp_max_depth": 3,
        "results": [
            {
                "engine": "ax_engine_mlx_ngram_accel",
                "prompt_case_id": "case_1",
                "decode_tok_s": {"median": 10.0},
                "ngram_acceleration_telemetry": {
                    "ax_mtp_draft_tokens": 4,
                    "ax_mtp_accepted_tokens": 2,
                    "ax_mtp_drafted_depth0": 4,
                    "ax_mtp_accepted_depth0": 2,
                },
                "trials": [{"output_token_ids": [1, 2]}],
            }
        ],
    }


def fake_mtplx_artifact() -> dict:
    return {
        "schema": "ax.mtplx.prompt_suite_mtp.v1",
        "engine": "mtplx",
        "depth": 3,
        "results": [
            {
                "prompt_id": "case_1",
                "summary": {
                    "decode_tok_s": {"median": 8.0},
                    "accepted_drafts": 2,
                    "drafted_tokens": 4,
                    "accept_rate": 0.5,
                },
                "runs": [
                    {
                        "tokens": [1, 2],
                        "accepted_by_depth": [2],
                        "drafted_by_depth": [4],
                    }
                ],
            }
        ],
    }


def fake_chart_summary() -> dict:
    return {
        "schema": "ax.qwen36_mtp_fair.v1",
        "contract": {
            "engines": ["mtplx", "ax_engine"],
            "suites": ["flappy", "long_code"],
        },
        "rows": [
            {
                "model": "27b-4bit",
                "model_label": "Qwen3.6 27B 4-bit",
                "suite": "flappy",
                "depth": 3,
                "engines": {
                    "mtplx": {
                        "decode_tok_s": 8.0,
                        "decode_tok_s_samples": [7.5, 8.0, 8.5],
                        "accept_rate": 0.50,
                        "accept_rate_samples": [0.45, 0.50, 0.55],
                    },
                    "ax_engine": {
                        "decode_tok_s": 10.0,
                        "decode_tok_s_samples": [9.5, 10.0, 10.5],
                        "accept_rate": 0.70,
                        "accept_rate_samples": [0.65, 0.70, 0.75],
                    },
                },
            },
            {
                "model": "27b-4bit",
                "model_label": "Qwen3.6 27B 4-bit",
                "suite": "long_code",
                "depth": 3,
                "engines": {
                    "mtplx": {
                        "decode_tok_s": 9.0,
                        "decode_tok_s_samples": [8.8, 9.0, 9.2],
                        "accept_rate": 0.60,
                        "accept_rate_samples": [0.58, 0.60, 0.62],
                    },
                    "ax_engine": {
                        "decode_tok_s": 11.0,
                        "decode_tok_s_samples": [10.8, 11.0, 11.2],
                        "accept_rate": 0.80,
                        "accept_rate_samples": [0.78, 0.80, 0.82],
                    },
                },
            },
        ],
    }


def fake_prefill_report() -> dict:
    return {
        "schema": "ax.mtp_prefill_ttft_report.v1",
        "contract": {"engines": ["mtplx", "lightning_mlx", "ax_engine"]},
        "rows": [
            {
                "model": "27b-4bit",
                "model_label": "Qwen3.6 27B 4-bit",
                "suite": "flappy",
                "engines": {
                    "mtplx": {
                        "prefill_tok_s": 100.0,
                        "prefill_tok_s_samples": [95.0, 100.0, 105.0],
                        "ttft_ms": 30.0,
                        "ttft_ms_samples": [28.0, 30.0, 32.0],
                    },
                    "lightning_mlx": {
                        "prefill_tok_s": 90.0,
                        "prefill_tok_s_samples": [88.0, 90.0, 92.0],
                        "ttft_ms": 34.0,
                        "ttft_ms_samples": [32.0, 34.0, 36.0],
                        "prefill_note": "approx_via_ttft",
                    },
                    "ax_engine": {
                        "prefill_tok_s": 110.0,
                        "prefill_tok_s_samples": [108.0, 110.0, 112.0],
                        "ttft_ms": 26.0,
                        "ttft_ms_samples": [24.0, 26.0, 28.0],
                    },
                },
            },
            {
                "model": "27b-4bit",
                "model_label": "Qwen3.6 27B 4-bit",
                "suite": "long_code",
                "engines": {
                    "mtplx": {
                        "prefill_tok_s": 120.0,
                        "prefill_tok_s_samples": [118.0, 120.0, 122.0],
                        "ttft_ms": 40.0,
                        "ttft_ms_samples": [38.0, 40.0, 42.0],
                    },
                    "lightning_mlx": {
                        "prefill_tok_s": 96.0,
                        "prefill_tok_s_samples": [94.0, 96.0, 98.0],
                        "ttft_ms": 44.0,
                        "ttft_ms_samples": [42.0, 44.0, 46.0],
                        "prefill_note": "approx_via_ttft",
                    },
                    "ax_engine": {
                        "prefill_tok_s": 130.0,
                        "prefill_tok_s_samples": [128.0, 130.0, 132.0],
                        "ttft_ms": 36.0,
                        "ttft_ms_samples": [34.0, 36.0, 38.0],
                    },
                },
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
