#!/usr/bin/env python3
"""Contract tests for the Flash Next same-pack peer harness."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_flash_next_peer as peer
import bench_omlx_prompt_suites as omlx_bench


class OmlxTokenizerTest(unittest.TestCase):
    def test_chat_template_mapping_uses_input_ids(self) -> None:
        self.assertEqual(
            omlx_bench.token_ids_from_encoded({"input_ids": [3, 4]}),
            [3, 4],
        )
        self.assertEqual(omlx_bench.token_ids_from_encoded([3, 4]), [3, 4])
        self.assertEqual(omlx_bench.token_ids_from_encoded([[3, 4]]), [3, 4])


class IndexViewTest(unittest.TestCase):
    def test_sidecar_keys_are_added_without_replacing_existing(self) -> None:
        merged = peer.merge_sidecar_into_index(
            {"weight_map": {"model.w": "model.safetensors"}},
            ["mtp.fc.weight", "model.w"],
            "mtp.safetensors",
        )
        self.assertEqual(merged["weight_map"]["mtp.fc.weight"], "mtp.safetensors")
        self.assertEqual(merged["weight_map"]["model.w"], "model.safetensors")


class OmlxSettingsTest(unittest.TestCase):
    def test_draft_width_follows_the_installed_field(self) -> None:
        current = omlx_bench.mtp_settings_kwargs({"mtp_fixed_depth"}, 3, False)
        self.assertEqual(current["mtp_fixed_depth"], 3)
        self.assertNotIn("mtp_num_draft_tokens", current)
        legacy = omlx_bench.mtp_settings_kwargs({"mtp_num_draft_tokens"}, 1, False)
        self.assertEqual(legacy["mtp_num_draft_tokens"], 1)

    def test_missing_draft_field_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            omlx_bench.mtp_settings_kwargs(set(), 1, False)


class EngineSelectionTest(unittest.TestCase):
    def test_qwen4_exp_uses_the_vlm_engine(self) -> None:
        self.assertEqual(omlx_bench.omlx_engine_kind("qwen4_exp", "auto"), "vlm")
        self.assertEqual(omlx_bench.omlx_engine_kind("qwen4_exp_text", "auto"), "vlm")

    def test_other_families_keep_the_text_engine(self) -> None:
        self.assertEqual(omlx_bench.omlx_engine_kind("qwen3_5", "auto"), "batched")
        self.assertEqual(omlx_bench.omlx_engine_kind(None, "auto"), "batched")
        self.assertEqual(omlx_bench.omlx_engine_kind("qwen4_exp", "batched"), "batched")


class PeerHarnessTest(unittest.TestCase):
    def test_extra_parameters_are_unsupported_not_a_speed(self) -> None:
        reason = peer.classify_load_failure(
            "ERROR: Received 384 parameters not in model: ngram_embedding.shards.0.weight."
        )
        self.assertIsNotNone(reason)
        self.assertNotIn("tok/s", reason or "")
        self.assertIsNone(peer.classify_load_failure("connection reset"))

    def test_mlx_lm_does_not_claim_qwen4_exp(self) -> None:
        self.assertFalse(peer.mlx_lm_supports("qwen4_exp"))
        self.assertFalse(peer.mlx_lm_supports("qwen4_exp_text"))

    def test_commands_pin_the_flash_next_pack_and_depth(self) -> None:
        commands = peer.lane_commands(
            python="python",
            omlx_python="python-omlx",
            pack=Path("/packs/flash-next"),
            prompts=Path("/prompts/flappy.jsonl"),
            output_dir=Path("/out"),
            repetitions=5,
            warmup_repetitions=2,
            max_tokens=256,
            cooldown=3.0,
            depth=3,
            mtplx_source=Path("/src/MTPLX"),
        )
        self.assertIn("--ax-ngram-accel", commands["ax"])
        self.assertIn("--ax-mtp-policy", commands["ax"])
        self.assertIn("required", commands["ax"])
        self.assertIn(peer.PACK_REPO, commands["ax"])
        self.assertEqual(commands["mtplx"][commands["mtplx"].index("--depth") + 1], "3")
        self.assertIn("--allow-unverified-model", commands["mtplx"])
        self.assertEqual(commands["omlx"][0], "python-omlx")
        self.assertIn("vlm", commands["omlx"])
        self.assertNotIn("--skip-mlx-lm", commands["mlx_lm"])

    def test_scrub_replaces_hostname_but_keeps_chip_names(self) -> None:
        payload = {
            "host": "secret-host.local",
            "hostname": "secret-host.local",
            "chip": "Apple M2 Ultra",
            "note": "built on secret-host.local",
            "toolchain": {"rustc": "host: aarch64-apple-darwin"},
        }
        cleaned = peer.scrub_host_identity(payload, "secret-host.local", "M2 Ultra 192 GB")
        self.assertEqual(cleaned["host"], "M2 Ultra 192 GB")
        self.assertEqual(cleaned["hostname"], "M2 Ultra 192 GB")
        self.assertEqual(cleaned["chip"], "Apple M2 Ultra")
        self.assertNotIn("secret-host.local", cleaned["note"])
        self.assertEqual(cleaned["toolchain"]["rustc"], "host: aarch64-apple-darwin")

    def test_median_uses_measured_repetitions(self) -> None:
        payload = {
            "results": [
                {
                    "prompt_tokens": 100,
                    "trials": [{"decode_tok_s": 10.0, "prefill_tok_s": 100.0}],
                    "runs": [
                        {"measured": False, "decode_tok_s": 1.0, "prompt_eval_time_s": 1.0},
                        {"measured": True, "decode_tok_s": 30.0, "prompt_eval_time_s": 0.5},
                    ],
                },
                {"decode_tok_s_values": [20.0, 40.0]},
            ]
        }
        # trials are consumed first and the function continues past runs only
        # when trials are absent. Split the shapes the runners actually emit.
        ax = {"results": [{"trials": [{"decode_tok_s": 10.0}, {"decode_tok_s": 30.0}]}]}
        mtplx = {
            "results": [
                {
                    "prompt_tokens": 100,
                    "runs": [
                        {"measured": False, "decode_tok_s": 1.0, "prompt_eval_time_s": 1.0},
                        {"measured": True, "decode_tok_s": 30.0, "prompt_eval_time_s": 0.5},
                    ],
                }
            ]
        }
        self.assertEqual(peer.measured_medians(ax)["decode_tok_s_median"], 20.0)
        self.assertEqual(peer.measured_medians(mtplx)["decode_tok_s_median"], 30.0)
        self.assertEqual(peer.measured_medians(mtplx)["prefill_tok_s_median"], 200.0)
        self.assertEqual(peer.measured_medians(payload)["measured_decode_samples"] > 0, True)

    def test_dry_run_writes_a_plan_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "pack"
            pack.mkdir()
            (pack / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            out = root / "out"
            code = peer.main(
                [
                    "--model-dir",
                    str(pack),
                    "--output-dir",
                    str(out),
                    "--dry-run",
                    "--lanes",
                    "ax",
                    "mlx_lm",
                ]
            )
            self.assertEqual(code, 0)
            plan = json.loads((out / "plan.json").read_text())
            self.assertEqual(plan["pack_revision"], peer.PACK_REVISION)
            self.assertFalse(plan["qualification"])
            self.assertIn("MTP-S", plan["open_gates"])
            self.assertNotIn("summary.json", {path.name for path in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
