#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next native execution oracle."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("check_flash_next_native_support.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_support", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class NativeSupportTests(unittest.TestCase):
    def test_frozen_native_support_receipt_remains_non_promoting_and_route_bound(self):
        artifact = SCRIPT.parents[1] / "benchmarks/results/qualification/2026-10-04-flash-next-mxfp4-native-support"
        result = json.loads((artifact / "result.json").read_text())
        receipt = json.loads((artifact / "build-receipt.json").read_text())
        inventory = json.loads((artifact / "pack-inventory.json").read_text())
        self.assertTrue(result["functional_controls_passed"])
        self.assertTrue(result["negative_control_result"]["activation_error_observed"])
        self.assertNotEqual(result["negative_control_result"]["exit_code"], 0)
        self.assertFalse(result["qualification"])
        self.assertFalse(result["release_ready"])
        self.assertTrue(all(value == "not_assessed" for value in receipt["mtp_gates"].values()))
        self.assertFalse(receipt["default_mtp_changed"])
        self.assertEqual(result["native_manifest_ready"], result["native_manifest_after"])
        self.assertTrue(result["native_manifest_ready"])
        self.assertTrue(result["hardware"]["storage"]["internal"])
        self.assertTrue(result["hardware"]["storage"]["solid_state"])
        self.assertEqual(result["inventory_sha256"], mod.sha256(artifact / "pack-inventory.json"))
        self.assertEqual(result["revision"], inventory["revision"])
        self.assertEqual(result["server_sha256"], receipt["server_sha256"])
        self.assertEqual(len(inventory["members"]), receipt["published_members"])
        for name, digest in receipt["raw_artifacts_sha256"].items():
            self.assertEqual(mod.sha256(artifact / name), digest, name)
        for mode in ("disabled", "required"):
            rows = result["results"][mode]
            self.assertEqual([row["name"] for row in rows], result["cases"])
            for row in rows:
                tokens = row["response"]["usage"]["completion_tokens"]
                mod.check_answer(row["name"], row["response"])
                mod.check_activation(row["metrics_before"], row["metrics_after"],
                                     mode == "required", tokens, row["request"]["max_tokens"])
                stored = json.loads((artifact / f"{mode}-{row['name']}.json").read_text())
                self.assertEqual(stored, row)
            stream = mod.parse_stream((artifact / f"{mode}-stream.sse").read_bytes())
            mod.check_answer("stream", stream)

    def test_dry_run_cannot_claim_qualification(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"],
                                check=True, capture_output=True, text=True)
        contract = json.loads(result.stdout)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])
        self.assertEqual(contract["modes"], ["disabled", "required"])

    def test_route_oracle_rejects_silent_direct_and_missing_metrics(self):
        before = {name: 0 for name in (mod.VERIFIED, *mod.ERRORS)}
        with self.assertRaisesRegex(ValueError, "without verified"):
            mod.check_activation(before, before, True, 12, 48)
        with self.assertRaisesRegex(ValueError, "without verified"):
            mod.check_activation(before, before, True, 1, 48)
        with self.assertRaisesRegex(ValueError, "missing required"):
            mod.check_activation({}, before, True, 12, 48)
        mod.check_activation(before, before, True, 1, 1)
        after = {**before, mod.VERIFIED: 7}
        mod.check_activation(before, after, True, 12, 48)
        with self.assertRaisesRegex(ValueError, "disabled-MTP"):
            mod.check_activation(before, after, False, 12, 48)

    def test_route_errors_and_output_budget_fail(self):
        before = {name: 0 for name in (mod.VERIFIED, *mod.ERRORS)}
        for error in mod.ERRORS:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, "route errors"):
                mod.check_activation(before, {**before, error: 1}, True, 1, 1)
        for count in (0, 49):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "request budget"):
                mod.check_activation(before, before, False, count, 48)

    def test_inventory_rejects_missing_shards_tampering_and_unbound_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            names = ["config.json", "axquant_manifest.json", "mtplx_runtime.json",
                     "mtp.safetensors", "model.safetensors.index.json", "model-1.safetensors"]
            for name in names:
                (root / name).write_bytes(b"test")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0": "model-1.safetensors"}}))
            members = [{"name": name, "size": (root / name).stat().st_size,
                        "sha256": mod.sha256(root / name),
                        "publisher_sha256": mod.sha256(root / name)} for name in names]
            inventory = {"repo_id": mod.PRIMARY_REPO, "revision": mod.PACK_REVISION,
                         "members": members, "storage": "internal SSD"}
            with patch.object(mod, "_mtp_metadata"):
                mod.validate_inventory(root, inventory)
                with self.assertRaisesRegex(ValueError, "storage medium"):
                    mod.validate_inventory(root, {**inventory, "storage": None})
                (root / "override.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "unbound model-directory"):
                    mod.validate_inventory(root, inventory)
                (root / "override.json").unlink()
                (root / "model-manifest.json").write_text("{}")
                self.assertEqual(mod.validate_inventory(root, inventory),
                                 {"model-manifest.json": mod.sha256(root / "model-manifest.json")})
                with self.assertRaisesRegex(ValueError, "omits indexed"):
                    mod.validate_inventory(root, {**inventory, "members": members[:-1]})
                members[-1]["publisher_sha256"] = "wrong"
                with self.assertRaisesRegex(ValueError, "publisher payload"):
                    mod.validate_inventory(root, inventory)
                members[-1]["publisher_sha256"] = members[-1]["sha256"]
                (root / "model-1.safetensors").write_bytes(b"evil")
                with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                    mod.validate_inventory(root, inventory)
                with self.assertRaisesRegex(ValueError, "immutable"):
                    mod.validate_inventory(root, {**inventory, "revision": "main"})

    def test_failure_log_is_scrubbed(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            (output / "server-required.log").write_text("listen 127.0.0.1\n")
            with patch.object(mod, "_run_mode", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    mod.run_mode(Path("server"), Path("model"), output, "required")
            self.assertNotIn("127.0.0.1", (output / "server-required.log").read_text())

    def test_blocked_required_must_observe_the_specific_error(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            process = unittest.mock.Mock()
            process.wait.return_value = 1
            process.poll.return_value = 1

            def spawn(*args, **kwargs):
                self.assertEqual(kwargs["env"]["AX_NO_SPEC"], "1")
                kwargs["stdout"].write(b"Error: Engine(MlxMtpRequiredButUnavailable)")
                return process

            with patch.object(mod.subprocess, "Popen", side_effect=spawn):
                self.assertTrue(mod.check_blocked_required(Path("server"), Path("model"), output)
                                ["activation_error_observed"])
                process.wait.return_value = 0
                with self.assertRaisesRegex(ValueError, "activation error"):
                    mod.check_blocked_required(Path("server"), Path("model"), output)

    def test_stream_requires_terminal_frames_and_correct_answer(self):
        chunks = [
            {"choices": [{"delta": {"content": "1, 2"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 4}},
        ]
        raw = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode()
        with self.assertRaisesRegex(ValueError, "DONE"):
            mod.parse_stream(raw)
        response = mod.parse_stream(raw + b"data: [DONE]\n\n")
        self.assertEqual(response["choices"][0]["message"]["content"], "1, 2")
        with self.assertRaisesRegex(ValueError, "incorrect content"):
            mod.check_answer("count", response)
        response["choices"][0]["message"]["content"] = ','.join(str(n) for n in range(1, 13))
        mod.check_answer("stream", response)
        response["choices"][0]["message"]["content"] = '```json\n{"color":"red","fruit":"apple"}\n```'
        mod.check_answer("json", response)

    def test_external_or_rotating_storage_is_rejected(self):
        for internal, solid in ((False, True), (True, False), (None, True)):
            info = {"SolidState": solid}
            if internal is not None:
                info["Internal"] = internal
            with patch.object(mod.subprocess, "check_output",
                              side_effect=["Filesystem Size Used Available Capacity Mounted\n/dev/disk3 1 1 0 100 /\n",
                                           mod.plistlib.dumps(info)]):
                with self.assertRaisesRegex(ValueError, "internal SSD"):
                    mod.storage_info(Path("model"))

    def test_form_factor_is_named_only_for_known_studio_models(self):
        for model, expected in (("Mac14,14", "Mac Studio"), ("Mac13,1", "Mac Studio"),
                                ("Mac14,8", "unrecognized (Mac14,8)"),
                                ("Mac16,5", "unrecognized (Mac16,5)")):
            with patch.object(mod.subprocess, "check_output", return_value=model + "\n"):
                self.assertEqual(mod.form_factor(), expected)


if __name__ == "__main__":
    unittest.main()
