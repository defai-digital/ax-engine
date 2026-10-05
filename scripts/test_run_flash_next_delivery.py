#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next fresh-delivery harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("run_flash_next_delivery.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("flash_delivery", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class DeliveryTests(unittest.TestCase):
    def test_fresh_cache_must_be_empty_unless_resuming(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            self.assertTrue(mod.check_fresh(root, False))
            root.mkdir()
            self.assertTrue(mod.check_fresh(root, False))
            (root / "leftover").write_text("x")
            with self.assertRaisesRegex(ValueError, "empty cache root"):
                mod.check_fresh(root, False)
            self.assertFalse(mod.check_fresh(root, True))
            file_root = Path(td) / "file"
            file_root.write_text("x")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                mod.check_fresh(file_root, False)

    def test_snapshot_dir_requires_exactly_the_pinned_revision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "exactly the pinned revision"):
                mod.snapshot_dir(root)
            snapshots = root / mod.REPO_DIR / "snapshots"
            (snapshots / mod.native.PACK_REVISION).mkdir(parents=True)
            self.assertEqual(mod.snapshot_dir(root), snapshots / mod.native.PACK_REVISION)
            (snapshots / "main").mkdir()
            with self.assertRaisesRegex(ValueError, "exactly the pinned revision"):
                mod.snapshot_dir(root)

    def test_environment_drops_inherited_overrides_and_sets_only_cache_and_python(self):
        with patch.dict(mod.os.environ, {"HF_HOME": "/x", "AX_STREAM_EXPERTS": "on",
                                         "MLX_X": "1", "XDG_CACHE_HOME": "/y", "KEEP": "1"}):
            env = mod.clean_env(Path("/cache"), Path("/py"))
        self.assertEqual(env["HF_HUB_CACHE"], "/cache")
        self.assertEqual(env["AX_ENGINE_PYTHON"], "/py")
        self.assertEqual(env["KEEP"], "1")
        for name in ("HF_HOME", "AX_STREAM_EXPERTS", "MLX_X", "XDG_CACHE_HOME"):
            self.assertNotIn(name, env)

    def test_terminal_record_is_the_last_json_object(self):
        stdout = '{"event":"progress","n":1}\nnot json\n{"event":"done","status":"ready"}\n'
        self.assertEqual(mod.terminal_record(stdout)["status"], "ready")
        with self.assertRaises(ValueError):
            mod.terminal_record("no json here\n")

    def test_scrub_removes_home_and_loopback_recursively(self):
        home = str(Path.home())
        out = mod.scrub({"a": [f"{home}/x", "127.0.0.1:1"], "b": {"c": 3}})
        self.assertEqual(out, {"a": ["[home]/x", "[loopback]:1"], "b": {"c": 3}})

    def test_delivery_requires_the_published_pack_and_a_ready_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            members = [{"name": "m", "size": 1, "sha256": "0"}]
            with patch.object(mod.native, "validate_inventory", return_value={}):
                with self.assertRaisesRegex(ValueError, "not the published pack"):
                    mod.verify_delivery(root, root, {"members": members})
                full = [{"name": f"f{i}", "size": 1, "sha256": "0"} for i in range(mod.PUBLISHED_MEMBERS)]
                full[0]["size"] = mod.PUBLISHED_BYTES - (mod.PUBLISHED_MEMBERS - 1)
                with self.assertRaisesRegex(ValueError, "no native model manifest"):
                    mod.verify_delivery(root, root, {"members": full})
                (root / "model-manifest.json").write_text(json.dumps(
                    {"runtime_status": {"ready": False, "blockers": ["x"]}}))
                with self.assertRaisesRegex(ValueError, "not runtime ready"):
                    mod.verify_delivery(root, root, {"members": full})
                (root / "model-manifest.json").write_text(json.dumps(
                    {"runtime_status": {"ready": True, "blockers": []}}))
                out = mod.verify_delivery(root, root, {"members": full})
                self.assertEqual(out["members_verified"], mod.PUBLISHED_MEMBERS)
                self.assertTrue(out["native_manifest_runtime_ready"])

    def test_dry_run_declares_all_three_control_modes_and_cannot_claim_qualification(self):
        out = subprocess.run([sys.executable, str(SCRIPT), "--dry-run"], capture_output=True,
                             text=True, check=True)
        contract = json.loads(out.stdout)
        self.assertEqual(contract["control_modes"], ["disabled", "required", "default"])
        self.assertEqual(contract["published_members"], 49)
        self.assertFalse(contract["qualification"])
        self.assertFalse(contract["release_ready"])

    def test_live_run_requires_all_inputs(self):
        done = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("live run requires", done.stderr)


if __name__ == "__main__":
    unittest.main()
