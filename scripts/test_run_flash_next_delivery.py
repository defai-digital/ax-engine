#!/usr/bin/env python3
"""Weight-free regressions for the Flash Next fresh-delivery harness."""
from __future__ import annotations

from argparse import Namespace
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

    def test_resuming_carries_the_failed_attempts_and_the_original_fresh_state(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache"
            cache.mkdir()
            with mod.lock_cache(cache) as descriptor:
                cache_state = mod.cache_resume_state(cache, descriptor)
            first = Path(td) / "attempt-1"
            first.mkdir()
            inventory = "a" * 64
            record = {"schema": mod.SCHEMA, "completed": False, "repo_id": mod.native.PRIMARY_REPO,
                      "revision": mod.native.PACK_REVISION, "inventory_sha256": inventory,
                      "resume_cache_sha256": cache_state,
                      "fresh_cache": True, "download": {"exit_code": 1, "elapsed_seconds": 450.0}}
            (first / "result.json").write_text(json.dumps(record))
            (first / "download.stderr.log").write_text("Error while downloading\nMax retries exceeded.\n")
            fresh, attempts = mod.load_previous_attempts(first, cache_state, inventory)
            self.assertTrue(fresh)
            self.assertEqual(attempts[0]["exit_code"], 1)
            self.assertIn("Max retries exceeded.", attempts[0]["stderr_tail"])
            second = Path(td) / "attempt-2"
            second.mkdir()
            (second / "result.json").write_text(json.dumps({
                **record, "previous_attempts": attempts, "resumed": True,
                "download": {"exit_code": 1, "elapsed_seconds": 90.0}}))
            _, chain = mod.load_previous_attempts(second, cache_state, inventory)
            self.assertEqual([a["exit_code"] for a in chain], [1, 1])
            # A cache that already holds data was not fresh to begin with.
            (first / "result.json").write_text(json.dumps({**record, "fresh_cache": False}))
            self.assertFalse(mod.load_previous_attempts(first, cache_state, inventory)[0])
            # Successful or foreign records cannot be resumed.
            (first / "result.json").write_text(json.dumps({**record, "completed": True}))
            with self.assertRaisesRegex(ValueError, "did not fail"):
                mod.load_previous_attempts(first, cache_state, inventory)
            (first / "result.json").write_text(json.dumps({**record, "schema": "other"}))
            with self.assertRaisesRegex(ValueError, "v2 delivery record"):
                mod.load_previous_attempts(first, cache_state, inventory)

    def test_resume_rejects_different_or_modified_cache_and_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "cache"
            cache.mkdir()
            attempt = root / "attempt"
            attempt.mkdir()
            partial = cache / "weight.incomplete"
            partial.write_bytes(b"first")
            inventory = "a" * 64
            with mod.lock_cache(cache) as descriptor:
                original = mod.cache_resume_state(cache, descriptor)
                (cache / ".DS_Store").write_bytes(b"finder metadata")
                self.assertEqual(mod.cache_resume_state(cache, descriptor), original)
                record = {"schema": mod.SCHEMA, "completed": False, "repo_id": mod.native.PRIMARY_REPO,
                          "revision": mod.native.PACK_REVISION, "inventory_sha256": inventory,
                          "download": {"exit_code": 1, "elapsed_seconds": 1},
                          "fresh_cache": True, "resume_cache_sha256": original}
                result = attempt / "result.json"
                result.write_text(json.dumps(record))
                self.assertTrue(mod.load_previous_attempts(attempt, original, inventory)[0])
                with self.assertRaisesRegex(ValueError, "different pack inventory"):
                    mod.load_previous_attempts(attempt, original, "b" * 64)
                original_stat = partial.stat()
                replacement = cache / "replacement"
                replacement.write_bytes(b"other")
                mod.os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                replacement.replace(partial)
                changed = mod.cache_resume_state(cache, descriptor)
                with self.assertRaisesRegex(ValueError, "unchanged cache root"):
                    mod.load_previous_attempts(attempt, changed, inventory)
                result.write_text(json.dumps({key: value for key, value in record.items()
                                              if key != "resume_cache_sha256"}))
                with self.assertRaisesRegex(ValueError, "unchanged cache root"):
                    mod.load_previous_attempts(attempt, original, inventory)
                result.write_text(json.dumps(record))
            another = root / "another"
            another.mkdir()
            (another / partial.name).write_bytes(b"first")
            with mod.lock_cache(another) as descriptor:
                unrelated = mod.cache_resume_state(another, descriptor)
                with self.assertRaisesRegex(ValueError, "unchanged cache root"):
                    mod.load_previous_attempts(attempt, unrelated, inventory)

    def test_shared_cache_lock_refuses_concurrent_owner_and_detects_path_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            root.mkdir()
            with mod.lock_cache(root) as descriptor:
                with self.assertRaisesRegex(ValueError, "another delivery"):
                    with mod.lock_cache(root):
                        self.fail("overlapping delivery acquired the lock")
                root.rename(root.with_name("moved"))
                root.mkdir()
                with self.assertRaisesRegex(ValueError, "cache root changed"):
                    mod.cache_resume_state(root, descriptor)
            with mod.lock_cache(root) as descriptor:
                mod.check_cache_directory(root, descriptor)

    def test_failed_download_binds_local_cache_for_next_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / "cache"
            cache.mkdir()
            cli = root / "ax-engine"
            cli.touch()
            inventory = root / "inventory.json"
            inventory.write_text('{"storage":"internal SSD"}')
            args = Namespace(cache_root=cache, previous_attempt=None, ax_engine=cli,
                             inventory=inventory, server_bin=cli, output=root / "first")
            contract = {"schema": mod.SCHEMA, "repo_id": mod.native.PRIMARY_REPO,
                        "revision": mod.native.PACK_REVISION}

            def failed_download(command, **kwargs):
                (cache / "download.incomplete").write_bytes(b"partial")
                return subprocess.CompletedProcess(command, 1)

            with (patch.object(mod.native, "storage_info", return_value={}) as storage_info,
                  patch.object(mod.native, "form_factor", return_value="MacBook Pro"),
                  patch.object(mod.shutil, "disk_usage", return_value=Namespace(free=1 << 40)),
                  patch.object(mod.subprocess, "run", side_effect=failed_download) as subprocess_run,
                  mod.lock_cache(cache) as descriptor):
                with self.assertRaisesRegex(RuntimeError, "downloader exited"):
                    mod.run_delivery(args, contract, 128 << 30, "Apple M5 Max", descriptor)
                receipt = json.loads((args.output / "result.json").read_text())
                cache_state = mod.cache_resume_state(cache, descriptor)
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["resume_cache_sha256"], cache_state)
                fresh, attempts = mod.load_previous_attempts(args.output, cache_state, mod.sha256_file(inventory))
                self.assertTrue(fresh)
                self.assertEqual(len(attempts), 1)
                self.assertNotIn(str(cache), json.dumps(receipt))
                subprocess_run.assert_called_once()
                self.assertEqual(subprocess_run.call_args.args[0],
                                 [str(cli), "download", mod.ALIAS, "--progress-json"])
                self.assertEqual(receipt["download"]["ax_engine_sha256"], mod.sha256_file(cli))
                self.assertNotIn("hub_client_version", receipt["download"])

                def populate_during_preflight(path):
                    (cache / "unrelated").write_bytes(b"pre-existing weights")
                    return {}

                args.previous_attempt = args.output
                args.output = root / "second"
                subprocess_run.reset_mock(side_effect=True)
                storage_info.side_effect = populate_during_preflight
                with self.assertRaisesRegex(ValueError, "cache contents changed"):
                    mod.run_delivery(args, contract, 128 << 30, "Apple M5 Max", descriptor)
                subprocess_run.assert_not_called()

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

    def test_environment_preserves_path_and_isolates_the_native_cache(self):
        inherited = {"HF_HOME": "/x", "AX_STREAM_EXPERTS": "on", "MLX_X": "1",
                     "XDG_CACHE_HOME": "/y", "PATH": "/usr/bin",
                     "AX_ENGINE_DOWNLOAD_HELPER": "/old", "HF_HUB_DOWNLOAD_TIMEOUT": "60"}
        with patch.dict(mod.os.environ, inherited, clear=True):
            self.assertEqual(mod.clean_env(Path("/cache")),
                             {"HF_HUB_CACHE": "/cache", "PATH": "/usr/bin"})

    def test_terminal_record_requires_an_actual_ready_pin_at_the_cache_destination(self):
        snapshot = Path("/cache") / mod.native.PACK_REVISION
        ready = {"schema_version": "ax.download_model.v1", "status": "ready", "errors": [],
                 "repo_id": mod.native.PRIMARY_REPO, "revision": mod.native.PACK_REVISION,
                 "resolved_revision": mod.native.PACK_REVISION, "dest": str(snapshot),
                 "manifest_present": True, "config_present": True}
        progress = '{"event":"progress","phase":"validate"}'
        self.assertEqual(mod.terminal_record(f"{progress}\n{json.dumps(ready)}\n", snapshot), ready)
        for key, bad in (("schema_version", "other"), ("repo_id", "other/model"),
                         ("errors", ["warning"]), ("resolved_revision", None),
                         ("status", "download_failed"), ("manifest_present", False),
                         ("config_present", False), ("revision", "main")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pinned native download"):
                mod.terminal_record(json.dumps({**ready, key: bad}), snapshot)
        with self.assertRaisesRegex(ValueError, "outside the pinned snapshot"):
            mod.terminal_record(json.dumps({**ready, "dest": "/elsewhere"}), snapshot)
        with self.assertRaisesRegex(ValueError, "pinned native download"):
            mod.terminal_record(progress, snapshot)
        with self.assertRaisesRegex(ValueError, "no JSON terminal record"):
            mod.terminal_record(f"{json.dumps(ready)}\nunexpected final line\n", snapshot)

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
        self.assertIn("native product CLI", contract["downloader"])
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
