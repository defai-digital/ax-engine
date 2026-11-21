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
            args = Namespace(cache_root=cache, previous_attempt=None, ax_engine=cli, python=Path(sys.executable),
                             helper=None, bench_bin=None, inventory=inventory, server_bin=cli, output=root / "first")
            contract = {"schema": mod.SCHEMA, "repo_id": mod.native.PRIMARY_REPO,
                        "revision": mod.native.PACK_REVISION}

            def failed_download(command, **kwargs):
                if command[1] == "-c":
                    return subprocess.CompletedProcess(command, 0, stdout="test-hub")
                (cache / "download.incomplete").write_bytes(b"partial")
                return subprocess.CompletedProcess(command, 1)

            with (patch.object(mod.native, "storage_info", return_value={}),
                  patch.object(mod.native, "form_factor", return_value="MacBook Pro"),
                  patch.object(mod.shutil, "disk_usage", return_value=Namespace(free=1 << 40)),
                  patch.object(mod.subprocess, "run", side_effect=failed_download) as subprocess_run,
                  mod.lock_cache(cache) as descriptor):
                with self.assertRaisesRegex(RuntimeError, "downloader exited"):
                    mod.run_delivery(args, contract, {}, 128 << 30, "Apple M5 Max", descriptor)
                receipt = json.loads((args.output / "result.json").read_text())
                cache_state = mod.cache_resume_state(cache, descriptor)
                self.assertFalse(receipt["completed"])
                self.assertEqual(receipt["resume_cache_sha256"], cache_state)
                fresh, attempts = mod.load_previous_attempts(args.output, cache_state, mod.sha256_file(inventory))
                self.assertTrue(fresh)
                self.assertEqual(len(attempts), 1)
                self.assertNotIn(str(cache), json.dumps(receipt))

                def populate_during_preflight(command, **kwargs):
                    (cache / "unrelated").write_bytes(b"pre-existing weights")
                    return subprocess.CompletedProcess(command, 0, stdout="test-hub")

                args.previous_attempt = args.output
                args.output = root / "second"
                subprocess_run.reset_mock(side_effect=True)
                subprocess_run.side_effect = populate_during_preflight
                with self.assertRaisesRegex(ValueError, "cache contents changed"):
                    mod.run_delivery(args, contract, {}, 128 << 30, "Apple M5 Max", descriptor)
                self.assertEqual(subprocess_run.call_count, 1)

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

    def test_source_builds_get_an_explicit_helper_and_bench_directory(self):
        with patch.dict(mod.os.environ, {"PATH": "/usr/bin"}):
            env = mod.clean_env(Path("/cache"), Path("/py"), Path("/repo/scripts/download_model.py"),
                                Path("/build/ax-engine-bench"))
            plain = mod.clean_env(Path("/cache"), Path("/py"))
        self.assertEqual(env["AX_ENGINE_DOWNLOAD_HELPER"], "/repo/scripts/download_model.py")
        self.assertTrue(env["PATH"].startswith("/build:"))
        self.assertNotIn("AX_ENGINE_DOWNLOAD_HELPER", plain)
        self.assertEqual(plain["PATH"], "/usr/bin")

    def test_transport_overrides_are_an_allowlist_of_integers_and_reach_the_environment(self):
        parsed = mod.parse_transport(["AX_ENGINE_HF_MAX_WORKERS=2", "HF_HUB_DOWNLOAD_TIMEOUT=60"])
        self.assertEqual(parsed, {"AX_ENGINE_HF_MAX_WORKERS": "2", "HF_HUB_DOWNLOAD_TIMEOUT": "60"})
        for bad in ("PATH=/x", "AX_ENGINE_HF_MAX_WORKERS", "AX_ENGINE_HF_MAX_WORKERS=two",
                    "HF_HUB_DOWNLOAD_TIMEOUT="):
            with self.assertRaises(ValueError, msg=bad):
                mod.parse_transport([bad])
        env = mod.clean_env(Path("/cache"), Path("/py"), transport=parsed)
        self.assertEqual(env["AX_ENGINE_HF_MAX_WORKERS"], "2")
        with patch.dict(mod.os.environ, {"HF_HUB_DOWNLOAD_TIMEOUT": "5"}):
            self.assertNotIn("HF_HUB_DOWNLOAD_TIMEOUT", mod.clean_env(Path("/cache"), Path("/py")))

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
