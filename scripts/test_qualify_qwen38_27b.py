#!/usr/bin/env python3
"""Unit tests for qualify_qwen38_27b.py (no weights)."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(__file__).with_name("qualify_qwen38_27b.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qualify_qwen38_27b", MODULE_PATH
)
assert MODULE_SPEC and MODULE_SPEC.loader
mod = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = mod
MODULE_SPEC.loader.exec_module(mod)


sys.path.insert(0, str(ROOT / "scripts"))
from qwen38_live_gate import run_live, validate_cells, validate_host, validate_paired_greedy


class LiveGateTest(unittest.TestCase):
    def test_only_designated_hardware_is_admitted(self):
        host = dict(system="Darwin", arch="arm64", chip="Apple M4 Pro",
                    memory_bytes=64 * 1024**3, model="Mac16,11")
        validate_host(host)
        for changes in ({"chip": "Apple M5 Max"}, {"memory_bytes": 32 * 1024**3},
                        {"model": "MacBookPro"}, {"system": "Linux"}):
            with self.assertRaises(ValueError):
                validate_host({**host, **changes})

    def test_missing_skipped_partial_or_fallback_never_passes(self):
        direct = dict(mode="direct", status="ok", surface_passed=True, qa_items=32, qa_hard_passed=32, mtp_draft_tokens=0, mtp_verify_tokens=0)
        mtp = {**direct, "mode": "mtp", "mtp_draft_tokens": 3, "mtp_verify_tokens": 4}
        validate_cells([direct, mtp])
        for cells in ([], [direct], [direct, direct], [direct, mtp, mtp],
                      [direct, {**mtp, "status": "skip"}],
                      [direct, {**mtp, "status": "model_quality"}],
                      [direct, {**mtp, "surface_passed": False}],
                      [direct, {**mtp, "mtp_draft_tokens": 0}]):
            with self.assertRaises(ValueError):
                validate_cells(cells)

    def test_paired_greedy_discloses_divergence_without_blocking_product_health(self):
        direct = dict(status="finished", prompt_tokens=list(range(1, 17)),
                      output_tokens=list(range(64)))
        matched = validate_paired_greedy(direct, direct)
        self.assertTrue(matched["matched"])
        self.assertEqual(matched["differences"], [])
        self.assertIsNone(matched["first_divergence"])
        self.assertEqual(matched["divergence_count"], 0)
        changed = list(direct["output_tokens"])
        changed[25] = 999
        changed[63] = 888
        result = validate_paired_greedy(direct, {**direct, "output_tokens": changed})
        self.assertFalse(result["matched"])
        self.assertFalse(result["release_blocking"])
        self.assertEqual(result["classification"], "diagnostic_only")
        self.assertEqual(result["first_divergence"], 25)
        self.assertEqual(result["divergence_count"], 2)
        self.assertEqual(result["differences"], [
            {"index": 25, "direct_token": 25, "mtp_token": 999},
            {"index": 63, "direct_token": 63, "mtp_token": 888},
        ])
        self.assertIsNone(result["logit_margins"])
        self.assertEqual(direct["output_tokens"], list(range(64)))

    def test_paired_greedy_rejects_empty_failed_partial_and_wrong_inputs(self):
        good = dict(status="finished", prompt_tokens=list(range(1, 17)),
                    output_tokens=list(range(64)))
        bad_cases = [{}, {**good, "status": "failed"},
                     {**good, "prompt_tokens": list(range(2, 18))},
                     {**good, "output_tokens": []},
                     {**good, "output_tokens": list(range(63))},
                     {**good, "output_tokens": [False] * 64},
                     {**good, "output_tokens": [-1] * 64}]
        for bad in bad_cases:
            for pair in ((bad, good), (good, bad), (bad, bad)):
                with self.subTest(pair=pair):
                    with self.assertRaisesRegex(ValueError, "missing, failed, or incomplete"):
                        validate_paired_greedy(*pair)

    def test_wrong_host_persists_failed_result(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result"
            with patch("qwen38_live_gate.platform.system", return_value="Darwin"), \
                 patch("qwen38_live_gate.platform.machine", return_value="arm64"), \
                 patch("qwen38_live_gate.subprocess.check_output", side_effect=[
                     "Apple M5 Max", "MacBookPro", str(128 * 1024**3), "26.6.2"]):
                code = run_live(SimpleNamespace(output=out), mod.contract(), ROOT)
            self.assertEqual(code, 1)
            result = json.loads((out / "qualification.json").read_text())
            self.assertEqual(result["schema"], 3)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["cells"], [])
            self.assertIn("requires Mac mini", result["error"])

    def test_live_report_preserves_diagnostic_split_but_rejects_invalid_or_fallback(self):
        # Fixture orchestration only: no hardware, installed wheel or model qualification.
        import run_qa_matrix as matrix

        for failure in (None, "partial", "fallback", "quality"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                package = root / "ax_engine"
                (package / "_bin").mkdir(parents=True)
                (root / "bin").mkdir()
                model = root / "model"
                model.mkdir()
                (model / "weights.safetensors").write_bytes(b"fixture weights")
                (package / "__init__.py").write_bytes(b"# fixture package\n")
                args = SimpleNamespace(
                    output=root / "result", model_dir=model, port=31494,
                    build_manifest=root / "manifest.json", wheel=root / "fixture.whl",
                    server_bin=package / "_bin/ax-engine-server",
                    bench_bin=package / "_bin/ax-engine-bench", cli=root / "bin/ax-engine",
                )
                for path in (args.server_bin, args.bench_bin, args.cli):
                    path.write_bytes(b"fixture executable")
                with zipfile.ZipFile(args.wheel, "w") as wheel:
                    wheel.write(package / "__init__.py", "ax_engine/__init__.py")
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                model_files = {"weights.safetensors": digest(model / "weights.safetensors")}
                manifest = dict(source_commit="fixture-commit", dirty=False,
                                model_revision=mod.PRIMARY_REVISION, model_files=model_files)
                for key, path in (("server", args.server_bin), ("bench", args.bench_bin),
                                  ("wheel", args.wheel), ("cli", args.cli)):
                    manifest[key + "_sha256"] = digest(path)
                args.build_manifest.write_text(json.dumps(manifest))
                contract = mod.contract()
                contract["model_manifest_sha256"] = hashlib.sha256(
                    json.dumps(model_files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                fake_package = ModuleType("ax_engine")
                fake_package.__file__ = str(package / "__init__.py")

                def doctor(command, **kwargs):
                    kwargs["stdout"].write(json.dumps({"result": "ready", "ready_for": ["model_checks"]}))
                    return SimpleNamespace(returncode=0)

                def run_cell(cell, **kwargs):
                    response = dict(status="finished", prompt_tokens=list(range(1, 17)),
                                    output_tokens=list(range(64)))
                    if cell.mode == "mtp":
                        response["output_tokens"][25] = 999
                        if failure == "partial":
                            response["output_tokens"].pop()
                    for prefix, value in (("server-route", response),
                                          ("server-route-request", {"fixture": True})):
                        (args.output / f"{prefix}-{cell.mode}-qwen3.8-27b.json").write_text(json.dumps(value))
                    cell.status = "ok"
                    cell.surface_passed = True
                    cell.qa_items = cell.qa_hard_passed = 32
                    if cell.mode == "mtp":
                        cell.mtp_draft_tokens = 0 if failure == "fallback" else 3
                        cell.mtp_verify_tokens = 4
                        if failure == "quality":
                            cell.qa_hard_passed = 31
                    return cell

                with patch.dict(sys.modules, {"ax_engine": fake_package,
                                "ax_engine._ax_engine": ModuleType("ax_engine._ax_engine")}), \
                     patch.dict("os.environ", {}, clear=True), \
                     patch("qwen38_live_gate.sys.executable", str(root / "bin/python")), \
                     patch("qwen38_live_gate.platform.system", return_value="Darwin"), \
                     patch("qwen38_live_gate.platform.machine", return_value="arm64"), \
                     patch("qwen38_live_gate.subprocess.check_output", side_effect=[
                         "Apple M4 Pro", "Mac16,11", str(64 * 1024**3), "26.6.2", "fixture-commit", ""]), \
                     patch("qwen38_live_gate.subprocess.run", side_effect=doctor), \
                     patch.object(matrix, "run_cell", side_effect=run_cell):
                    code = run_live(args, contract, ROOT)
                result = json.loads((args.output / "qualification.json").read_text())
                self.assertEqual(result["schema"], 3)
                self.assertEqual(result["mtp_certification"],
                                 {gate: "not_assessed" for gate in ("MTP-S", "MTP-P", "MTP-D")})
                self.assertEqual(len(result["paired_greedy_artifacts"]), 4)
                for name, expected in result["paired_greedy_artifacts"].items():
                    self.assertEqual(digest(args.output / name), expected)
                self.assertEqual(code, 1 if failure else 0)
                self.assertEqual(result["status"], "failed" if failure else "passed")
                if failure:
                    self.assertIn({"partial": "incomplete", "fallback": "requested route",
                                   "quality": "QA failed"}[failure], result["error"])
                else:
                    self.assertFalse(result["paired_greedy"]["matched"])
                    self.assertEqual(result["paired_greedy"]["first_divergence"], 25)
                    self.assertFalse(result["paired_greedy"]["release_blocking"])

    def test_run_requires_all_provenance_arguments(self):
        with self.assertRaises(SystemExit):
            mod.parse_args(["--run"])


class QualifyQwen38Test(unittest.TestCase):
    def test_contract_pins_primary_pack(self) -> None:
        payload = mod.contract()
        self.assertEqual(payload["alias"], "qwen3.8-27b:axq")
        self.assertEqual(
            payload["repo_id"],
            "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        )
        self.assertEqual(
            payload["revision"],
            "3e290738e96972307c6aeb9934ab170ca0eae1c1",
        )
        self.assertIn("27B weights", payload["ci"])
        self.assertEqual(payload["host_class"], "Mac mini M4 Pro, 64 GB")

    def test_contract_separates_product_health_from_mtp_certification(self):
        payload = mod.contract()
        self.assertIn("complete paired 64-token direct/MTP greedy probe artifacts", payload["release_blocking"])
        self.assertFalse(any("identical" in gate for gate in payload["release_blocking"]))
        self.assertIn("not a ship gate", payload["diagnostic_only"][0])
        self.assertEqual(set(payload["mtp_gates"]), {"MTP-S", "MTP-P", "MTP-D"})
        self.assertIn("Separate shipping safety gate", payload["mtp_gates"]["MTP-S"])

    def test_dry_run_cli_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--dry-run", "--json"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["alias"], "qwen3.8-27b:axq")

    def test_live_preflight_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "wrongrev"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                mod._live_preflight(model_dir)

    def test_live_preflight_accepts_pinned_snapshot_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / mod.PRIMARY_REVISION
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            mod._live_preflight(model_dir)


if __name__ == "__main__":
    unittest.main()
