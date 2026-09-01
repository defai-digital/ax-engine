#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).with_name("download_model.py")
spec = importlib.util.spec_from_file_location("download_model", SCRIPT_PATH)
download_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_model)


# Minimal dense language-model roles the native readiness gate requires.
# Packed attention/FFN keeps fixture size small while still passing role checks.
_MINIMAL_READY_ROLES: list[tuple[str, int | None]] = [
    ("token_embedding", None),
    ("final_norm", None),
    ("attention_norm", 0),
    ("ffn_norm", 0),
    ("attention_qkv_packed", 0),
    ("attention_o", 0),
    ("ffn_gate_up_packed", 0),
    ("ffn_down", 0),
]


def write_safetensors(path: Path, payload: bytes | None = None) -> None:
    """Write a multi-tensor safetensors file matching ``write_manifest``.

    ``payload`` is ignored (kept for call-site compatibility). The fixture uses
    the smallest shapes accepted by the native runtime validator.
    """
    del payload  # historical API
    element_bytes = 4
    shapes = {
        "token_embedding": [1, 1],
        "final_norm": [1],
        "attention_norm": [1],
        "ffn_norm": [1],
        "attention_qkv_packed": [3, 1],
        "attention_o": [1, 1],
        "ffn_gate_up_packed": [2, 1],
        "ffn_down": [1, 1],
    }
    header: dict[str, object] = {}
    body = bytearray()
    for index, (role, _layer) in enumerate(_MINIMAL_READY_ROLES):
        name = f"t{index}_{role}"
        shape = shapes[role]
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        start = len(body)
        body.extend(b"\0" * (element_count * element_bytes))
        end = len(body)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start, end],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + bytes(body))


def write_manifest(path: Path) -> None:
    weights = path.parent / "model.safetensors"
    raw = weights.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    data_base_offset = 8 + header_size
    header = json.loads(raw[8 : 8 + header_size])
    tensors: list[dict[str, object]] = []
    for index, (role, layer_index) in enumerate(_MINIMAL_READY_ROLES):
        name = f"t{index}_{role}"
        entry = header[name]
        start, end = entry["data_offsets"]
        tensor: dict[str, object] = {
            "name": name,
            "role": role,
            "dtype": "f32",
            "shape": entry["shape"],
            "file": weights.name,
            "offset_bytes": data_base_offset + start,
            "length_bytes": end - start,
        }
        if layer_index is not None:
            tensor["layer_index"] = layer_index
        tensors.append(tensor)
    path.write_text(
        json.dumps(
            {
                "schema_version": download_model.NATIVE_MANIFEST_SCHEMA_VERSION,
                "model_family": "qwen3",
                "tensor_format": "safetensors",
                "layer_count": 1,
                "hidden_size": 1,
                "intermediate_size": 1,
                "attention_head_count": 1,
                "attention_head_dim": 1,
                "kv_head_count": 1,
                "vocab_size": 1,
                "tie_word_embeddings": True,
                "tensors": tensors,
            }
        )
    )


def write_provenance(path: Path, repo_id: str, revision: str | None = None) -> None:
    download_model._write_download_provenance(path, repo_id, revision)


class DownloadModelScriptTest(unittest.TestCase):
    def test_standalone_repo_parser_fallback_matches_packaged_contract(self) -> None:
        cases = {
            "owner/repo": ("owner/repo", None),
            "owner/repo.git@feature/downloads": (
                "owner/repo",
                "feature/downloads",
            ),
            "https://hf.co/owner/repo/tree/refs%2Fpr%2F123": (
                "owner/repo",
                "refs/pr/123",
            ),
            "https://hf.co:443/owner/repo/tree/v2?download=true#files": (
                "owner/repo",
                "v2",
            ),
            "https://hf.co/owner/repo?ignored=\tvalue": ("owner/repo", None),
            "\towner/repo\n": ("owner/repo", None),
        }
        with patch.object(download_model, "_load_repo_ref_module", return_value=None):
            for value, expected in cases.items():
                with self.subTest(value=value):
                    self.assertEqual(download_model._parse_repo_ref(value), expected)
            for invalid in (
                "owner/re--po",
                "owner/repo@../escape",
                "https://example.com/owner/repo",
                "owner/repo@bad%ZZ",
                "https://hf.co:/owner/repo",
                "https://hf.co:invalid/owner/repo",
                "https://hf.co:65536/owner/repo",
                "https://hf.co:443:extra/owner/repo",
                "https://hf.\nco/owner/repo",
                "https://hf.co/owner/re\tpo",
                "\0https://hf.co/owner/repo",
                "owner/repo@feature\u0080other",
                "\x1cowner/repo",
                "C:/owner/repo",
                r"owner\repo/model",
                r"owner/repo@C:\temp",
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    download_model._parse_repo_ref(invalid)

    def test_default_destination_uses_mlx_lm_cache_root(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--mlx-community--Qwen3-4B-4bit",
            )

        with patch.dict(os.environ, {"HF_HOME": "/tmp/hf-home"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-home/hub/models--mlx-community--Qwen3-4B-4bit"),
            )

        with patch.dict(os.environ, {"HF_HUB_CACHE": "/tmp/hf-hub"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/hf-hub/models--mlx-community--Qwen3-4B-4bit"),
            )

        with patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/xdg-cache"}, clear=True):
            self.assertEqual(
                download_model.default_mlx_lm_repo_cache_dir("mlx-community/Qwen3-4B-4bit"),
                Path("/tmp/xdg-cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit"),
            )

    def test_json_summary_for_existing_ready_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_manifest(model_dir / "model-manifest.json")
            write_provenance(model_dir, "mlx-community/Qwen3-4B-4bit")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "mlx-community/Qwen3-4B-4bit",
                    "--dest",
                    str(model_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            summary = json.loads(result.stdout)
            self.assertEqual(summary["schema_version"], "ax.download_model.v1")
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["dest"], str(model_dir))
            self.assertTrue(summary["manifest_present"])
            self.assertEqual(summary["safetensors_count"], 1)
            self.assertEqual(
                summary["server_command"],
                [
                    "ax-engine-server",
                    "--mlx",
                    "--mlx-model-artifacts-dir",
                    str(model_dir),
                    "--port",
                    "31418",
                ],
            )

    def test_progress_json_emits_events_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_manifest(model_dir / "model-manifest.json")
            write_provenance(model_dir, "mlx-community/Qwen3-4B-4bit")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "mlx-community/Qwen3-4B-4bit",
                    "--dest",
                    str(model_dir),
                    "--progress-json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0]["event"], "progress")
            self.assertEqual(lines[-1]["status"], "ready")

    def test_progress_json_failure_is_machine_readable_without_json_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            sentinel = model_dir / "keep.txt"
            sentinel.write_text("do not replace")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "owner/repo",
                    "--dest",
                    str(model_dir),
                    "--progress-json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0]["event"], "progress")
            self.assertEqual(lines[-1]["status"], "download_failed")
            self.assertEqual(sentinel.read_text(), "do not replace")

    def test_progress_json_argument_errors_emit_one_terminal_record(self) -> None:
        cases = [
            ["--progress-json"],
            ["owner/repo", "--progress-json", "--unknown"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT_PATH), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 1)
                terminal = json.loads(result.stdout)
                self.assertEqual(terminal["schema_version"], "ax.download_model.v1")
                self.assertEqual(terminal["status"], "download_failed")

    def test_progress_json_long_invalid_ref_still_emits_terminal_record(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "x" * 5000,
                "--progress-json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        terminal = json.loads(result.stdout)
        self.assertEqual(terminal["schema_version"], "ax.download_model.v1")
        self.assertEqual(terminal["status"], "download_failed")

    def test_json_summary_for_unrecognized_nonempty_dest_is_hermetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            sentinel = model_dir / "keep.txt"
            sentinel.write_text("do not replace")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "owner/repo",
                    "--dest",
                    str(model_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "download_failed")
            self.assertIn("not a matching", "\n".join(summary["errors"]))
            self.assertEqual(sentinel.read_text(), "do not replace")

    def test_download_reuses_existing_cache_snapshot(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--mlx-community--Qwen3-4B-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo_cache / "refs").mkdir()
            (repo_cache / "refs" / "main").write_text("abc123")
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")

            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                calls.append(model)
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [])
            self.assertEqual(resolved, snapshot)

    def test_local_only_reuses_exact_cached_revision_without_network(self) -> None:
        repo_id = "owner/repo"
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "hub" / "models--owner--repo" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download") as fetch,
            ):
                resolved = download_model.download(
                    repo_id,
                    None,
                    revision=revision,
                    local_only=True,
                    quiet=True,
                )

            self.assertEqual(resolved, snapshot)
            fetch.assert_not_called()

    def test_local_only_cache_miss_fails_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, {"HF_HOME": tmp}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download") as fetch,
                self.assertRaisesRegex(RuntimeError, "local-only forbids network downloads"),
            ):
                download_model.download(
                    "owner/repo",
                    None,
                    revision="b" * 40,
                    local_only=True,
                    quiet=True,
                )

            fetch.assert_not_called()

    def test_local_only_rejects_force(self) -> None:
        with self.assertRaisesRegex(ValueError, "force cannot be combined"):
            download_model.download(
                "owner/repo",
                None,
                force=True,
                local_only=True,
                quiet=True,
            )

    def test_download_forwards_revision_and_resolves_ref(self) -> None:
        repo_id = "owner/repo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--owner--repo"
            snapshot = repo_cache / "snapshots" / "def456"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")
            (repo_cache / "refs").mkdir(parents=True)
            (repo_cache / "refs" / "v2").write_text("def456")

            # Existing cache hit resolves the named ref without a download.
            with patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True):
                resolved = download_model.download(repo_id, None, revision="v2", quiet=True)
            self.assertEqual(resolved, snapshot)

            # A miss forwards the revision to snapshot_download.
            calls: list[tuple[str, str | None]] = []

            def fake_hf_download(
                model: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                calls.append((model, revision))
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                download_model.download(repo_id, None, revision="v3", quiet=True)

            self.assertEqual(calls, [(repo_id, "v3")])

    def test_download_parses_url_and_embedded_revision_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")
            calls: list[tuple[str, str | None]] = []

            def fake_hf_download(
                repo_id: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                calls.append((repo_id, revision))
                return snapshot

            with (
                patch.object(
                    download_model,
                    "_latest_mlx_lm_snapshot",
                    return_value=None,
                ),
                patch.object(
                    download_model,
                    "_run_hf_snapshot_download",
                    side_effect=fake_hf_download,
                ),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: None,
                ),
            ):
                resolved = download_model.download(
                    "https://huggingface.co/owner/repo/tree/v2",
                    None,
                    quiet=True,
                )

            self.assertEqual(resolved, snapshot)
            self.assertEqual(calls, [("owner/repo", "v2")])

    def test_dest_copy_is_atomic_and_repairs_partial_dest(self) -> None:
        repo_id = "owner/repo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            # Partial/corrupt contents from an interrupted older download.
            (dest / "model.safetensors").write_bytes(b"partial")
            write_provenance(dest, repo_id)
            snapshot = root / "snapshot"

            def fake_hf_download(
                model: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
                write_safetensors(snapshot / "model.safetensors", b"complete")
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, dest, quiet=True)

            self.assertEqual(resolved, dest)
            # The staged snapshot replaced the partial dest; payload markers are
            # ignored by write_safetensors, so assert structural readiness instead.
            self.assertGreater((dest / "model.safetensors").stat().st_size, len(b"partial"))
            self.assertIsNone(download_model._safetensors_file_error(dest / "model.safetensors"))
            self.assertTrue((dest / "config.json").is_file())
            # No temp/backup dirs survive the swap.
            leftovers = [p.name for p in root.iterdir() if p.name.startswith(".dest")]
            self.assertEqual(leftovers, [])

    def test_atomic_copy_restores_previous_dest_when_activation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")
            dest = root / "dest"
            dest.mkdir()
            sentinel = dest / "keep.txt"
            sentinel.write_text("old")
            (dest / "model-manifest.json").write_text("{}")

            real_rename = Path.rename

            def fail_activation(path: Path, target: Path) -> Path:
                if path.name.startswith(".dest.download-tmp-"):
                    raise OSError("injected activation failure")
                return real_rename(path, target)

            with (
                patch.object(Path, "rename", fail_activation),
                self.assertRaisesRegex(OSError, "injected activation failure"),
            ):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision=None,
                    force=True,
                )

            self.assertEqual(sentinel.read_text(), "old")
            leftovers = [path.name for path in root.iterdir() if path.name.startswith(".dest.")]
            self.assertEqual(leftovers, [])

    def test_atomic_copy_never_deletes_legacy_reserved_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")
            dest = root / "dest"
            legacy_tmp = root / ".dest.download-tmp"
            legacy_backup = root / ".dest.previous"
            legacy_tmp.mkdir()
            legacy_backup.mkdir()
            (legacy_tmp / "keep.txt").write_text("tmp")
            (legacy_backup / "keep.txt").write_text("backup")

            download_model._copy_snapshot_to_dest(
                snapshot,
                dest,
                repo_id="owner/repo",
                revision=None,
            )

            self.assertEqual((legacy_tmp / "keep.txt").read_text(), "tmp")
            self.assertEqual((legacy_backup / "keep.txt").read_text(), "backup")

    def test_atomic_copy_rejects_destination_created_while_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")
            dest = root / "dest"

            def create_competing_destination(_stage: Path) -> None:
                dest.mkdir()
                (dest / "important.txt").write_text("keep")

            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision=None,
                    prepare_destination=create_competing_destination,
                )

            self.assertEqual((dest / "important.txt").read_text(), "keep")
            leftovers = [
                path.name for path in root.iterdir() if path.name.startswith(".dest.")
            ]
            self.assertEqual(leftovers, [])

    def test_atomic_copy_rejects_directory_and_out_of_cache_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--owner--repo"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            outside_dir = root / "outside-dir"
            outside_dir.mkdir()
            (outside_dir / "secret.txt").write_text("secret")
            linked_dir = snapshot / "linked-dir"
            linked_dir.symlink_to(outside_dir, target_is_directory=True)
            dest = root / "dest-dir-link"

            with self.assertRaisesRegex(RuntimeError, "symlinked directory"):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision=None,
                )
            self.assertFalse(download_model._path_exists(dest))

            linked_dir.unlink()
            outside_file = root / "outside-secret.txt"
            outside_file.write_text("secret")
            (snapshot / "linked-file").symlink_to(outside_file)
            dest = root / "dest-file-link"
            with self.assertRaisesRegex(RuntimeError, "symlink outside"):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision=None,
                )
            self.assertFalse(download_model._path_exists(dest))

            (snapshot / "linked-file").unlink()
            fifo = snapshot / "named-pipe"
            os.mkfifo(fifo)
            dest = root / "dest-special-file"
            with self.assertRaisesRegex(RuntimeError, "special file"):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision=None,
                )
            self.assertFalse(download_model._path_exists(dest))

    def test_atomic_copy_materializes_canonical_hub_blob_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--owner--repo"
            blob = repo_cache / "blobs" / "weights-hash"
            blob.parent.mkdir(parents=True)
            write_safetensors(blob)
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").symlink_to(blob)
            dest = root / "dest"

            download_model._copy_snapshot_to_dest(
                snapshot,
                dest,
                repo_id="owner/repo",
                revision=None,
            )

            copied = dest / "model.safetensors"
            self.assertTrue(copied.is_file())
            self.assertFalse(copied.is_symlink())
            self.assertEqual(copied.read_bytes(), blob.read_bytes())

    def test_atomic_copy_rejects_link_to_different_cached_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--owner--repo"
            snapshot = repo_cache / "snapshots" / "requested-revision"
            other_snapshot = repo_cache / "snapshots" / "different-revision"
            snapshot.mkdir(parents=True)
            other_snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(other_snapshot / "model.safetensors")
            (snapshot / "model.safetensors").symlink_to(
                other_snapshot / "model.safetensors"
            )
            dest = root / "dest"

            with self.assertRaisesRegex(RuntimeError, "symlink outside"):
                download_model._copy_snapshot_to_dest(
                    snapshot,
                    dest,
                    repo_id="owner/repo",
                    revision="requested-revision",
                )

            self.assertFalse(download_model._path_exists(dest))

    def test_dest_none_rejects_out_of_cache_snapshot_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--owner--repo"
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            outside = root / "outside.safetensors"
            write_safetensors(outside)
            (snapshot / "model.safetensors").symlink_to(outside)

            for source in ("cache", "fresh"):
                with (
                    self.subTest(source=source),
                    patch.object(
                        download_model,
                        "_latest_mlx_lm_snapshot",
                        return_value=snapshot if source == "cache" else None,
                    ),
                    patch.object(
                        download_model,
                        "_run_hf_snapshot_download",
                        return_value=snapshot,
                    ) as fetch,
                    patch.object(
                        download_model,
                        "_total_repo_bytes",
                        return_value=None,
                    ),
                    self.assertRaisesRegex(RuntimeError, "symlink outside"),
                ):
                    download_model.download("owner/repo", None, quiet=True)
                if source == "cache":
                    fetch.assert_not_called()
                else:
                    fetch.assert_called_once()

    def test_atomic_copy_replaces_file_and_broken_symlink_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")

            for kind in ("file", "broken_symlink"):
                dest = root / kind
                if kind == "file":
                    dest.write_text("old")
                else:
                    dest.symlink_to(root / "missing-target")

                with self.subTest(kind=kind):
                    download_model._copy_snapshot_to_dest(
                        snapshot,
                        dest,
                        repo_id="owner/repo",
                        revision=None,
                        force=True,
                    )
                    self.assertTrue(dest.is_dir())
                    self.assertTrue((dest / "model.safetensors").is_file())
                    leftovers = [
                        path.name for path in root.iterdir() if path.name.startswith(f".{kind}.")
                    ]
                    self.assertEqual(leftovers, [])

    def test_invalid_fresh_snapshot_never_replaces_existing_dest(self) -> None:
        repo_id = "owner/repo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            sentinel = dest / "keep.txt"
            sentinel.write_text("old")
            (dest / "model-manifest.json").write_text('{"stale":true}')
            snapshot = root / "invalid-snapshot"
            snapshot.mkdir()
            write_safetensors(snapshot / "model.safetensors")

            with (
                patch.dict(os.environ, {"HF_HOME": str(root / "hf")}, clear=True),
                patch.object(
                    download_model,
                    "_run_hf_snapshot_download",
                    return_value=snapshot,
                ),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: None,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "downloaded snapshot.*invalid",
                ),
            ):
                download_model.download(
                    repo_id,
                    dest,
                    force=True,
                    quiet=True,
                )

            self.assertEqual(sentinel.read_text(), "old")

    def test_manifest_failure_never_replaces_existing_dest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            sentinel = dest / "keep.txt"
            sentinel.write_text("working model")
            (dest / "model-manifest.json").write_text('{"stale":true}')

            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")

            stdout = io.StringIO()
            argv = [
                "download_model.py",
                "owner/repo",
                "--dest",
                str(dest),
                "--force",
                "--json",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    download_model,
                    "_run_hf_snapshot_download",
                    return_value=snapshot,
                ),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: None,
                ),
                patch.object(download_model, "_try_generate_manifest", return_value=False),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "download_failed")
            self.assertIn("previous destination was preserved", "\n".join(summary["errors"]))
            self.assertEqual(sentinel.read_text(), "working model")
            self.assertFalse((dest / "model.safetensors").exists())

    def test_explicit_revision_cannot_reuse_different_dest_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "config.json").write_text("{}")
            write_safetensors(dest / "model.safetensors")
            write_manifest(dest / "model-manifest.json")
            write_provenance(dest, "owner/repo", "v1")

            with self.assertRaisesRegex(RuntimeError, "not a matching.*revision v2"):
                download_model.download(
                    "owner/repo",
                    dest,
                    revision="v2",
                    quiet=True,
                )

    def test_revision_path_escape_is_rejected_before_cache_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_model = Path(tmp) / "unrelated"
            local_model.mkdir()
            (local_model / "config.json").write_text("{}")
            write_safetensors(local_model / "model.safetensors")

            with self.assertRaises(ValueError):
                download_model.download(
                    "owner/repo",
                    None,
                    revision=str(local_model),
                    quiet=True,
                )

    def test_unsafe_broad_destinations_are_rejected_before_download_or_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "cwd"
            home = root / "home"
            cwd.mkdir()
            home.mkdir()
            cwd_sentinel = cwd / "keep.txt"
            home_sentinel = home / "keep.txt"
            root_sentinel = root / "keep.txt"
            cwd_sentinel.write_text("cwd")
            home_sentinel.write_text("home")
            root_sentinel.write_text("root")

            with (
                patch.object(download_model.Path, "cwd", return_value=cwd),
                patch.object(download_model.Path, "home", return_value=home),
                patch.object(download_model, "_latest_mlx_lm_snapshot") as latest,
                patch.object(download_model, "_run_hf_snapshot_download") as fetch,
                patch.object(download_model, "_preflight_disk_space") as preflight,
                patch.object(download_model, "_copy_snapshot_to_dest") as copy,
            ):
                for unsafe in (Path("/"), root, cwd, home, Path(".")):
                    with (
                        self.subTest(unsafe=unsafe),
                        self.assertRaisesRegex(RuntimeError, "unsafe destination"),
                    ):
                        download_model.download("owner/repo", unsafe, force=True, quiet=True)

            latest.assert_not_called()
            fetch.assert_not_called()
            preflight.assert_not_called()
            copy.assert_not_called()
            self.assertEqual(cwd_sentinel.read_text(), "cwd")
            self.assertEqual(home_sentinel.read_text(), "home")
            self.assertEqual(root_sentinel.read_text(), "root")

    def test_force_rejects_unrelated_nonempty_directory_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "unrelated"
            dest.mkdir()
            sentinel = dest / "keep.txt"
            sentinel.write_text("keep")
            (dest / "looks-like-model.safetensors").mkdir()

            with (
                patch.object(download_model, "_latest_mlx_lm_snapshot") as latest,
                patch.object(download_model, "_run_hf_snapshot_download") as fetch,
                patch.object(download_model, "_preflight_disk_space") as preflight,
                patch.object(download_model, "_copy_snapshot_to_dest") as copy,
                self.assertRaisesRegex(RuntimeError, "unrelated non-empty directory"),
            ):
                download_model.download(
                    "owner/repo",
                    dest,
                    force=True,
                    quiet=True,
                )

            latest.assert_not_called()
            fetch.assert_not_called()
            preflight.assert_not_called()
            copy.assert_not_called()
            self.assertEqual(sentinel.read_text(), "keep")

    def test_snapshot_destination_overlap_is_rejected_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            write_safetensors(snapshot / "model.safetensors")

            with (
                patch.object(
                    download_model,
                    "_latest_mlx_lm_snapshot",
                    return_value=snapshot,
                ),
                patch.object(download_model, "_preflight_disk_space") as preflight,
                patch.object(download_model, "_copy_snapshot_to_dest") as copy,
                self.assertRaisesRegex(RuntimeError, "overlaps source snapshot"),
            ):
                download_model.download(
                    "owner/repo",
                    snapshot / "nested-dest",
                    quiet=True,
                )

            preflight.assert_not_called()
            copy.assert_not_called()

            containing_dest = root / "containing-dest"
            containing_snapshot = containing_dest / "snapshot"
            containing_snapshot.mkdir(parents=True)
            (containing_snapshot / "config.json").write_text("{}")
            write_safetensors(containing_snapshot / "model.safetensors")
            write_provenance(containing_dest, "owner/repo")
            with (
                patch.object(
                    download_model,
                    "_latest_mlx_lm_snapshot",
                    return_value=containing_snapshot,
                ),
                patch.object(download_model, "_preflight_disk_space") as preflight,
                patch.object(download_model, "_copy_snapshot_to_dest") as copy,
                self.assertRaisesRegex(RuntimeError, "overlaps source snapshot"),
            ):
                download_model.download("owner/repo", containing_dest, quiet=True)

            preflight.assert_not_called()
            copy.assert_not_called()

            write_provenance(snapshot, "owner/repo")
            with (
                patch.object(
                    download_model,
                    "_run_hf_snapshot_download",
                    return_value=snapshot,
                ),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    return_value=None,
                ),
                patch.object(download_model, "_copy_snapshot_to_dest") as copy,
                self.assertRaisesRegex(RuntimeError, "overlaps source snapshot"),
            ):
                download_model.download(
                    "owner/repo",
                    snapshot,
                    force=True,
                    quiet=True,
                )
            copy.assert_not_called()

    def test_latest_snapshot_ignores_symlinked_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--owner--repo"
            snapshots = repo_cache / "snapshots"
            snapshots.mkdir(parents=True)
            escaped = root / "outside"
            escaped.mkdir()
            (snapshots / "malicious").symlink_to(escaped, target_is_directory=True)

            with patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True):
                self.assertIsNone(download_model._latest_mlx_lm_snapshot("owner/repo"))

    def test_disk_preflight_aborts_when_download_does_not_fit(self) -> None:
        repo_id = "owner/repo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: 10**15,
                ),
                self.assertRaisesRegex(RuntimeError, "insufficient disk space"),
            ):
                download_model.download(repo_id, None, quiet=True)

    def test_disk_preflight_aggregates_cache_and_dest_on_same_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            usage = shutil.disk_usage(root)
            constrained_usage = type(usage)(usage.total, usage.used, 150)
            with (
                patch.dict(os.environ, {"HF_HOME": str(root / "hf")}, clear=True),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: 100,
                ),
                patch.object(
                    download_model.shutil,
                    "disk_usage",
                    return_value=constrained_usage,
                ),
                self.assertRaisesRegex(RuntimeError, "needs.*210 B"),
            ):
                download_model._preflight_disk_space(
                    "owner/repo",
                    dest,
                    revision="v2",
                )

    def test_download_uses_huggingface_hub_snapshot_download(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "hub" / "models--mlx-community--Qwen3-4B-4bit"
            snapshot = repo_cache / "snapshots" / "abc123"
            (repo_cache / "refs").mkdir(parents=True)
            (repo_cache / "refs" / "main").write_text("abc123")

            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                calls.append(model)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}")
                write_safetensors(snapshot / "model.safetensors")
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [repo_id])
            self.assertEqual(resolved, snapshot)

    def test_gemma4_unified_download_does_not_invoke_mlx_lm_generation(self) -> None:
        repo_id = "mlx-community/gemma-4-12B-it-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            calls: list[str] = []

            def fake_hf_download(
                model: str,
                *,
                revision: str | None = None,
                force_download: bool = False,
                quiet: bool = False,
                progress_json: bool = False,
                progress_bar: bool = False,
                total_bytes: int | None = None,
            ) -> Path:
                calls.append(model)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text('{"model_type":"gemma4_unified"}')
                write_safetensors(snapshot / "model.safetensors")
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": tmp}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(calls, [repo_id])
            self.assertEqual(resolved, snapshot)

    def test_force_refreshes_dest_without_deleting_other_cached_revisions(self) -> None:
        repo_id = "mlx-community/Qwen3-4B-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached_revision = (
                root
                / "hub"
                / "models--mlx-community--Qwen3-4B-4bit"
                / "snapshots"
                / "keep-me"
                / "cached.bin"
            )
            cached_revision.parent.mkdir(parents=True)
            cached_revision.write_bytes(b"still referenced")
            dest = root / "dest"
            dest.mkdir()
            # Artifacts left from a prior (possibly different) model.
            (dest / "model-manifest.json").write_text('{"stale":true}')
            write_safetensors(dest / "old.safetensors", b"old!")

            snapshot = root / "snapshot"

            def fake_hf_download(
                model,
                *,
                revision=None,
                force_download=False,
                quiet=False,
                progress_json=False,
                progress_bar=False,
                total_bytes=None,
            ):
                self.assertTrue(force_download)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
                write_safetensors(snapshot / "model.safetensors", b"new!")
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, dest, force=True, quiet=True)

            self.assertEqual(resolved, dest)
            # Stale manifest is dropped so main() regenerates it against the new weights.
            self.assertFalse((dest / "model-manifest.json").exists())
            self.assertTrue((dest / "model.safetensors").exists())
            self.assertEqual(cached_revision.read_bytes(), b"still referenced")

    def test_force_preserves_manifest_shipped_by_snapshot(self) -> None:
        repo_id = "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "dest"
            dest.mkdir()
            (dest / "model-manifest.json").write_text('{"stale":true}')
            snapshot = root / "snapshot"

            def fake_hf_download(
                model,
                *,
                revision=None,
                force_download=False,
                quiet=False,
                progress_json=False,
                progress_bar=False,
                total_bytes=None,
            ):
                self.assertEqual(model, repo_id)
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text('{"model_type":"qwen3_5"}')
                write_safetensors(snapshot / "model.safetensors", b"new!")
                write_manifest(snapshot / "model-manifest.json")
                return snapshot

            with (
                patch.dict(os.environ, {"HF_HOME": str(root)}, clear=True),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model, "_total_repo_bytes", lambda _repo, _revision=None: None
                ),
            ):
                resolved = download_model.download(repo_id, dest, force=True, quiet=True)

            self.assertEqual(resolved, dest)
            manifest = json.loads((dest / "model-manifest.json").read_text())
            self.assertEqual(
                manifest["schema_version"],
                download_model.NATIVE_MANIFEST_SCHEMA_VERSION,
            )

    def test_qwen_visual_manifest_is_marked_for_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5_moe", "vision_config": {}})
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "language_model.model.embed_tokens.weight": "model.safetensors",
                            "vision_tower.patch_embed.proj.weight": "model.safetensors",
                        }
                    }
                )
            )
            (model_dir / "model-manifest.json").write_text(
                json.dumps({"tensors": [{"name": "language_model.model.embed_tokens.weight"}]})
            )

            self.assertTrue(download_model.manifest_needs_media_rebuild(model_dir))

    def test_gemma_visual_manifest_requires_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "gemma4", "vision_config": {}})
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "vision_tower.patch_embedder.input_proj.weight": "model.safetensors",
                            "embed_vision.embedding_projection.weight": "model.safetensors",
                        }
                    }
                )
            )
            (model_dir / "model-manifest.json").write_text(
                json.dumps({"tensors": [{"name": "vision_tower.patch_embedder.input_proj.weight"}]})
            )

            self.assertTrue(download_model.manifest_needs_media_rebuild(model_dir))

    def test_gemma_unified_visual_manifest_is_marked_for_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "gemma4_unified", "vision_config": {}})
            )
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "language_model.model.embed_tokens.weight": "model.safetensors",
                            "vision_embedder.patch_dense.weight": "optiq/optiq_vision.safetensors",
                            "embed_vision.embedding_projection.weight": (
                                "optiq/optiq_vision.safetensors"
                            ),
                        }
                    }
                )
            )
            (model_dir / "model-manifest.json").write_text(
                json.dumps({"tensors": [{"name": "language_model.model.embed_tokens.weight"}]})
            )

            self.assertTrue(download_model.manifest_needs_media_rebuild(model_dir))

    def test_embedding_repos_use_standard_download_flow(self) -> None:
        repo_id = "AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"

            def fake_hf_download(
                model,
                *,
                revision=None,
                force_download=False,
                quiet=False,
                progress_json=False,
                progress_bar=False,
                total_bytes=None,
            ):
                self.assertEqual(model, repo_id)
                snapshot.mkdir()
                (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
                write_safetensors(snapshot / "model.safetensors")
                return snapshot

            with (
                # Hermetic: the developer machine may have this AutomatosX
                # repo in its real HF cache; force the fetch path.
                patch.object(
                    download_model, "_latest_mlx_lm_snapshot", lambda repo, revision=None: None
                ),
                patch.object(download_model, "_run_hf_snapshot_download", fake_hf_download),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    lambda _repo, _revision=None: None,
                ),
            ):
                resolved = download_model.download(repo_id, None, quiet=True)

            self.assertEqual(resolved, snapshot)

    def test_manifest_generation_uses_local_release_binary_before_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            local_bin = root / "target" / "release" / "generate-manifest"
            local_bin.parent.mkdir(parents=True)
            local_bin.write_text("#!/bin/sh\n")
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(download_model, "REPO_ROOT", root),
                patch.object(
                    download_model.shutil,
                    "which",
                    side_effect=lambda name: "cargo" if name == "cargo" else None,
                ),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            self.assertEqual(calls, [[str(local_bin), "--validate", str(model_dir)]])

    def test_manifest_generation_falls_back_after_installed_bench_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            local_bin = root / "target" / "debug" / "generate-manifest"
            local_bin.parent.mkdir(parents=True)
            local_bin.write_text("#!/bin/sh\n")
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if command[0] == "ax-engine-bench":
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(download_model, "REPO_ROOT", root),
                patch.object(
                    download_model.shutil,
                    "which",
                    side_effect=lambda name: (
                        "/usr/bin/ax-engine-bench" if name == "ax-engine-bench" else None
                    ),
                ),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            self.assertEqual(
                calls,
                [
                    ["ax-engine-bench", "generate-manifest", "--validate", str(model_dir)],
                    [str(local_bin), "--validate", str(model_dir)],
                ],
            )

    def test_manifest_generation_prefers_bundled_binary_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            bundled = "/wheel/ax_engine/_bin/ax-engine-bench"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(download_model, "_bundled_bench_bin", return_value=bundled),
                patch.object(
                    download_model.shutil,
                    "which",
                    side_effect=lambda name: "/usr/bin/ax-engine-bench",
                ),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(download_model._try_generate_manifest(model_dir, quiet=True))

            # The bundled binary is used; the stale PATH binary is never invoked.
            self.assertEqual(
                calls, [[bundled, "generate-manifest", "--validate", str(model_dir)]]
            )

    def test_manifest_validation_is_read_only_and_uses_native_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            manifest_blob = Path(tmp) / "manifest-blob"
            manifest_blob.write_text("{}")
            manifest_path = model_dir / "model-manifest.json"
            manifest_path.symlink_to(manifest_blob)
            bundled = "/wheel/ax_engine/_bin/ax-engine-bench"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                self.assertTrue(manifest_path.is_symlink())
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch.object(download_model, "_bundled_bench_bin", return_value=bundled),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(download_model._try_validate_manifest(model_dir, quiet=True))

            self.assertEqual(
                calls,
                [[bundled, "generate-manifest", "--validate", str(model_dir)]],
            )
            self.assertTrue(manifest_path.is_symlink())
            self.assertEqual(manifest_blob.read_text(), "{}")

    def test_manifest_validation_prefers_source_workspace_over_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text("[workspace]\n")
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model-manifest.json").write_text("{}")
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                self.assertEqual(kwargs["cwd"], str(root))
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with (
                patch.object(download_model, "REPO_ROOT", root),
                patch.object(download_model, "_bundled_bench_bin", return_value=None),
                patch.object(
                    download_model.shutil,
                    "which",
                    side_effect=lambda name: f"/usr/bin/{name}",
                ),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(download_model._try_validate_manifest(model_dir, quiet=True))

            self.assertEqual(
                calls[0][0:8],
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "ax-engine-core",
                    "--bin",
                    "generate-manifest",
                    "--",
                ],
            )

    def test_manifest_generation_absolutizes_option_like_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = "/wheel/ax_engine/_bin/ax-engine-bench"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(download_model.os, "getcwd", return_value=str(root)),
                patch.object(download_model, "_bundled_bench_bin", return_value=bundled),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(
                    download_model._try_generate_manifest(Path("-models"), quiet=True)
                )

            self.assertEqual(
                calls,
                [[bundled, "generate-manifest", "--validate", str(root / "-models")]],
            )

    def test_manifest_command_launch_failure_allows_fallback(self) -> None:
        with patch.object(
            download_model.subprocess,
            "run",
            side_effect=OSError("not executable"),
        ):
            self.assertFalse(
                download_model._run_manifest_command(
                    ["/broken/ax-engine-bench", "generate-manifest", "/tmp/model"],
                    quiet=True,
                    label="broken generator",
                )
            )

    def test_manifest_regeneration_detaches_hub_blob_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_cache = root / "models--owner--repo"
            manifest_blob = repo_cache / "blobs" / "manifest-hash"
            manifest_blob.parent.mkdir(parents=True)
            original_blob = b'{"stale":true}'
            manifest_blob.write_bytes(original_blob)
            snapshot = repo_cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            write_safetensors(snapshot / "model.safetensors")
            manifest_path = snapshot / "model-manifest.json"
            manifest_path.symlink_to(manifest_blob)

            def fake_run(command, **kwargs):
                self.assertFalse(manifest_path.is_symlink())
                write_manifest(manifest_path)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(
                    download_model,
                    "_bundled_bench_bin",
                    return_value="/wheel/ax-engine-bench",
                ),
                patch.object(download_model.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(
                    download_model._try_generate_manifest(
                        snapshot,
                        quiet=True,
                        force=True,
                    )
                )

            self.assertEqual(manifest_blob.read_bytes(), original_blob)
            self.assertTrue(manifest_path.is_file())
            self.assertFalse(manifest_path.is_symlink())

    def test_manifest_generation_force_replaces_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            bundled = "/wheel/ax_engine/_bin/ax-engine-bench"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with (
                patch.object(download_model, "_bundled_bench_bin", return_value=bundled),
                patch.object(download_model.subprocess, "run", fake_run),
            ):
                self.assertTrue(
                    download_model._try_generate_manifest(model_dir, quiet=True, force=True)
                )

            self.assertEqual(
                calls,
                [
                    [
                        bundled,
                        "generate-manifest",
                        "--force",
                        "--validate",
                        str(model_dir),
                    ]
                ],
            )

    def test_validation_rejects_missing_shards_declared_by_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model-00001-of-00002.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model-00001-of-00002.safetensors",
                            "model.layers.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )

            errors = download_model._validation_errors(model_dir)

            self.assertEqual(
                errors,
                [f"missing safetensors shard model-00002-of-00002.safetensors in {model_dir}"],
            )

    def test_safetensors_files_include_nested_optiq_and_skip_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            write_safetensors(model_dir / "model.safetensors")
            (model_dir / "optiq").mkdir()
            write_safetensors(model_dir / "optiq" / "optiq_vision.safetensors")
            (model_dir / "assistant").mkdir()
            write_safetensors(model_dir / "assistant" / "model.safetensors")

            relative = [
                path.relative_to(model_dir).as_posix()
                for path in download_model._safetensors_files(model_dir)
            ]
            self.assertEqual(
                relative,
                ["model.safetensors", "optiq/optiq_vision.safetensors"],
            )

    def test_validation_ignores_stale_index_for_differently_sharded_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"mistral3"}')
            write_safetensors(model_dir / "model-00001-of-00003.safetensors")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"model.layers.0.weight": "model-00001-of-00010.safetensors"}}
                )
            )

            self.assertEqual(download_model._validation_errors(model_dir), [])

    def test_validation_rejects_truncated_safetensors_and_malformed_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{not json")
            (model_dir / "model.safetensors").write_bytes(b"\0")

            errors = "\n".join(download_model._validation_errors(model_dir))

            self.assertIn("truncated safetensors header", errors)
            self.assertIn("unable to read config.json", errors)

    def test_main_rebuilds_corrupt_manifest_before_reporting_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            (model_dir / "model-manifest.json").write_text("{not json")
            write_provenance(model_dir, "owner/repo")
            force_values: list[bool] = []

            def fake_generate(
                dest: Path,
                *,
                quiet: bool = False,
                force: bool = False,
            ) -> bool:
                force_values.append(force)
                write_manifest(dest / "model-manifest.json")
                return True

            argv = [
                "download_model.py",
                "owner/repo",
                "--dest",
                str(model_dir),
                "--json",
            ]
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    download_model,
                    "_try_generate_manifest",
                    side_effect=fake_generate,
                ),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 0)
            self.assertEqual(force_values, [True])
            self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

    def test_main_rejects_blocked_manifest_and_regenerates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_manifest(model_dir / "model-manifest.json")
            manifest_path = model_dir / "model-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["runtime_status"] = {
                "ready": False,
                "blockers": ["unsupported projection"],
            }
            manifest_path.write_text(json.dumps(manifest))
            write_provenance(model_dir, "owner/repo")
            force_values: list[bool] = []

            def fake_generate(
                dest: Path,
                *,
                quiet: bool = False,
                force: bool = False,
            ) -> bool:
                force_values.append(force)
                write_manifest(dest / "model-manifest.json")
                return True

            stdout = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "download_model.py",
                        "owner/repo",
                        "--dest",
                        str(model_dir),
                        "--json",
                    ],
                ),
                patch.object(
                    download_model,
                    "_try_generate_manifest",
                    side_effect=fake_generate,
                ),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 0)
            self.assertEqual(force_values, [True])
            self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

    def test_main_does_not_reuse_manifest_rejected_by_native_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_manifest(model_dir / "model-manifest.json")
            manifest_path = model_dir / "model-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            qkv = next(
                tensor
                for tensor in manifest["tensors"]
                if tensor["role"] == "attention_qkv_packed"
            )
            qkv["role"] = "attention_qa"
            manifest_path.write_text(json.dumps(manifest))
            write_provenance(model_dir, "owner/repo")

            # The Python approximation accepts this partial MLA role, while
            # NativeModelArtifacts rejects MLA tensors for the qwen3 family.
            self.assertFalse(download_model._manifest_needs_rebuild(model_dir))
            self.assertFalse(download_model._try_validate_manifest(model_dir, quiet=True))

            stdout = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "download_model.py",
                        "owner/repo",
                        "--dest",
                        str(model_dir),
                        "--json",
                    ],
                ),
                patch.object(
                    download_model,
                    "_try_validate_manifest",
                    return_value=False,
                ) as validate,
                patch.object(
                    download_model,
                    "_try_generate_manifest",
                    return_value=False,
                ) as generate,
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "manifest_missing")
            validate.assert_called_once_with(model_dir, quiet=True)
            generate.assert_called_once_with(model_dir, quiet=True, force=True)

    def test_main_does_not_report_ready_when_generator_writes_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_provenance(model_dir, "owner/repo")
            stdout = io.StringIO()

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "download_model.py",
                        "owner/repo",
                        "--dest",
                        str(model_dir),
                        "--json",
                    ],
                ),
                patch.object(download_model, "_try_generate_manifest", return_value=True),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "manifest_missing")
            self.assertIn("still invalid", "\n".join(summary["errors"]))

    def test_force_staged_manifest_is_generated_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(snapshot / "model.safetensors")
            dest = root / "dest"
            generated: list[Path] = []
            stdout = io.StringIO()

            def fake_generate(
                candidate: Path,
                *,
                quiet: bool = False,
                force: bool = False,
            ) -> bool:
                generated.append(candidate)
                write_manifest(candidate / "model-manifest.json")
                return True

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "download_model.py",
                        "owner/repo",
                        "--dest",
                        str(dest),
                        "--force",
                        "--json",
                    ],
                ),
                patch.object(
                    download_model,
                    "_run_hf_snapshot_download",
                    return_value=snapshot,
                ),
                patch.object(
                    download_model,
                    "_total_repo_bytes",
                    return_value=None,
                ),
                patch.object(
                    download_model,
                    "_try_generate_manifest",
                    side_effect=fake_generate,
                ),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 0)
            self.assertEqual(len(generated), 1)
            self.assertTrue((dest / "model-manifest.json").is_file())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

    def test_main_serializes_filesystem_failure_as_download_summary(self) -> None:
        argv = ["download_model.py", "owner/repo", "--json"]
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.object(download_model, "download", side_effect=OSError("disk full")),
            redirect_stdout(stdout),
        ):
            code = download_model.main()

        self.assertEqual(code, 1)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "download_failed")
        self.assertIn("disk full", "\n".join(summary["errors"]))

    def test_main_returns_nonzero_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text('{"model_type":"qwen3"}')
            write_safetensors(model_dir / "model.safetensors")
            write_provenance(model_dir, "mlx-community/Qwen3-4B-4bit")

            argv = [
                "download_model.py",
                "mlx-community/Qwen3-4B-4bit",
                "--dest",
                str(model_dir),
                "--json",
            ]
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_model, "_try_generate_manifest", return_value=False),
                redirect_stdout(stdout),
            ):
                code = download_model.main()

            self.assertEqual(code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "manifest_missing")
            self.assertFalse(summary["manifest_present"])


class ManifestHeaderBindingTest(unittest.TestCase):
    @staticmethod
    def _write_bound_fixture(
        model_dir: Path,
        *,
        source_dtype: str = "F32",
        manifest_dtype: str = "f32",
        source_shape: list[int] | None = None,
        source_length: int | None = None,
        include_unrelated: bool = False,
        overlap_unrelated: bool = False,
        include_rank_zero_other: bool = False,
    ) -> dict[str, object]:
        dtype_widths = {
            "F16": 2,
            "BF16": 2,
            "F32": 4,
            "I8": 1,
            "U8": 1,
            "U32": 4,
            "I64": 8,
        }
        weight_shape = [1] if source_shape is None else source_shape
        weight_length = dtype_widths[source_dtype] if source_length is None else source_length
        # Primary binding tensor first so mutation tests still target tensors[0].
        entries: list[tuple[str, str, list[int], bytes, str, int | None]] = [
            ("weight", source_dtype, weight_shape, bytes(weight_length), "token_embedding", None)
        ]
        for role, layer_index in _MINIMAL_READY_ROLES[1:]:
            name = f"{role}_w"
            entries.append(
                (
                    name,
                    source_dtype,
                    [1],
                    bytes(dtype_widths[source_dtype]),
                    role,
                    layer_index,
                )
            )
        if include_rank_zero_other:
            # Rank-0 scalars are only legal for role=other (extension tensors).
            # Empty product still occupies one element of storage in safetensors.
            entries.append(
                (
                    "scalar_other",
                    source_dtype,
                    [],
                    bytes(dtype_widths[source_dtype]),
                    "other",
                    None,
                )
            )
        if include_unrelated:
            entries.extend(
                [
                    (
                        "unused_float",
                        "F16",
                        [1],
                        bytes(dtype_widths["F16"]),
                        "__unrelated__",
                        None,
                    ),
                    (
                        "unused_counter",
                        "I64",
                        [1],
                        bytes(dtype_widths["I64"]),
                        "__unrelated__",
                        None,
                    ),
                ]
            )

        payload = bytearray()
        header: dict[str, object] = {}
        relative_offsets: dict[str, tuple[int, int]] = {}
        for name, dtype, shape, tensor_bytes, _role, _layer in entries:
            start = len(payload)
            payload.extend(tensor_bytes)
            end = len(payload)
            relative_offsets[name] = (start, end)
            overlaps_weight = overlap_unrelated and name == "unused_float"
            declared_start = start - 1 if overlaps_weight else start
            declared_end = end - 1 if overlaps_weight else end
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [declared_start, declared_end],
            }

        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        header_bytes += b" " * ((-len(header_bytes)) % 8)
        weights = model_dir / "model.safetensors"
        weights.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + bytes(payload))

        data_base_offset = 8 + len(header_bytes)
        tensors: list[dict[str, object]] = []
        for name, _dtype, shape, _tensor_bytes, role, layer_index in entries:
            if role == "__unrelated__":
                continue
            start, end = relative_offsets[name]
            tensor: dict[str, object] = {
                "name": name,
                "role": role,
                "dtype": manifest_dtype,
                "shape": shape,
                "file": weights.name,
                "offset_bytes": data_base_offset + start,
                "length_bytes": end - start,
            }
            if layer_index is not None:
                tensor["layer_index"] = layer_index
            tensors.append(tensor)
        manifest: dict[str, object] = {
            "schema_version": download_model.NATIVE_MANIFEST_SCHEMA_VERSION,
            "model_family": "qwen3",
            "tensor_format": "safetensors",
            "layer_count": 1,
            "hidden_size": 4,
            "attention_head_count": 1,
            "attention_head_dim": 4,
            "kv_head_count": 1,
            "vocab_size": 1,
            "tie_word_embeddings": True,
            "tensors": tensors,
        }
        (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_manifest_tensor_binding_matches_every_source_field_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)
            tensor = manifest["tensors"][0]
            mutations = {
                "name": "other_weight",
                "file": "other.safetensors",
                "dtype": "f16",
                "shape": [2],
                "offset_bytes": tensor["offset_bytes"] + 1,
                "length_bytes": tensor["length_bytes"] + 1,
            }

            self.assertFalse(download_model._manifest_needs_rebuild(model_dir))
            for field, value in mutations.items():
                with self.subTest(field=field):
                    mutated = json.loads(json.dumps(manifest))
                    mutated["tensors"][0][field] = value
                    (model_dir / "model-manifest.json").write_text(json.dumps(mutated))
                    self.assertTrue(download_model._manifest_needs_rebuild(model_dir))

    def test_manifest_rejects_duplicate_declared_source_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)
            duplicate = json.loads(json.dumps(manifest["tensors"][0]))
            duplicate["role"] = "lm_head"
            manifest["tensors"].append(duplicate)
            (model_dir / "model-manifest.json").write_text(json.dumps(manifest))

            self.assertTrue(download_model._manifest_needs_rebuild(model_dir))

    def test_manifest_allows_rank_zero_other_tensor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir, include_rank_zero_other=True)
            self.assertFalse(download_model._manifest_needs_rebuild(model_dir))

            # Language roles must stay rank-positive.
            for tensor in manifest["tensors"]:
                if tensor["role"] == "other":
                    tensor["role"] = "token_embedding"
                    break
            (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(download_model._manifest_needs_rebuild(model_dir))

    def test_manifest_rejects_token_embedding_only_as_incomplete(self) -> None:
        """P1: structural binding alone must not mark a role-incomplete manifest ready."""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)
            self.assertFalse(download_model._manifest_needs_rebuild(model_dir))

            # Drop every role except token_embedding; bindings stay exact.
            manifest["tensors"] = [
                tensor
                for tensor in manifest["tensors"]
                if tensor["role"] == "token_embedding"
            ]
            (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(download_model._manifest_needs_rebuild(model_dir))
            reason = download_model._manifest_missing_required_roles(manifest)
            self.assertIsNotNone(reason)
            self.assertIn("final_norm", reason)

    def test_manifest_may_omit_unrelated_source_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self._write_bound_fixture(model_dir, include_unrelated=True)

            self.assertFalse(download_model._manifest_needs_rebuild(model_dir))

    def test_manifest_readiness_allows_value_from_key_attention_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)
            packed = next(
                tensor
                for tensor in manifest["tensors"]
                if tensor["role"] == "attention_qkv_packed"
            )
            packed["role"] = "attention_q"
            manifest["tensors"].append({**packed, "role": "attention_k"})
            manifest["attention_value_from_key_layers"] = [0]

            self.assertIsNone(download_model._manifest_missing_required_roles(manifest))

    def test_manifest_readiness_rejects_invalid_value_from_key_layer_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)

            for layer_indices in ([-1], [1], [True], ["0"]):
                with self.subTest(layer_indices=layer_indices):
                    manifest["attention_value_from_key_layers"] = layer_indices
                    reason = download_model._manifest_missing_required_roles(manifest)
                    self.assertEqual(reason, "invalid attention_value_from_key_layers")

    def test_manifest_readiness_rejects_conflicting_value_from_key_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            manifest = self._write_bound_fixture(model_dir)
            manifest["attention_value_from_key_layers"] = [0]

            reason = download_model._manifest_missing_required_roles(manifest)
            self.assertIn("attention_qkv_packed", reason or "")

            packed = next(
                tensor
                for tensor in manifest["tensors"]
                if tensor["role"] == "attention_qkv_packed"
            )
            packed["role"] = "attention_q"
            manifest["tensors"].append({**packed, "role": "attention_k"})
            manifest["tensors"].append({**packed, "role": "attention_v"})

            reason = download_model._manifest_missing_required_roles(manifest)
            self.assertIn("without attention_v", reason or "")

    def test_supported_safetensors_dtypes_bind_to_manifest_names(self) -> None:
        supported_dtypes = {
            "F16": "f16",
            "BF16": "bf16",
            "F32": "f32",
            "I8": "i8",
            "U8": "u8",
            "U32": "u32",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source_dtype, manifest_dtype in supported_dtypes.items():
                with self.subTest(source_dtype=source_dtype):
                    model_dir = root / source_dtype.lower()
                    model_dir.mkdir()
                    self._write_bound_fixture(
                        model_dir,
                        source_dtype=source_dtype,
                        manifest_dtype=manifest_dtype,
                    )
                    self.assertFalse(download_model._manifest_needs_rebuild(model_dir))

    def test_manifest_binding_fails_closed_on_unreadable_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self._write_bound_fixture(model_dir)
            (model_dir / "model.safetensors").write_bytes(b"\0")

            self.assertTrue(download_model._manifest_needs_rebuild(model_dir))

    def test_supported_tensor_rejects_inconsistent_shape_and_length(self) -> None:
        cases = {
            "wrong_length": ([1], 8),
            "shape_exceeds_payload": ([100], 4),
            "zero_dimension_with_data": ([0, 10**100], 4),
            "huge_product": ([sys.maxsize, sys.maxsize], 4),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, (shape, length) in cases.items():
                with self.subTest(label=label):
                    model_dir = root / label
                    model_dir.mkdir()
                    self._write_bound_fixture(
                        model_dir,
                        source_shape=shape,
                        source_length=length,
                    )

                    self.assertIsNotNone(
                        download_model._safetensors_file_error(model_dir / "model.safetensors")
                    )
                    self.assertTrue(download_model._manifest_needs_rebuild(model_dir))

    def test_overlapping_source_ranges_invalidate_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self._write_bound_fixture(
                model_dir,
                include_unrelated=True,
                overlap_unrelated=True,
            )

            error = download_model._safetensors_file_error(model_dir / "model.safetensors")
            self.assertIsNotNone(error)
            self.assertIn("overlapping tensor data ranges", error)
            self.assertTrue(download_model._manifest_needs_rebuild(model_dir))


if __name__ == "__main__":
    unittest.main()
