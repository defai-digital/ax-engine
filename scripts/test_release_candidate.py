from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.release_candidate import CandidateError, create_manifest, verify_manifest


TAG = "v6.9.0"
COMMIT = "a" * 40


class ReleaseCandidateTests(unittest.TestCase):
    def make_candidate(self, root: Path) -> dict[str, object]:
        bin_dir = root / "bin"
        wheel_dir = root / "wheel"
        bin_dir.mkdir()
        wheel_dir.mkdir()
        for name in ("ax-engine", "ax-engine-server", "ax-engine-bench"):
            (bin_dir / name).write_bytes(f"binary:{name}".encode())
        (wheel_dir / "ax_engine-6.9.0-cp310-abi3-macosx_26_0_arm64.whl").write_bytes(
            b"wheel"
        )
        return create_manifest(root, TAG, COMMIT)

    def test_create_and_verify_each_asset_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_candidate(root)

            binaries = verify_manifest(root, manifest, TAG, COMMIT, "binaries")
            wheel = verify_manifest(root, manifest, TAG, COMMIT, "wheel")
            all_assets = verify_manifest(root, manifest, TAG, COMMIT, "all")

            self.assertEqual(3, len(binaries))
            self.assertEqual(1, len(wheel))
            self.assertEqual(4, len(all_assets))

    def test_tampered_asset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_candidate(root)
            (root / "bin" / "ax-engine").write_bytes(b"tampered")

            with self.assertRaisesRegex(CandidateError, "mismatch"):
                verify_manifest(root, manifest, TAG, COMMIT, "binaries")

    def test_wrong_commit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_candidate(root)

            with self.assertRaisesRegex(CandidateError, "git_commit mismatch"):
                verify_manifest(root, manifest, TAG, "b" * 40, "all")

    def test_manifest_path_cannot_escape_candidate_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_candidate(root)
            binaries = manifest["binaries"]
            assert isinstance(binaries, dict)
            record = binaries["ax-engine"]
            assert isinstance(record, dict)
            record["path"] = "../outside"

            with self.assertRaisesRegex(CandidateError, "escapes root"):
                verify_manifest(root, manifest, TAG, COMMIT, "binaries")

    def test_manifest_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.make_candidate(Path(tmp))
            self.assertEqual(manifest, json.loads(json.dumps(manifest)))


if __name__ == "__main__":
    unittest.main()
