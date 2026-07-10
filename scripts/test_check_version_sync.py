import json
import pathlib
import tempfile
import unittest

from scripts import check_version_sync


class VersionSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self._write_versions("1.2.3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_versions(self, version: str) -> None:
        self._write("Cargo.toml", f'[workspace.package]\nversion = "{version}"\n')
        self._write("pyproject.toml", f'[project]\nversion = "{version}"\n')
        self._write(
            "sdk/javascript/package.json",
            json.dumps({"version": version}),
        )
        self._write(
            "sdk/ruby/lib/ax_engine/version.rb",
            f'AX_ENGINE_VERSION = "ignored"\nVERSION = "{version}"\n',
        )
        self._write(
            "sdk/go/axengine/client.go",
            f'package axengine\nconst Version = "{version}"\n',
        )
        self._write(
            "sdk/swift/Sources/AxEngine/AxEngineClient.swift",
            f'public static let version = "{version}"\n',
        )

    def test_accepts_all_six_aligned_versions(self):
        self.assertEqual(check_version_sync.verify_versions(self.root), "1.2.3")
        self.assertEqual(
            check_version_sync.verify_versions(self.root, "v1.2.3"),
            "1.2.3",
        )

    def test_rejects_a_mismatched_sdk_version(self):
        self._write(
            "sdk/go/axengine/client.go",
            'package axengine\nconst Version = "1.2.4"\n',
        )

        with self.assertRaisesRegex(
            check_version_sync.VersionSyncError,
            "sdk/go/axengine/client.go=1.2.4",
        ):
            check_version_sync.verify_versions(self.root)

    def test_rejects_an_unparseable_version_surface(self):
        self._write(
            "sdk/swift/Sources/AxEngine/AxEngineClient.swift",
            'public static let packageVersion = "1.2.3"\n',
        )

        with self.assertRaisesRegex(
            check_version_sync.VersionSyncError,
            "could not parse version from .*AxEngineClient.swift",
        ):
            check_version_sync.verify_versions(self.root)


if __name__ == "__main__":
    unittest.main()
