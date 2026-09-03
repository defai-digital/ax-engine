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
        self._write(
            "pyproject.toml",
            "\n".join(
                (
                    "[project]",
                    f'version = "{version}"',
                    'requires-python = ">=3.12"',
                    "classifiers = [",
                    '  "Programming Language :: Python :: 3.12",',
                    '  "Programming Language :: Python :: 3.13",',
                    "]",
                    "[tool.ruff]",
                    'target-version = "py312"',
                    "[tool.mypy]",
                    'python_version = "3.12"',
                    "",
                )
            ),
        )
        self._write(
            "crates/ax-engine-py/Cargo.toml",
            'pyo3 = { version = "0.29", features = ["abi3-py312"] }\n',
        )
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
        for path in (
            "README.md",
            "docs/GETTING-STARTED.md",
            "crates/ax-engine-py/README.md",
            "docs/sdk/python.md",
        ):
            self._write(
                path,
                f'python3 -m pip install "ax-engine[download]>={version},<2"\n',
            )
        self._write(
            "docs/sdk/swift.md",
            f'The current version is `{version}`.\n',
        )

    def test_accepts_aligned_package_and_install_versions(self):
        self.assertEqual(check_version_sync.verify_versions(self.root), "1.2.3")
        self.assertEqual(
            check_version_sync.verify_versions(self.root, "v1.2.3"),
            "1.2.3",
        )
        self.assertEqual(check_version_sync.verify_python_policy(self.root), "3.12")

    def test_rejects_misaligned_python_abi_floor(self):
        self._write(
            "crates/ax-engine-py/Cargo.toml",
            'pyo3 = { version = "0.29", features = ["abi3-py310"] }\n',
        )

        with self.assertRaisesRegex(
            check_version_sync.VersionSyncError,
            "abi3-py312",
        ):
            check_version_sync.verify_python_policy(self.root)

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

    def test_rejects_a_mismatched_install_version(self):
        self._write(
            "README.md",
            'python3 -m pip install "ax-engine[download]>=1.2.2,<2"\n',
        )

        with self.assertRaisesRegex(
            check_version_sync.VersionSyncError,
            "README.md=1.2.2",
        ):
            check_version_sync.verify_versions(self.root)

    def test_rejects_inconsistent_install_versions_in_one_guide(self):
        self._write(
            "crates/ax-engine-py/README.md",
            "\n".join(
                (
                    'pip install "ax-engine[download]>=1.2.3,<2"',
                    'pip install "ax-engine[openai]>=1.2.2,<2"',
                )
            ),
        )

        with self.assertRaisesRegex(
            check_version_sync.VersionSyncError,
            "inconsistent install versions in crates/ax-engine-py/README.md",
        ):
            check_version_sync.verify_versions(self.root)


if __name__ == "__main__":
    unittest.main()
