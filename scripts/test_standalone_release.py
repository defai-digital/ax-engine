"""Integration tests for the relocatable macOS standalone release layout."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_RUNTIME = ROOT / "scripts" / "prepare-mlx-release-runtime.sh"
PREPARE_STANDALONE = ROOT / "scripts" / "prepare-standalone-release.sh"
VALIDATE_STANDALONE = ROOT / "scripts" / "validate-standalone.sh"
MACHO_NAMES = (
    "libmlx.dylib",
    "libjaccl.dylib",
    "ax-engine",
    "ax-engine-server",
    "ax-engine-bench",
)


@unittest.skipUnless(sys.platform == "darwin", "standalone releases are macOS-only")
class StandaloneReleaseTests(unittest.TestCase):
    def run_command(
        self,
        *args: str | Path,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(arg) for arg in args],
            check=check,
            capture_output=True,
            text=True,
            env=env,
        )

    def dylib_linkage(self, path: Path) -> tuple[str, tuple[str, ...]]:
        install_name_lines = self.run_command("otool", "-D", path).stdout.splitlines()
        linked_lines = self.run_command("otool", "-L", path).stdout.splitlines()
        install_name = install_name_lines[1].strip()
        linked = tuple(line.strip().split()[0] for line in linked_lines[1:] if line.strip())
        return install_name, linked

    def build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        for tool in ("clang", "codesign", "install_name_tool", "lipo", "otool", "vtool"):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} is unavailable")

        source = root / "source"
        binaries = root / "binaries"
        site_packages = root / "site-packages"
        mlx_lib = site_packages / "mlx" / "lib"
        mlx_include = site_packages / "mlx" / "include" / "mlx"
        mlx_dist_info = site_packages / "mlx-0.32.1.dist-info"
        mlx_license = mlx_dist_info / "licenses"
        metal_dist_info = site_packages / "mlx_metal-0.32.1.dist-info"
        source.mkdir()
        binaries.mkdir()
        mlx_lib.mkdir(parents=True)
        mlx_include.mkdir(parents=True)
        mlx_license.mkdir(parents=True)
        metal_dist_info.mkdir(parents=True)

        (source / "jaccl.c").write_text("int jaccl_value(void) { return 1; }\n")
        (source / "mlx.c").write_text(
            "extern int jaccl_value(void);\nint mlx_value(void) { return jaccl_value() + 1; }\n"
        )
        (source / "main.c").write_text(
            "extern int mlx_value(void);\nint main(void) { return mlx_value() == 2 ? 0 : 1; }\n"
        )

        self.run_command(
            "clang",
            "-dynamiclib",
            "-mmacosx-version-min=15.0",
            source / "jaccl.c",
            "-Wl,-install_name,@rpath/libjaccl.dylib",
            "-o",
            mlx_lib / "libjaccl.dylib",
        )
        self.run_command(
            "clang",
            "-dynamiclib",
            "-mmacosx-version-min=15.0",
            source / "mlx.c",
            f"-L{mlx_lib}",
            "-ljaccl",
            "-Wl,-install_name,@rpath/libmlx.dylib",
            "-o",
            mlx_lib / "libmlx.dylib",
        )
        for name in ("ax-engine", "ax-engine-server", "ax-engine-bench"):
            self.run_command(
                "clang",
                "-mmacosx-version-min=15.0",
                source / "main.c",
                f"-L{mlx_lib}",
                "-lmlx",
                f"-Wl,-rpath,{mlx_lib}",
                "-Wl,-headerpad_max_install_names",
                "-o",
                binaries / name,
            )
        (mlx_lib / "mlx.metallib").write_bytes(b"test metallib")
        (mlx_include / "version.h").write_text(
            "#define MLX_VERSION_MAJOR 0\n"
            "#define MLX_VERSION_MINOR 32\n"
            "#define MLX_VERSION_PATCH 1\n"
        )
        (mlx_dist_info / "WHEEL").write_text("Tag: cp312-cp312-macosx_15_0_arm64\n")
        (metal_dist_info / "WHEEL").write_text("Tag: py3-none-macosx_15_0_arm64\n")
        (mlx_license / "LICENSE").write_text("MIT test fixture\n")
        return binaries, mlx_lib, site_packages

    def prepare_payload(self, root: Path) -> Path:
        binaries, mlx_lib, _ = self.build_fixture(root)
        runtime = root / "runtime"
        payload = root / "payload"
        self.run_command("bash", STAGE_RUNTIME, mlx_lib, runtime)
        for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib"):
            self.assertEqual((mlx_lib / name).read_bytes(), (runtime / name).read_bytes())
        self.run_command("bash", PREPARE_STANDALONE, binaries, runtime, payload)
        for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib"):
            self.assertEqual((runtime / name).read_bytes(), (payload / name).read_bytes())
        for name in MACHO_NAMES:
            self.run_command("codesign", "--force", "--sign", "-", payload / name)
        for name in ("libmlx.dylib", "libjaccl.dylib"):
            self.assertEqual(
                self.dylib_linkage(runtime / name),
                self.dylib_linkage(payload / name),
            )
        self.assertEqual(
            (runtime / "mlx.metallib").read_bytes(),
            (payload / "mlx.metallib").read_bytes(),
        )
        return payload

    def test_payload_runs_in_archive_and_homebrew_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self.prepare_payload(root)

            server_loads = self.run_command("otool", "-L", payload / "ax-engine-server").stdout
            mlx_loads = self.run_command("otool", "-L", payload / "libmlx.dylib").stdout
            server_commands = self.run_command("otool", "-l", payload / "ax-engine-server").stdout
            self.assertIn("@rpath/libmlx.dylib", server_loads)
            self.assertIn("@rpath/libjaccl.dylib", mlx_loads)
            self.assertIn("@loader_path", server_commands)
            self.assertIn("@loader_path/../libexec", server_commands)
            self.assertNotIn(str(root / "site-packages"), server_commands)

            self.run_command("bash", VALIDATE_STANDALONE, payload)

            prefix = root / "homebrew-prefix"
            bin_dir = prefix / "bin"
            libexec_dir = prefix / "libexec"
            bin_dir.mkdir(parents=True)
            libexec_dir.mkdir()
            for name in ("ax-engine", "ax-engine-server", "ax-engine-bench"):
                shutil.copy2(payload / name, bin_dir / name)
            for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib"):
                shutil.copy2(payload / name, libexec_dir / name)

            clean_env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(root),
                "TMPDIR": str(root),
            }
            self.run_command(bin_dir / "ax-engine-server", "--help", env=clean_env)

    def test_validator_rejects_incomplete_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self.prepare_payload(Path(tmp))
            (payload / "mlx.metallib").unlink()

            result = self.run_command(
                "bash",
                VALIDATE_STANDALONE,
                payload,
                check=False,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("required MLX runtime asset is missing", result.stderr)

    def test_runtime_staging_rejects_wrong_mlx_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mlx_lib, site_packages = self.build_fixture(root)
            version_header = site_packages / "mlx" / "include" / "mlx" / "version.h"
            version_header.write_text(
                "#define MLX_VERSION_MAJOR 0\n"
                "#define MLX_VERSION_MINOR 31\n"
                "#define MLX_VERSION_PATCH 0\n"
            )

            result = self.run_command(
                "bash",
                STAGE_RUNTIME,
                mlx_lib,
                root / "runtime",
                check=False,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("does not match repository pin", result.stderr)

    def test_runtime_staging_accepts_newer_dylib_minos_with_warning(self) -> None:
        # Upstream wheels can under-claim: dylib minos NEWER than the wheel
        # tag (mlx 0.32.0 shipped minos 26.2 under a macosx_26_0 tag). The
        # digest checks pin the exact bytes, so staging accepts this and
        # surfaces the real runtime floor instead of failing the release.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mlx_lib, site_packages = self.build_fixture(root)
            for distribution in ("mlx-0.32.1", "mlx_metal-0.32.1"):
                wheel = site_packages / f"{distribution}.dist-info" / "WHEEL"
                wheel.write_text(wheel.read_text().replace("macosx_15_0", "macosx_14_0"))

            result = self.run_command(
                "bash",
                STAGE_RUNTIME,
                mlx_lib,
                root / "runtime",
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("effective runtime floor is 15.0", result.stderr)

    def test_runtime_staging_rejects_dylib_minos_below_wheel_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mlx_lib, site_packages = self.build_fixture(root)
            for distribution in ("mlx-0.32.1", "mlx_metal-0.32.1"):
                wheel = site_packages / f"{distribution}.dist-info" / "WHEEL"
                wheel.write_text(wheel.read_text().replace("macosx_15_0", "macosx_16_0"))

            result = self.run_command(
                "bash",
                STAGE_RUNTIME,
                mlx_lib,
                root / "runtime",
                check=False,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("targets macOS 15.0; wheel requires 16.0", result.stderr)


if __name__ == "__main__":
    unittest.main()
