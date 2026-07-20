"""Regression tests for scripts/check-mlx-version.sh.

The admitted pip mlx pin currently ships LC_BUILD_VERSION minos 15.0 while
still embedding NAX kernels. An earlier gate treated minos < 26.2 as proof that
NAX was compiled out and hard-failed the official wheel — a false positive that
blocked a healthy repo .venv. The gate must accept pip wheels and refuse only
Homebrew-resolved dylibs (mirroring mlx-sys/build.rs).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check-mlx-version.sh"


def _extract_is_homebrew_mlx_path() -> str:
    """Pull the shipped classifier out of the gate script (no reimplementation)."""
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"\nis_homebrew_mlx_path\(\) \{\n.*?\n\}\n",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("is_homebrew_mlx_path not found in check-mlx-version.sh")
    return match.group(0)


class CheckMlxVersionPathTests(unittest.TestCase):
    """Unit-test the Homebrew path classifier shipped in the shell gate."""

    def _classify(self, path: str) -> int:
        # Disable the brew --prefix branch so pure path patterns are hermetic.
        script = (
            "brew() { return 1; }\n"
            + _extract_is_homebrew_mlx_path()
            + f'is_homebrew_mlx_path "{path}"; echo $?\n'
        )
        result = subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout.strip())

    def test_homebrew_cellar_and_opt_paths_are_rejected(self) -> None:
        self.assertEqual(
            self._classify("/opt/homebrew/opt/mlx/lib/libmlx.dylib"),
            0,
        )
        self.assertEqual(
            self._classify("/opt/homebrew/Cellar/mlx/0.32.0/lib/libmlx.dylib"),
            0,
        )
        self.assertEqual(
            self._classify("/usr/local/Cellar/mlx/0.31.2/lib/libmlx.dylib"),
            0,
        )

    def test_pip_site_packages_path_is_accepted(self) -> None:
        pip_path = (
            "/Users/dev/.venv/lib/python3.12/site-packages/mlx/lib/libmlx.dylib"
        )
        self.assertEqual(self._classify(pip_path), 1)


class CheckMlxVersionScriptTests(unittest.TestCase):
    """Drive the real shell entry point against the repo toolchain."""

    def test_script_passes_on_repo_venv_even_when_minos_is_below_26_2(self) -> None:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python3"
        if not venv_python.is_file():
            self.skipTest("repo .venv python not present")

        probe = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import mlx, pathlib; "
                "print(pathlib.Path(list(mlx.__path__)[0]) / 'lib' / 'libmlx.dylib')",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            self.skipTest(f"repo .venv cannot import mlx: {probe.stderr}")
        dylib = Path(probe.stdout.strip())
        if not dylib.is_file():
            self.skipTest(f"missing libmlx.dylib at {dylib}")

        minos = ""
        otool = subprocess.run(
            ["otool", "-l", str(dylib)],
            check=False,
            capture_output=True,
            text=True,
        )
        if otool.returncode == 0:
            match = re.search(
                r"LC_BUILD_VERSION.*?minos\s+(\S+)",
                otool.stdout,
                flags=re.DOTALL,
            )
            if match:
                minos = match.group(1)

        env = os.environ.copy()
        # Force the gate to use the repo venv interpreter under test.
        env.pop("PYTHON_BIN", None)

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        combined = result.stdout + result.stderr
        self.assertEqual(
            result.returncode,
            0,
            f"check-mlx-version failed on pip wheel (minos={minos or 'unknown'}):\n"
            f"{combined}",
        )
        self.assertIn("MLX toolchain OK", combined)
        # When the admitted wheel still ships minos < 26.2, prove we exercised
        # the regression (hard-fail on low minos would have failed above).
        if minos:
            major = int(minos.split(".")[0])
            if major < 26:
                self.assertIn("informational", combined)

    def test_script_rejects_empty_pin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mlx.version").write_text("\n", encoding="utf-8")
            scripts = root / "scripts"
            scripts.mkdir()
            # Minimal common.sh for AX_REPO_ROOT / AX_PYTHON_BIN.
            (scripts / "lib").mkdir()
            (scripts / "lib" / "common.sh").write_text(
                f'AX_REPO_ROOT="{root}"\n'
                f'AX_PYTHON_BIN="{REPO_ROOT / ".venv" / "bin" / "python3"}"\n'
                "ax_tmp_dir() { mktemp -d; }\n"
                "ax_rm_rf() { rm -rf \"$@\"; }\n"
                "ax_run_cleanup() { :; }\n",
                encoding="utf-8",
            )
            script_body = SCRIPT.read_text(encoding="utf-8")
            # Point the sourced common.sh at the fixture.
            fixture_script = scripts / "check-mlx-version.sh"
            fixture_script.write_text(script_body, encoding="utf-8")
            result = subprocess.run(
                ["bash", str(fixture_script)],
                check=False,
                capture_output=True,
                text=True,
                cwd=root,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mlx.version is empty", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
