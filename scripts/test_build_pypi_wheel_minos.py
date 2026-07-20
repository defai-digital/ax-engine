"""Regression: wheel minos floor must not reject vendored pip MLX dylibs.

``scripts/build-pypi-wheel.sh`` step 5c enforces ``MACOSX_DEPLOYMENT_TARGET``
on ax-engine product Mach-O only. Delocate vendors the admitted pip
``libmlx.dylib`` / ``libjaccl.dylib`` (currently minos 15.0 with NAX). A
blanket floor on every bundled Mach-O would hard-fail a correct wheel build.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WHEEL_SCRIPT = REPO_ROOT / "scripts" / "build-pypi-wheel.sh"


def _extract_function(name: str) -> str:
    text = WHEEL_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        rf"\n{re.escape(name)}\(\) \{{\n.*?\n\}}\n",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{name}() not found in build-pypi-wheel.sh")
    return match.group(0)


def _extract_version_ge() -> str:
    text = WHEEL_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"\nversion_ge\(\) \{\n.*?\n\}\n",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("version_ge() not found in build-pypi-wheel.sh")
    return match.group(0)


class ProductMachoClassifierTests(unittest.TestCase):
    """Drive the shipped ``is_ax_engine_product_macho`` classifier."""

    def _classify(self, rel: str) -> int:
        script = (
            _extract_function("is_ax_engine_product_macho")
            + f'is_ax_engine_product_macho "{rel}"; echo $?\n'
        )
        result = subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(result.stdout.strip())

    def test_product_bins_and_extension_are_enforced(self) -> None:
        self.assertEqual(self._classify("ax_engine/_bin/ax-engine-server"), 0)
        self.assertEqual(self._classify("ax_engine/_bin/ax-engine-bench"), 0)
        self.assertEqual(self._classify("ax_engine/_bin/ax-engine"), 0)
        self.assertEqual(
            self._classify("ax_engine/_ax_engine.abi3.so"),
            0,
        )
        self.assertEqual(self._classify("ax_engine/something.so"), 0)

    def test_vendored_mlx_dylibs_are_excluded(self) -> None:
        self.assertEqual(self._classify("ax_engine.dylibs/libmlx.dylib"), 1)
        self.assertEqual(self._classify("ax_engine.dylibs/libjaccl.dylib"), 1)
        self.assertEqual(self._classify("libmlx.dylib"), 1)
        self.assertEqual(self._classify("libjaccl.dylib"), 1)


class Step5cSimulatedInspectTests(unittest.TestCase):
    """Simulate step 5c against real pip dylibs + a product binary path."""

    def _pip_mlx_lib_dir(self) -> Path | None:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python3"
        if not venv_python.is_file():
            return None
        probe = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import mlx, pathlib; print(pathlib.Path(list(mlx.__path__)[0]) / 'lib')",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if probe.returncode != 0:
            return None
        lib = Path(probe.stdout.strip())
        if not (lib / "libmlx.dylib").is_file():
            return None
        return lib

    def _run_step5c_on_inspect_dir(self, inspect_dir: Path, floor: str) -> subprocess.CompletedProcess[str]:
        """Run the shipped classifier + version_ge loop (not a reimplementation)."""
        script = f"""
set -euo pipefail
MACOSX_DEPLOYMENT_TARGET="{floor}"
INSPECT_DIR="{inspect_dir}"
{_extract_version_ge()}
{_extract_function("is_ax_engine_product_macho")}
bad_binaries=()
skipped_vendor=()
while IFS= read -r -d '' f; do
    if file "$f" | grep -q "Mach-O"; then
        rel="${{f#"$INSPECT_DIR"/}}"
        minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/{{f=1}} f && /minos/{{print $2; exit}}')"
        [[ -z "$minos" ]] && continue
        if ! is_ax_engine_product_macho "$rel"; then
            if ! version_ge "$minos" "$MACOSX_DEPLOYMENT_TARGET"; then
                skipped_vendor+=("$rel (minos $minos, vendored)")
            fi
            continue
        fi
        if ! version_ge "$minos" "$MACOSX_DEPLOYMENT_TARGET"; then
            bad_binaries+=("$rel (minos $minos)")
        fi
    fi
done < <(find "$INSPECT_DIR" -type f -print0)
if [[ ${{#skipped_vendor[@]}} -gt 0 ]]; then
    echo "vendor_skipped=${{#skipped_vendor[@]}}"
    printf 'VENDOR %s\\n' "${{skipped_vendor[@]}}"
fi
if [[ ${{#bad_binaries[@]}} -gt 0 ]]; then
    echo "error: product binaries below floor"
    printf 'BAD %s\\n' "${{bad_binaries[@]}}"
    exit 1
fi
echo "product_ok=1"
"""
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_pip_mlx_vendored_below_floor_does_not_fail_step5c(self) -> None:
        lib = self._pip_mlx_lib_dir()
        if lib is None:
            self.skipTest("repo .venv pip mlx not available")

        mlx = lib / "libmlx.dylib"
        jaccl = lib / "libjaccl.dylib"
        minos = ""
        otool = subprocess.run(
            ["otool", "-l", str(mlx)],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(
            r"LC_BUILD_VERSION.*?minos\s+(\S+)",
            otool.stdout,
            flags=re.DOTALL,
        )
        if match:
            minos = match.group(1)
        # Prove we are on the low-minos pin when possible (regression condition).
        if minos and int(minos.split(".")[0]) >= 26:
            self.skipTest(f"pip libmlx minos is already {minos}; not the regression case")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dylibs = root / "ax_engine.dylibs"
            dylibs.mkdir(parents=True)
            shutil.copy2(mlx, dylibs / "libmlx.dylib")
            if jaccl.is_file():
                shutil.copy2(jaccl, dylibs / "libjaccl.dylib")
            # Vendor-only tree: product binaries are covered by
            # test_low_minos_product_binary_still_fails. Including a local
            # debug/release binary here would conflate its own minos with the
            # vendored-MLX regression.

            result = self._run_step5c_on_inspect_dir(root, "26.2")
            combined = result.stdout + result.stderr
            self.assertEqual(
                result.returncode,
                0,
                f"step 5c rejected vendored pip MLX (minos={minos}):\n{combined}",
            )
            self.assertIn("product_ok=1", combined)
            self.assertIn("libmlx.dylib", combined)
            self.assertIn("vendor_skipped=", combined)

    def test_low_minos_product_binary_still_fails(self) -> None:
        """Floor still protects ax-engine product Mach-O."""
        lib = self._pip_mlx_lib_dir()
        if lib is None:
            self.skipTest("repo .venv pip mlx not available")
        # Reuse a known low-minos Mach-O as a stand-in product binary.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            product = root / "ax_engine" / "_bin"
            product.mkdir(parents=True)
            shutil.copy2(lib / "libmlx.dylib", product / "ax-engine-server")
            result = self._run_step5c_on_inspect_dir(root, "26.2")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("product binaries below floor", result.stdout + result.stderr)
            self.assertIn("ax-engine-server", result.stdout + result.stderr)


class ScriptSourceContractTests(unittest.TestCase):
    def test_step5c_comments_exclude_vendored_mlx(self) -> None:
        text = WHEEL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("is_ax_engine_product_macho", text)
        self.assertIn("libmlx.dylib", text)
        self.assertIn("vendored", text.lower())
        # Must not reintroduce a blanket "all bundled Mach-O" floor message.
        self.assertNotIn(
            "all bundled Mach-O binaries have minos",
            text,
        )


if __name__ == "__main__":
    unittest.main()
