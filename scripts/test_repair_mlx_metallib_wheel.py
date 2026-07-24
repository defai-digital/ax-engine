from __future__ import annotations

import base64
import csv
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

if __package__:
    from scripts.repair_mlx_metallib_wheel import (
        BUNDLED_LIBMLX,
        FINAL_METALLIB,
        STAGED_METALLIB,
        WheelRepairError,
        repair_wheel,
    )
else:
    from repair_mlx_metallib_wheel import (
        BUNDLED_LIBMLX,
        FINAL_METALLIB,
        STAGED_METALLIB,
        WheelRepairError,
        repair_wheel,
    )


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


REPO_ROOT = Path(__file__).resolve().parent.parent


class RepairMlxMetallibWheelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.wheel = Path(self.tempdir.name) / "ax_engine-6.12.0-cp310-abi3-macosx_26_0_arm64.whl"

    def _write_wheel(
        self,
        *,
        include_anchor: bool = True,
        include_staged: bool = True,
        include_final: bool = False,
        signed: bool = False,
        tampered_member: bool = False,
        duplicate_record_row: bool = False,
        unsafe_member: bool = False,
    ) -> None:
        members = {
            "ax_engine/__init__.py": b"VALUE = 1\n",
            "ax_engine-6.12.0.dist-info/METADATA": b"Name: ax-engine\nVersion: 6.12.0\n",
            "ax_engine-6.12.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        }
        if include_anchor:
            members[BUNDLED_LIBMLX] = b"libmlx"
        if include_staged:
            members[STAGED_METALLIB] = b"metal-library"
        if include_final:
            members[FINAL_METALLIB] = b"final-metal-library"
        if unsafe_member:
            members["../outside-wheel"] = b"unsafe"
        record_path = "ax_engine-6.12.0.dist-info/RECORD"
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        for name, data in members.items():
            writer.writerow((name, _record_hash(data), len(data)))
        if duplicate_record_row:
            data = members["ax_engine/__init__.py"]
            writer.writerow(("ax_engine/__init__.py", _record_hash(data), len(data)))
        writer.writerow((record_path, "", ""))
        with zipfile.ZipFile(self.wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                if tampered_member and name == "ax_engine/__init__.py":
                    data = b"VALUE = 2\n"
                archive.writestr(name, data)
            archive.writestr(record_path, output.getvalue())
            if signed:
                archive.writestr(f"{record_path}.jws", b"signature")

    def test_repairs_location_and_record_then_is_idempotent(self) -> None:
        self._write_wheel()

        self.assertTrue(repair_wheel(self.wheel))
        self.assertFalse(repair_wheel(self.wheel))

        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
            self.assertNotIn(STAGED_METALLIB, names)
            self.assertEqual(archive.read(FINAL_METALLIB), b"metal-library")
            record_path = "ax_engine-6.12.0.dist-info/RECORD"
            rows = {
                row[0]: (row[1], row[2])
                for row in csv.reader(io.TextIOWrapper(archive.open(record_path), encoding="utf-8"))
            }
            self.assertEqual(
                rows[FINAL_METALLIB],
                (_record_hash(b"metal-library"), str(len(b"metal-library"))),
            )
            self.assertNotIn(STAGED_METALLIB, rows)
            self.assertEqual(rows[record_path], ("", ""))

    def test_requires_delocated_libmlx_anchor(self) -> None:
        self._write_wheel(include_anchor=False)
        with self.assertRaisesRegex(WheelRepairError, "missing ax_engine/.dylibs/libmlx"):
            repair_wheel(self.wheel)

    def test_rejects_ambiguous_metallib_members(self) -> None:
        self._write_wheel(include_final=True)
        with self.assertRaisesRegex(WheelRepairError, "both temporary and final"):
            repair_wheel(self.wheel)

    def test_rejects_signed_record(self) -> None:
        self._write_wheel(signed=True)
        with self.assertRaisesRegex(WheelRepairError, "signed RECORD"):
            repair_wheel(self.wheel)

    def test_rejects_tampered_member_without_rewriting_wheel(self) -> None:
        self._write_wheel(tampered_member=True)
        original = self.wheel.read_bytes()

        with self.assertRaisesRegex(WheelRepairError, "digest or size mismatch"):
            repair_wheel(self.wheel)

        self.assertEqual(self.wheel.read_bytes(), original)

    def test_rejects_duplicate_record_paths(self) -> None:
        self._write_wheel(duplicate_record_row=True)
        with self.assertRaisesRegex(WheelRepairError, "duplicate path"):
            repair_wheel(self.wheel)

    def test_rejects_unsafe_member_paths(self) -> None:
        self._write_wheel(unsafe_member=True)
        with self.assertRaisesRegex(WheelRepairError, "unsafe member path"):
            repair_wheel(self.wheel)

    def test_rejects_symlinked_wheel(self) -> None:
        self._write_wheel()
        link = self.wheel.with_name("linked.whl")
        link.symlink_to(self.wheel)
        with self.assertRaisesRegex(WheelRepairError, "symlinked wheel"):
            repair_wheel(link)


class RepairMlxMetallibReleaseContractTests(unittest.TestCase):
    def test_release_build_repairs_before_final_asset_guards(self) -> None:
        script = (REPO_ROOT / "scripts" / "build-pypi-wheel.sh").read_text(encoding="utf-8")
        repair = 'python3 scripts/repair_mlx_metallib_wheel.py "$DELOCATED"'
        final_guard = 'verify_wheel_member "$DELOCATED" "ax_engine/.dylibs/mlx.metallib"'

        self.assertIn(repair, script)
        self.assertIn(final_guard, script)
        self.assertLess(script.index("delocate-wheel --require-archs arm64"), script.index(repair))
        self.assertLess(script.index(repair), script.index(final_guard))

    def test_repository_script_gate_runs_repair_regressions(self) -> None:
        gate = (REPO_ROOT / "scripts" / "check-scripts.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/repair_mlx_metallib_wheel.py", gate)
        self.assertIn("scripts/test_repair_mlx_metallib_wheel.py", gate)


if __name__ == "__main__":
    unittest.main()
