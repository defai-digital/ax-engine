#!/usr/bin/env python3
"""Place MLX's Metal library beside delocate's bundled libmlx.dylib.

Maturin cannot stage ``mlx.metallib`` directly in ``ax_engine/.dylibs``:
delocate rejects a wheel when its destination directory already exists.  The
build therefore stages the file in ``ax_engine.dylibs`` and runs this helper
after delocate.  The helper performs an atomic wheel rewrite and regenerates
the wheel ``RECORD`` so the installed distribution remains verifiable.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import os
import stat
import tempfile
import zipfile
from pathlib import Path

STAGED_METALLIB = "ax_engine.dylibs/mlx.metallib"
BUNDLED_DYLIB_DIR = "ax_engine/.dylibs"
BUNDLED_LIBMLX = f"{BUNDLED_DYLIB_DIR}/libmlx.dylib"
FINAL_METALLIB = f"{BUNDLED_DYLIB_DIR}/mlx.metallib"
RECORD_HASH_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})


class WheelRepairError(RuntimeError):
    """Raised when a wheel is not safe to repair."""


def _record_digest(digest: bytes) -> str:
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _record_bytes(rows: list[tuple[str, str, str]], record_path: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows([*rows, (record_path, "", "")])
    return stream.getvalue().encode("utf-8")


def _member_record_values(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, algorithm: str
) -> tuple[str, str]:
    digest = hashlib.new(algorithm)
    size = 0
    with archive.open(info) as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
    return f"{algorithm}={encoded}", str(size)


def _validate_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    path = name[:-1] if name.endswith("/") else name
    if (
        not path
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise WheelRepairError(f"wheel contains unsafe member path: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise WheelRepairError(f"wheel contains symbolic-link member: {name}")


def _verify_record(wheel: Path, members: list[zipfile.ZipInfo], record_path: str) -> None:
    files = {member.filename: member for member in members if not member.is_dir()}
    with zipfile.ZipFile(wheel) as archive:
        try:
            record_text = archive.read(record_path).decode("utf-8")
            parsed_rows = list(csv.reader(io.StringIO(record_text, newline=""), strict=True))
        except (UnicodeDecodeError, csv.Error) as error:
            raise WheelRepairError(f"wheel RECORD is not valid UTF-8 CSV: {error}") from error

        rows: dict[str, tuple[str, str]] = {}
        for index, row in enumerate(parsed_rows, start=1):
            if len(row) != 3 or not row[0]:
                raise WheelRepairError(f"wheel RECORD has malformed row {index}")
            if row[0] in rows:
                raise WheelRepairError(f"wheel RECORD has duplicate path: {row[0]}")
            rows[row[0]] = (row[1], row[2])

        missing = sorted(files.keys() - rows.keys())
        unexpected = sorted(rows.keys() - files.keys())
        if missing:
            raise WheelRepairError("wheel RECORD is missing members: " + ", ".join(missing))
        if unexpected:
            raise WheelRepairError(
                "wheel RECORD references absent members: " + ", ".join(unexpected)
            )
        if rows[record_path] != ("", ""):
            raise WheelRepairError("RECORD must not hash itself")

        for name, info in files.items():
            if name == record_path:
                continue
            recorded_hash, recorded_size = rows[name]
            algorithm, separator, encoded_digest = recorded_hash.partition("=")
            if separator != "=" or algorithm not in RECORD_HASH_ALGORITHMS or not encoded_digest:
                raise WheelRepairError(f"RECORD has unsupported hash for {name}")
            if (recorded_hash, recorded_size) != _member_record_values(archive, info, algorithm):
                raise WheelRepairError(f"RECORD digest or size mismatch for {name}")


def _copy_member(
    source: zipfile.ZipFile,
    destination: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    output_name: str,
) -> tuple[str, str, str] | None:
    output_info = copy.copy(info)
    output_info.filename = output_name
    if info.is_dir():
        destination.writestr(output_info, b"")
        return None

    digest = hashlib.sha256()
    size = 0
    with source.open(info, "r") as reader, destination.open(output_info, "w") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return output_name, _record_digest(digest.digest()), str(size)


def _wheel_members(wheel: Path) -> tuple[list[zipfile.ZipInfo], str]:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
    for member in members:
        _validate_member(member)
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise WheelRepairError("wheel contains duplicate ZIP member names")
    records = [name for name in names if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise WheelRepairError(f"wheel must contain exactly one RECORD; found {len(records)}")
    record_path = records[0]
    signature_paths = {f"{record_path}.jws", f"{record_path}.p7s"}
    found_signatures = sorted(signature_paths.intersection(names))
    if found_signatures:
        raise WheelRepairError(
            "refusing to invalidate signed RECORD: " + ", ".join(found_signatures)
        )
    if BUNDLED_LIBMLX not in names:
        raise WheelRepairError(f"delocated wheel is missing {BUNDLED_LIBMLX}")
    return members, record_path


def _verify_repaired_wheel(wheel: Path) -> None:
    members, record_path = _wheel_members(wheel)
    _verify_record(wheel, members, record_path)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if STAGED_METALLIB in names:
            raise WheelRepairError(f"temporary staged member remains: {STAGED_METALLIB}")
        if FINAL_METALLIB not in names:
            raise WheelRepairError(f"repaired wheel is missing {FINAL_METALLIB}")


def repair_wheel(wheel: Path) -> bool:
    """Repair ``wheel`` atomically; return whether a rewrite was required."""

    if wheel.is_symlink():
        raise WheelRepairError(f"refusing to replace symlinked wheel: {wheel}")
    wheel = wheel.resolve(strict=True)
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise WheelRepairError(f"not a wheel file: {wheel}")

    members, record_path = _wheel_members(wheel)
    _verify_record(wheel, members, record_path)
    names = {member.filename for member in members}
    staged = STAGED_METALLIB in names
    final = FINAL_METALLIB in names
    if staged and final:
        raise WheelRepairError(
            f"wheel contains both temporary and final metallib members: "
            f"{STAGED_METALLIB}, {FINAL_METALLIB}"
        )
    if not staged:
        if not final:
            raise WheelRepairError(f"wheel is missing staged member {STAGED_METALLIB}")
        _verify_repaired_wheel(wheel)
        return False

    original_mode = stat.S_IMODE(wheel.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{wheel.name}.", suffix=".tmp", dir=wheel.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)

        with (
            zipfile.ZipFile(wheel, "r") as source,
            zipfile.ZipFile(temporary_path, "w", allowZip64=True) as destination,
        ):
            record_info = next(info for info in members if info.filename == record_path)
            rows: list[tuple[str, str, str]] = []
            for info in members:
                if info.filename == record_path:
                    continue
                output_name = FINAL_METALLIB if info.filename == STAGED_METALLIB else info.filename
                row = _copy_member(source, destination, info, output_name=output_name)
                if row is not None:
                    rows.append(row)
            output_record_info = copy.copy(record_info)
            destination.writestr(output_record_info, _record_bytes(rows, record_path))

        os.chmod(temporary_path, original_mode)
        _verify_repaired_wheel(temporary_path)
        os.replace(temporary_path, wheel)
        temporary_path = None
        _verify_repaired_wheel(wheel)
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    changed = repair_wheel(args.wheel)
    print(f"{'repaired' if changed else 'verified'}: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
